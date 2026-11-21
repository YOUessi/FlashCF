"""Replay native BF16 + real additive 4D Qwen2.5-VL Attention fixtures on Tang.

The pretrained Qwen capture stores GQA-expanded K/V. Collapse every Nth
head to *native* [B,Hkv,L,D], then run the native-GQA Triton kernel without
repeat_interleave inside it. Check original pinned CFPO helper, dense native
BF16 reference, Triton BF16 forward and Q/K/V gradients separately.

Raw fixtures remain only on Tang. This script emits sanitized JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_bf16 import flashcf_native,forward_native
from tools.p0_official_cmve_oracle import official_dense_cfpo,UPSTREAM_COMMIT


def run_one(path,source):
    item=torch.load(path,map_location="cpu",weights_only=True)
    group=int(item["gqa_repeat"])
    q=item["q"].to("cuda").contiguous().detach().requires_grad_(True)
    krep=item["k"].to("cuda").contiguous()
    vrep=item["v"].to("cuda").contiguous()
    k=krep[:,::group].contiguous().detach().requires_grad_(True)
    v=vrep[:,::group].contiguous().detach().requires_grad_(True)
    B,H,L,D=q.shape
    assert H//k.shape[1]==group
    if not torch.equal(k.repeat_interleave(group,1),krep) or not torch.equal(v.repeat_interleave(group,1),vrep):
        raise AssertionError("Captured Qwen GQA expansion is not regular repetition")
    mask=item["additive_mask"].to("cuda").contiguous()
    spans=item["image_spans"]
    meta=dense_native_cmve(q,k,v,spans,mask,thres_mode="high",mean_mode="image",return_metadata=True)
    expected,details=meta
    # Official source expects repeated heads, as done in the published CFPO adapter.
    source_output,_,official_gates,_=official_dense_cfpo(
        q,k.repeat_interleave(group,1),v.repeat_interleave(group,1),mask,spans,
        thres_mode="high",mean_mode="image",source=source)
    exact_source_err=(expected.float()-source_output.float()).abs().max().item()
    official_mask_equal=bool(torch.equal(details["cmve_mask"],official_gates))
    actual=flashcf_native(q,k,v,spans,mask,thres_mode="high",mean_mode="image",block_k=32)
    weight=torch.randn_like(actual)
    gref=torch.autograd.grad((expected*weight).sum(),(q,k,v),retain_graph=True)
    gtri=torch.autograd.grad((actual*weight).sum(),(q,k,v))
    diff=actual.float()-expected.float()
    grad_diffs=[(a.float()-b.float()) for a,b in zip(gtri,gref)]
    result={
      "layer":int(item["layer"]), "original_dtype":str(q.dtype),
      "shape_q":[B,H,L,D], "shape_k_native":list(k.shape),
      "gqa_group_size":group, "mask_shape":list(mask.shape),
      "image_spans":spans,
      "exact_upstream_source_sha":UPSTREAM_COMMIT,
      "dense_vs_original_bf16_max_abs":exact_source_err,
      "original_mask_equal_dense":official_mask_equal,
      "triton_bf16_max_abs":diff.abs().max().item(),
      "triton_bf16_rmse":float(diff.square().mean().sqrt()),
      "triton_bf16_relative_l2":float(torch.linalg.vector_norm(diff)/
                                      torch.linalg.vector_norm(expected.float()).clamp(min=1)),
      "triton_grad_q_max_abs":grad_diffs[0].abs().max().item(),
      "triton_grad_k_native_max_abs":grad_diffs[1].abs().max().item(),
      "triton_grad_v_native_max_abs":grad_diffs[2].abs().max().item(),
      "finite":bool(torch.isfinite(actual).all() and all(torch.isfinite(x).all() for x in gtri)),
    }
    result["pass_forward_relative_l2_under_2pct"]=result["triton_bf16_relative_l2"]<0.02
    result["pass_gradient_max_abs_under_0_5"]=max(
        result["triton_grad_q_max_abs"],result["triton_grad_k_native_max_abs"],
        result["triton_grad_v_native_max_abs"])<0.5
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixture-dir",type=Path,
                   default=Path.home()/".cache/flashcf_fixtures/qwen2_5_vl_3b")
    p.add_argument("--original-source",type=Path,
                   default=Path.home()/".cache/flashcf_upstream/qwen2_vl_cmve_98ca5a.py")
    p.add_argument("--summary",type=Path,
                   default=Path.home()/".cache/flashcf_results/p1_native_bf16_parity_local.json")
    args=p.parse_args()
    torch.manual_seed(20261121)
    results=[run_one(args.fixture_dir/f"native_qwen_layer_{i:02d}.pt",args.original_source)
             for i in [0,8]]
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps({
        "status":"FINISHED","torch":torch.__version__,
        "gpu":torch.cuda.get_device_name(0),"real_pretrained_fixture":True,
        "BF16_operator_compared_to_original":True,
        "GQA_native_no_repeat_in_Triton":True,
        "Triton_native_BF16_full_model_training":False,
        "results":results},indent=2)+"\n")
    if not all(x["pass_forward_relative_l2_under_2pct"] and x["pass_gradient_max_abs_under_0_5"] and
               x["finite"] and x["original_mask_equal_dense"] and x["dense_vs_original_bf16_max_abs"]==0
               for x in results):
        raise SystemExit("Real pretrained QKV parity gate failed")
