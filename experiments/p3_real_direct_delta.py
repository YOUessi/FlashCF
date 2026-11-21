"""P3: real pretrained Qwen BF16 QKV parity of direct correction-only CMVE.

Real native-GQA fixtures are stored only on Tang. Compare the operator
to the forced FlashAttention factual plus original explicit CMVE branch,
including isolated input Q/K/V gradients and output relative L2.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from reference.native_sdpa_cmve import native_sdpa_cmve
from kernels.triton_cmve_delta import flashcf_factual_plus_delta,flashcf_cmve_delta
from kernels.hybrid_flash_factual import hybrid_flash_factual


def evaluate_fixture(path):
    data=torch.load(path,map_location="cpu",weights_only=True)
    group=int(data["gqa_repeat"])
    q=data["q"].to("cuda").contiguous().detach().requires_grad_(True)
    k=data["k"][:,::group].to("cuda").contiguous().detach().requires_grad_(True)
    v=data["v"][:,::group].to("cuda").contiguous().detach().requires_grad_(True)
    mask=data["additive_mask"].to("cuda").contiguous()
    spans=data["image_spans"]
    opts=dict(thres_mode="high",mean_mode="image",factual_backend="flash_causal")
    baseline=native_sdpa_cmve(q,k,v,spans,mask,**opts)
    direct=flashcf_factual_plus_delta(q,k,v,spans,mask,**opts)
    hybrid=hybrid_flash_factual(q,k,v,spans,mask,
                                thres_mode="high",mean_mode="image")
    torch.manual_seed(20261121+int(data["layer"]))
    direction=torch.randn_like(q)
    refgrads=torch.autograd.grad((baseline*direction).sum(),(q,k,v),retain_graph=True)
    record={"layer":int(data["layer"]),"shape":list(q.shape),
            "native_kv_shape":list(k.shape),"GQA_repeat":group,
            "comparison":"Forced Flash factual + original explicit BF16 CMVE"}
    for name,y in (("direct",direct),("hybrid_diagnostic",hybrid)):
        gs=torch.autograd.grad((y*direction).sum(),(q,k,v),retain_graph=True)
        diff=(y-baseline).float()
        norms=[(torch.linalg.vector_norm((a-b).float())/
                torch.linalg.vector_norm(b.float()).clamp(min=1)).item()
               for a,b in zip(gs,refgrads)]
        record[name]={
            "output_max_abs":float(diff.abs().max()),
            "output_rel_l2":float(torch.linalg.vector_norm(diff)/
                                  torch.linalg.vector_norm(baseline.float()).clamp(min=1)),
            "grad_rel_l2":{"q":norms[0],"k":norms[1],"v":norms[2]}
        }
    print(json.dumps(record),flush=True)
    return record


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",type=Path,default=Path.home()/".cache"/"flashcf_fixtures"/"qwen2_5_vl_3b")
    p.add_argument("--output",type=Path,default=Path.home()/".cache"/"flashcf_results"/"p3_real_direct_delta.json")
    a=p.parse_args()
    results=[evaluate_fixture(a.fixtures/f"native_qwen_layer_{n:02}.pt") for n in (0,8)]
    payload={"model":"Qwen/Qwen2.5-VL-3B-Instruct","precision":"BF16",
             "gpu":torch.cuda.get_device_name(0),
             "git_source_commit":os.environ.get("FLASHCF_SOURCE_COMMIT","immutable snapshot"),
             "note":"Isolated QKV gradients; no full pretrained model or RL training in this test",
             "results":results}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2)+"\n")
