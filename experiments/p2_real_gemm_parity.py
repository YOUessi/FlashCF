"""P2 real-pretrained Qwen BF16 QKV parity: full Tensor Core Q/K/V tiled kernel.

Uses local Tang fixtures captured previously from pinned Qwen2.5-VL-3B.
Checks the original hard-gate tensor, forward, dQ/dK/dV relative-L2,
and quantized threshold. No full model loading or optimizer step.
"""
from __future__ import annotations
import argparse,json,os,torch
from pathlib import Path
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_qtile import flashcf_gemm_all
from kernels.triton_native_debug import diagnostic_triton_gate


def real_layer(path):
    t=torch.load(path,map_location="cpu",weights_only=True)
    group=int(t["gqa_repeat"])
    q=t["q"].cuda().contiguous().detach().requires_grad_(True)
    k=t["k"][:,::group].cuda().contiguous().detach().requires_grad_(True)
    v=t["v"][:,::group].cuda().contiguous().detach().requires_grad_(True)
    mask=t["additive_mask"].cuda().contiguous()
    spans=t["image_spans"]
    torch.manual_seed(20261121+int(t["layer"]))
    dy=torch.randn_like(q)
    yref,meta=dense_native_cmve(q,k,v,spans,mask,
                                 thres_mode="high",mean_mode="image",
                                 return_metadata=True)
    gout=torch.autograd.grad((yref*dy).sum(),(q,k,v))
    y=flashcf_gemm_all(q,k,v,spans,mask,
                       thres_mode="high",mean_mode="image",
                       row_tile=16,key_tile=64,grad_q_tile=16,grad_k_tile=64)
    gtr=torch.autograd.grad((y*dy).sum(),(q,k,v))
    gate,state=diagnostic_triton_gate(q.detach(),k.detach(),v.detach(),
                                     spans,mask,thres_mode="high",
                                     mean_mode="image",row_tile=16,key_tile=64)
    original_gate=meta["cmve_mask"].bool()
    mask_flips=int((original_gate!=gate).sum())
    def errors(x,r):
        dx=x.float()-r.float()
        return {"max_abs":dx.abs().max().item(),
                "rel_l2":(torch.linalg.vector_norm(dx)/
                         torch.linalg.vector_norm(r.float()).clamp(min=1)).item(),
                "rmse":dx.square().mean().sqrt().item()}
    return {
        "layer":int(t["layer"]),"shape":list(q.shape),
        "native_kv_shape":list(k.shape),"gqa_ratio":group,
        "mask_shape":list(mask.shape),
        "reference_threshold":float(meta["thresholds"][0][3]),
        "tensor_core_threshold":float(state["thresholds"][0]),
        "reference_gate_count":int(original_gate.sum()),
        "tensor_core_gate_count":int(gate.sum()),
        "mask_flips":mask_flips,
        "output":errors(y,yref),
        "gradients":{name:errors(actual,reference) for name,actual,reference
                     in zip(("q","k","v"),gtr,gout)},
        "pass":mask_flips==0,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fixtures",type=Path,
                   default=Path.home()/".cache"/"flashcf_fixtures"/"qwen2_5_vl_3b")
    p.add_argument("--output",type=Path,default=Path.home()/".cache"/"flashcf_results"/"p2_real_gemm_all_local.json")
    args=p.parse_args()
    results=[real_layer(args.fixtures/f"native_qwen_layer_{l:02d}.pt") for l in (0,8)]
    payload={"repo_commit":os.environ.get("FLASHCF_SOURCE_COMMIT"),
             "gpu":torch.cuda.get_device_name(0),
             "model":"Qwen/Qwen2.5-VL-3B-Instruct",
             "model_revision":"66285546d2b821cf421d4f5eb2576359d3770cd3",
             "precision":"BF16","source":"real pretrained post-MRoPE QKV captured earlier",
             "note":"single isolated operator; NOT full patched CFPO RL training",
             "layers":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2),flush=True)
    assert all(x["pass"] for x in results), "hard-mask drift detected"
