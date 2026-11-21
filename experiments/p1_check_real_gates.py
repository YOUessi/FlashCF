"""Compare each BF16 CFPO hard gate on actual Qwen2.5-VL-3B activations.

For debugging only: creates a [B,H,L,L] temporary bool tensor to count
hard-mask disagreements between CFPO dense reference and the native Triton.
No diagnostic tensors are uploaded; sanitized numeric aggregates only.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_debug import diagnostic_triton_gate
from kernels.triton_native_bf16 import flashcf_native
from kernels.triton_native_backward_v2 import flashcf_native_v2


def run(path):
    item=torch.load(path,map_location="cpu",weights_only=True)
    ratio=int(item["gqa_repeat"])
    q=item["q"].cuda().contiguous().detach().requires_grad_(True)
    k=item["k"][:,::ratio].cuda().contiguous().detach().requires_grad_(True)
    v=item["v"][:,::ratio].cuda().contiguous().detach().requires_grad_(True)
    mask=item["additive_mask"].cuda().contiguous()
    spans=item["image_spans"]
    yref,meta=dense_native_cmve(q,k,v,spans,mask,return_metadata=True,
                                 thres_mode="high",mean_mode="image")
    gates,state=diagnostic_triton_gate(q.detach(),k.detach(),v.detach(),
                                       spans,mask,thres_mode="high",
                                       mean_mode="image")
    gt=gates!=meta["cmve_mask"].bool()
    yt=flashcf_native(q,k,v,spans,mask,thres_mode="high",mean_mode="image")
    generator=torch.Generator(device="cuda").manual_seed(20261121)
    weights=torch.randn(yt.shape,device="cuda",dtype=yt.dtype,generator=generator)
    gref=torch.autograd.grad((yref*weights).sum(),(q,k,v))
    gtr=torch.autograd.grad((yt*weights).sum(),(q,k,v))
    yv2=flashcf_native_v2(q,k,v,spans,mask,thres_mode="high",mean_mode="image")
    gv2=torch.autograd.grad((yv2*weights).sum(),(q,k,v))
    def error(a,b):
        a=a.float(); b=b.float()
        d=a-b
        return dict(max_abs=d.abs().max().item(),
                    rel_l2=(torch.linalg.vector_norm(d)/
                            torch.linalg.vector_norm(b).clamp(min=1)).item())
    t0=meta["thresholds"][0][3] if meta["thresholds"] else None
    t1=state["thresholds"][0].item() if state["num_spans"] else None
    return dict(layer=int(item["layer"]),shape=list(q.shape),gqa_ratio=ratio,
        image_spans=spans,
        reference_gate_count=int(meta["cmve_mask"].bool().sum().item()),
        triton_gate_count=int(gates.sum().item()),
        mask_flipped_positions=int(gt.sum().item()),
        mask_flip_fraction=gt.float().mean().item(),
        original_threshold=t0,triton_threshold=t1,
        output=error(yt,yref),
        gradients={name:error(a,b) for name,a,b in zip(("q","k_native","v_native"),gtr,gref)},
        gradients_v2={name:error(a,b) for name,a,b in zip(("q","k_native","v_native"),gv2,gref)},
        output_v2=error(yv2,yref))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dir",type=Path,default=Path.home()/".cache/flashcf_fixtures/qwen2_5_vl_3b")
    p.add_argument("--output",type=Path,default=Path.home()/".cache/flashcf_results/p1_real_gate_diag_local.json")
    args=p.parse_args()
    data=[run(args.dir/f"native_qwen_layer_{i:02}.pt") for i in (0,8)]
    print(json.dumps(data,indent=2),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+"\n")
