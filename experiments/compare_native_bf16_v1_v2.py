"""Benchmark BF16 native-GQA backward V1 vs V2 on Tang.

Synthetic full-prefill, 4D masked attention, and the actual Qwen-style shape
(Hq=16, Hkv=2, head_dim=128). Measures complete forward+backward, not model RL.
V2 is expected to be more numerically faithful but may cost more QK recompute.
"""
import argparse,json
from pathlib import Path
import sys,torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_bf16 import flashcf_native
from kernels.triton_native_backward_v2 import flashcf_native_v2
from experiments.benchmark_native_bf16 import measure


def run(L,repeats):
    torch.manual_seed(20261121+L)
    B,Hq,Hkv,D=1,16,2,128
    q,k,v=[torch.randn((B,h,L,D),device="cuda",dtype=torch.bfloat16,
                        requires_grad=True) for h in (Hq,Hkv,Hkv)]
    mask=torch.zeros((B,1,L,L),device="cuda",dtype=torch.bfloat16)
    mask.masked_fill_(torch.ones((L,L),device="cuda",dtype=torch.bool).triu(1)[None,None],
                      torch.finfo(mask.dtype).min)
    span=[[(L//8,L//2)]]
    kw=dict(thres_mode="high",mean_mode="image")
    fns={
      "dense":lambda q,k,v:dense_native_cmve(q,k,v,span,mask,**kw),
      "v1":lambda q,k,v:flashcf_native(q,k,v,span,mask,block_k=32,**kw),
      "v2":lambda q,k,v:flashcf_native_v2(q,k,v,span,mask,block_k=32,**kw),
    }
    w=torch.randn_like(q)
    y0=fns["dense"](q,k,v)
    gd=torch.autograd.grad((y0*w).sum(),(q,k,v))
    row={"L":L,"Hq":Hq,"Hkv":Hkv,"D":D,"dtype":"BF16",
         "mask":"4D causal [1,1,L,L]", "image_span":span[0]}
    for name,fn in fns.items():
        out=fn(q,k,v)
        grads=torch.autograd.grad((out*w).sum(),(q,k,v))
        err=(out.float()-y0.float())
        rel=[(torch.linalg.vector_norm((a-b).float())/
              torch.linalg.vector_norm(b.float()).clamp(min=1)).item()
             for a,b in zip(grads,gd)]
        ms,mem=measure(fn,(q,k,v),w,repeats)
        row[name]={"median_full_step_ms":ms,
                   "peak_extra_mib":mem,
                   "max_output_abs":float(err.abs().max()),
                   "output_rel_l2":float(torch.linalg.vector_norm(err)/
                                         torch.linalg.vector_norm(y0.float()).clamp(min=1)),
                   "gradient_rel_l2":{"q":rel[0],"k":rel[1],"v":rel[2]}}
    print(json.dumps(row),flush=True)
    return row


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",nargs="+",type=int,default=[97,256,512])
    p.add_argument("--repeat",type=int,default=5)
    p.add_argument("--summary",type=Path,
                   default=Path.home()/".cache/flashcf_results/p1_v1_v2_bench_local.json")
    args=p.parse_args()
    data=[run(L,args.repeat) for L in args.lengths]
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps({"machine":"Tang RTX4090 Laptop","torch":torch.__version__,
        "precision":"BF16","benchmark":"CUDA events, synthetic, no full RL","results":data},indent=2)+"\n")
