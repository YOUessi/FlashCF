"""Sweep a small, reproducible BF16 P2 Tensor Core tiling grid on Tang.

This is an architecture-specific *measurement* (Ada RTX4090 Laptop),
not a universal hard-coded tiling rule. Use candidate configs only after
re-verifying dense forward and input-gradient parity.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import torch
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_qtile import flashcf_gemm_all
from experiments.benchmark_native_bf16 import measure

CONFIGS=[
 (16,64,16,64),
 (32,64,16,64),
 (16,128,16,64),
 (32,128,32,64),
 (16,64,32,64),
 (16,64,16,128),
 (32,64,32,128),
 (16,128,32,128),
 (32,32,32,32),
 (16,32,16,32),
 (32,64,16,128),
 (16,64,32,128),
]


def run(L,reps):
    torch.manual_seed(651+L)
    B,Hq,Hkv,D=1,16,2,128
    q,k,v=[torch.randn((B,h,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
           for h in (Hq,Hkv,Hkv)]
    weights=torch.randn_like(q)
    mask=torch.zeros((B,1,L,L),device="cuda",dtype=torch.bfloat16)
    mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],
                      torch.finfo(mask.dtype).min)
    spans=[[(L//8,L//2)]]
    opts=dict(thres_mode="high",mean_mode="image")
    ref=dense_native_cmve(q,k,v,spans,mask,**opts)
    gr=torch.autograd.grad((ref*weights).sum(),(q,k,v))
    rows=[]
    for row,key,qgrad,kgrad in CONFIGS:
        fn=lambda a,b,c:flashcf_gemm_all(a,b,c,spans,mask,row_tile=row,
                     key_tile=key,grad_q_tile=qgrad,grad_k_tile=kgrad,
                     block_k=32,**opts)
        config={"row_tile":row,"key_tile":key,"grad_q_tile":qgrad,"grad_k_tile":kgrad}
        try:
            y=fn(q,k,v)
            g=torch.autograd.grad((y*weights).sum(),(q,k,v))
            ferr=(torch.linalg.vector_norm((y-ref).float())/
                  torch.linalg.vector_norm(ref.float()).clamp(min=1)).item()
            gerr=max((torch.linalg.vector_norm((a-b).float())/
                      torch.linalg.vector_norm(b.float()).clamp(min=1)).item() for a,b in zip(g,gr))
            if ferr>0.01 or gerr>0.03:
                raise RuntimeError(f"Parity gate: forward rel {ferr}, grads {gerr}")
            ms,mem=measure(fn,(q,k,v),weights,reps)
            config.update({"status":"PASS","median_step_ms":ms,
                           "peak_extra_mib":mem,
                           "forward_relative_error":ferr,"max_grad_relative_error":gerr})
        except Exception as e:
            config.update({"status":"FAIL","error":type(e).__name__+": "+str(e)[:180]})
        print("TILE_SWEEP",json.dumps({"L":L,**config}),flush=True)
        rows.append(config)
    winners=sorted((x for x in rows if x["status"]=="PASS"),key=lambda x:x["median_step_ms"])
    return {"L":L,"configs":rows,"top3":winners[:3]}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",nargs="+",type=int,default=[512,1024])
    p.add_argument("--repeat",type=int,default=5)
    p.add_argument("--output",type=Path,
                   default=Path.home()/".cache"/"flashcf_results"/"p2_tile_sweep_local.json")
    args=p.parse_args()
    results=[run(L,args.repeat) for L in args.lengths]
    payload={"source_commit":os.environ.get("FLASHCF_SOURCE_COMMIT","immutable snapshot"),
             "device":torch.cuda.get_device_name(),"torch":torch.__version__,
             "note":"synthetic BF16, single causal image span, small grid, one Tang GPU; tuning not universally optimal",
             "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print("WINNERS",json.dumps([{ "L":x["L"],"top3":x["top3"]} for x in results]),flush=True)
