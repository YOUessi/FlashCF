"""Native BF16/GQA/real 4D-mask gradient and speed stress on Tang.

Compares genuine [B,Hq,L,D] / [B,Hkv,L,D] BF16 input to dense
Qwen/CFPO-style eager reference, without expanding KV in the Triton call.
Outputs absolute and relative errors, not just overly permissive tolerances.

This is a synthetic operator-level benchmark, NOT a full CFPO RL benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import subprocess
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.native_bf16_gqa import dense_native_cmve
from kernels.triton_native_bf16 import flashcf_native


def measure(fn,parts,weights,repeats=8):
    for _ in range(3):
        y=fn(*parts)
        grads=torch.autograd.grad((y*weights).sum(),parts)
        del y,grads
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_mem=torch.cuda.memory_allocated()
    records=[]
    for _ in range(repeats):
        a=torch.cuda.Event(enable_timing=True)
        b=torch.cuda.Event(enable_timing=True)
        a.record()
        y=fn(*parts)
        g=torch.autograd.grad((y*weights).sum(),parts)
        b.record()
        b.synchronize()
        records.append(a.elapsed_time(b))
        del y,g
    return round(float(torch.tensor(records).median()),4),round(
        (torch.cuda.max_memory_allocated()-start_mem)/2**20,3)


def run_one(L,Hq,Hkv,D,scenario,seed,repeat):
    torch.manual_seed(seed)
    q,k,v=[
        torch.randn((1,h,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
        for h in (Hq,Hkv,Hkv)
    ]
    mask=torch.zeros((1,1,L,L),device="cuda",dtype=torch.bfloat16)
    triangle=torch.ones((L,L),device="cuda",dtype=torch.bool).triu(1)
    mask.masked_fill_(triangle[None,None],torch.finfo(mask.dtype).min)
    if scenario=="padding":
        mask[:,:,:,int(L*.86):]=torch.finfo(mask.dtype).min
    elif scenario=="packing":
        split=L//2
        mask[:,:,:split,split:]=torch.finfo(mask.dtype).min
        mask[:,:,split:,:split]=torch.finfo(mask.dtype).min
    elif scenario=="causal":
        pass
    else:raise ValueError(scenario)
    if scenario=="packing":
        spans=[[(max(1,L//8),max(2,L//4)),(L//2+3,3*L//4)]]
    else:
        spans=[[(max(1,L//8),L//2)]]
    args={"thres_mode":"high","mean_mode":"image"}
    dense=lambda q,k,v:dense_native_cmve(q,k,v,spans,mask,**args)
    native=lambda q,k,v:flashcf_native(q,k,v,spans,mask,block_k=32,**args)
    wt=torch.randn_like(q)
    y0=dense(q,k,v)
    y1=native(q,k,v)
    gd=torch.autograd.grad((y0*wt).sum(),(q,k,v))
    gt=torch.autograd.grad((y1*wt).sum(),(q,k,v))
    def error(a,b):
        a=a.float()
        b=b.float()
        dif=a-b
        return {"max_abs":dif.abs().max().item(),
                "rmse":dif.square().mean().sqrt().item(),
                "rel_l2":(torch.linalg.vector_norm(dif)/
                          torch.linalg.vector_norm(b).clamp(min=1)).item()}
    ferror=error(y1,y0)
    errors={name:error(a,b) for name,a,b in zip(("q","k","v"),gt,gd)}
    t0,m0=measure(dense,(q,k,v),wt,repeat)
    t1,m1=measure(native,(q,k,v),wt,repeat)
    result={"sequence_length":L,"Hq":Hq,"Hkv":Hkv,"D":D,"scenario":scenario,
            "fp16_or_bf16":"BF16","precision":"BF16 QKV, FP32 tile reductions",
            "forward":ferror,"gradient":errors,
            "dense_step_ms":t0,"triton_step_ms":t1,
            "dense_peak_extra_MiB":m0,"triton_peak_extra_MiB":m1,
            "triton_vs_dense_speedup":round(t0/t1,3)}
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",type=int,nargs="+",default=[97,256,512])
    p.add_argument("--repeat",type=int,default=6)
    p.add_argument("--summary",type=Path,
                   default=Path.home()/".cache"/"flashcf_results"/"p1_native_stress_local.json")
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    results=[]
    for l in args.lengths:
        for scenario in ("causal","packing","padding") if l<=256 else ("causal",):
            results.append(run_one(l,16,2,128,scenario,20261121+l,args.repeat))
    run={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip() if (ROOT/".git").exists() else "immutable source snapshot",
         "torch":torch.__version__,"gpu":torch.cuda.get_device_name(0),
         "model_weights_used":False,
         "note":"BF16 and native GQA synthetic CUDA operator benchmark, not CFPO RL.",
         "results":results}
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(run,indent=2)+"\n")
