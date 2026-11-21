"""Measure forward+backward of FlashCF Triton vs *non-optimized* dense PyTorch.

Not an apples-to-apples optimized FlashAttention baseline. Synthetic FP32,
causal attention, no dropout, one image region. GPU events and peak allocations.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P))
from reference.cfpo_reference import dense_cfpo
from kernels.triton_backward import flashcf_autograd


def measure(method, q,k,v, spans, weight, n):
    for _ in range(2):
        y=method(q,k,v,spans)
        grad=torch.autograd.grad((y*weight).sum(),(q,k,v))
        del y,grad
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated=torch.cuda.memory_allocated()
    times=[]
    for _ in range(n):
        begin=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        begin.record()
        y=method(q,k,v,spans)
        gq,gk,gv=torch.autograd.grad((y*weight).sum(),(q,k,v))
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end))
        del y,gq,gk,gv
    peak=torch.cuda.max_memory_allocated()-allocated
    return float(torch.tensor(times).median()), round(peak/(2**20),3)


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--lengths",nargs="+",type=int,default=[64,128,256])
    arg.add_argument("--repeat",type=int,default=6)
    arg.add_argument("--output",type=Path,default=None)
    args=arg.parse_args()
    assert torch.cuda.is_available()
    results=[]
    for L in args.lengths:
        B,H,D=1,4,64
        torch.manual_seed(2026+L)
        q,k,v=[torch.randn((B,H,L,D),device="cuda",dtype=torch.float32,
                          requires_grad=True) for _ in range(3)]
        weight=torch.randn_like(q)
        spans=[[(max(1,L//8),L//2)]]
        kwargs=dict(thres_mode="medium",mean_mode="image")
        baseline=lambda q,k,v,spans:dense_cfpo(q,k,v,spans,**kwargs)
        candidate=lambda q,k,v,spans:flashcf_autograd(q,k,v,spans,block_k=32,**kwargs)
        y0=baseline(q,k,v,spans); y1=candidate(q,k,v,spans)
        gd=torch.autograd.grad((y0*weight).sum(),(q,k,v),retain_graph=False)
        gc=torch.autograd.grad((y1*weight).sum(),(q,k,v))
        oe=(y0-y1).abs().max().item()
        ge=max((a-b).abs().max().item() for a,b in zip(gd,gc))
        if oe>2e-4 or ge>1e-3:raise AssertionError(f"L={L} err=({oe},{ge})")
        dt,dp=measure(baseline,q,k,v,spans,weight,args.repeat)
        kt,kp=measure(candidate,q,k,v,spans,weight,args.repeat)
        row={"length":L,"heads":H,"head_dim":D,
             "forward_max_abs":oe,"gradient_max_abs":ge,
             "dense_train_step_ms":dt,"flashcf_train_step_ms":kt,
             "dense_peak_extra_mib":dp,"flashcf_peak_extra_mib":kp,
             "speedup_dense_over_flashcf":round(dt/kt,3)}
        print(json.dumps(row),flush=True)
        results.append(row)
    meta={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
          "git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=P,text=True).strip(),
          "caveat":"dense PyTorch reference not optimized. FlashCF uses atomic gradient prototype, no production benchmark.",
          "results":results}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(meta,indent=2)+"\n")
    print("DONE",flush=True)
if __name__=="__main__":main()
