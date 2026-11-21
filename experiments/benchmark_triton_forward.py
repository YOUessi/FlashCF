"""Forward performance + peak allocation probe on Tang; no model downloads.

Dense PyTorch baseline is the *reference*, not an optimized competing kernel.
All measurements are synthetic, CUDA events, warmed up, and exclude setup.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.cfpo_reference import dense_cfpo
from kernels.triton_forward import flashcf_forward


def measure(fn, repetitions=12):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before=torch.cuda.memory_allocated()
    times=[]
    for _ in range(repetitions):
        start=torch.cuda.Event(enable_timing=True)
        stop=torch.cuda.Event(enable_timing=True)
        start.record()
        output=fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
        del output
    peak=torch.cuda.max_memory_allocated()-before
    arr=torch.tensor(times)
    return float(torch.median(arr)),int(peak)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",nargs="+",type=int,default=[128,256,512])
    p.add_argument("--repeat",type=int,default=12)
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args()
    assert torch.cuda.is_available()
    import triton
    rows=[]
    for L in args.lengths:
        B,H,D=1,4,64
        torch.manual_seed(1234+L)
        q,k,v=[torch.randn((B,H,L,D),device="cuda",dtype=torch.float32) for _ in range(3)]
        spans=[[(max(1,L//8),L//2)]]
        kwargs=dict(thres_mode="medium",mean_mode="image")
        dense=lambda:dense_cfpo(q,k,v,spans,**kwargs)
        kernel=lambda:flashcf_forward(q,k,v,spans,block_k=32,**kwargs)
        o0=dense()
        o1=kernel()
        maxerr=(o0-o1).abs().max().item()
        if maxerr>1e-4:raise AssertionError((L,maxerr))
        dt,dp=measure(dense,args.repeat)
        kt,kp=measure(kernel,args.repeat)
        row={"sequence_length":L,"heads":H,"head_dim":D,"max_abs_error":maxerr,
             "dense_reference_median_ms":dt,"triton_prototype_median_ms":kt,
             "dense_reference_peak_delta_mib":round(dp/(2**20),3),
             "triton_prototype_peak_delta_mib":round(kp/(2**20),3),
             "speed_ratio_dense_over_triton":dt/kt if kt else None}
        print(json.dumps(row))
        rows.append(row)
    result={"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
            "triton":triton.__version__,
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip(),
            "note":"Initial correctness kernel; dense Python reference is NOT a competitive optimized baseline. No fused backward or CFPO RL.",
            "measurements":rows}
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("DONE")
if __name__=="__main__":main()
