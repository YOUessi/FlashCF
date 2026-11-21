"""Repeated, alternating-order BF16 FlashCF vs forced-Flash + explicit CMVE.

Not a full RL training benchmark. CUDA event timing and peak allocated memory
for real Qwen-shaped synthetic BF16 Q/K/V with controlled shape/seed.
Alternating AB/BA controls order bias. Report medians and p10/p90, not
a single cherry-picked run. No raw tensors or model activations are exported.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from reference.native_sdpa_cmve import native_sdpa_cmve
from kernels.triton_native_qtile import flashcf_gemm_all


def gpu_state():
    try:
        result=subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,memory.used",
             "--format=csv,noheader"],text=True,timeout=8).strip()
        return result
    except Exception as exc:
        return "unavailable: "+type(exc).__name__


def one(seed,L,rounds,steps):
    torch.manual_seed(seed)
    B,Hq,Hkv,D=1,16,2,128
    q,k,v=[torch.randn((B,h,L,D),device="cuda",dtype=torch.bfloat16,
                        requires_grad=True) for h in (Hq,Hkv,Hkv)]
    dy=torch.randn_like(q)
    mask=torch.zeros((B,1,L,L),device="cuda",dtype=torch.bfloat16)
    mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],
                      torch.finfo(torch.bfloat16).min)
    spans=[[(L//8,L//2)]]
    kw=dict(thres_mode="high",mean_mode="image")
    methods={
      "flash_cmve":lambda:native_sdpa_cmve(q,k,v,spans,mask,
                       factual_backend="flash_causal",**kw),
      "flashcf_gemm_all":lambda:flashcf_gemm_all(q,k,v,spans,mask,
                       row_tile=16,key_tile=64,grad_q_tile=16,grad_k_tile=64,
                       block_k=32,**kw)
    }
    def workload(func):
        y=func()
        grads=torch.autograd.grad((y*dy).sum(),(q,k,v))
        del y,grads
    for method in methods.values():
        for _ in range(5):workload(method)
    torch.cuda.synchronize()
    values={k:[] for k in methods}
    memory={}
    order_seed=seed%2
    for r in range(rounds):
        order=["flash_cmve","flashcf_gemm_all"]
        if (r+order_seed)%2: order.reverse()
        for name in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            prior=torch.cuda.memory_allocated()
            for _ in range(steps):
                begin=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                begin.record()
                workload(methods[name])
                end.record()
                end.synchronize()
                values[name].append(float(begin.elapsed_time(end)))
            memory.setdefault(name,[]).append(
                round((torch.cuda.max_memory_allocated()-prior)/2**20,3))
    result={"seed":seed,"L":L,"gpu_before":gpu_state(),"methods":{}}
    for name,measurements in values.items():
        arr=np.asarray(measurements)
        result["methods"][name]={
            "n":len(arr),
            "p10_ms":round(float(np.percentile(arr,10)),4),
            "p50_ms":round(float(np.percentile(arr,50)),4),
            "p90_ms":round(float(np.percentile(arr,90)),4),
            "peak_extra_mib_max":max(memory[name]),
        }
    f=result["methods"]["flash_cmve"]["p50_ms"]
    t=result["methods"]["flashcf_gemm_all"]["p50_ms"]
    result["p50_speedup_forced_flash_over_flashcf"]=round(f/t,4)
    result["gpu_after"]=gpu_state()
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",nargs="+",type=int,default=[512,1024,2048])
    p.add_argument("--seeds",nargs="+",type=int,default=[20261121,20261122,20261123])
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--steps",type=int,default=12)
    p.add_argument("--output",type=Path,
                   default=Path.home()/".cache"/"flashcf_results"/"p2_repeated_flash_benchmark.json")
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA GPU on Tang")
    results=[one(seed,L,args.rounds,args.steps) for L in args.lengths for seed in args.seeds]
    payload={
      "repo_sha":os.environ.get("FLASHCF_SOURCE_COMMIT") or "pinned immutable runner",
      "torch":torch.__version__,"device":torch.cuda.get_device_name(0),
      "rounds":args.rounds,"steps_per_round":args.steps,
      "model_weights_used":False,
      "baseline":"PyTorch forced SDPBackend.FLASH_ATTENTION causal factual + dense CMVE",
      "candidate":"BF16 all-TensorCore tiled FlashCF Q/K/V",
      "disclaimer":"No original CFPO verl RL, no official FlashAttention2 package, synthetic QKV",
      "results":results,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    print("DONE",flush=True)
