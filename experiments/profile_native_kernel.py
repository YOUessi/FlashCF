"""Isolate BF16 FlashCF compute bottlenecks on Tang with CUDA profiler.

This profiles a *single* native-GQA FP32/BF16 forward + backward at a time.
Profiler table entries are reported to stdout, no raw model data saved.
Not an end-to-end model-training benchmark.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.profiler import profile, ProfilerActivity
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.triton_native_backward_v2 import flashcf_native_v2

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--length",type=int,default=512)
    p.add_argument("--max-rows",type=int,default=30)
    args=p.parse_args()
    torch.manual_seed(712)
    B,Hq,Hkv,L,D=1,16,2,args.length,128
    q,k,v=[torch.randn((B,h,L,D),device="cuda",dtype=torch.bfloat16,requires_grad=True)
           for h in (Hq,Hkv,Hkv)]
    mask=torch.zeros((B,1,L,L),device="cuda",dtype=torch.bfloat16)
    mask.masked_fill_(torch.ones(L,L,device="cuda",dtype=torch.bool).triu(1)[None,None],torch.finfo(mask.dtype).min)
    spans=[[(L//8,L//2)]]
    def run():
        y=flashcf_native_v2(q,k,v,spans,mask,thres_mode="high",mean_mode="image",block_k=32)
        grad=torch.autograd.grad((y.float().square()).mean(),(q,k,v))
        return y,grad
    for _ in range(4):run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:
        run()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_device_time_total",row_limit=args.max_rows),flush=True)
    kernelrows=[]
    for evt in prof.key_averages():
        name=evt.key
        if "_get_" in name or "_grad" in name or "_vjp" in name or "NativeFlashCF" in name or "aten::" in name:
            kernelrows.append({"event":name,"cuda_total_us":float(getattr(evt,"device_time_total",0)),
                               "cuda_self_us":float(getattr(evt,"self_device_time_total",0)),
                               "calls":int(evt.count)})
    kernelrows.sort(key=lambda x:x["cuda_self_us"],reverse=True)
    out={"device":torch.cuda.get_device_name(),"length":L,"Hq":Hq,"Hkv":Hkv,"D":D,
         "dtype":"BF16", "profile":kernelrows[:45],
         "note":"torch.profiler single run including tracing overhead; rank of kernels only, not accurate benchmark elapsed time"}
    print("JSON_SUMMARY",json.dumps(out),flush=True)

if __name__=="__main__":main()
