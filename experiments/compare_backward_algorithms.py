"""Compare Dense, atomic row-scatter, race-free key-gather FlashCF training steps.

All implementations match a restricted FP32 causal CFPO reference.
CUDA timing with warmup; Python baseline is not a fully optimized competitor.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.cfpo_reference import dense_cfpo
from reference.cfpo_sdpa_baseline import sdpa_cmve_baseline
from kernels.triton_backward import flashcf_autograd
from kernels.triton_backward_gather import flashcf_gather


def measure(fn, xs, weights, spans, repeats):
    for _ in range(2):
        y=fn(*xs,spans)
        g=torch.autograd.grad((y*weights).sum(),xs)
        del y,g
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_alloc=torch.cuda.memory_allocated()
    ms=[]
    for _ in range(repeats):
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        y=fn(*xs,spans)
        g=torch.autograd.grad((y*weights).sum(),xs)
        end.record()
        end.synchronize()
        ms.append(start.elapsed_time(end))
        del g,y
    return float(torch.tensor(ms).median()),round((torch.cuda.max_memory_allocated()-start_alloc)/2**20,3)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",nargs="+",type=int,default=[128,256,512,1024])
    p.add_argument("--repeat",type=int,default=4)
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args()
    rows=[]
    for L in args.lengths:
        torch.manual_seed(L+911)
        xs=[torch.randn(1,4,L,64,device="cuda",dtype=torch.float32,requires_grad=True)
            for _ in range(3)]
        weights=torch.randn_like(xs[0])
        spans=[[(L//8,L//2)]]
        opts={"thres_mode":"medium","mean_mode":"image"}
        methods={
          "dense":lambda *z:dense_cfpo(*z,**opts),
          "sdpa_cmve":lambda *z:sdpa_cmve_baseline(*z,**opts),
          "atomic":lambda *z:flashcf_autograd(*z,block_k=32,**opts),
          "gather":lambda *z:flashcf_gather(*z,block_k=32,**opts),
        }
        expected=methods["dense"](*xs,spans)
        ge=torch.autograd.grad((expected*weights).sum(),xs)
        rec={"length":L,"heads":4,"head_dim":64}
        for name,method in methods.items():
            y=method(*xs,spans)
            grads=torch.autograd.grad((y*weights).sum(),xs)
            ferr=(expected-y).detach().abs().max().item()
            gerr=max((a-b).detach().abs().max().item() for a,b in zip(ge,grads))
            if ferr>2e-4 or gerr>1e-3:raise AssertionError((name,L,ferr,gerr))
            ms,mem=measure(method,xs,weights,spans,args.repeat)
            rec[name]={"median_ms":round(ms,4),"peak_delta_mib":mem,
                       "max_forward_error":round(ferr,9),"max_grad_error":round(gerr,9)}
        print(json.dumps(rec),flush=True)
        rows.append(rec)
    artifact={"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "gpu":torch.cuda.get_device_name(),
              "torch":torch.__version__,
              "note":"synthetic FP32, causal, no dropout; PyTorch Dense reference not optimized strong baseline; no official CFPO fixtures",
              "measurements":rows}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(artifact,indent=2)+"\n")
    print("DONE",flush=True)
if __name__=="__main__":main()
