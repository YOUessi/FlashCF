"""Diagnose threshold and softmax differences at long sequence lengths.

Diagnostic only; allocates O(L^2) as required for reference comparison.
Do NOT use for production attention at extreme L.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kernels.triton_forward import flashcf_forward
from reference.cfpo_reference import dense_cfpo


@torch.no_grad()
def one(L):
    B,H,D=1,4,64
    first=L//8
    last=L//2
    spans=[[(first,last)]]
    torch.manual_seed(L+911)
    q,k,v=[torch.randn(1,H,L,D,device="cuda",dtype=torch.float32) for _ in range(3)]
    out,meta=flashcf_forward(q,k,v,spans,thres_mode="medium",mean_mode="image",
                            block_k=32,return_diagnostics=True,return_internal=True)
    s=q@k.transpose(-1,-2)/math.sqrt(D)
    allowed=torch.ones((L,L),device="cuda",dtype=torch.bool).tril()
    p=torch.softmax(s.masked_fill(~allowed,None if False else -float("inf")),dim=-1)
    pspan=p[:,:,last:,first:last]
    valid=pspan>1e-8
    selected=pspan[valid]
    exact_mean=selected.mean()
    exact_std=selected.std()
    td=exact_mean+exact_std
    tt=meta["thresholds"][0]
    approximate=torch.exp(s[:,:,last:,first:last]-meta["_lse"][:,:,last:,None])
    mask_dense=(pspan>td)&valid
    mask_approx=(approximate>tt)&(approximate>1.e-8)
    flips=(mask_dense!=mask_approx)
    lse_dense=torch.logsumexp(s.masked_fill(~allowed,-float("inf")),dim=-1)
    lse_t=meta["_lse"]
    yref=dense_cfpo(q,k,v,spans,thres_mode="medium",mean_mode="image")
    diag={
      "L":L,
      "eligible":int(valid.sum()),
      "total_region":int(pspan.numel()),
      "dense_threshold":float(td),
      "triton_threshold":float(tt),
      "threshold_diff":float(tt-td),
      "dense_mean":float(exact_mean),
      "dense_std":float(exact_std),
      "triton_count":int(meta["count"][0]),
      "triton_lse_max_abs":float((lse_dense-lse_t).abs().max()),
      "triton_normalized_prob_max_abs":float((pspan-approximate).abs().max()),
      "flipped_mask_entries":int(flips.sum()),
      "dense_selected":int(mask_dense.sum()),
      "approx_selected":int(mask_approx.sum()),
      "output_max_abs":float((out-yref).abs().max()),
    }
    print(json.dumps(diag),flush=True)
    return diag

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--lengths",type=int,nargs="+",default=[1024,2048,4096])
    p.add_argument("--output",type=Path,default=None)
    args=p.parse_args()
    result=[one(L) for L in args.lengths]
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
