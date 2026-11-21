"""More faithful eager baseline for the CFPO execution pattern on causal FP32.

CFPO's official source separately materializes attention probabilities for CMVE
and calls FlashAttention for the original output. In contrast, the basic
Dense reference reuses materialized probabilities for its original output.

This surrogate uses PyTorch scaled_dot_product_attention for normal output,
but PyTorch chooses its own backend; use profiler to verify fused vs math.
No full official model, no dropout, and no arbitrary 4D masks yet.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from reference.cfpo_reference import _validate,_replacement_scalar,_threshold


def sdpa_cmve_baseline(q,k,v,spans,*,thres_mode="high",mean_mode="image"):
    B,H,L,D=_validate(q,k,v,spans)
    assert q.dtype==torch.float32 and q.is_contiguous()
    logits=q@k.transpose(-1,-2)/math.sqrt(D)
    allowed=torch.ones((L,L),dtype=torch.bool,device=q.device).tril()
    logits=logits.masked_fill(~allowed[None,None,:,:],-float("inf"))
    p=torch.softmax(logits,dim=-1,dtype=torch.float32)
    replacement=_replacement_scalar(v,spans,mean_mode)
    mask=torch.zeros_like(p)
    for b,regions in enumerate(spans):
        for first,last in regions:
            if last>=L:continue
            slice_=p[b,:,last:,first:last]
            valid=slice_>1.0e-8
            if bool(valid.any()):
                cutoff=_threshold(slice_[valid],thres_mode)
                mask[b,:,last:,first:last]=(valid&(slice_>cutoff)).to(p.dtype)
    delta=(p*mask)@(replacement-v)
    normal=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
    return normal+delta
