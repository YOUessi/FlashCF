"""Exact eager-CMVE reference for prefill attention with native grouped KV heads.

Implements the pinned CFPO's deterministic (noise=False, dropout=0) BF16
mask/replacement semantics without loading verl or duplicating its source.

Input Q: [B,Hq,L,D], K/V: [B,Hkv,L,D], 4D additive mask:
[B,1,L,L] or [B,Hq,L,L]. This reference MATERIALIZES the attention matrix
for correctness tests. Production Triton kernels must not.

Important numerical details: native torch.matmul yields BF16 logits, scaling
and mask addition stay BF16, softmax computes FP32 then returns BF16, and
mean/std + threshold + replacement are calculated in input dtype.
"""
from __future__ import annotations

import math
import torch


def dense_native_cmve(q,k,v,spans,mask,*,thres_mode="high",mean_mode="image",
                      return_metadata=False):
    if any(x.ndim!=4 for x in (q,k,v)):raise ValueError("expected 4D tensors")
    B,Hq,L,D=q.shape
    Hkv=k.shape[1]
    if k.shape!=(B,Hkv,L,D) or v.shape!=k.shape or Hq%Hkv:
        raise ValueError("invalid native GQA heads or key/value tensor shape")
    if mask is None or mask.shape not in ((B,1,L,L),(B,Hq,L,L)):
        raise ValueError("requires real 4D additive attention mask [B,1|Hq,L,L]")
    if mask.device!=q.device or q.device!=k.device or q.device!=v.device:
        raise ValueError("device mismatch")
    if q.dtype!=k.dtype or k.dtype!=v.dtype or q.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError("only matched fp32/bf16 input supported")
    if thres_mode not in ("high","medium","low","extra") or mean_mode not in ("image","text"):
        raise ValueError("invalid CMVE config")
    if len(spans)!=B:raise ValueError("spans batch mismatch")
    group=Hq//Hkv
    kr=k.repeat_interleave(group,dim=1)
    vr=v.repeat_interleave(group,dim=1)
    score=(q@kr.transpose(-1,-2))/math.sqrt(D)
    score=score+mask
    probs=torch.softmax(score,dim=-1,dtype=torch.float32).to(q.dtype)

    selected=torch.zeros((B,L),device=q.device,dtype=torch.bool)
    for b,items in enumerate(spans):
        for st,en in items:
            if not (0<=st<en<=L):raise ValueError("invalid span")
            selected[b,st:en]=True
    if mean_mode=="text": selected=~selected
    count=selected.sum(-1).clamp(min=1).to(v.dtype).view(B,1,1,1)
    masked=vr.where(selected[:,None,:,None],0.)
    replace=masked.sum((2,3),keepdim=True)/(count*D)
    # Precise official mask semantics: one mean/std per batch+span, across heads.
    mask_cmve=torch.zeros_like(probs)
    threshold_records=[]
    mode_multiplier=dict(low=0,medium=1,high=2,extra=3)[thres_mode]
    for b,items in enumerate(spans):
        for st,en in items:
            if en>=L:continue
            prob_slice=probs[b:b+1,:,en:,st:en]
            eligible=prob_slice>1e-8
            if not bool(eligible.any()):continue
            vals=prob_slice[eligible]
            mu=vals.mean()
            sd=vals.std()  # sample standard deviation, CFPO default
            threshold=mu+mode_multiplier*sd
            current=(prob_slice>threshold)&eligible
            mask_cmve[b:b+1,:,en:,st:en]=current.to(probs.dtype)
            threshold_records.append((b,st,en,float(threshold.detach().float()),int(current.detach().sum())))
    base=probs@vr
    delta=(probs*mask_cmve)@(replace-vr)
    out=base+delta
    if return_metadata:
        return out,{"probs":probs,"cmve_mask":mask_cmve,"thresholds":threshold_records,
                    "factual":base,"value_mean":replace,"counterfactual_delta":delta}
    return out
