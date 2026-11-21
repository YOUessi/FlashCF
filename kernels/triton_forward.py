"""First CUDA/Triton FlashCF forward: exact causal/value-intervention *prototype*.

Computes regular attention, per-span global statistics and CFPO-style value correction
using O(B*H*L*D + S*H*L) intermediate storage rather than O(B*H*L*L).

Scope: contiguous [B,H,L,D] Q/K/V, causal, no padding, no dropout, no GQA
unless KV heads have already been repeated. FP32 forward only. No backward yet.
The original CFPO value-delta formula is not our innovation.

Reference: Raven-July/CFPO / verl/models/transformers/qwen2_vl_cmve.py
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import triton
import triton.language as tl

COEFFICIENT = {"low": 0.0, "medium": 1.0, "high": 2.0, "extra": 3.0}


@triton.jit
def _base_lse(Q, K, V, O, LSE,
              L:tl.constexpr, D:tl.constexpr, H:tl.constexpr,
              SCALE:tl.constexpr, BK:tl.constexpr, BD:tl.constexpr):
    i=tl.program_id(0)
    bh=tl.program_id(1)
    ds=tl.arange(0, BD)
    qs=tl.load(Q+(bh*L+i)*D+ds, mask=ds<D, other=0).to(tl.float32)
    m=tl.full((), -float("inf"), tl.float32)
    z=tl.full((), 0.0, tl.float32)
    acc=tl.full((BD,), 0.0, tl.float32)
    for base in range(0, i+1, BK):
        js=base+tl.arange(0, BK)
        valid=js<=i
        ptr=(bh*L+js[:,None])*D+ds[None,:]
        kk=tl.load(K+ptr, mask=valid[:,None] & (ds[None,:]<D), other=0).to(tl.float32)
        vv=tl.load(V+ptr, mask=valid[:,None] & (ds[None,:]<D), other=0).to(tl.float32)
        scores=tl.sum(kk*qs[None,:], 1)*SCALE
        scores=tl.where(valid, scores, -float("inf"))
        next_m=tl.maximum(m, tl.max(scores,0))
        old_scale=tl.exp(m-next_m)
        w=tl.exp(scores-next_m)
        acc=acc*old_scale+tl.sum(w[:,None]*vv,axis=0)
        z=z*old_scale+tl.sum(w,0)
        m=next_m
    tl.store(O+(bh*L+i)*D+ds, acc/z, mask=ds<D)
    tl.store(LSE+bh*L+i,m+tl.log(z))


@triton.jit
def _span_moments(Q,K,LSE,SPANS,PART_SUM,PART_SQ,PART_COUNT,
                  L:tl.constexpr,D:tl.constexpr,H:tl.constexpr,
                  SCALE:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    h=tl.program_id(1)
    si=tl.program_id(2)
    b=tl.load(SPANS+si*3+0)
    first=tl.load(SPANS+si*3+1)
    last=tl.load(SPANS+si*3+2)
    index=(si*H+h)*L+i
    s1=tl.full((),0.0,tl.float32)
    s2=tl.full((),0.0,tl.float32)
    cnt=tl.full((),0.0,tl.float32)
    if i>=last:
        ds=tl.arange(0,BD)
        bh=b*H+h
        q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
        lse=tl.load(LSE+bh*L+i)
        for j0 in range(first,last,BK):
            js=j0+tl.arange(0,BK)
            kt=tl.load(K+((bh*L+js[:,None])*D+ds[None,:]),
                       mask=(js[:,None]<last)&(ds[None,:]<D),other=0).to(tl.float32)
            logits=tl.sum(kt*q[None,:],1)*SCALE
            p=tl.exp(logits-lse)
            eligible=(js<last)&(p>1.0e-8)
            p=tl.where(eligible,p,0.)
            s1+=tl.sum(p,0)
            s2+=tl.sum(p*p,0)
            cnt+=tl.sum(eligible.to(tl.float32),0)
    tl.store(PART_SUM+index,s1)
    tl.store(PART_SQ+index,s2)
    tl.store(PART_COUNT+index,cnt)


@triton.jit
def _correction(Q,K,V,LSE,SPANS,THRESH,REPLACEMENT,DELTA,
                L:tl.constexpr,D:tl.constexpr,H:tl.constexpr,
                SCALE:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    h=tl.program_id(1)
    si=tl.program_id(2)
    b=tl.load(SPANS+si*3+0)
    first=tl.load(SPANS+si*3+1)
    last=tl.load(SPANS+si*3+2)
    thresh=tl.load(THRESH+si)
    ds=tl.arange(0,BD)
    accum=tl.full((BD,),0.0,tl.float32)
    if i>=last:
        bh=b*H+h
        q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
        lse=tl.load(LSE+bh*L+i)
        replacement=tl.load(REPLACEMENT+bh)
        for j0 in range(first,last,BK):
            js=j0+tl.arange(0,BK)
            valid=js<last
            ptr=(bh*L+js[:,None])*D+ds[None,:]
            kk=tl.load(K+ptr,mask=valid[:,None] & (ds[None,:]<D),other=0).to(tl.float32)
            vv=tl.load(V+ptr,mask=valid[:,None] & (ds[None,:]<D),other=0).to(tl.float32)
            p=tl.exp(tl.sum(kk*q[None,:],1)*SCALE-lse)
            active=valid & (p>1.0e-8) & (p>thresh)
            coeff=tl.where(active,p,0.)
            accum+=tl.sum(coeff[:,None]*(replacement-vv),axis=0)
    tl.store(DELTA+((si*H+h)*L+i)*D+ds,accum,mask=ds<D)


def _pack_spans(spans, batch_size, length, device):
    flat=[]
    for b,group in enumerate(spans):
        for st,en in group:
            if not (0 <= st < en <= length):
                raise ValueError(f"invalid image span {(b,st,en)} for L={length}")
            flat.append((b,st,en))
    if not flat:
        return None,flat
    # Differently overlapping spans would overwrite the same intervention positions.
    for b,group in enumerate(spans):
        ordered=sorted(group)
        if any(a[1]>b_[0] for a,b_ in zip(ordered,ordered[1:])):
            raise ValueError(f"overlapping spans in batch {b}")
    return torch.tensor(flat,device=device,dtype=torch.int32),flat


@torch.no_grad()
def flashcf_forward(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,spans:Sequence,
                    *,thres_mode="high",mean_mode="image",block_k=32,
                    return_diagnostics=False):
    """Run true tiled GPU forward with bounded attention intermediates.

    Input Q,K,V: contiguous float32 [B,H,L,D], pre-rotated if needed.
    Spans: lists of (start,end) half-open visual token intervals for each B.
    Results: float32 [B,H,L,D].
    """
    if q.device.type!="cuda" or any(t.device!=q.device for t in (k,v)):
        raise ValueError("FlashCF Triton forward requires CUDA Q/K/V")
    if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=4:
        raise ValueError("expected equal [B,H,L,D] tensors, with GQA pre-repeated")
    if any(t.dtype!=torch.float32 or not t.is_contiguous() for t in (q,k,v)):
        raise ValueError("v1 requires contiguous FP32 Q/K/V")
    B,H,L,D=q.shape
    if len(spans)!=B:
        raise ValueError("one list of visual spans per batch item is required")
    if thres_mode not in COEFFICIENT or mean_mode not in ("image","text"):
        raise ValueError("invalid threshold or value mean mode")
    if block_k not in (16,32,64,128):
        raise ValueError("block_k must be 16,32,64 or 128")
    packed,flat=_pack_spans(spans,B,L,q.device)
    bd=triton.next_power_of_2(D)
    base=torch.empty_like(v)
    lse=torch.empty((B,H,L),device=q.device,dtype=torch.float32)
    grid=(L,B*H)
    _base_lse[grid](q,k,v,base,lse,L,D,H,1/math.sqrt(D),block_k,bd,num_warps=4)

    if not flat:
        return (base,{"num_spans":0,"attention_materialized":False}) if return_diagnostics else base

    is_image=torch.zeros((B,L),device=q.device,dtype=torch.bool)
    for b,st,en in flat:
        is_image[b,st:en]=True
    choose=is_image if mean_mode=="image" else ~is_image
    selected=v*choose[:,None,:,None]
    denom=choose.sum(dim=-1).clamp(min=1).to(torch.float32)[:,None]*D
    replacement=selected.sum(dim=(2,3))/denom  # [B,H], one scalar per B,H

    S=len(flat)
    partial_shape=(S,H,L)
    part_sum=torch.empty(partial_shape,device=q.device,dtype=torch.float32)
    part_sq=torch.empty_like(part_sum)
    part_count=torch.empty_like(part_sum)
    _span_moments[(L,H,S)](q,k,lse,packed,part_sum,part_sq,part_count,
                          L,D,H,1/math.sqrt(D),block_k,bd,num_warps=4)
    count=part_count.sum(dim=(1,2))
    sums=part_sum.sum(dim=(1,2))
    sqs=part_sq.sum(dim=(1,2))
    n=count.clamp(min=1)
    mu=sums/n
    variance=(sqs-sums*sums/n)/(count-1).clamp(min=1)
    std=torch.sqrt(torch.clamp(variance,min=0))
    thresholds=torch.where(count>1, mu+COEFFICIENT[thres_mode]*std, torch.full_like(mu,float("inf")))
    delta=torch.empty((S,H,L,D),device=q.device,dtype=torch.float32)
    _correction[(L,H,S)](q,k,v,lse,packed,thresholds,replacement.contiguous(),delta,
                         L,D,H,1/math.sqrt(D),block_k,bd,num_warps=4)
    out=base
    for si,(b,st,en) in enumerate(flat):
        out[b] += delta[si]
    if return_diagnostics:
        return out,{"num_spans":S,"attention_materialized":False,
                    "thresholds":thresholds.detach().clone(),"count":count.detach().clone(),
                    "intermediate_bytes":lse.numel()*4+part_sum.numel()*12+delta.numel()*4}
    return out
