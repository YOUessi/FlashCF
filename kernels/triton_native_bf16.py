"""Native BF16 + grouped KV + real 4D additive-mask FlashCF Triton kernels.

First functional, bounded-intermediate implementation for CFPO-style CMVE.

Q [B,Hq,L,D], K/V [B,Hkv,L,D] with Hq % Hkv == 0.
Mask [B,1,L,L] or [B,Hq,L,L], 4D additive (no implicit causal
assumption: each valid row is defined by the supplied mask).

BF16 parity mode:
- BF16 QK score, scale and mask addition are rounded as native Torch eager;
- softmax exp/normalization is in FP32, probability quantized to BF16;
- replacement is computed from native BF16 V values, including denom rounding;
- output is rounded to input dtype.
Floating point reductions may still differ from PyTorch, and hard CMVE
threshold ties can alter a few mask entries: monitor actual gate disagreement.

Autograd uses an analytic fixed-gate straight-through Jacobian with FP32
intermediate accumulation and returns gradients in input dtype. This does not
claim bit-identical PyTorch BF16 backward at every operation/cast boundary.

No KV tensor repeat in forward. Backward computes per-query-head dK/dV into
O(B*Hq*L*D) float32 partials then sums the GQA group, not O(L^2) state.

Known limits: q_len==k_len full prefill; dropout=0, noise=False, no sliding
window except what the input mask already encodes; contiguous tensors;
requires at least one unmasked key per query row; no second derivatives.
"""
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

THRESH_COEFF={"low":0, "medium":1, "high":2, "extra":3}


@triton.jit
def _attention_logits(qvec, kblock, Mask, b, h, i, js,
                      L:tl.constexpr,MH:tl.constexpr,D:tl.constexpr,
                      SCALE:tl.constexpr,BF16:tl.constexpr):
    z=tl.sum(kblock*qvec[None,:],axis=1)
    if BF16:
        z=z.to(tl.bfloat16).to(tl.float32)
    z=z*SCALE
    if BF16:
        z=z.to(tl.bfloat16).to(tl.float32)
    mh=0 if MH==1 else h
    bias=tl.load(Mask+((b*MH+mh)*L+i)*L+js,
                 mask=js<L,other=-float("inf")).to(tl.float32)
    # In the real Qwen2.5 mask disallowed positions are -3e38 or -inf.
    # Finite nonzero additive biases are retained. Do not silently replace
    # an arbitrary mask with a triangular causal mask.
    valid=(js<L)&(bias>-1.e4)
    z=z+tl.where(valid,bias,0.)
    if BF16:
        z=z.to(tl.bfloat16).to(tl.float32)
    z=tl.where(valid,z,-float("inf"))
    return z,valid


@triton.jit
def _prob(qvec,kblock,Mask,lse,b,h,i,js,
          L:tl.constexpr,MH:tl.constexpr,D:tl.constexpr,
          SCALE:tl.constexpr,BF16:tl.constexpr):
    z,valid=_attention_logits(qvec,kblock,Mask,b,h,i,js,L,MH,D,SCALE,BF16)
    p=tl.exp(z-lse)
    p=tl.where(valid,p,0.)
    if BF16:
        p=p.to(tl.bfloat16).to(tl.float32)
    return p,valid


@triton.jit
def _get_lse(Q,K,Mask,LSE,
             L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
             MH:tl.constexpr,GROUP:tl.constexpr,
             SCALE:tl.constexpr,BF16:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    ds=tl.arange(0,BD)
    q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    peak=tl.full((),-float("inf"),tl.float32)
    den=tl.full((),0.,tl.float32)
    for st in range(0,L,BK):
        js=st+tl.arange(0,BK)
        kk=tl.load(K+(kv*L+js[:,None])*D+ds[None,:],
                 mask=(js[:,None]<L)&(ds[None,:]<D),other=0).to(tl.float32)
        score,valid=_attention_logits(q,kk,Mask,b,h,i,js,L,MH,D,SCALE,BF16)
        if tl.sum(valid.to(tl.int32),0)>0:
            newpeak=tl.maximum(peak,tl.max(score,0))
            scale=tl.where(den>0.,tl.exp(peak-newpeak),0.)
            den=den*scale+tl.sum(tl.exp(score-newpeak),0)
            peak=newpeak
    # At least one valid key per query is required by wrapper.
    tl.store(LSE+bh*L+i,peak+tl.log(den))


@triton.jit
def _get_base(Q,K,V,Mask,LSE,BASE,
              L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
              MH:tl.constexpr,GROUP:tl.constexpr,
              SCALE:tl.constexpr,BF16:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    ds=tl.arange(0,BD)
    q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    lse=tl.load(LSE+bh*L+i)
    acc=tl.full((BD,),0.,tl.float32)
    for st in range(0,L,BK):
        js=st+tl.arange(0,BK)
        ptr=(kv*L+js[:,None])*D+ds[None,:]
        kk=tl.load(K+ptr,mask=(js[:,None]<L)&(ds[None,:]<D),
                   other=0).to(tl.float32)
        vv=tl.load(V+ptr,mask=(js[:,None]<L)&(ds[None,:]<D),
                   other=0).to(tl.float32)
        p,_=_prob(q,kk,Mask,lse,b,h,i,js,L,MH,D,SCALE,BF16)
        acc+=tl.sum(p[:,None]*vv,0)
    tl.store(BASE+(bh*L+i)*D+ds,acc,mask=ds<D)


@triton.jit
def _get_stats(Q,K,Mask,LSE,SPANS,PART_SUM,PART_SQ,PART_COUNT,
               L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
               MH:tl.constexpr,GROUP:tl.constexpr,
               SCALE:tl.constexpr,BF16:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    h=tl.program_id(1)
    si=tl.program_id(2)
    b=tl.load(SPANS+si*3)
    st=tl.load(SPANS+si*3+1)
    en=tl.load(SPANS+si*3+2)
    idx=(si*HQ+h)*L+i
    s1=tl.full((),0.,tl.float32)
    s2=tl.full((),0.,tl.float32)
    n=tl.full((),0.,tl.float32)
    if i>=en:
        bh=b*HQ+h
        kv=b*HK+h//GROUP
        ds=tl.arange(0,BD)
        q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
        lse=tl.load(LSE+bh*L+i)
        for j0 in range(st,en,BK):
            js=j0+tl.arange(0,BK)
            kk=tl.load(K+(kv*L+js[:,None])*D+ds[None,:],
                      mask=(js[:,None]<en)&(ds[None,:]<D),
                      other=0).to(tl.float32)
            p,_=_prob(q,kk,Mask,lse,b,h,i,js,L,MH,D,SCALE,BF16)
            active=(js<en)&(p>1.e-8)
            p=tl.where(active,p,0.)
            s1+=tl.sum(p,0)
            s2+=tl.sum(p*p,0)
            n+=tl.sum(active.to(tl.float32),0)
    tl.store(PART_SUM+idx,s1)
    tl.store(PART_SQ+idx,s2)
    tl.store(PART_COUNT+idx,n)


@triton.jit
def _get_delta(Q,K,V,Mask,LSE,SPANS,THRESH,REPL,DELTA,
               L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
               MH:tl.constexpr,GROUP:tl.constexpr,
               SCALE:tl.constexpr,BF16:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    h=tl.program_id(1)
    si=tl.program_id(2)
    b=tl.load(SPANS+si*3)
    st=tl.load(SPANS+si*3+1)
    en=tl.load(SPANS+si*3+2)
    cutoff=tl.load(THRESH+si)
    ds=tl.arange(0,BD)
    acc=tl.full((BD,),0.,tl.float32)
    if i>=en:
        bh=b*HQ+h
        kv=b*HK+h//GROUP
        q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
        lse=tl.load(LSE+bh*L+i)
        replacement=tl.load(REPL+bh).to(tl.float32)
        for j0 in range(st,en,BK):
            js=j0+tl.arange(0,BK)
            ptr=(kv*L+js[:,None])*D+ds[None,:]
            kk=tl.load(K+ptr,mask=(js[:,None]<en)&(ds[None,:]<D),
                      other=0).to(tl.float32)
            vv=tl.load(V+ptr,mask=(js[:,None]<en)&(ds[None,:]<D),
                      other=0).to(tl.float32)
            p,_=_prob(q,kk,Mask,lse,b,h,i,js,L,MH,D,SCALE,BF16)
            hit=(js<en)&(p>1.e-8)&(p>cutoff)
            diff=replacement-vv
            if BF16:
                diff=diff.to(tl.bfloat16).to(tl.float32)
            acc+=tl.sum(tl.where(hit,p,0.)[:,None]*diff,0)
    tl.store(DELTA+((si*HQ+h)*L+i)*D+ds,acc,mask=ds<D)


@triton.jit
def _grad_q(Q,K,V,Mask,Y,DY,LSE,SPANS,THRESH,REPL,DQ,
            L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
            MH:tl.constexpr,GROUP:tl.constexpr,NS:tl.constexpr,
            SCALE:tl.constexpr,BF16:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    ds=tl.arange(0,BD)
    q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    y=tl.load(Y+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    go=tl.load(DY+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    lse=tl.load(LSE+bh*L+i)
    repl=tl.load(REPL+bh).to(tl.float32)
    dq=tl.full((BD,),0.,tl.float32)
    for j0 in range(0,L,BK):
        js=j0+tl.arange(0,BK)
        ptr=(kv*L+js[:,None])*D+ds[None,:]
        kk=tl.load(K+ptr,mask=(js[:,None]<L)&(ds[None,:]<D),other=0).to(tl.float32)
        vv=tl.load(V+ptr,mask=(js[:,None]<L)&(ds[None,:]<D),other=0).to(tl.float32)
        p,_=_prob(q,kk,Mask,lse,b,h,i,js,L,MH,D,SCALE,BF16)
        selected=tl.full((BK,),False,tl.int1)
        for si in range(NS):
            sb=tl.load(SPANS+si*3)
            st=tl.load(SPANS+si*3+1)
            en=tl.load(SPANS+si*3+2)
            cutoff=tl.load(THRESH+si)
            selected=selected | ((sb==b)&(i>=en)&(js>=st)&(js<en)&
                                  (p>1.e-8)&(p>cutoff))
        diff=repl-vv
        if BF16:
            diff=diff.to(tl.bfloat16).to(tl.float32)
        w=vv+tl.where(selected[:,None],diff,0.)
        dscores=p*tl.sum((w-y[None,:])*go[None,:],1)
        dq+=tl.sum(dscores[:,None]*kk,0)*SCALE
    tl.store(DQ+(bh*L+i)*D+ds,dq,mask=ds<D)


@triton.jit
def _grad_kv(Q,K,V,Mask,Y,DY,LSE,SPANS,THRESH,REPL,DKTMP,DVTMP,GC_PART,
             L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
             MH:tl.constexpr,GROUP:tl.constexpr,NS:tl.constexpr,
             SCALE:tl.constexpr,BF16:tl.constexpr,BQ:tl.constexpr,BD:tl.constexpr):
    j=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    ds=tl.arange(0,BD)
    k=tl.load(K+(kv*L+j)*D+ds,mask=ds<D,other=0).to(tl.float32)
    v=tl.load(V+(kv*L+j)*D+ds,mask=ds<D,other=0).to(tl.float32)
    repl=tl.load(REPL+bh).to(tl.float32)
    dk=tl.full((BD,),0.,tl.float32)
    dv=tl.full((BD,),0.,tl.float32)
    dmean=tl.full((),0.,tl.float32)
    for i0 in range(0,L,BQ):
        is_=i0+tl.arange(0,BQ)
        valid=is_<L
        ptr=(bh*L+is_[:,None])*D+ds[None,:]
        qs=tl.load(Q+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        ys=tl.load(Y+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        gs=tl.load(DY+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        lse=tl.load(LSE+bh*L+is_,mask=valid,other=float("inf"))
        # Scalar key broadcasts against a BQ-query tile.
        score=tl.sum(qs*k[None,:],1)
        if BF16:
            score=score.to(tl.bfloat16).to(tl.float32)
        score=score*SCALE
        if BF16:
            score=score.to(tl.bfloat16).to(tl.float32)
        mh=0 if MH==1 else h
        bias=tl.load(Mask+((b*MH+mh)*L+is_)*L+j,mask=valid,
                     other=-float("inf")).to(tl.float32)
        allowed=valid&(bias>-1.e4)
        score=score+tl.where(allowed,bias,0.)
        if BF16:
            score=score.to(tl.bfloat16).to(tl.float32)
        p=tl.exp(score-lse)
        p=tl.where(allowed,p,0.)
        if BF16:
            p=p.to(tl.bfloat16).to(tl.float32)
        selected=tl.full((BQ,),False,tl.int1)
        for si in range(NS):
            sb=tl.load(SPANS+si*3)
            st=tl.load(SPANS+si*3+1)
            en=tl.load(SPANS+si*3+2)
            cutoff=tl.load(THRESH+si)
            selected=selected | ((sb==b)&(is_>=en)&(j>=st)&(j<en)&
                                  (p>1.e-8)&(p>cutoff))
        diff=repl-v
        if BF16:
            diff=diff.to(tl.bfloat16).to(tl.float32)
        w=v[None,:]+tl.where(selected[:,None],diff[None,:],0.)
        dscores=p*tl.sum((w-ys)*gs,1)
        dk+=tl.sum(dscores[:,None]*qs,0)*SCALE
        dv+=tl.sum(p[:,None]*tl.where(selected[:,None],0.,gs),0)
        dmean+=tl.sum(tl.where(selected,p,0.)*tl.sum(gs,1),0)
    tl.store(DKTMP+(bh*L+j)*D+ds,dk,mask=ds<D)
    tl.store(DVTMP+(bh*L+j)*D+ds,dv,mask=ds<D)
    tl.store(GC_PART+bh*L+j,dmean)


def _check(q,k,v,mask,spans,mean_mode,mode,block):
    if q.device.type!="cuda" or any(x.device!=q.device for x in (k,v,mask)):
        raise ValueError("Q/K/V/mask must reside on CUDA")
    if any(x.ndim!=4 for x in (q,k,v,mask)):
        raise ValueError("expected rank-4 Q/K/V/mask")
    B,Hq,L,D=q.shape
    Hkv=k.shape[1]
    if k.shape!=(B,Hkv,L,D) or v.shape!=k.shape or Hq%Hkv:
        raise ValueError("full-prefill native GQA requires [B,Hq,L,D] and [B,Hkv,L,D]")
    if mask.shape not in ((B,1,L,L),(B,Hq,L,L)):
        raise ValueError("4D additive mask shape must be [B,1|Hq,L,L]")
    if any(not x.is_contiguous() for x in (q,k,v,mask)):
        raise ValueError("v1 requires contiguous Q/K/V/mask")
    if not (q.dtype==k.dtype==v.dtype and q.dtype in (torch.float32,torch.bfloat16)):
        raise ValueError("v1 supports matched FP32/BF16 Q/K/V")
    if mask.dtype not in (torch.float32,torch.bfloat16):
        raise ValueError("mask must be additive FP32/BF16 (not boolean)")
    if D>256 or block not in (16,32,64):
        raise ValueError("v1 head_dim <= 256; block_k in (16,32,64)")
    if mode not in THRESH_COEFF or mean_mode not in ("image","text"):
        raise ValueError("unsupported CMVE config")
    if len(spans)!=B:
        raise ValueError("batch-level spans expected")
    # Fail rather than approximate when a whole query row has no valid keys.
    # This validation runs on a PROVIDED dense 4D mask, not on QK probabilities.
    valid=mask>-1.e4
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("all-masked query rows are not supported in v1")
    flat=[]
    for b,items in enumerate(spans):
        previous=0
        for st,en in sorted(items):
            if not (previous<=st<en<=L):
                raise ValueError("visual spans must be non-overlapping and in range")
            flat.append((b,st,en))
            previous=en
    return B,Hq,Hkv,L,D,flat


@torch.no_grad()
def forward_native(q,k,v,spans,mask,*,thres_mode="high",
                   mean_mode="image",block_k=32,return_internal=False):
    """GQA/4D-mask forward. Returns same dtype as input Q, never repeats KV."""
    B,Hq,Hkv,L,D,flat=_check(q,k,v,mask,spans,mean_mode,thres_mode,block_k)
    ratio=Hq//Hkv
    S=len(flat)
    bf16=q.dtype==torch.bfloat16
    bd=triton.next_power_of_2(D)
    LSE=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
    base=torch.empty((B,Hq,L,D),device=q.device,dtype=q.dtype)
    _get_lse[(L,B*Hq)](q,k,mask,LSE,L,D,Hq,Hkv,mask.shape[1],ratio,
                        1/math.sqrt(D),bf16,block_k,bd,num_warps=4)
    _get_base[(L,B*Hq)](q,k,v,mask,LSE,base,L,D,Hq,Hkv,mask.shape[1],
                         ratio,1/math.sqrt(D),bf16,block_k,bd,num_warps=4)
    if not S:
        if return_internal:
            return base,dict(lse=LSE,packed=torch.empty((0,3),device=q.device,dtype=torch.int32),
                             thresholds=torch.empty(0,device=q.device,dtype=torch.float32),
                             replacement=torch.zeros((B,Hq),device=q.device,dtype=q.dtype),
                             num_spans=0)
        return base

    packed=torch.tensor(flat,device=q.device,dtype=torch.int32).contiguous()
    partial=torch.empty((S,Hq,L),device=q.device,dtype=torch.float32)
    psq=torch.empty_like(partial)
    pn=torch.empty_like(partial)
    _get_stats[(L,Hq,S)](q,k,mask,LSE,packed,partial,psq,pn,L,D,Hq,Hkv,
                          mask.shape[1],ratio,1/math.sqrt(D),bf16,block_k,bd,num_warps=4)
    count=pn.sum(dim=(1,2))
    total=partial.sum(dim=(1,2))
    total_sq=psq.sum(dim=(1,2))
    avg=total/count.clamp(min=1)
    std=torch.sqrt(torch.clamp(
        (total_sq-total*total/count.clamp(min=1))/(count-1).clamp(min=1),min=0))
    if bf16:
        avg=avg.to(torch.bfloat16)
        std=std.to(torch.bfloat16)
    threshold=avg+THRESH_COEFF[thres_mode]*std
    thresholds=torch.where(count>1,threshold.to(torch.float32),
                            torch.full_like(threshold.to(torch.float32),float("inf")))
    select=torch.zeros((B,L),device=q.device,dtype=torch.bool)
    for b,st,en in flat:
        select[b,st:en]=True
    if mean_mode=="text":select=~select
    # Original CFPO computes a scalar per (B,Hq) from native GQA-repeated V.
    # The source count is in original value dtype; this matters for BF16 rounding.
    denom=select.sum(-1).clamp(min=1).to(v.dtype).view(B,1,1,1)*D
    masked=v.where(select[:,None,:,None],0.)
    replkv=(masked.sum(dim=(2,3),keepdim=True)/denom).view(B,Hkv)
    replacement=replkv.repeat_interleave(ratio,dim=1).contiguous()
    delta=torch.empty((S,Hq,L,D),device=q.device,dtype=q.dtype)
    _get_delta[(L,Hq,S)](q,k,v,mask,LSE,packed,thresholds,replacement,delta,
                         L,D,Hq,Hkv,mask.shape[1],ratio,
                         1/math.sqrt(D),bf16,block_k,bd,num_warps=4)
    y=base.clone()
    for si,(b,st,en) in enumerate(flat):
        y[b]=(y[b]+delta[si]).to(y.dtype)
    if return_internal:
        return y,dict(lse=LSE,packed=packed,thresholds=thresholds,
                      replacement=replacement,num_spans=S,
                      partial_stats_bytes=3*partial.numel()*4,
                      delta_bytes=delta.numel()*delta.element_size())
    return y


class NativeFlashCF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,spans,mode,mean_mode,block_k):
        y,state=forward_native(q,k,v,spans,mask,thres_mode=mode,
                               mean_mode=mean_mode,block_k=block_k,return_internal=True)
        ctx.save_for_backward(q,k,v,mask,y,state["lse"],state["packed"],
                              state["thresholds"],state["replacement"])
        ctx.spans=spans
        ctx.mean_mode=mean_mode
        ctx.nspan=state["num_spans"]
        ctx.bk=block_k
        return y

    @staticmethod
    def backward(ctx,dy):
        q,k,v,mask,y,lse,packed,cutoff,repl=ctx.saved_tensors
        B,Hq,L,D=q.shape
        Hkv=k.shape[1]
        ratio=Hq//Hkv
        bk=ctx.bk
        bd=triton.next_power_of_2(D)
        bf16=q.dtype==torch.bfloat16
        # FP32 partials are essential: source BF16 grads are returned only
        # after accumulating all query heads that share one native KV head.
        dq=torch.empty((B,Hq,L,D),device=q.device,dtype=torch.float32)
        dkparts=torch.empty_like(dq)
        dvparts=torch.empty_like(dq)
        gcpart=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        grid=(L,B*Hq)
        _grad_q[grid](q,k,v,mask,y,dy.contiguous(),lse,packed,cutoff,repl,dq,
                      L,D,Hq,Hkv,mask.shape[1],ratio,ctx.nspan,
                      1/math.sqrt(D),bf16,bk,bd,num_warps=4)
        _grad_kv[grid](q,k,v,mask,y,dy.contiguous(),lse,packed,cutoff,repl,
                       dkparts,dvparts,gcpart,L,D,Hq,Hkv,mask.shape[1],ratio,
                       ctx.nspan,1/math.sqrt(D),bf16,bk,bd,num_warps=4)
        dk=dkparts.view(B,Hkv,ratio,L,D).sum(dim=2)
        dv=dvparts.view(B,Hkv,ratio,L,D).sum(dim=2)
        source=torch.zeros((B,L),device=q.device,dtype=torch.bool)
        for b,group in enumerate(ctx.spans):
            for st,en in group:source[b,st:en]=True
        if ctx.mean_mode=="text":source=~source
        count=source.sum(-1).clamp(min=1).to(torch.float32)
        gmean=gcpart.sum(-1).view(B,Hkv,ratio).sum(dim=2)
        dv=dv+(gmean/(count[:,None]*D))[:,:,None,None]*source[:,None,:,None]
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None,None,None,None,None


def flashcf_native(q,k,v,spans,mask,*,thres_mode="high",mean_mode="image",block_k=32):
    """Native grouped heads + 4D additive mask + BF16/FP32 autograd operator."""
    return NativeFlashCF.apply(q,k,v,mask,spans,thres_mode,mean_mode,block_k)
