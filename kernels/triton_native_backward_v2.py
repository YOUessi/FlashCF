"""Numerically faithful BF16 native-GQA Triton backward v2 (experimental).

The v1 backward used dlogit = P_bf16 * dot(dY,W-Y_bf16). That identity
holds in full precision but NOT exactly across the BF16 output rounding
and P_fp32 -> P_bf16 cast in the original CFPO eager call graph.

This version instead:
1. recomputes FP32 pre-cast softmax probabilities;
2. reconstructs BF16 dL/dP from the two original GEMM branches;
3. computes one row-wise softmax Jacobian reduction z=sum(P_fp32*dL/dP);
4. computes dlogit=P_fp32*(dL/dP-z) and emulates BF16 gradient casts;
5. gathers gradients by key for native GQA heads with no atomics.

No O(L²) saved tensor. Extra row-vjp kernel increases arithmetic.
Do not assume this variant is faster; measure separately.
"""
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

from kernels.triton_native_bf16 import (
    forward_native, _attention_logits, _prob,
)


@triton.jit
def _grad_p(qvec,vblock,go,coeff,repl,
            BF16:tl.constexpr):
    """dL/dP through (P@V) + ((P*gate)@(repl-V)), with BF16 boundaries."""
    base=tl.sum(vblock*go[None,:],1)
    diff=repl-vblock
    if BF16:
        diff=diff.to(tl.bfloat16).to(tl.float32)
        base=base.to(tl.bfloat16).to(tl.float32)
    add=tl.sum(diff*go[None,:],1)
    if BF16:
        add=add.to(tl.bfloat16).to(tl.float32)
    value=base+tl.where(coeff,add,0.)
    if BF16:
        value=value.to(tl.bfloat16).to(tl.float32)
    return value


@triton.jit
def _get_score_grads(dprobs,psoftmax,z,
                     SCALE:tl.constexpr,BF16:tl.constexpr):
    dscore=psoftmax*(dprobs-z)
    if BF16:
        dscore=dscore.to(tl.bfloat16).to(tl.float32)
    dx=dscore*SCALE
    if BF16:
        dx=dx.to(tl.bfloat16).to(tl.float32)
    return dx


@triton.jit
def _vjp_rows(Q,K,V,Mask,LSE,SPANS,THRESH,REPL,DY,ROWDOT,
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
    gy=tl.load(DY+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    lse=tl.load(LSE+bh*L+i)
    c=tl.load(REPL+bh).to(tl.float32)
    z=tl.full((),0.,tl.float32)
    for st in range(0,L,BK):
        js=st+tl.arange(0,BK)
        ptr=(kv*L+js[:,None])*D+ds[None,:]
        kk=tl.load(K+ptr,mask=(js[:,None]<L)&(ds[None,:]<D),other=0).to(tl.float32)
        vv=tl.load(V+ptr,mask=(js[:,None]<L)&(ds[None,:]<D),other=0).to(tl.float32)
        score,allowed=_attention_logits(q,kk,Mask,b,h,i,js,L,MH,D,SCALE,BF16)
        p32=tl.where(allowed,tl.exp(score-lse),0.)
        p= p32.to(tl.bfloat16).to(tl.float32) if BF16 else p32
        hit=tl.full((BK,),False,tl.int1)
        for si in range(NS):
            sb=tl.load(SPANS+si*3)
            a=tl.load(SPANS+si*3+1)
            e=tl.load(SPANS+si*3+2)
            cut=tl.load(THRESH+si)
            hit=hit|((sb==b)&(i>=e)&(js>=a)&(js<e)&(p>1.e-8)&(p>cut))
        grad_p=_grad_p(q,vv,gy,hit,c,BF16)
        z+=tl.sum(p32*grad_p,0)
    tl.store(ROWDOT+bh*L+i,z)


@triton.jit
def _grad_query_v2(Q,K,V,Mask,LSE,SPANS,THRESH,REPL,DY,ROWDOT,DQ,
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
    gy=tl.load(DY+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    lse=tl.load(LSE+bh*L+i)
    c=tl.load(REPL+bh).to(tl.float32)
    z=tl.load(ROWDOT+bh*L+i)
    accum=tl.full((BD,),0.,tl.float32)
    for st in range(0,L,BK):
        js=st+tl.arange(0,BK)
        ptr=(kv*L+js[:,None])*D+ds[None,:]
        kk=tl.load(K+ptr,mask=(js[:,None]<L)&(ds[None,:]<D),other=0).to(tl.float32)
        vv=tl.load(V+ptr,mask=(js[:,None]<L)&(ds[None,:]<D),other=0).to(tl.float32)
        score,allowed=_attention_logits(q,kk,Mask,b,h,i,js,L,MH,D,SCALE,BF16)
        p32=tl.where(allowed,tl.exp(score-lse),0.)
        p=p32.to(tl.bfloat16).to(tl.float32) if BF16 else p32
        hit=tl.full((BK,),False,tl.int1)
        for si in range(NS):
            sb=tl.load(SPANS+si*3)
            a=tl.load(SPANS+si*3+1)
            e=tl.load(SPANS+si*3+2)
            cut=tl.load(THRESH+si)
            hit=hit|((sb==b)&(i>=e)&(js>=a)&(js<e)&(p>1.e-8)&(p>cut))
        grad_p=_grad_p(q,vv,gy,hit,c,BF16)
        dx=_get_score_grads(grad_p,p32,z,SCALE,BF16)
        accum+=tl.sum(dx[:,None]*kk,0)
    tl.store(DQ+(bh*L+i)*D+ds,accum,mask=ds<D)


@triton.jit
def _grad_key_v2(Q,K,V,Mask,LSE,SPANS,THRESH,REPL,DY,ROWDOT,DKTMP,DVTMP,GCPART,
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
    c=tl.load(REPL+bh).to(tl.float32)
    dk=tl.full((BD,),0.,tl.float32)
    dv=tl.full((BD,),0.,tl.float32)
    gc=tl.full((),0.,tl.float32)
    for st in range(0,L,BQ):
        is_=st+tl.arange(0,BQ)
        valid=is_<L
        ptr=(bh*L+is_[:,None])*D+ds[None,:]
        qs=tl.load(Q+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        gy=tl.load(DY+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        lse=tl.load(LSE+bh*L+is_,mask=valid,other=float('inf'))
        z=tl.load(ROWDOT+bh*L+is_,mask=valid,other=0.)
        score=tl.sum(qs*k[None,:],1)
        if BF16:
            score=score.to(tl.bfloat16).to(tl.float32)
        score=score*SCALE
        if BF16:
            score=score.to(tl.bfloat16).to(tl.float32)
        mh=0 if MH==1 else h
        bias=tl.load(Mask+((b*MH+mh)*L+is_)*L+j,mask=valid,
                     other=-float('inf')).to(tl.float32)
        allowed=valid&(bias>-1.e4)
        score=score+tl.where(allowed,bias,0.)
        if BF16:
            score=score.to(tl.bfloat16).to(tl.float32)
        p32=tl.where(allowed,tl.exp(score-lse),0.)
        p=p32.to(tl.bfloat16).to(tl.float32) if BF16 else p32
        hit=tl.full((BQ,),False,tl.int1)
        for si in range(NS):
            sb=tl.load(SPANS+si*3)
            a=tl.load(SPANS+si*3+1)
            e=tl.load(SPANS+si*3+2)
            cut=tl.load(THRESH+si)
            hit=hit|((sb==b)&(is_>=e)&(j>=a)&(j<e)&(p>1.e-8)&(p>cut))
        base=tl.sum(gy*v[None,:],1)
        diff=c-v
        if BF16:
            base=base.to(tl.bfloat16).to(tl.float32)
            diff=diff.to(tl.bfloat16).to(tl.float32)
        add=tl.sum(gy*diff[None,:],1)
        if BF16:
            add=add.to(tl.bfloat16).to(tl.float32)
        grad_p=base+tl.where(hit,add,0.)
        if BF16:
            grad_p=grad_p.to(tl.bfloat16).to(tl.float32)
        dx=_get_score_grads(grad_p,p32,z,SCALE,BF16)
        dk+=tl.sum(dx[:,None]*qs,0)
        dv+=tl.sum(p[:,None]*tl.where(hit[:,None],0.,gy),0)
        gc+=tl.sum(tl.where(hit,p,0.)*tl.sum(gy,1),0)
    tl.store(DKTMP+(bh*L+j)*D+ds,dk,mask=ds<D)
    tl.store(DVTMP+(bh*L+j)*D+ds,dv,mask=ds<D)
    tl.store(GCPART+bh*L+j,gc)


class NativeFlashCFV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,spans,mode,mean_mode,block_k):
        y,state=forward_native(q,k,v,spans,mask,thres_mode=mode,
                                mean_mode=mean_mode,block_k=block_k,return_internal=True)
        ctx.save_for_backward(q,k,v,mask,state["lse"],state["packed"],
                              state["thresholds"],state["replacement"])
        ctx.spans=spans
        ctx.mean_mode=mean_mode
        ctx.nspan=state["num_spans"]
        ctx.bk=block_k
        return y

    @staticmethod
    def backward(ctx,dy):
        q,k,v,mask,lse,packed,thresholds,repl=ctx.saved_tensors
        B,Hq,L,D=q.shape
        Hkv=k.shape[1]
        ratio=Hq//Hkv
        bk=ctx.bk
        bd=triton.next_power_of_2(D)
        bf16=q.dtype==torch.bfloat16
        dquery=torch.empty((B,Hq,L,D),device=q.device,dtype=torch.float32)
        dkparts=torch.empty_like(dquery)
        dvparts=torch.empty_like(dquery)
        gcpart=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        rowdot=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        shape=(L,B*Hq)
        _vjp_rows[shape](q,k,v,mask,lse,packed,thresholds,repl,dy.contiguous(),rowdot,
                         L,D,Hq,Hkv,mask.shape[1],ratio,ctx.nspan,
                         1/math.sqrt(D),bf16,bk,bd,num_warps=4)
        _grad_query_v2[shape](q,k,v,mask,lse,packed,thresholds,repl,dy.contiguous(),rowdot,dquery,
                              L,D,Hq,Hkv,mask.shape[1],ratio,ctx.nspan,
                              1/math.sqrt(D),bf16,bk,bd,num_warps=4)
        _grad_key_v2[shape](q,k,v,mask,lse,packed,thresholds,repl,dy.contiguous(),rowdot,
                            dkparts,dvparts,gcpart,
                            L,D,Hq,Hkv,mask.shape[1],ratio,ctx.nspan,
                            1/math.sqrt(D),bf16,bk,bd,num_warps=4)
        dk=dkparts.view(B,Hkv,ratio,L,D).sum(dim=2)
        dv=dvparts.view(B,Hkv,ratio,L,D).sum(dim=2)
        selected=torch.zeros((B,L),device=q.device,dtype=torch.bool)
        for b,group in enumerate(ctx.spans):
            for st,en in group:selected[b,st:en]=True
        if ctx.mean_mode=="text":selected=~selected
        cnt=selected.sum(-1).clamp(min=1).to(torch.float32)
        gmean=gcpart.sum(-1).view(B,Hkv,ratio).sum(-1)
        dv=dv+(gmean/(cnt[:,None]*D))[:,:,None,None]*selected[:,None,:,None]
        return dquery.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None,None,None,None,None


def flashcf_native_v2(q,k,v,spans,mask,*,thres_mode="high",mean_mode="image",block_k=32):
    return NativeFlashCFV2.apply(q,k,v,mask,spans,thres_mode,mean_mode,block_k)
