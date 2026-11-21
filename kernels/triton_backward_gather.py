"""Race-free two-pass Triton backward for FlashCF.

Pass Q: each query row owns dQ and gathers contributions from all earlier keys.
Pass KV: each key position owns dK/dV and gathers contributions from later rows.
No atomics and no [L,L] attention materialization.

This is a research prototype: FP32, causal, zero dropout, padded masks not yet
supported, and no second derivative. It trades extra QK recomputation and
memory reads for deterministic gradient accumulation and better long-length scaling.
"""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl
from kernels.triton_forward import flashcf_forward


@triton.jit
def _grad_query(Q,K,V,Y,DY,LSE,SPANS,THRESH,REPLACEMENT,GQ,
                L:tl.constexpr,D:tl.constexpr,H:tl.constexpr,NSPANS:tl.constexpr,
                SCALE:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//H
    ds=tl.arange(0,BD)
    q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    y=tl.load(Y+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    go=tl.load(DY+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    lse=tl.load(LSE+bh*L+i)
    c=tl.load(REPLACEMENT+bh)
    dq=tl.full((BD,),0.0,tl.float32)
    for start in range(0,i+1,BK):
        js=start+tl.arange(0,BK)
        valid=js<=i
        ptr=(bh*L+js[:,None])*D+ds[None,:]
        k=tl.load(K+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        v=tl.load(V+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        p=tl.exp(tl.sum(k*q[None,:],1)*SCALE-lse)
        p=tl.where(valid,p,0.0)
        active=tl.full((BK,),False,tl.int1)
        for si in range(NSPANS):
            bs=tl.load(SPANS+si*3)
            a=tl.load(SPANS+si*3+1)
            e=tl.load(SPANS+si*3+2)
            threshold=tl.load(THRESH+si)
            active=active | ((bs==b)&(i>=e)&(js>=a)&(js<e)&(p>1.e-8)&(p>threshold))
        w=tl.where(active[:,None],c,v)
        dscore=p*tl.sum((w-y[None,:])*go[None,:],1)
        dq+=tl.sum(dscore[:,None]*k,0)*SCALE
    tl.store(GQ+(bh*L+i)*D+ds,dq,mask=ds<D)


@triton.jit
def _grad_keys(Q,K,V,Y,DY,LSE,SPANS,THRESH,REPLACEMENT,GK,GV,GC_PART,
               L:tl.constexpr,D:tl.constexpr,H:tl.constexpr,NSPANS:tl.constexpr,
               SCALE:tl.constexpr,BQ:tl.constexpr,BD:tl.constexpr):
    j=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//H
    ds=tl.arange(0,BD)
    k=tl.load(K+(bh*L+j)*D+ds,mask=ds<D,other=0).to(tl.float32)
    v=tl.load(V+(bh*L+j)*D+ds,mask=ds<D,other=0).to(tl.float32)
    c=tl.load(REPLACEMENT+bh)
    dk=tl.full((BD,),0.0,tl.float32)
    dv=tl.full((BD,),0.0,tl.float32)
    gc=tl.full((),0.0,tl.float32)
    for start in range(j,L,BQ):
        rows=start+tl.arange(0,BQ)
        valid=rows<L
        ptr=(bh*L+rows[:,None])*D+ds[None,:]
        qs=tl.load(Q+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        ys=tl.load(Y+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        gs=tl.load(DY+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        lse=tl.load(LSE+bh*L+rows,mask=valid,other=0.0)
        p=tl.exp(tl.sum(qs*k[None,:],1)*SCALE-lse)
        p=tl.where(valid,p,0.)
        active=tl.full((BQ,),False,tl.int1)
        for si in range(NSPANS):
            bs=tl.load(SPANS+si*3)
            a=tl.load(SPANS+si*3+1)
            e=tl.load(SPANS+si*3+2)
            threshold=tl.load(THRESH+si)
            active=active | ((bs==b)&(rows>=e)&(j>=a)&(j<e)&
                             (p>1.e-8)&(p>threshold))
        w=tl.where(active[:,None],c,v[None,:])
        dscore=p*tl.sum((w-ys)*gs,1)
        dk+=tl.sum(dscore[:,None]*qs,0)*SCALE
        dv+=tl.sum(p[:,None]*tl.where(active[:,None],0.,gs),0)
        gc+=tl.sum(tl.where(active,p,0.)*tl.sum(gs,1),0)
    tl.store(GK+(bh*L+j)*D+ds,dk,mask=ds<D)
    tl.store(GV+(bh*L+j)*D+ds,dv,mask=ds<D)
    tl.store(GC_PART+bh*L+j,gc)


class FlashCFGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,spans,mode,mean_mode,block_k):
        out,state=flashcf_forward(q,k,v,spans,thres_mode=mode,mean_mode=mean_mode,
                                  block_k=block_k,return_diagnostics=True,return_internal=True)
        ctx.save_for_backward(q,k,v,out,state["_lse"],state["_packed"],
                              state["_thresholds"],state["_replacement"])
        ctx.spans=spans
        ctx.mean_mode=mean_mode
        ctx.block_k=block_k
        ctx.nspans=state["num_spans"]
        return out

    @staticmethod
    def backward(ctx,go):
        q,k,v,y,lse,packed,thresholds,repl=ctx.saved_tensors
        B,H,L,D=q.shape
        dk=torch.empty_like(k)
        dq=torch.empty_like(q)
        dv=torch.empty_like(v)
        gcp=torch.empty((B,H,L),device=q.device,dtype=torch.float32)
        bd=triton.next_power_of_2(D)
        bk=ctx.block_k
        _grad_query[(L,B*H)](q,k,v,y,go.contiguous(),lse,packed,thresholds,repl,dq,
                             L,D,H,ctx.nspans,1/math.sqrt(D),bk,bd,num_warps=4)
        _grad_keys[(L,B*H)](q,k,v,y,go.contiguous(),lse,packed,thresholds,repl,
                            dk,dv,gcp,L,D,H,ctx.nspans,1/math.sqrt(D),bk,bd,num_warps=4)
        gc=gcp.sum(-1)
        mask=torch.zeros((B,L),device=q.device,dtype=torch.bool)
        for b,group in enumerate(ctx.spans):
            for st,en in group:
                mask[b,st:en]=True
        if ctx.mean_mode=="text":mask=~mask
        den=mask.sum(-1).clamp(min=1).to(torch.float32)*D
        dv=dv+(gc/den[:,None])[:,:,None,None]*mask[:,None,:,None]
        return dq,dk,dv,None,None,None,None


def flashcf_gather(q,k,v,spans,*,thres_mode="high",mean_mode="image",block_k=32):
    return FlashCFGather.apply(q,k,v,spans,thres_mode,mean_mode,block_k)
