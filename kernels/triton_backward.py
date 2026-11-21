"""FlashCF initial GPU backward, based on causal forward's compact saved state.

The first version launches one row/head CTA and uses atomic adds to K/V
gradients. It is a correctness prototype, NOT the optimized reduction kernel.
It does NOT materialize [B,H,L,L] attention probabilities.

Supported: contiguous FP32, causal attention, zero dropout, pre-repeated GQA
heads, per-span hard threshold. No second derivatives.
"""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl

from kernels.triton_forward import flashcf_forward


@triton.jit
def _bwd_rows(Q,K,V,Y,DY,LSE,SPANS,THRESH,REPLACEMENT,GQ,GK,GV,GMEAN,
              L:tl.constexpr,D:tl.constexpr,H:tl.constexpr,
              NSPANS:tl.constexpr,SCALE:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//H
    h=bh%H
    ds=tl.arange(0,BD)
    q=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    y=tl.load(Y+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    gy=tl.load(DY+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    gy_sum=tl.sum(gy,0)
    lse=tl.load(LSE+bh*L+i)
    c=tl.load(REPLACEMENT+bh)
    dq=tl.full((BD,),0.,tl.float32)
    gmean=tl.full((),0.,tl.float32)
    for start in range(0,i+1,BK):
        js=start+tl.arange(0,BK)
        valid=js<=i
        ptr=(bh*L+js[:,None])*D+ds[None,:]
        k=tl.load(K+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        v=tl.load(V+ptr,mask=valid[:,None]&(ds[None,:]<D),other=0).to(tl.float32)
        score=tl.sum(k*q[None,:],1)*SCALE
        p=tl.exp(score-lse)
        p=tl.where(valid,p,0.)
        active=tl.full((BK,),False,tl.int1)
        for si in range(NSPANS):
            bb=tl.load(SPANS+si*3+0)
            a=tl.load(SPANS+si*3+1)
            e=tl.load(SPANS+si*3+2)
            threshold=tl.load(THRESH+si)
            active=active | ((bb==b)&(i>=e)&(js>=a)&(js<e)
                             &(p>1.0e-8)&(p>threshold))
        # When selected, W_j is the scalar (per-batch/head) mean of the
        # original source V tokens. Otherwise W_j is original V_j.
        w=tl.where(active[:,None],c,v)
        dlogit=p*tl.sum((w-y[None,:])*gy[None,:],1)
        dq+=tl.sum(dlogit[:,None]*k,0)*SCALE
        gk=dlogit[:,None]*q[None,:]*SCALE
        gv=p[:,None]*tl.where(active[:,None],0.,gy[None,:])
        tl.atomic_add(GK+ptr,gk,mask=valid[:,None]&(ds[None,:]<D),sem="relaxed")
        tl.atomic_add(GV+ptr,gv,mask=valid[:,None]&(ds[None,:]<D),sem="relaxed")
        gmean+=tl.sum(tl.where(active,p,0.),0)*gy_sum
    tl.store(GQ+(bh*L+i)*D+ds,dq,mask=ds<D)
    tl.atomic_add(GMEAN+bh,gmean,sem="relaxed")


class FlashCFAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,spans,thres_mode,mean_mode,block_k):
        y,metadata=flashcf_forward(q,k,v,spans,thres_mode=thres_mode,
                                  mean_mode=mean_mode,block_k=block_k,
                                  return_diagnostics=True,return_internal=True)
        lse=metadata["_lse"]
        packed=metadata["_packed"]
        thresh=metadata["_thresholds"]
        replacement=metadata["_replacement"]
        ctx.save_for_backward(q,k,v,y,lse,packed,thresh,replacement)
        ctx.spans=spans
        ctx.mean_mode=mean_mode
        ctx.block_k=block_k
        ctx.num_spans=metadata["num_spans"]
        return y

    @staticmethod
    def backward(ctx,grad_output):
        q,k,v,y,lse,packed,thresh,replacement=ctx.saved_tensors
        B,H,L,D=q.shape
        gq=torch.empty_like(q)
        gk=torch.zeros_like(k)
        gv=torch.zeros_like(v)
        gmean=torch.zeros((B,H),device=q.device,dtype=torch.float32)
        BK=ctx.block_k
        BD=triton.next_power_of_2(D)
        _bwd_rows[(L,B*H)](
            q,k,v,y,grad_output.contiguous(),lse,packed,thresh,replacement,
            gq,gk,gv,gmean,L,D,H,ctx.num_spans,1/math.sqrt(D),BK,BD,
            num_warps=4)
        # The hard threshold masks have no derivative; only the Value mean
        # replacement has a global derivative through all selected source V.
        choose=torch.zeros((B,L),device=q.device,dtype=torch.bool)
        for b,group in enumerate(ctx.spans):
            for st,en in group:choose[b,st:en]=True
        if ctx.mean_mode=="text":choose=~choose
        den=choose.sum(-1).clamp(min=1).to(torch.float32)*D
        gv=gv + (gmean/den[:,None])[:,:,None,None]*choose[:,None,:,None]
        return gq,gk,gv,None,None,None,None


def flashcf_autograd(q,k,v,spans,*,thres_mode="high",mean_mode="image",block_k=32):
    """Custom forward/backward CUDA path (v1, no double backward)."""
    return FlashCFAutograd.apply(q,k,v,spans,thres_mode,mean_mode,block_k)
