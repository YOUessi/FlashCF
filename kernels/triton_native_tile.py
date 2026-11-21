"""Tensor-Core query-tiled CFPO Attention factual path.

v3 replaces per-query scalar QK scanning in the first two forward passes
with 16/32-query x 64/128-key BF16 GEMM tiles. The original span-moments
and intervention correction kernels remain unchanged for controlled ablation.

Full [L,L] attention is NEVER materialized by this kernel. It is explicitly
a BF16 prefill optimization. FP32 uses the existing non-tiled reference path.

This implementation retains the original half precision rounding of
QK dot -> scaled logits -> additive mask and normalized softmax P.
"""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_logits(Q, K, MASK, b, bh, h, kv, qi, kj, ds,
                  L:tl.constexpr, D:tl.constexpr, MH:tl.constexpr,
                  SCALE:tl.constexpr, BM:tl.constexpr, BK:tl.constexpr):
    q=tl.load(Q+((bh*L+qi[:,None])*D+ds[None,:]),
              mask=(qi[:,None]<L)&(ds[None,:]<D),other=0)
    k=tl.load(K+((kv*L+kj[:,None])*D+ds[None,:]),
              mask=(kj[:,None]<L)&(ds[None,:]<D),other=0)
    x=tl.dot(q,tl.trans(k))
    # PyTorch original matmul on BF16 returns BF16 logits, then scaling
    # and additive bias are performed with BF16 rounding.
    x=x.to(tl.bfloat16).to(tl.float32)
    x=(x*SCALE).to(tl.bfloat16).to(tl.float32)
    mh=0 if MH==1 else h
    bias=tl.load(MASK+((b*MH+mh)*L+qi[:,None])*L+kj[None,:],
                 mask=(qi[:,None]<L)&(kj[None,:]<L),other=-float("inf"))
    bias=bias.to(tl.float32)
    valid=(qi[:,None]<L)&(kj[None,:]<L)&(bias>-1.e4)
    x=(x+tl.where(valid,bias,0.)).to(tl.bfloat16).to(tl.float32)
    return tl.where(valid,x,-float("inf"))


@triton.jit
def _tile_lse(Q,K,MASK,LSE,
              L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
              MH:tl.constexpr,GROUP:tl.constexpr,SCALE:tl.constexpr,
              BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    m_tile=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    qi=m_tile*BM+tl.arange(0,BM)
    ds=tl.arange(0,BD)
    maximum=tl.full((BM,),-float("inf"),tl.float32)
    denom=tl.full((BM,),0.,tl.float32)
    for st in range(0,L,BK):
        kj=st+tl.arange(0,BK)
        x=_tiled_logits(Q,K,MASK,b,bh,h,kv,qi,kj,ds,L,D,MH,SCALE,BM,BK)
        newmax=tl.maximum(maximum,tl.max(x,1))
        newmax=tl.where(qi<L,newmax,0.)
        prev=tl.where(denom>0.,tl.exp(maximum-newmax),0.)
        e=tl.exp(x-newmax[:,None])
        denom=denom*prev+tl.sum(e,1)
        maximum=newmax
    lse=maximum+tl.log(denom)
    tl.store(LSE+bh*L+qi,lse,mask=qi<L)


@triton.jit
def _tile_base(Q,K,V,MASK,LSE,BASE,
               L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
               MH:tl.constexpr,GROUP:tl.constexpr,SCALE:tl.constexpr,
               BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    m_tile=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    qi=m_tile*BM+tl.arange(0,BM)
    ds=tl.arange(0,BD)
    lse=tl.load(LSE+bh*L+qi,mask=qi<L,other=0.)
    accumulator=tl.full((BM,BD),0.,tl.float32)
    for st in range(0,L,BK):
        kj=st+tl.arange(0,BK)
        x=_tiled_logits(Q,K,MASK,b,bh,h,kv,qi,kj,ds,L,D,MH,SCALE,BM,BK)
        # Native CFPO converts original float32 normalized probs back into
        # BF16 *before* the factual Value GEMM.
        prob=tl.exp(x-lse[:,None]).to(tl.bfloat16)
        val=tl.load(V+((kv*L+kj[:,None])*D+ds[None,:]),
                    mask=(kj[:,None]<L)&(ds[None,:]<D),other=0)
        accumulator+=tl.dot(prob,val)
    tl.store(BASE+((bh*L+qi[:,None])*D+ds[None,:]),accumulator,
             mask=(qi[:,None]<L)&(ds[None,:]<D))


def tiled_factual(q,k,v,mask,*,row_tile=16,key_tile=64):
    """Return factual BF16 output + LSE using 2 tiled Tensor Core passes.

    Only for contiguous, full-prefill BF16 and masks [B,1|Hq,L,L].
    Caller validates all rows have >=1 valid key.
    """
    if q.dtype!=torch.bfloat16 or k.dtype!=torch.bfloat16 or v.dtype!=torch.bfloat16:
        raise ValueError("Tiled v3 currently supports BF16 only")
    B,Hq,L,D=q.shape
    Hkv=k.shape[1]
    if Hq%Hkv or v.shape!=k.shape or mask.shape not in ((B,1,L,L),(B,Hq,L,L)):
        raise ValueError("invalid tiled attention dimensions")
    if row_tile not in (16,32) or key_tile not in (32,64,128):
        raise ValueError("row tile=16/32; key tile=32/64/128")
    if any(not x.is_contiguous() for x in (q,k,v,mask)):
        raise ValueError("tiled v3 requires contiguous tensors")
    if D>256:
        raise ValueError("max supported head dimension 256")
    bd=triton.next_power_of_2(D)
    lse=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
    base=torch.empty((B,Hq,L,D),device=q.device,dtype=torch.bfloat16)
    grid=(triton.cdiv(L,row_tile),B*Hq)
    args=(L,D,Hq,Hkv,mask.shape[1],Hq//Hkv,1/math.sqrt(D),
          row_tile,key_tile,bd)
    _tile_lse[grid](q,k,mask,lse,*args,num_warps=4)
    _tile_base[grid](q,k,v,mask,lse,base,*args,num_warps=4)
    return base,lse


class NativeFlashCFTiled(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,spans,mode,mean_mode,block_k,row_tile,key_tile):
        from kernels.triton_native_bf16 import forward_native
        y,state=forward_native(q,k,v,spans,mask,thres_mode=mode,
                               mean_mode=mean_mode,block_k=block_k,
                               row_tile=row_tile,key_tile=key_tile,
                               return_internal=True)
        ctx.save_for_backward(q,k,v,mask,state["lse"],state["packed"],
                              state["thresholds"],state["replacement"])
        ctx.spans=spans
        ctx.mean_mode=mean_mode
        ctx.nspan=state["num_spans"]
        ctx.bk=block_k
        return y

    @staticmethod
    def backward(ctx,grad_y):
        from kernels.triton_native_backward_v2 import NativeFlashCFV2
        return (*NativeFlashCFV2.backward(ctx,grad_y),None,None)


def flashcf_tiled(q,k,v,spans,mask,*,thres_mode="high",mean_mode="image",
                  block_k=32,row_tile=16,key_tile=64):
    """BF16 experimental tiled forward + verified V2 gather backward."""
    return NativeFlashCFTiled.apply(q,k,v,mask,spans,thres_mode,mean_mode,
                                     block_k,row_tile,key_tile)
