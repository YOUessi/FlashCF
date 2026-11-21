"""Tensor Core tiled BF16 softmax-VJP and dQ gradients, plus GEMM dK/dV.

Each program owns a 16/32-query tile and processes grouped key tiles with
matrix multiplies. No full attention matrix is saved; recomputation maintains
BF16 operator semantics under native GQA and real 4D additive masks.
"""
from __future__ import annotations

import math
import torch
import triton
import triton.language as tl

from kernels.triton_native_tile import _tiled_logits
from kernels.triton_native_gemm_kv import _gather_grad_kv_gemm
from kernels.triton_native_bf16 import forward_native


@triton.jit
def _scores_and_dp(Q,K,V,MASK,LSE,SPANS,THRESH,REPL,DY,
                   b,bh,h,kv,qi,kj,ds,
                   L:tl.constexpr,D:tl.constexpr,MH:tl.constexpr,
                   NS:tl.constexpr,SCALE:tl.constexpr,
                   BM:tl.constexpr,BK:tl.constexpr,DELTA_ONLY:tl.constexpr=False):
    vv=tl.load(V+((kv*L+kj[:,None])*D+ds[None,:]),
               mask=(kj[:,None]<L)&(ds[None,:]<D),other=0)
    gy=tl.load(DY+((bh*L+qi[:,None])*D+ds[None,:]),
               mask=(qi[:,None]<L)&(ds[None,:]<D),other=0)
    kk=tl.load(K+((kv*L+kj[:,None])*D+ds[None,:]),
               mask=(kj[:,None]<L)&(ds[None,:]<D),other=0)
    scores=_tiled_logits(Q,K,MASK,b,bh,h,kv,qi,kj,ds,
                          L,D,MH,SCALE,BM,BK)
    lse=tl.load(LSE+bh*L+qi,mask=qi<L,other=float("inf"))
    p32=tl.exp(scores-lse[:,None])
    p32=tl.where(qi[:,None]<L,p32,0.)
    p=p32.to(tl.bfloat16).to(tl.float32)
    chosen=tl.full((BM,BK),False,tl.int1)
    for si in range(NS):
        sb=tl.load(SPANS+si*3)
        first=tl.load(SPANS+si*3+1)
        last=tl.load(SPANS+si*3+2)
        cutoff=tl.load(THRESH+si)
        chosen=chosen | ((sb==b)&(qi[:,None]>=last)&
                         (kj[None,:]>=first)&(kj[None,:]<last)&
                         (p>1.e-8)&(p>cutoff))
    c=tl.load(REPL+bh).to(tl.float32)
    diff=(c-vv.to(tl.float32)).to(tl.bfloat16)
    extra=tl.dot(gy,tl.trans(diff)).to(tl.bfloat16).to(tl.float32)
    if DELTA_ONLY:
        dp=tl.where(chosen,extra,0.).to(tl.bfloat16).to(tl.float32)
    else:
        base=tl.dot(gy,tl.trans(vv)).to(tl.bfloat16).to(tl.float32)
        dp=(base+tl.where(chosen,extra,0.)).to(tl.bfloat16).to(tl.float32)
    return p32,dp,kk


@triton.jit
def _vjp_tile(Q,K,V,MASK,LSE,SPANS,THRESH,REPL,DY,ROWDOT,
              L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
              MH:tl.constexpr,GROUP:tl.constexpr,NS:tl.constexpr,
              SCALE:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,
              DELTA_ONLY:tl.constexpr=False):
    mtile=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    qi=mtile*BM+tl.arange(0,BM)
    ds=tl.arange(0,BD)
    z=tl.full((BM,),0.,tl.float32)
    for st in range(0,L,BK):
        kj=st+tl.arange(0,BK)
        p32,dp,_=_scores_and_dp(Q,K,V,MASK,LSE,SPANS,THRESH,REPL,DY,
                                b,bh,h,kv,qi,kj,ds,L,D,MH,NS,SCALE,BM,BK,DELTA_ONLY)
        z+=tl.sum(p32*dp,1)
    tl.store(ROWDOT+bh*L+qi,z,mask=qi<L)


@triton.jit
def _grad_q_tile(Q,K,V,MASK,LSE,SPANS,THRESH,REPL,DY,ROWDOT,DQ,
                 L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                 MH:tl.constexpr,GROUP:tl.constexpr,NS:tl.constexpr,
                 SCALE:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,
                 DELTA_ONLY:tl.constexpr=False):
    mtile=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    qi=mtile*BM+tl.arange(0,BM)
    ds=tl.arange(0,BD)
    z=tl.load(ROWDOT+bh*L+qi,mask=qi<L,other=0.)
    accum=tl.full((BM,BD),0.,tl.float32)
    for st in range(0,L,BK):
        kj=st+tl.arange(0,BK)
        p32,dp,kk=_scores_and_dp(Q,K,V,MASK,LSE,SPANS,THRESH,REPL,DY,
                                 b,bh,h,kv,qi,kj,ds,L,D,MH,NS,SCALE,BM,BK,DELTA_ONLY)
        dscore=(p32*(dp-z[:,None])).to(tl.bfloat16).to(tl.float32)
        dx=(dscore*SCALE).to(tl.bfloat16)
        accum+=tl.dot(dx,kk)
    tl.store(DQ+((bh*L+qi[:,None])*D+ds[None,:]),accum,
             mask=(qi[:,None]<L)&(ds[None,:]<D))


class NativeFlashCFGemmAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,spans,mode,mean_mode,block_k,
                row_tile,key_tile,grad_q_tile,grad_k_tile):
        y,state=forward_native(q,k,v,spans,mask,thres_mode=mode,
                               mean_mode=mean_mode,block_k=block_k,
                               row_tile=row_tile,key_tile=key_tile,
                               return_internal=True)
        ctx.save_for_backward(q,k,v,mask,state["lse"],state["packed"],
                              state["thresholds"],state["replacement"])
        ctx.spans=spans
        ctx.mean_mode=mean_mode
        ctx.nspan=state["num_spans"]
        ctx.grad_q_tile=grad_q_tile
        ctx.grad_k_tile=grad_k_tile
        return y

    @staticmethod
    def backward(ctx,dy):
        q,k,v,mask,lse,packed,thresholds,repl=ctx.saved_tensors
        B,Hq,L,D=q.shape
        Hkv=k.shape[1]
        group=Hq//Hkv
        bd=triton.next_power_of_2(D)
        grad=dy.contiguous()
        rowdot=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        dq=torch.empty((B,Hq,L,D),device=q.device,dtype=torch.float32)
        dkparts=torch.empty_like(dq)
        dvparts=torch.empty_like(dq)
        gcpart=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        gp=(L,D,Hq,Hkv,mask.shape[1],group,ctx.nspan,
            1/math.sqrt(D),ctx.grad_q_tile,ctx.grad_k_tile,bd)
        grid=(triton.cdiv(L,ctx.grad_q_tile),B*Hq)
        _vjp_tile[grid](q,k,v,mask,lse,packed,thresholds,repl,grad,rowdot,
                        *gp,num_warps=4)
        _grad_q_tile[grid](q,k,v,mask,lse,packed,thresholds,repl,grad,rowdot,dq,
                           *gp,num_warps=4)
        _gather_grad_kv_gemm[(triton.cdiv(L,32),B*Hq)](
            q,k,v,mask,lse,packed,thresholds,repl,grad,rowdot,
            dkparts,dvparts,gcpart,
            L,D,Hq,Hkv,mask.shape[1],group,ctx.nspan,
            1/math.sqrt(D),16,32,bd,num_warps=4)
        dk=dkparts.view(B,Hkv,group,L,D).sum(2)
        dv=dvparts.view(B,Hkv,group,L,D).sum(2)
        selected=torch.zeros((B,L),device=q.device,dtype=torch.bool)
        for b,sps in enumerate(ctx.spans):
            for st,en in sps:selected[b,st:en]=True
        if ctx.mean_mode=="text":selected=~selected
        count=selected.sum(-1).clamp(min=1).float()
        gmean=gcpart.sum(-1).view(B,Hkv,group).sum(-1)
        dv=dv+(gmean/(count[:,None]*D))[:,:,None,None]*selected[:,None,:,None]
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None,None,None,None,None,None,None,None,None


def flashcf_gemm_all(q,k,v,spans,mask,*,thres_mode="high",mean_mode="image",
                     block_k=32,row_tile=16,key_tile=64,
                     grad_q_tile=16,grad_k_tile=64):
    """Experimental Tensor-Core multirow QK forward and Q/K/V backward."""
    if q.dtype!=torch.bfloat16:
        raise ValueError("Native Tensor Core kernel requires BF16")
    if grad_q_tile not in (16,32) or grad_k_tile not in (32,64,128):
        raise ValueError("unsupported backward tile configuration")
    return NativeFlashCFGemmAll.apply(q,k,v,mask,spans,thres_mode,mean_mode,
                                       block_k,row_tile,key_tile,grad_q_tile,grad_k_tile)
