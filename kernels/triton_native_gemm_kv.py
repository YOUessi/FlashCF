"""Tensor-Core tiled K/V gradient gather for BF16 grouped attention.

BF16 native GQA, additive 4D mask, fixed hard CFPO intervention mask.
One CTA owns a 16/32-key tile and one query head; it scans 16/32-query tiles.
Each query/key pair is revisited only in low-scratch recomputation, never
materializes [B,Hq,L,L] attention or gradient.

Numerical caution: GEMM accumulation can round differently from the
reference's scalar-vectors, and a BF16 hard gate comparison is discontinuous.
The stored threshold and LSE come from the exact same native-v2 forward.
"""
from __future__ import annotations
import math
import torch
import triton
import triton.language as tl
from kernels.triton_native_backward_v2 import _vjp_rows, _grad_query_v2
from kernels.triton_native_bf16 import forward_native


@triton.jit
def _gather_grad_kv_gemm(
    Q,K,V,Mask,LSE,SPANS,THRESH,REPL,DY,ROWDOT,DKTMP,DVTMP,GCPART,
    L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
    MH:tl.constexpr,GROUP:tl.constexpr,NS:tl.constexpr,
    SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    tile=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    kj=tile*BN+tl.arange(0,BN)
    ds=tl.arange(0,BD)
    k=tl.load(K+(kv*L+kj[:,None])*D+ds[None,:],
              mask=(kj[:,None]<L)&(ds[None,:]<D),other=0)
    v=tl.load(V+(kv*L+kj[:,None])*D+ds[None,:],
              mask=(kj[:,None]<L)&(ds[None,:]<D),other=0)
    replacement=tl.load(REPL+bh).to(tl.float32)
    diff=(replacement-v.to(tl.float32)).to(tl.bfloat16)
    gk=tl.full((BN,BD),0.,tl.float32)
    gv=tl.full((BN,BD),0.,tl.float32)
    gc=tl.full((BN,),0.,tl.float32)
    for start in range(0,L,BM):
        qi=start+tl.arange(0,BM)
        q=tl.load(Q+(bh*L+qi[:,None])*D+ds[None,:],
                  mask=(qi[:,None]<L)&(ds[None,:]<D),other=0)
        gy=tl.load(DY+(bh*L+qi[:,None])*D+ds[None,:],
                   mask=(qi[:,None]<L)&(ds[None,:]<D),other=0)
        lse=tl.load(LSE+bh*L+qi,mask=qi<L,other=float("inf"))
        z=tl.load(ROWDOT+bh*L+qi,mask=qi<L,other=0.)
        scores=tl.dot(q,tl.trans(k)).to(tl.bfloat16).to(tl.float32)
        scores=(scores*SCALE).to(tl.bfloat16).to(tl.float32)
        mh=0 if MH==1 else h
        additive=tl.load(Mask+((b*MH+mh)*L+qi[:,None])*L+kj[None,:],
                         mask=(qi[:,None]<L)&(kj[None,:]<L),
                         other=-float("inf")).to(tl.float32)
        valid=(qi[:,None]<L)&(kj[None,:]<L)&(additive>-1.e4)
        scores=(scores+tl.where(valid,additive,0.)).to(tl.bfloat16).to(tl.float32)
        scores=tl.where(valid,scores,-float("inf"))
        probs32=tl.where(valid,tl.exp(scores-lse[:,None]),0.)
        probs=probs32.to(tl.bfloat16)
        active=tl.full((BM,BN),False,tl.int1)
        for si in range(NS):
            sb=tl.load(SPANS+si*3)
            first=tl.load(SPANS+si*3+1)
            last=tl.load(SPANS+si*3+2)
            cutoff=tl.load(THRESH+si)
            active=active | ((sb==b)&(qi[:,None]>=last)&
                             (kj[None,:]>=first)&(kj[None,:]<last)&
                             (probs.to(tl.float32)>1.e-8)&
                             (probs.to(tl.float32)>cutoff))
        grad_base=tl.dot(gy,tl.trans(v)).to(tl.bfloat16).to(tl.float32)
        grad_extra=tl.dot(gy,tl.trans(diff)).to(tl.bfloat16).to(tl.float32)
        grad_prob=(grad_base+tl.where(active,grad_extra,0.)).to(tl.bfloat16).to(tl.float32)
        dlogit=(probs32*(grad_prob-z[:,None])).to(tl.bfloat16).to(tl.float32)
        dx=(dlogit*SCALE).to(tl.bfloat16)
        gk+=tl.dot(tl.trans(dx),q)
        masked_prob=tl.where(active,0.,probs.to(tl.float32)).to(tl.bfloat16)
        gv+=tl.dot(tl.trans(masked_prob),gy)
        gc+=tl.sum(tl.where(active,probs.to(tl.float32),0.)*
                    tl.sum(gy.to(tl.float32),1)[:,None],0)
    tl.store(DKTMP+((bh*L+kj[:,None])*D+ds[None,:]),gk,
             mask=(kj[:,None]<L)&(ds[None,:]<D))
    tl.store(DVTMP+((bh*L+kj[:,None])*D+ds[None,:]),gv,
             mask=(kj[:,None]<L)&(ds[None,:]<D))
    tl.store(GCPART+bh*L+kj,gc,mask=kj<L)


class NativeFlashCFGemmKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,spans,mode,mean_mode,block_k,row_tile,key_tile,
                grad_query_tile,grad_key_tile):
        y,state=forward_native(q,k,v,spans,mask,thres_mode=mode,
                               mean_mode=mean_mode,block_k=block_k,
                               row_tile=row_tile,key_tile=key_tile,return_internal=True)
        ctx.save_for_backward(q,k,v,mask,state["lse"],state["packed"],
                              state["thresholds"],state["replacement"])
        ctx.spans=spans
        ctx.mean_mode=mean_mode
        ctx.nspan=state["num_spans"]
        ctx.bk=block_k
        ctx.grad_query_tile=grad_query_tile
        ctx.grad_key_tile=grad_key_tile
        return y

    @staticmethod
    def backward(ctx,dy):
        q,k,v,mask,lse,packed,thresholds,repl=ctx.saved_tensors
        B,Hq,L,D=q.shape
        Hkv=k.shape[1]
        group=Hq//Hkv
        bd=triton.next_power_of_2(D)
        src=[q,k,v,mask,lse,packed,thresholds,repl]
        go=dy.contiguous()
        rowdot=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        dq=torch.empty((B,Hq,L,D),device=q.device,dtype=torch.float32)
        dkparts=torch.empty_like(dq)
        dvparts=torch.empty_like(dq)
        gcpart=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        rowgrid=(L,B*Hq)
        args=(L,D,Hq,Hkv,mask.shape[1],group,ctx.nspan,1/math.sqrt(D),
              True,ctx.bk,bd)
        _vjp_rows[rowgrid](q,k,v,mask,lse,packed,thresholds,repl,go,rowdot,
                           *args,num_warps=4)
        _grad_query_v2[rowgrid](q,k,v,mask,lse,packed,thresholds,repl,go,rowdot,dq,
                                *args,num_warps=4)
        _gather_grad_kv_gemm[(triton.cdiv(L,ctx.grad_key_tile),B*Hq)](
            q,k,v,mask,lse,packed,thresholds,repl,go,rowdot,dkparts,dvparts,gcpart,
            L,D,Hq,Hkv,mask.shape[1],group,ctx.nspan,1/math.sqrt(D),
            ctx.grad_query_tile,ctx.grad_key_tile,bd,num_warps=4)
        dk=dkparts.view(B,Hkv,group,L,D).sum(dim=2)
        dv=dvparts.view(B,Hkv,group,L,D).sum(dim=2)
        sel=torch.zeros((B,L),device=q.device,dtype=torch.bool)
        for b,group_spans in enumerate(ctx.spans):
            for start,end in group_spans:sel[b,start:end]=True
        if ctx.mean_mode=="text":sel=~sel
        norm=sel.sum(-1).clamp(min=1).to(torch.float32)
        gmean=gcpart.sum(-1).view(B,Hkv,group).sum(-1)
        dv=dv+(gmean/(norm[:,None]*D))[:,:,None,None]*sel[:,None,:,None]
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None,None,None,None,None,None,None,None,None


def flashcf_gemm_kv(q,k,v,spans,mask,*,thres_mode="high",mean_mode="image",
                    block_k=32,row_tile=16,key_tile=64,
                    grad_query_tile=16,grad_key_tile=32):
    """Tensor Core factual+KV-backward; v2 query backward unchanged."""
    if q.dtype!=torch.bfloat16:
        raise ValueError("Tensor Core GEMM KV kernel currently BF16-only")
    if grad_query_tile not in (16,32) or grad_key_tile not in (16,32,64):
        raise ValueError("gradient query/key tiles 16/32 and 16/32/64")
    return NativeFlashCFGemmKV.apply(q,k,v,mask,spans,thres_mode,mean_mode,
                                      block_k,row_tile,key_tile,grad_query_tile,grad_key_tile)
