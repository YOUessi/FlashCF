"""Diagnostic-only hard CMVE gate extraction for native BF16 Triton.

WARNING: intentionally materializes [B,Hq,L,L] UINT8 mask to compare
elementwise against the original CFPO dense mask on small test sequences.
This module is NEVER called by the production bounded-memory FlashCF path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernels.triton_native_bf16 import _prob,forward_native


@triton.jit
def _write_mask(Q,K,Mask,LSE,SPANS,THRESH,OUT,
                L:tl.constexpr,D:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,
                MH:tl.constexpr,GROUP:tl.constexpr,NS:tl.constexpr,
                SCALE:tl.constexpr,BF16:tl.constexpr,
                BK:tl.constexpr,BD:tl.constexpr):
    i=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//HQ
    h=bh%HQ
    kv=b*HK+h//GROUP
    ds=tl.arange(0,BD)
    qs=tl.load(Q+(bh*L+i)*D+ds,mask=ds<D,other=0).to(tl.float32)
    lse=tl.load(LSE+bh*L+i)
    for j0 in range(0,L,BK):
        js=j0+tl.arange(0,BK)
        kk=tl.load(K+(kv*L+js[:,None])*D+ds[None,:],
                   mask=(js[:,None]<L)&(ds[None,:]<D),other=0).to(tl.float32)
        p,_=_prob(qs,kk,Mask,lse,b,h,i,js,L,MH,D,SCALE,BF16)
        hit=tl.full((BK,),False,tl.int1)
        for si in range(NS):
            sb=tl.load(SPANS+si*3)
            st=tl.load(SPANS+si*3+1)
            en=tl.load(SPANS+si*3+2)
            threshold=tl.load(THRESH+si)
            hit=hit | ((sb==b)&(i>=en)&(js>=st)&(js<en)&
                         (p>1.e-8)&(p>threshold))
        tl.store(OUT+bh*L*L+i*L+js,hit.to(tl.uint8),mask=js<L)


@torch.no_grad()
def diagnostic_triton_gate(q,k,v,spans,mask,*,thres_mode="high",
                           mean_mode="image",block_k=32):
    y,state=forward_native(q,k,v,spans,mask,thres_mode=thres_mode,
                            mean_mode=mean_mode,block_k=block_k,
                            return_internal=True)
    B,Hq,L,D=q.shape
    Hkv=k.shape[1]
    from math import sqrt
    result=torch.empty((B,Hq,L,L),device=q.device,dtype=torch.uint8)
    _write_mask[(L,B*Hq)](
        q,k,mask,state["lse"],state["packed"],state["thresholds"],result,
        L,D,Hq,Hkv,mask.shape[1],Hq//Hkv,state["num_spans"],
        1/sqrt(D),q.dtype==torch.bfloat16,
        block_k,triton.next_power_of_2(D),num_warps=4)
    return result.bool(),state
