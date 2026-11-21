"""Direct bounded-memory CMVE correction with factual FlashAttention.

No double evaluation of the factual branch: the factual branch is run once
through PyTorch CUDA FlashAttention and the correction-only branch is run once
through Triton. The custom correction-only backward differentiates the
selected value delta, while ordinary factual autograd is handled by PyTorch.

BF16, native GQA, prefill and additive 4D mask; zero dropout/noise.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend,sdpa_kernel
from kernels.triton_native_bf16 import forward_native,_check
from kernels.triton_native_qtile import _vjp_tile,_grad_q_tile
from kernels.triton_native_gemm_kv import _gather_grad_kv_gemm


class CMVEDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,spans,mode,mean_mode,block_k,row_tile,key_tile,grad_q_tile,grad_k_tile):
        correction,state=forward_native(q,k,v,spans,mask,
            thres_mode=mode,mean_mode=mean_mode,block_k=block_k,
            row_tile=row_tile,key_tile=key_tile,return_internal=True,
            delta_only=True)
        ctx.save_for_backward(q,k,v,mask,state["lse"],state["packed"],
                              state["thresholds"],state["replacement"])
        ctx.spans=spans
        ctx.mean_mode=mean_mode
        ctx.num_spans=state["num_spans"]
        ctx.grad_q_tile=grad_q_tile
        ctx.grad_k_tile=grad_k_tile
        return correction

    @staticmethod
    def backward(ctx,grad_output):
        q,k,v,mask,lse,packed,thresholds,replacement=ctx.saved_tensors
        B,Hq,L,D=q.shape
        Hkv=k.shape[1]
        group=Hq//Hkv
        bd=triton.next_power_of_2(D)
        dy=grad_output.contiguous()
        z=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        dq=torch.empty((B,Hq,L,D),device=q.device,dtype=torch.float32)
        dkpartial=torch.empty_like(dq)
        dvpartial=torch.empty_like(dq)
        meanpartial=torch.empty((B,Hq,L),device=q.device,dtype=torch.float32)
        common=(L,D,Hq,Hkv,mask.shape[1],group,ctx.num_spans,
                1/math.sqrt(D),ctx.grad_q_tile,ctx.grad_k_tile,bd,True)
        grid=(triton.cdiv(L,ctx.grad_q_tile),B*Hq)
        _vjp_tile[grid](q,k,v,mask,lse,packed,thresholds,replacement,dy,z,
                        *common,num_warps=4)
        _grad_q_tile[grid](q,k,v,mask,lse,packed,thresholds,replacement,dy,z,dq,
                           *common,num_warps=4)
        _gather_grad_kv_gemm[(triton.cdiv(L,32),B*Hq)](
            q,k,v,mask,lse,packed,thresholds,replacement,dy,z,
            dkpartial,dvpartial,meanpartial,
            L,D,Hq,Hkv,mask.shape[1],group,ctx.num_spans,1/math.sqrt(D),
            16,32,bd,True,num_warps=4)
        dk=dkpartial.reshape(B,Hkv,group,L,D).sum(2)
        dv=dvpartial.reshape(B,Hkv,group,L,D).sum(2)
        select=torch.zeros((B,L),device=q.device,dtype=torch.bool)
        for b,regions in enumerate(ctx.spans):
            for start,end in regions:select[b,start:end]=True
        if ctx.mean_mode=="text":select=~select
        source_count=select.sum(-1).clamp(min=1).float()*D
        gmean=meanpartial.sum(-1).reshape(B,Hkv,group).sum(-1)
        dv=dv+(gmean/source_count[:,None])[:,:,None,None]*select[:,None,:,None]
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None,None,None,None,None,None,None,None,None


def flashcf_cmve_delta(q,k,v,spans,mask,*,thres_mode="high",mean_mode="image",
                       block_k=32,row_tile=16,key_tile=64,
                       grad_q_tile=16,grad_k_tile=64):
    if q.dtype!=torch.bfloat16:raise ValueError("Native direct CMVE supports BF16 only")
    if grad_q_tile not in (16,32) or grad_k_tile not in (32,64,128):
        raise ValueError("Invalid query/key gradient tile")
    return CMVEDelta.apply(q,k,v,mask,spans,thres_mode,mean_mode,
                           block_k,row_tile,key_tile,grad_q_tile,grad_k_tile)


def flashcf_factual_plus_delta(q,k,v,spans,mask,*,thres_mode="high",
                               mean_mode="image",factual_backend="flash_causal",
                               block_k=32,row_tile=16,key_tile=64,
                               grad_q_tile=16,grad_k_tile=64):
    B,Hq,Hkv,L,D,regions=_check(q,k,v,mask,spans,mean_mode,thres_mode,block_k)
    ratio=Hq//Hkv
    kr=k.repeat_interleave(ratio,dim=1)
    vr=v.repeat_interleave(ratio,dim=1)
    if factual_backend=="flash_causal":
        causal=torch.arange(L,device=q.device)[:,None]>=torch.arange(L,device=q.device)[None,:]
        if not bool(((mask>-1.e4)==causal[None,None]).all()):
            raise ValueError("flash_causal requires simple unpadded causal 4D mask")
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            factual=F.scaled_dot_product_attention(q,kr,vr,attn_mask=None,
                                                   dropout_p=0.0,is_causal=True)
    elif factual_backend=="auto":
        factual=F.scaled_dot_product_attention(q,kr,vr,attn_mask=mask,
                                               dropout_p=0.0,is_causal=False)
    else:
        raise ValueError("factual_backend should be flash_causal or auto")
    correction=flashcf_cmve_delta(q,k,v,spans,mask,thres_mode=thres_mode,
                                  mean_mode=mean_mode,block_k=block_k,
                                  row_tile=row_tile,key_tile=key_tile,
                                  grad_q_tile=grad_q_tile,grad_k_tile=grad_k_tile)
    return factual+correction
