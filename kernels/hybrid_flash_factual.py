"""Diagnostic: FlashAttention factual + bounded-memory Triton CMVE delta.

The published CFPO model computes its ordinary factual output with a
FlashAttention kernel and adds a separately computed counterfactual Value
correction. Current FlashCF all-tiled uses its own factual implementation,
which can have different BF16 rounding.

This diagnostic forces the SAME PyTorch Flash causal factual path as the
strong surrogate baseline, while obtaining the CMVE delta by subtracting
Triton all-tiled ordinary from Triton all-tiled intervened output.

WARNING: This method executes TWO tiled attention forward/backward graphs
plus Flash factual. It is a numerical ablation and NOT a performance kernel.
A production hybrid must compute correction directly in one tiled pass and
implement its own backward to avoid this expensive subtraction trick.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from kernels.triton_native_qtile import flashcf_gemm_all


def hybrid_flash_factual(q,k,v,spans,mask,*,thres_mode="high",
                         mean_mode="image"):
    if q.dtype!=torch.bfloat16:
        raise ValueError("Diagnostic hybrid requires native BF16")
    B,Hq,L,D=q.shape
    Hkv=k.shape[1]
    if Hq%Hkv or mask.shape not in ((B,1,L,L),(B,Hq,L,L)):
        raise ValueError("Invalid GQA/mask")
    causal=torch.ones((L,L),device=q.device,dtype=torch.bool).tril()
    if not bool(((mask>-1.e4)==causal[None,None,:,:]).all()):
        raise ValueError("Hybrid factual requires simple causal unpadded mask")
    kw=dict(thres_mode=thres_mode,mean_mode=mean_mode,
            row_tile=16,key_tile=64,grad_q_tile=16,grad_k_tile=64)
    cmve=flashcf_gemm_all(q,k,v,spans,mask,**kw)
    no_spans=[[] for _ in range(B)]
    no_cmve=flashcf_gemm_all(q,k,v,no_spans,mask,**kw)
    delta=(cmve.float()-no_cmve.float()).to(q.dtype)
    kr=k.repeat_interleave(Hq//Hkv,dim=1)
    vr=v.repeat_interleave(Hq//Hkv,dim=1)
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        factual=F.scaled_dot_product_attention(q,kr,vr,attn_mask=None,
                                               dropout_p=0.0,is_causal=True)
    return factual+delta
