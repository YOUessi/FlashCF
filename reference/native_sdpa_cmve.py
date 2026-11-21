"""Closer-to-original native Qwen2.5-VL BF16 CFPO execution surrogate.

Regular factual attention is executed by PyTorch SDPA (where the CUDA backend
may use FlashAttention), while the counterfactual branch *materializes*
attention probabilities and applies CFPO's global hard saliency mask and
scalar value-mean prior as in the upstream source.

This baseline exposes the real bottleneck: the factual SDPA may be fast,
but the CFPO correction still allocates a [B,Hq,L,L] attention tensor.

Not a replacement for official verl CFPO training or identical FlashAttention
configuration. Record the selected CUDA SDPA backend with a profiler.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def native_sdpa_cmve(q,k,v,spans,mask,*,thres_mode="high",mean_mode="image"):
    B,Hq,L,D=q.shape
    Hkv=k.shape[1]
    assert q.dtype==k.dtype==v.dtype
    if Hq%Hkv or v.shape!=k.shape or mask.shape not in ((B,1,L,L),(B,Hq,L,L)):
        raise ValueError("native GQA and 4D additive mask required")
    kr=k.repeat_interleave(Hq//Hkv,dim=1)
    vr=v.repeat_interleave(Hq//Hkv,dim=1)
    normal=F.scaled_dot_product_attention(q,kr,vr,attn_mask=mask,
                                           dropout_p=0.,is_causal=False)
    # The next ops mirror the original CMVE branch, not the optimized kernel.
    scores=(q@kr.transpose(-1,-2))/math.sqrt(D)
    scores=scores+mask
    if q.dtype==torch.float16:
        scores=torch.where(torch.isinf(scores),torch.zeros_like(scores),scores)
    p=torch.softmax(scores,dim=-1,dtype=torch.float32).to(q.dtype)
    if thres_mode not in ("low","medium","high","extra"):
        raise ValueError("invalid saliency threshold")
    if mean_mode not in ("image","text"):
        raise ValueError("invalid value replacement mode")
    image_mask=torch.zeros((B,L),device=q.device,dtype=torch.bool)
    for b,sps in enumerate(spans):
        for start,end in sps:
            if not (0<=start<end<=L):
                raise ValueError("invalid image interval")
            image_mask[b,start:end]=True
    is_value=image_mask if mean_mode=="image" else ~image_mask
    num=is_value.sum(-1).clamp(min=1).to(vr.dtype).view(B,1,1,1)
    zero=vr.where(is_value[:,None,:,None],0.)
    mean=zero.sum(dim=(2,3),keepdim=True)/(num*D)
    mask_cmve=torch.zeros_like(p)
    coeff=dict(low=0.,medium=1.,high=2.,extra=3.)[thres_mode]
    for b,sps in enumerate(spans):
        for start,end in sps:
            if end>=L:continue
            values=p[b:b+1,:,end:,start:end]
            good=values>1.e-8
            if bool(good.any()):
                selected=values[good]
                thresh=selected.mean()+coeff*selected.std()
                mask_cmve[b:b+1,:,end:,start:end]=((values>thresh)&good).to(p.dtype)
    correction=(p*mask_cmve)@(mean-vr)
    return normal+correction
