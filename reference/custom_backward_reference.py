"""Memory-bounded educational custom backward for deterministic CFPO-CMVE.

This is intentionally slow Python/PyTorch, not a kernel. It stores only
Q/K/V, output, LSE, per-span thresholds, and O(tile) temporary tensors.
It analytically backpropagates through softmax and through the per-head scalar
Value replacement mean. Hard saliency threshold comparisons have zero gradient,
which matches PyTorch's comparison-based GMM_mask convention.

Main caveat: no dropout, no diffusion noise, no sliding-window mask yet.
"""
import math
import torch
from cfpo_reference import (
    streaming_cfpo, _replacement_scalar, _allowed_chunk, _has_valid_query,
)


class FlashCFReference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, image_spans, thres_mode, mean_mode,
                tile_k, valid_token_mask, sample_ids):
        with torch.no_grad():
            y, lse, thresholds = streaming_cfpo(
                q,k,v,image_spans,thres_mode=thres_mode,mean_mode=mean_mode,
                valid_token_mask=valid_token_mask,sample_ids=sample_ids,
                tile_k=tile_k,return_metadata=True)
            replacement = _replacement_scalar(v,image_spans,mean_mode)
        ctx.save_for_backward(q,k,v,y,lse,replacement)
        ctx.image_spans=image_spans
        ctx.thres_mode=thres_mode
        ctx.mean_mode=mean_mode
        ctx.tile_k=tile_k
        ctx.valid_token_mask=valid_token_mask
        ctx.sample_ids=sample_ids
        ctx.thresholds=thresholds
        return y

    @staticmethod
    def backward(ctx, grad_out):
        q,k,v,y,lse,replacement=ctx.saved_tensors
        image_spans=ctx.image_spans
        valid_mask=ctx.valid_token_mask
        sample_ids=ctx.sample_ids
        thresholds=ctx.thresholds
        tile=ctx.tile_k
        B,H,L,D=q.shape
        gq=torch.zeros_like(q)
        gk=torch.zeros_like(k)
        gv=torch.zeros_like(v)
        scale=1/math.sqrt(D)
        # For each head the replacement is a scalar mean over input values.
        # This derivative requires one global scalar gradient per (B,H).
        for b in range(B):
            is_image=torch.zeros((L,),device=v.device,dtype=torch.bool)
            for start,end in image_spans[b]:is_image[start:end]=True
            source_mask = is_image if ctx.mean_mode=='image' else ~is_image
            source_count=int(source_mask.sum())
            for h in range(H):
                grad_replacement=q.new_zeros(())
                c=replacement[b,h,0,0]
                for i in range(L):
                    if not _has_valid_query(b,i,valid_mask):continue
                    gy=grad_out[b,h,i]
                    gy_sum=gy.sum()
                    target=y[b,h,i]
                    for j in range(0,i+1,tile):
                        end=min(i+1,j+tile)
                        allowed=_allowed_chunk(b,i,j,end,q.device,valid_mask,sample_ids)
                        if not bool(allowed.any()):continue
                        logits=(k[b,h,j:end]@q[b,h,i])*scale
                        p=torch.exp(logits-lse[b,h,i]).masked_fill(~allowed,0.)
                        m=torch.zeros((end-j,),device=q.device,dtype=torch.bool)
                        for si,(s,e) in enumerate(image_spans[b]):
                            if i < e or s >= end or e <= j:continue
                            threshold=thresholds[(b,si)]
                            if threshold is None:continue
                            selected=(torch.arange(j,end,device=q.device)>=s) & (torch.arange(j,end,device=q.device)<e)
                            m |= selected & (p>1e-8)&(p>threshold)
                        mf=m.to(p.dtype)
                        w = v[b,h,j:end] + mf[:,None]*(c-v[b,h,j:end])
                        # dscore = p * dO . (W_j - O_i), fixed hard mask.
                        ds = p * ((w-target[None,:])*gy[None,:]).sum(-1)
                        gq[b,h,i] += (ds[:,None]*k[b,h,j:end]).sum(0)*scale
                        gk[b,h,j:end] += ds[:,None]*q[b,h,i][None,:]*scale
                        gv[b,h,j:end] += (p*(1-mf))[:,None]*gy[None,:]
                        grad_replacement += (p*mf).sum()*gy_sum
                if source_count:
                    # d(c)/d(V_{j,d}) = 1/(#selected tokens * head_dim).
                    gv[b,h,source_mask] += grad_replacement/(source_count*D)
        return (gq,gk,gv,None,None,None,None,None,None)


def flashcf_custom(q,k,v,spans,thres_mode='high',mean_mode='image',tile_k=4,
                   valid_token_mask=None,sample_ids=None):
    return FlashCFReference.apply(q,k,v,spans,thres_mode,mean_mode,
                                  tile_k,valid_token_mask,sample_ids)