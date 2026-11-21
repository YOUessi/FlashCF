"""CFPO CMVE reference (dense + three-pass streaming) for correctness testing.

This is an independently written educational reference matching the deterministic,
no-dropout, full-sequence CFPO CMVE branch for already position-rotated Q/K.
It is NOT the official CFPO source, a CUDA kernel, or an O(L) autograd implementation.

Tensor layout [batch, heads, sequence, head_dim]. A list of image token spans
per batch item uses half-open intervals [start, end). More than one span can
be intervened on simultaneously, with *separate* global thresholds per span.

Original reference implementation: Raven-July/CFPO,
  verl/models/transformers/qwen2_vl_cmve.py
"""
from __future__ import annotations

import math
from typing import List, Tuple
import torch

Span = Tuple[int, int]
Spans = List[List[Span]]


def _validate(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spans: Spans):
    assert q.ndim == k.ndim == v.ndim == 4, 'expected [B,H,L,D]'
    assert q.shape == k.shape == v.shape, 'repeat_kv beforehand for GQA'
    B, H, L, D = q.shape
    assert len(spans) == B
    for item in spans:
        for start, end in item:
            assert 0 <= start < end <= L, (start, end, L)
    return B,H,L,D


def _allowed(B: int, L: int, dev, valid_token_mask=None, sample_ids=None):
    """Additional synthetic sample-packing/padding masks for reference tests.

    The *real* Qwen2-VL path uses the model's 4-D additive attention mask;
    final integration must instead honor that mask exactly.
    """
    allowed = torch.ones((B,L,L),dtype=torch.bool,device=dev).tril()
    if valid_token_mask is not None:
        assert valid_token_mask.shape == (B,L)
        allowed &= valid_token_mask[:,:,None] & valid_token_mask[:,None,:]
    if sample_ids is not None:
        assert sample_ids.shape == (B,L)
        allowed &= (sample_ids[:,:,None] == sample_ids[:,None,:])
    return allowed



def _allowed_chunk(b,i,start,end,device,valid_token_mask=None,sample_ids=None):
    """Return only one tile of the attention mask (no [L,L] allocation)."""
    js=torch.arange(start,end,device=device)
    allowed=js <= i
    if valid_token_mask is not None:
        allowed=allowed & valid_token_mask[b,i] & valid_token_mask[b,start:end]
    if sample_ids is not None:
        allowed=allowed & (sample_ids[b,start:end]==sample_ids[b,i])
    return allowed


def _has_valid_query(b,i,valid_token_mask=None):
    # Causal self-attention always permits a token to attend itself unless padded.
    return valid_token_mask is None or bool(valid_token_mask[b,i])

def _replacement_scalar(v, spans, mean_mode):
    """Matches official mean_mode image/text: scalar per B/H across tokens and channels.

    NOTE: the official implementation does not remove padding from the mean
    in text mode. Padding is likewise not removed here (by design).
    """
    assert mean_mode in ('image','text')
    B,H,L,D = v.shape
    masks=[]
    for b in range(B):
        is_image = torch.zeros((L,),device=v.device,dtype=torch.bool)
        for start,end in spans[b]: is_image[start:end] = True
        select = is_image if mean_mode=='image' else ~is_image
        masks.append(select)
    masks = torch.stack(masks)
    masked = v * masks[:,None,:,None].to(v.dtype)
    sums = masked.sum(dim=(2,3), keepdim=True)  # [B,H,1,1]
    denom = masks.sum(-1).clamp(min=1)[:,None,None,None] * D
    return sums / denom


def _threshold(x, mode):
    """x is 1-D valid attention probabilities of one visual span.
    Uses torch.std() sample correction, like CFPO's GMM_mask.
    """
    assert mode in ('high','medium','low','extra')
    if x.numel()==0: return None
    mean = x.mean()
    sd = torch.std(x)  # sample standard deviation, default correction=1
    return mean + {'high':2, 'medium':1,'low':0,'extra':3}[mode]*sd


def dense_cfpo(q, k, v, image_spans: Spans, *, thres_mode='high', mean_mode='image',
               valid_token_mask=None, sample_ids=None):
    B,H,L,D=_validate(q,k,v,image_spans)
    allow=_allowed(B,L,q.device,valid_token_mask,sample_ids)
    scores=(q@k.transpose(-1,-2))/math.sqrt(D)
    scores=scores.masked_fill(~allow[:,None,:,:],-float('inf'))
    # Torch.softmax on an all-masked row returns NaN, so replace such rows
    # with zero logits and zero the result after softmax.
    any_allowed=allow.any(dim=-1)
    scores=torch.where(any_allowed[:,None,:,None], scores, torch.zeros_like(scores))
    p=torch.softmax(scores,dim=-1)
    p=torch.where(allow[:,None,:,:],p,torch.zeros_like(p))
    replaced=_replacement_scalar(v,image_spans,mean_mode)
    delta_v = replaced-v
    final_mask=torch.zeros_like(p)
    for b,spans in enumerate(image_spans):
        for start,end in spans:
            if end >= L: continue
            a=p[b,:,end:,start:end]
            eligible=a>1e-8
            if not torch.any(eligible): continue
            threshold=_threshold(a[eligible],thres_mode)
            m=(a>threshold)&eligible
            final_mask[b,:,end:,start:end]=m.to(p.dtype)
    return p@v+(p*final_mask)@delta_v


def streaming_cfpo(q,k,v,image_spans: Spans,*,thres_mode='high',mean_mode='image',
                   valid_token_mask=None,sample_ids=None,tile_k=4,return_metadata=False):
    """3-pass CPU mathematical reference without a materialized [B,H,L,L] tensor.

    Important: ordinary PyTorch autograd retains each tiled operation; this
    function is a forward-memory reference only. To keep training activations
    linear in L requires a custom recomputing backward (future GPU work).
    """
    B,H,L,D=_validate(q,k,v,image_spans)
    assert tile_k>0
    replaced=_replacement_scalar(v,image_spans,mean_mode)
    delta_v=replaced-v
    rows=[[None]*H for _ in range(B)]
    lses=[[None]*H for _ in range(B)]
    # Pass 1: online causal softmax + normal output.
    for b in range(B):
        for h in range(H):
            out=[]; lse=[]
            for i in range(L):
                if not _has_valid_query(b,i,valid_token_mask):
                    out.append(q.new_zeros((D,)))
                    lse.append(q.new_tensor(float('inf')))
                    continue
                peak=q.new_tensor(float('-inf'))
                denominator=q.new_zeros(())
                weighted=q.new_zeros((D,))
                for j in range(0,i+1,tile_k):
                    end=min(i+1,j+tile_k)
                    if not bool(_allowed_chunk(b,i,j,end,q.device,valid_token_mask,sample_ids).any()):continue
                    logits=(k[b,h,j:end]@q[b,h,i])/math.sqrt(D)
                    logits=logits.masked_fill(~_allowed_chunk(b,i,j,end,q.device,valid_token_mask,sample_ids),float('-inf'))
                    new_peak=torch.maximum(peak,logits.max())
                    adjust=torch.exp(peak-new_peak)
                    unnorm=torch.exp(logits-new_peak)
                    weighted=weighted*adjust+(unnorm[:,None]*v[b,h,j:end]).sum(0)
                    denominator=denominator*adjust+unnorm.sum()
                    peak=new_peak
                out.append(weighted/denominator)
                lse.append(peak+torch.log(denominator))
            rows[b][h]=torch.stack(out)
            lses[b][h]=torch.stack(lse)
    base=torch.stack([torch.stack(heads) for heads in rows])
    logsumexp=torch.stack([torch.stack(heads) for heads in lses])
    thresholds={}
    # Pass 2: one threshold per (batch, visual span) across ALL heads/queries.
    for b,spans in enumerate(image_spans):
        for si,(start,end) in enumerate(spans):
            count=0
            mean=q.new_zeros(())
            M2=q.new_zeros(())
            if end>=L:
                thresholds[(b,si)]=None;continue
            for h in range(H):
                for i in range(end,L):
                    if not _has_valid_query(b,i,valid_token_mask):continue
                    for j in range(start,end,tile_k):
                        je=min(end,j+tile_k)
                        logits=(k[b,h,j:je]@q[b,h,i])/math.sqrt(D)
                        probs=torch.exp(logits-logsumexp[b,h,i]).masked_fill(~_allowed_chunk(b,i,j,je,q.device,valid_token_mask,sample_ids),0.)
                        vals=probs[probs>1e-8]
                        m=vals.numel()
                        if not m:continue
                        mu=vals.mean()
                        M2_tile=((vals-mu)**2).sum()
                        new_count=count+m
                        offset=mu-mean
                        mean=mean+offset*(m/new_count)
                        M2=M2+M2_tile+offset.square()*(count*m/new_count)
                        count=new_count
            if not count:
                thresholds[(b,si)]=None;continue
            std=torch.sqrt(M2/(count-1)) if count>1 else q.new_tensor(float('nan'))
            coeff={'high':2,'medium':1,'low':0,'extra':3}[thres_mode]
            thresholds[(b,si)]=mean+coeff*std
    # Pass 3: recompute only visual attention tiles to apply the same masks.
    result=[]
    for b,spans in enumerate(image_spans):
        hresult=[]
        for h in range(H):
            increments=[q.new_zeros((D,)) for _ in range(L)]
            for si,(start,end) in enumerate(spans):
                threshold=thresholds[(b,si)]
                if threshold is None:continue
                for i in range(end,L):
                    if not _has_valid_query(b,i,valid_token_mask):continue
                    block_sum=q.new_zeros((D,))
                    for j in range(start,end,tile_k):
                        je=min(end,j+tile_k)
                        logits=(k[b,h,j:je]@q[b,h,i])/math.sqrt(D)
                        p=torch.exp(logits-logsumexp[b,h,i]).masked_fill(~_allowed_chunk(b,i,j,je,q.device,valid_token_mask,sample_ids),0.)
                        active=(p>1e-8)&(p>threshold)
                        block_sum=block_sum+(p[:,None]*active[:,None].to(p.dtype)*delta_v[b,h,j:je]).sum(0)
                    increments[i]=increments[i]+block_sum
            hresult.append(base[b,h]+torch.stack(increments))
        result.append(torch.stack(hresult))
    y=torch.stack(result)
    if return_metadata:return y,logsumexp,thresholds
    return y