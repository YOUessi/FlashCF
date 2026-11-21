"""Replay a captured CFPO Q/K/V fixture in the official deterministic dense math.

This replay does NOT import Qwen2VL/FlashAttention; it mirrors the official
CMVE dense-math path after rotary positions and repeat_kv are applied.
It reports a comparison to captured FlashAttention factual output (when present).

Input: .pt file produced by capture_official_qkv.py. Load only trusted files.
Supported: full Q/K/V causal self-attention; 4-D additive mask; no dropout,
no diffusion noise, no sliding window, q_len == k_len.

Usage: python replay_official_fixture.py fixtures/rank0/cfpo_cmve_000.pt
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import torch


def reference_official_dense(q, k, v, attention_mask, image_pos, *, thres_mode,
                             noise=False, mean_mode='image', attention_dropout=0.0,
                             sliding_window=None):
    if noise or attention_dropout != 0 or sliding_window is not None:
        raise NotImplementedError('Replay first verifies deterministic no-window CMVE')
    if q.shape != k.shape or k.shape != v.shape or q.ndim != 4:
        raise NotImplementedError('This replay requires already repeated GQA KV and full self-attention')
    b, h, L, d = q.shape
    if attention_mask is None:
        raise ValueError('Actual 4-D additive mask needed to check official masked branch')
    if attention_mask.ndim != 4 or attention_mask.shape[-2:] != (L,L):
        raise ValueError(f'Expected 4-D additive attention mask [B,1,L,L], got {attention_mask.shape}')
    scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(d)
    scores = scores + attention_mask[:, :, :, :L].to(scores.dtype)
    if q.dtype == torch.float16:
        scores = torch.where(torch.isinf(scores), torch.zeros_like(scores), scores)
    p = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    # Replacement mean exactly as values_noise_multiply() deterministic mode:
    if mean_mode == 'image':
        select = torch.zeros((b,L),dtype=torch.bool,device=v.device)
    elif mean_mode == 'text':
        select = torch.ones((b,L),dtype=torch.bool,device=v.device)
    else:
        raise ValueError(mean_mode)
    for batch, intervals in enumerate(image_pos):
        for interval in intervals:
            start, end = interval['image_token_start'], interval['image_token_end']
            if start < L and end <= L:
                select[batch,start:end] = (mean_mode == 'image')
    replaced = (v * select[:,None,:,None].to(v.dtype)).sum((2,3),keepdim=True)
    replaced = replaced / (select.sum(-1).clamp(min=1)[:,None,None,None]*d)
    final_mask = torch.zeros_like(p)
    for batch, intervals in enumerate(image_pos):
        for interval in intervals:
            start, end = interval['image_token_start'], interval['image_token_end']
            if end >= L or not start < end:
                continue
            sl = p[batch:batch+1,:,end:,start:end]
            valid = sl > 1e-8
            if not valid.any():
                continue
            vals = sl[valid]
            mean, std = vals.mean(), vals.std()
            multiplier = {'high':2,'medium':1,'low':0,'extra':3}[thres_mode]
            threshold = mean + multiplier*std
            mask = (sl > threshold) & valid
            final_mask[batch:batch+1,:,end:,start:end] = mask.to(p.dtype)
    return p@v + (p*final_mask)@(replaced-v), p, final_mask


def load_and_run(path: str|Path, max_len=2048):
    # Torch pickle only from files produced by your own capture script.
    x = torch.load(path, map_location='cpu', weights_only=True)
    q,k,v = (x[n].float().detach().requires_grad_(True) for n in ('q','k','v'))
    if q.size(-2) > max_len:
        raise ValueError('Fixture too long for dense CPU replay; capture shorter input')
    mask = x.get('attention_mask')
    if mask is not None:
        mask = mask.float()
    y,p,cmask = reference_official_dense(q,k,v,mask,x['image_pos'],
        thres_mode=x['thres_mode'],noise=x['noise'],mean_mode=x['mean_mode'],
        attention_dropout=x['dropout'],sliding_window=x['sliding_window'])
    print('fixture=',path,'shape=',tuple(q.shape),'dtype=',str(x['q'].dtype))
    print('dense reference output shape=',tuple(y.shape))
    print('counterfactual selection fraction=',round(cmask.float().mean().item(),7))
    if 'flash_output' in x and isinstance(x['flash_output'],torch.Tensor):
        normal=p@v
        off=x['flash_output'].float()
        if off.ndim==4 and off.shape==(q.size(0),q.size(2),q.size(1),q.size(3)):
            off=off.transpose(1,2)
        if normal.shape == off.shape:
            print('normal FlashAttention-vs-dense max abs error=',float((normal-off).abs().max()))
        else:
            print('flash_output shape differs: ',tuple(off.shape),' vs ',tuple(normal.shape))
    return y,p,cmask


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('fixture',type=Path)
    args=parser.parse_args()
    load_and_run(args.fixture)