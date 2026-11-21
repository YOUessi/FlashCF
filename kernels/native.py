"""Stable public native FlashCF API.

The default BF16 training operator is the numerically improved v2 backward.
V1 remains accessible for ablation / performance comparison but must not be
silently substituted for training-quality parity.

Input layout:
  Q: [B,Hq,L,D] BF16/FP32, post-M-RoPE
  K,V: [B,Hkv,L,D] BF16/FP32, *without* KV head repetition
  attention_mask: [B,1|Hq,L,L] additive BF16/FP32, including causal/padding
  image_spans: list[list[(start,end)]] with half-open visual token intervals

Returns [B,Hq,L,D] same dtype and native-shape autograd gradients.
"""
from __future__ import annotations

from kernels.triton_native_bf16 import forward_native
from kernels.triton_native_backward_v2 import flashcf_native_v2
from kernels.triton_cmve_delta import flashcf_factual_plus_delta


def flashcf_native(q,k,v,image_spans,attention_mask,*,thres_mode="high",
                   mean_mode="image",block_k=32):
    """Preferred production-EXPERIMENTAL BF16 operator (V2 VJP)."""
    return flashcf_native_v2(q,k,v,image_spans,attention_mask,
                             thres_mode=thres_mode,mean_mode=mean_mode,
                             block_k=block_k)


def flashcf_direct(q,k,v,image_spans,attention_mask,*,thres_mode="high",
                   mean_mode="image",factual_backend="flash_causal",block_k=32,
                   row_tile=16,key_tile=64):
    """P3 correction-only experimental API: Flash factual once + Triton CMVE delta.

    No double factual computation, but full 3B CFPO training parity remains unproven.
    """
    return flashcf_factual_plus_delta(
        q,k,v,image_spans,attention_mask,thres_mode=thres_mode,
        mean_mode=mean_mode,factual_backend=factual_backend,block_k=block_k,
        row_tile=row_tile,key_tile=key_tile)


__all__=["flashcf_native","flashcf_direct","forward_native"]
