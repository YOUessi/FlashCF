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


def flashcf_native(q,k,v,image_spans,attention_mask,*,thres_mode="high",
                   mean_mode="image",block_k=32):
    """Preferred production-EXPERIMENTAL BF16 operator (V2 VJP)."""
    return flashcf_native_v2(q,k,v,image_spans,attention_mask,
                             thres_mode=thres_mode,mean_mode=mean_mode,
                             block_k=block_k)


__all__=["flashcf_native","forward_native"]
