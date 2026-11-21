# P1: Native BF16 / GQA / real 4D additive Attention mask

**Status (2026-11-21):** Implemented and GPU-executed on Tang. Native BF16
forward, native grouped KV-head mapping, additive-mask interpretation, and
custom backward all run in Triton. **Not yet an optimized production kernel or
proof of bit-exact CFPO training parity.**

## Exact developer-facing interface

```python
from kernels.native import flashcf_native

# Shapes and inputs, all contiguous on the same CUDA device:
# q: BF16 [B, Hq, L, D] (after M-RoPE)
# k: BF16 [B, Hkv, L, D] (NO expanded repeat_kv)
# v: BF16 [B, Hkv, L, D]
# mask: BF16/FP32 [B,1,L,L] or [B,Hq,L,L] genuine additive Attention mask
# spans: list of visual half-open intervals per batch sample
y = flashcf_native(q, k, v, spans, mask,
                   thres_mode="high", mean_mode="image", block_k=32)
# y is BF16 [B,Hq,L,D]; dQ [B,Hq,L,D], dK/dV [B,Hkv,L,D].
```

The public wrapper chooses **V2** for more faithful BF16 Softmax gradients.
V1 remains accessible at `kernels.triton_native_bf16.flashcf_native` only
as an ablation. The v2 implementation is `kernels/triton_native_backward_v2.py`.

## Algorithm and compute path

1. **LSE:** query-row tiled dot product, mapped to native KV head
   `kv_head = query_head // (Hq // Hkv)`. For BF16, apply native rounding
   at logits / scaling / additive-mask addition boundaries.
2. **Factual output:** recompute valid tiles and aggregate BF16-rounded
   Softmax probabilities against native, non-repeated V.
3. **Global CMVE statistics:** for each visual span, obtain sum of
   BF16-rounded probabilities, squared sum, and valid count. Reduce across
   query positions and attention heads, using sample variance; derive the
   official mean + lambda*std threshold. For source BF16, threshold and the
   scalar replacement average are quantized in the input dtype.
4. **Counterfactual Value correction:** generate hard mask in tile based on
   `p>1e-8 and p>threshold`; accumulate without materializing full L²
   Attention weights. Preserve independent intervals, head-aware additive
   masks, and native grouped KV head mapping.
5. **Backward:** V2 computes `dL/dP` for each row and the exact **FP32
   pre-cast Softmax VJP** `sum(P_float32*dL/dP)`. This fixes the
   `P_bf16 * (W-Y_bf16)` cancellation problem of V1. Query-gradient and
   key-centric gradients are recomputed with FP32 accumulations; temporary
   per-query-head dK/dV partials are reduced across GQA groups and the global
   Value-mean gradient is distributed to source tokens.

The kernel does **not** allocate an extra full LxL Attention matrix.
The user-supplied 4D mask itself is already quadratic, however. Thus
the algorithm's **incremental scratch space**, not total input bytes, is
bounded by O(B*Hq*L*D + S*Hq*L*D) for S visual intervals.

## Accepted / rejected modes

| Feature | P1 status |
|---|---|
| BF16 Q/K/V on CUDA | **Implemented, tested** |
| FP32 Q/K/V | **Implemented, tested** |
| GQA Hq divisible by Hkv, no explicit KV repeat in forward | **Implemented, tested** |
| dK/dV returned in original [B,Hkv,L,D] shape | **Implemented, tested** |
| Real 4D additive mask [B,1|Hq,L,L] | **Implemented, tested** |
| Causal, padding-key restrictions, packed segments, finite Attention bias | **Implemented, tested** |
| Disjoint multiple image intervals, text/image replacement | **Implemented, tested** |
| No available Key in an entire Query row | **Rejected explicitly in P1** |
| Noncontiguous Q/K/V | **Rejected explicitly in P1** |
| Query length distinct from Key length / decode-with-cache | **Unsupported in P1** |
| Dropout, diffusion noise, model-native sliding-window bookkeeping | **Unsupported / not silently enabled** |
| FP16, FP8, INT8 weights | **Unsupported; BF16/FP32 only** |
| BF16 bitwise-exact backward parity | **Not achieved** |
| End-to-end official original verl/CFPO multi-layer training | **Not yet executed** |

**Hard-mask discontinuity:** the single-precision 4096-token tests from
P0 identified gate flips caused by very small probability rounding changes.
That remains a distinct correctness and numerical-reliability research gate.

## Tang validation

**27/27 regression tests passed** as of commit `9842f5d`, including
native-BF16 GQA/4D masks and more faithful V2 backwards.

### Real pretrained Qwen2.5-VL-3B Q/K/V (layers 0 and 8)

- Model weights SHA256 validated, repo revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Tensor layout: Q [1,16,97,128], native K/V [1,2,97,128].
- Real attention mask: [1,1,97,97], no KV repeat before the P1 operator.
- Hard-gate choices: **zero flips** for both layers. Original and Triton
  thresholds equal: 0.0048828125 (layer 0), 0.016357421875 (layer 8).

| Metric | Layer 0 | Layer 8 |
|---|---:|---:|
| BF16 output relative L2 | 0.00042035 | 0.0000009067 |
| Q-gradient relative L2 V1 | 0.07577664 | 0.00429327 |
| Q-gradient relative L2 **V2** | **0.00179335** | **0.00003582** |
| K-gradient relative L2 **V2** | 0.00298385 | 0.00290358 |
| V-gradient relative L2 **V2** | 0.00250779 | 0.00304611 |

This shows **V2 is materially closer to original BF16 autograd than V1**
on recorded real pretrained activations, while not being bitwise exact.

### Synthetic BF16 native GQA training-step CUDA event medians

B=1,Hq=16,Hkv=2,D=128, one causal 4D mask.

| L | Dense step | Triton V2 step | Dense peak *extra* | Triton V2 peak *extra* |
|---:|---:|---:|---:|---:|
| 97 | 1.0045 ms | 1.0281 ms | 4.490 MiB | 3.717 MiB |
| 256 | 1.2660 ms | 2.4740 ms | 19.753 MiB | 9.801 MiB |
| 512 | 1.1018 ms | 7.0328 ms | 72.502 MiB | 19.598 MiB |

**Current bottleneck:** row-wise tile scanning, redundant QK recomputations,
and the separate per-row VJP. This P1 is a **functional/correctness milestone**,
not a competitive speedup. Dense here is eager PyTorch, not optimized CFPO
training or FlashAttention-4.

### Real pretrained model integration (frozen weights)

Qwen2.5-VL-3B layer 8 was patched through the native
`eager_attention_forward` hook; dense vs native Triton V2 was propagated
to final-model logits, and a deterministic logit loss was backpropagated to
the *selected layer's detached Q/K/V* (full 3B weights frozen).

- Peak allocated GPU memory: **7.394 GiB**.
- Final logits relative L2 difference: **0.0062436** (~0.624%).
- Selected-layer Q/K/V gradient relative L2 differences:
  **0.04437 / 0.04072 / 0.05852** (~4.4% / 4.1% / 5.9%).

These are **higher** than local operator gradients because subsequent
Transformer layers amplify small BF16 differences. Do not claim full model
gradient equivalence.

## Source and reproducibility

- BF16 reference: [native_bf16_gqa.py](../reference/native_bf16_gqa.py)
- Triton forward + V1 backward: [triton_native_bf16.py](../kernels/triton_native_bf16.py)
- Improved V2 backward: [triton_native_backward_v2.py](../kernels/triton_native_backward_v2.py)
- Public API: [native.py](../kernels/native.py)
- Native hard mask debugging (diagnostic materializes L² mask deliberately):
  [triton_native_debug.py](../kernels/triton_native_debug.py)
- Real pretrained fixture parity: [p1_real_bf16_gqa_parity.py](../experiments/p1_real_bf16_gqa_parity.py)
- Hard-gate parity: [p1_check_real_gates.py](../experiments/p1_check_real_gates.py)
- Synthetic step benchmark: [compare_native_bf16_v1_v2.py](../experiments/compare_native_bf16_v1_v2.py)
- Frozen pretrained model layer backward:
  [p1_full_model_layer_backward.py](../experiments/p1_full_model_layer_backward.py)
- Sanitized results:
  [P1 full JSON log](../experiments/results/tang_2026-11-21_p1_native_bf16_gqa_mask.json).

No raw QKV fixtures, image inputs, model weights, or large checkpoint files
were committed.

## Next technical gate: performance AND model parity

1. Replace per-row serial QK loops with multi-row/tiled matrix multiplication
   (Tensor Core where supported), reuse online-softmax LSE and BF16 statistics.
2. Improve native BF16 forward parity especially at low-level Qwen layer 0,
   then inspect final-logit and whole-downstream-gradient amplification.
3. Optimize backward dK/dV group aggregation and/or fuse VJP reduction without
   materializing Attention scores.
4. Profile *strong* stock CFPO + FlashAttention baseline and peak-memory
   allocation on matched shapes and dtypes; calculate full training throughput.
5. Only then consider full model CFPO RL training on 80GB GPUs.

Current policy: **Tang remains the GPU validation runner; GitHub is source of
truth; do not rent a server or claim production training speedup yet.**
