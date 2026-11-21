# FlashCF

**FlashCF — IO-aware Counterfactual Attention for Multimodal RL (research prototype)**

Goal: compute CFPO-style counterfactual attention **without materializing an entire `[B,H,L,L]` attention tensor**, while preserving the reference forward, threshold statistics, and gradients. This repository is experimental. **Real Triton forward/backward kernels now run on Tang** in a restricted FP32 causal setting. There are numerical GPU microbenchmarks, but **no official CFPO training speedup or paper-quality result yet**.

## Development responsibilities

- **GitHub `YOUessi/FlashCF` is the source of truth** for all implementation, experiments, and documentation.
- **Mac** is the development environment (editor, reviews, commits).
- **Tang** is an NVIDIA **RTX 4090 Laptop 16 GB** Linux GPU runner for early Triton compilation, correctness, and microbenchmarks. Tang's local checkout is a disposable execution copy, **not a separate source tree**.
- Full CFPO RL training or comparisons needing large VRAM will move to rented 80 GB GPU(s) only after single-GPU validation.

## What is implemented

`reference/`: independent dense/streaming PyTorch numerical references, custom backward, and a split-SDPA+CMVE surrogate. They cover a **restricted deterministic subset** of CFPO CMVE (pre-rotated Q/K/V, causal attention, zero dropout and no diffusion).

`kernels/`: **actual Triton GPU implementation**: online-softmax + span-moment reduction + correction; atomic-gradient v1 and race-free query/key-centric backward v2. Both use bounded intermediate memory. Their current FP32 causal semantics are **not proof of full official CFPO equivalence**.

`tools/`: diagnostic hooks and replay for the **official** CFPO attention module. Captured `Q/K/V` fixtures must stay local/private; they can disclose representations derived from inputs.

`tests/`: correctness smoke tests. Synthetic tests **do not substitute for real CFPO model fixtures**.

`experiments/`: reproducible Tang GPU benchmarks, long-context numerical diagnostics, and measured results; see [kernel design + limitations](docs/TRITON_KERNEL_DESIGN.md).

## First run

```bash
python -m pip install torch pytest
python -m unittest discover -s tests -v
python experiments/gpu_probe.py
python experiments/compare_backward_algorithms.py --lengths 256 512 1024 2048
```

The GPU probe requires CUDA and Triton and exits with a clear error when unavailable. No model weights are downloaded by the smoke tests.

## Official scientific baseline

- CFPO original implementation: https://github.com/Raven-July/CFPO
- CFPO attention entry point: `verl/models/transformers/qwen2_vl_cmve.py`
- Official baseline commit reviewed: `98ca5a4123be2e85e94855fead46c9a385611136`
- Reviewed attention file blob: `2b9e77ea947f3b07801c9e6a863b774ac2521a5a`

**The value-only correction identity is already in CFPO and is not claimed as a new contribution.** Our intended research contribution is the **IO-aware reduction, correction and backward execution algorithm** and its measured impact on real multimodal RL training.

## P0 real pretrained Qwen2.5-VL validation — 2026-11-21

**Completed on Tang RTX 4090 16GB:** SHA-verified pretrained Qwen2.5-VL-3B
weights, native BF16 M-RoPE Attention captures (layers 0/8), exact original
CFPO CMVE function replay, FP32 Triton parity on real QKV, and native Qwen
layer-8 CMVE intervention with observable model-logit changes. Model
forward peak allocated memory was **7.131 GiB**. Raw activations remain local.

**Not yet complete:** BF16 Triton kernel, arbitrary padded/packed masks,
published multi-layer CFPO training, whole-model gradients or optimized
end-to-end throughput. Strict 4096-token hard-mask numerical boundary remains.

Details: [P0 real model report](docs/P0_REAL_MODEL_VALIDATION.md) and
[raw-free evidence](experiments/results/tang_2026-11-21_p0_real_qwen_cfpo_operator.json).

## P1 native BF16 / GQA / 4D-mask (2026-11-21)

**Implemented and tested on Tang:**

- Native BF16 Q/K/V, with non-expanded grouped KV heads (GQA).
- True 4D additive attention masks of shape `[B,1|Hq,L,L]`, including
  causal, padding-key, block-packed segments and finite per-head bias.
- BF16-aware CMVE statistics and scalar Value replacement.
- Triton Forward and custom Backward in original native head layouts.
- Default training candidate: `from kernels.native import flashcf_native`
  (V2 BF16 pre-softmax VJP). V1 remains an ablation only.
- **27/27** local tests passed. Original CFPO hard gate vs Triton had **zero**
  flips on two real pretrained Qwen2.5-VL-3B decoder layer fixtures.

**Important limits:** Native BF16 backward is not bitwise identical to eager
PyTorch (real QKV local gradient relative errors ~0.003 or better for K/V,
and 0.0018/0.000036 for Q). In a frozen pretrained *whole-model layer
injection* gradient deviations can be 4-6%. The current BF16 kernel is
**slower than the Dense reference at L=256/512** despite lower peak
intermediate memory; it is NOT ready for claims of training speedup.

Full verified data and kernel design:
[P1 Native BF16/GQA/4D Mask](docs/P1_NATIVE_BF16_GQA_MASK.md).

## P3 direct correction-only CMVE (2026-11-21)

**New execution architecture:** preserve the original factual FlashAttention
kernel, and run one bounded-intermediate Triton kernel graph for the
CFPO-style CMVE correction (with custom backward). No duplicate full
factual computation, unlike the previous diagnostic double-run hybrid.

```python
from kernels.native import flashcf_direct

# Native BF16 pre-rotated Q/K/V, real additive 4D mask,
# B,Hq,L,D; B,Hkv,L,D; B,1|Hq,L,L.
output = flashcf_direct(q,k,v,image_spans,attention_mask,
                        factual_backend="flash_causal")
```

- **46/46 tests** passed for correction-only operator and earlier kernels.
- Real pretrained Qwen2.5-VL-3B layers 0 and 8: identical BF16 forward
  against forced factual Flash+original explicit CMVE for those real QKV,
  with local Q-gradient difference 0 and native K/V relative differences
  generally under 0.3%.
- Frozen 3B model single-layer comparison (layer 0) against the same
  forced Flash baseline: final logits and Q gradients exactly matched on
  one synthetic test, K/V relative gradients ~0.269%/0.014%.
- 36-layer pretrained model **forward and backward call graphs run on Tang**,
  with full model weights frozen (no full RL training). All-layer gradient
  deviations still range ~3–8%; **training equivalence not established**.
- Repeated BF16 synthetic 2048-token full Attention step:
  ~12.79–12.80ms Direct vs ~16.58–16.59ms forced Flash+explicit CMVE,
  peak **additional** allocated memory ~106MiB vs ~1100MiB.
- 512-token Direct remains slower; do not assert universal speedup.
- **Full official CFPO/verl RL, original FlashAttention2 configuration,
  and training quality are NOT YET validated.**

Detailed architecture, reproducible metrics, and full-model limitations:
[P3 Direct Correction Results](docs/P3_DIRECT_CORRECTION_RESULTS.md).
The P2 all-Triton operator remains an explicit ablation in
[P2 Tensor Core Results](docs/P2_TENSORCORE_RESULTS.md).

## Research gates

1. Correctness vs official CFPO, including actual attention masks, image spans, value statistics and gradients.
2. **DONE for FP32 causal synthetic input:** Triton tiled reduction and correction (no full attention matrix).
3. **DONE for FP32 pre-repeated KV heads:** two GPU custom backward algorithms. **NOT DONE** for end-to-end BF16, padding/packing and model-native GQA semantics.
4. Real Qwen2.5-VL / CFPO integration with controlled compute and memory comparisons.
5. End-to-end training and cross-GPU evaluation before publication claims.

## Verified Tang experiments (2026-11-21)

- 18/18 unit tests passed, including 5 Triton Forward and 8 GPU Backward variants.
- For **synthetic FP32** shape B=1,H=4,D=64,L=2048, gather backward completed
  forward+backward in ~5.79 ms with ~13.10 MiB peak extra allocation; split-SDPA+CMVE
  surrogate took ~6.86 ms / ~288.04 MiB in that *single microbenchmark run*.
- At 4096 tokens, two of ~12.58 million hard threshold comparisons flipped due
  minute softmax rounding differences. The global threshold matched, yet
  max output absolute error was ~0.00159 (RMSE ~7.56e-6). See
  [numerical diagnostic](experiments/results/tang_2026-11-21_4096_threshold_numerics.json).
  Do **not** silently change the intervention rule or declare strict equivalence
  at all sequence lengths.
- Full results: [Dense / Atomic / Gather](experiments/results/tang_2026-11-21_triton_dense_atomic_gather.json)
  and [SDPA+CMVE comparison](experiments/results/tang_2026-11-21_sdpa_baseline_comparison.json).

See [docs/M1_REAL_CFPO_CAPTURE.md](docs/M1_REAL_CFPO_CAPTURE.md) for official fixture capture, and [experiments/README.md](experiments/README.md) for Tang runs.

## License and attribution

Reference implementations in this repository are original research scaffolding; do not copy upstream CFPO/FlashAttention source without respecting their licenses. Cite CFPO and FlashAttention when using results or discussing the approach.
