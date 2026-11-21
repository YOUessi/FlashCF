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
