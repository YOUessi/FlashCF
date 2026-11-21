# FlashCF

**FlashCF — IO-aware Counterfactual Attention for Multimodal RL (research prototype)**

Goal: compute CFPO-style counterfactual attention **without materializing an entire `[B,H,L,L]` attention tensor**, while preserving the reference forward, threshold statistics, and gradients. This repository is experimental; it contains **no verified training speedup or paper results**.

## Development responsibilities

- **GitHub `YOUessi/FlashCF` is the source of truth** for all implementation, experiments, and documentation.
- **Mac** is the development environment (editor, reviews, commits).
- **Tang** is an NVIDIA **RTX 4090 Laptop 16 GB** Linux GPU runner for early Triton compilation, correctness, and microbenchmarks. Tang's local checkout is a disposable execution copy, **not a separate source tree**.
- Full CFPO RL training or comparisons needing large VRAM will move to rented 80 GB GPU(s) only after single-GPU validation.

## What is implemented

`reference/`: independent dense/streaming PyTorch numerical references and custom backward. They currently cover a **restricted deterministic subset** of the CFPO CMVE path (pre-rotated Q/K/V, causal attention, zero dropout and no diffusion). They are **not** CUDA kernels nor proof of full CFPO equivalence.

`tools/`: diagnostic hooks and replay for the **official** CFPO attention module. Captured `Q/K/V` fixtures must stay local/private; they can disclose representations derived from inputs.

`tests/`: correctness smoke tests. Synthetic tests **do not substitute for real CFPO model fixtures**.

`experiments/`: reproducible runner scripts and measurement logs.

## First run

```bash
python -m pip install torch pytest
python -m pytest -q
python experiments/gpu_probe.py
```

The GPU probe requires CUDA and Triton and exits with a clear error when unavailable. No model weights are downloaded by the smoke tests.

## Official scientific baseline

- CFPO original implementation: https://github.com/Raven-July/CFPO
- CFPO attention entry point: `verl/models/transformers/qwen2_vl_cmve.py`
- Official baseline commit reviewed: `98ca5a4123be2e85e94855fead46c9a385611136`
- Reviewed attention file blob: `2b9e77ea947f3b07801c9e6a863b774ac2521a5a`

**The value-only correction identity is already in CFPO and is not claimed as a new contribution.** Our intended research contribution is the **IO-aware reduction, correction and backward execution algorithm** and its measured impact on real multimodal RL training.

## Research gates

1. Correctness vs official CFPO, including actual attention masks, image spans, value statistics and gradients.
2. CUDA/Triton tiled reduction and correction (no full attention matrix).
3. Custom backward under GQA/mixed precision; match dense reference within documented tolerances.
4. Real Qwen2.5-VL / CFPO integration with controlled compute and memory comparisons.
5. End-to-end training and cross-GPU evaluation before publication claims.

See [docs/M1_REAL_CFPO_CAPTURE.md](docs/M1_REAL_CFPO_CAPTURE.md) for official fixture capture, and [experiments/README.md](experiments/README.md) for Tang runs.

## License and attribution

Reference implementations in this repository are original research scaffolding; do not copy upstream CFPO/FlashAttention source without respecting their licenses. Cite CFPO and FlashAttention when using results or discussing the approach.
