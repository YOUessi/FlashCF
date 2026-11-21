# Tang GPU execution — reproducible baseline

**Source of truth:** `https://github.com/YOUessi/FlashCF`.

Tang is an **execution-only** checkout. Make source edits as commits to GitHub, then pull the exact commit on Tang. Do not store unique source code in Tang's checkout.

## Environment checked 2026-11-21

- Tang: Ubuntu 6.8, NVIDIA GeForce RTX 4090 Laptop 16 GB (Compute Capability 8.9)
- Driver 580.178.04; torch 2.10.0+cu128; Triton 3.6.0
- NVCC on PATH is older 11.5; Triton JIT smoke did work independently
- ~35 GB disk free when checked; do not install models or large dependencies without checking free space
- Qwen2.5-VL-3B not cached; `verl`, `flash_attn` not installed
- Initial testing requires only torch, triton and small random tensors

## Commands

```bash
git clone https://github.com/YOUessi/FlashCF.git ~/projects/FlashCF
cd ~/projects/FlashCF
git rev-parse HEAD
python3 -m unittest discover -s tests -v
python3 experiments/gpu_probe.py --json experiments/results/tang_gpu_probe.json
```

Run manifest should record **Git commit SHA**, driver, GPU, torch, Triton, device memory, tested shapes, output/gradient errors, process exit code. Commit the **sanitized JSON summary** (not private fixtures or model activations) via GitHub.

## Gates

- GPU smoke: Triton kernel compiles, reference forward and gradients agree.
- Official CFPO gate: capture real Q/K/V only after compatible model and space available.
- Memory reduction / speedup can be claimed **only after implementing and benchmarking a real fused kernel**.

`experiments/results/` is for summarized results. Large tensors, private images, model weights, and raw captures are gitignored.
