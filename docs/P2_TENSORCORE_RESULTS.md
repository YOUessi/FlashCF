# P2 — Tensor Core Multirow FlashCF: GPU Measurements and Research Gates

**Date:** 2026-11-21  
**Canonical repo:** https://github.com/YOUessi/FlashCF  
**GPU:** Tang RTX 4090 Laptop 16 GB (Ada SM 8.9)  
**Precision:** BF16 input/output, grouped K/V heads, 4D additive mask  
**Conclusion:** Functional multirow Tensor Core implementation substantially improves
long-sequence FlashCF over the original rowwise kernels. The current implementation
has credible **isolated operator** peak-memory improvement and a measured
long-context performance advantage against a split, FlashAttention-forced
factual + explicit CMVE surrogate. **No original verl/CFPO end-to-end RL training
or A-conference claim has been established.**

## 1. Computational changes (four code paths)

Original P1 BF16 v2:

- Rowwise LSE, factual Attention output, CMVE statistical reduction and correction
- Rowwise pre-cast BF16 Softmax VJP for backward
- One key position per CTA gathering all query positions for dK/dV
- Asymptotic attention compute O(B*Hq*L²*D), with no explicitly stored L² Attention
- Measured at 512: 7.63ms full operator step (synthetic, Qwen-shaped).

P2 staged optimizations:

| Phase | New file / algorithm | What changes |
|---|---|---|
| **P2-A** | [triton_native_tile.py](../kernels/triton_native_tile.py) | 16/32 Query rows × 32/64/128 Key tile Tensor Core QK; tiled LSE and factual P@V instead of independent scalar dot-product Query rows |
| **P2-B** | [triton_native_gemm_kv.py](../kernels/triton_native_gemm_kv.py) | Key-tile dK/dV matrix reductions, retaining explicit per-row VJP and dQ |
| **P2-C** | [triton_native_qtile.py](../kernels/triton_native_qtile.py) | Multirow Softmax VJP plus multirow dQ Tensor Core, combined with P2-B grouped dK/dV |
| **Control** | [triton_native_backward_v2.py](../kernels/triton_native_backward_v2.py) | Numerically careful, slower rowwise BF16 VJP; still the default training prototype until full model gradients are sufficiently close |

These optimization steps preserve CFPO's value-delta decomposition and global
hard saliency threshold. Neither formula nor Tensor Core matmul itself is claimed
as a new theory contribution. Their algorithmic target is the **entire
statistics-conditioned counterfactual Attention execution graph** without
materializing all Attention weights in global memory.

### Complexity and memory

- Existing, supplied input 4D additive mask can occupy O(B*L²) or O(B*Hq*L²).
  The current P2 implementation does NOT compress this input mask.
- Additional saved scratch (Q/K/V already supplied) is approximately
  O(B*Hq*L*D + S*Hq*L*D), with S disjoint image spans, versus O(B*Hq*L²)
  attention/grad intermediates for explicit CMVE.
- Work remains O(B*Hq*L²*D) up to repeated tiled passes, though BF16 Tensor
  Core matrix products are much more efficient than scalar QK loops.
- Tensor Core dQ and dK/dV do not require global atomic_add accumulation.
- Hard comparison mask values can flip near numerical ties: 4096 FP32
  numerical boundary remains open; it is not removed by these optimizations.

## 2. Tang: direct phase-by-phase latency results

**Scope:** Synthetic Qwen-like tensors [B=1,Hq=16,Hkv=2,L,D=128], BF16,
single image span [L/8,L/2], 4D unpadded causal mask, entire forward plus
custom backward; 6 CUDA-event repetitions (one execution). Numbers vary
with clocks and benchmark ordering and are not a final statistically
controlled dataset.

| Tokens | Rowwise V2 | Tiled factual only | Tiled factual + GEMM KV | Full tiled Q+KV (P2-C) | Dense PyTorch |
|---:|---:|---:|---:|---:|---:|
| 97 | 1.046ms | 0.924ms | 0.986ms | **0.907ms** | 0.888ms |
| 256 | 2.443ms | 2.059ms | 1.357ms | **0.970ms** | 0.920ms |
| 512 | 7.632ms | 6.111ms | 3.373ms | **1.153ms** | 1.041ms |
| 1024 | 29.752ms | 23.382ms | 13.312ms | **3.235ms** | 4.178ms |

P2-C is ~6.62x faster than the original BF16 rowwise V2 at 512 and
~9.20x at 1024. Again, the Dense PyTorch baseline does not represent
upstream CFPO's complete runtime and is not the strongest control.

## 3. Stronger factual-attention controls (important distinction)

The original CFPO explicitly runs FlashAttention for the factual branch and
a separate dense materialized Attention computation for its counterfactual
statistics and Value correction. We tested two mathematical surrogates:

- **SDPA+CMVE:** native PyTorch SDPA factual plus explicit CMVE, with the
  user's 4D additive mask. Tang profiler confirmed PyTorch dispatches to
  **Efficient Attention**, NOT a FlashAttention-specific implementation.
- **Forced Flash+CMVE:** factual uses PyTorch's explicitly
  `SDPBackend.FLASH_ATTENTION` with a verified simple causal unpadded mask;
  the CMVE correction remains explicitly materialized and uses the identical
  actual 4D mask. This is **closer** to the original CFPO factual execution,
  but still not its exact legacy flash-attn + verl training stack.

| Length | P2-C full step | Forced Flash+CMVE full step | P2-C peak extra | Forced Flash+CMVE extra |
|---:|---:|---:|---:|---:|
| 256 | 0.9394ms | 1.1703ms | 9.801MiB | 22.018MiB |
| 512 | 1.3207ms | **1.0342ms** | 19.598MiB | 76.033MiB |
| 1024 | 4.0356ms | 4.1820ms | 39.192MiB | 283.066MiB |
| 2048 | **13.5389ms** | 16.5837ms | 78.380MiB | 1100.129MiB |

At L=2048, P2-C achieved about **1.22x** faster full step and about
**92.9%** lower peak *extra* allocated memory in this synthetic run versus
forced Flash+CMVE. Shorter lengths do NOT consistently favor P2-C.
At L=512, P2-C was slower by about 27.7%.

**Caveats:** the performance numbers are single-run medians of ~7 event
repetitions (warmup included), with GPU clocks not locked. Full official
CFPO training, real data and model-quality endpoints are NOT measured.

## 4. Real pretrained Qwen2.5-VL-3B numerical results

Pinned model revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`,
head layout Q=[1,16,97,128], native K/V=[1,2,97,128].
Source original CFPO CMVE: pinned commit `98ca5a4123be2e85e94855fead46c9a385611136`.

| Metric | Real decoder layer 0 | Real decoder layer 8 |
|---|---:|---:|
| Exact hard CFPO selected positions | 243 | 206 |
| P2-C hard-mask flips | **0** | **0** |
| BF16 output relative L2 | 0.0002118 | **0** |
| Q gradient relative L2 | 0.0011213 | 0.0000069 |
| Native K gradient relative L2 | 0.0025853 | 0.0027371 |
| Native V gradient relative L2 | 0.0024748 | 0.0025418 |

These gradients treat captured Q/K/V as independent inputs. They are not
entire model parameter gradients or a full CFPO training comparison.

### Frozen full model layer injection

A single selected native Qwen2.5-VL-3B layer 8 was run with Dense BF16 CMVE
versus P2-C, and the selected layer gradients were propagated from the
complete pretrained model's final logits, while model weights stayed frozen.

- Peak model GPU allocation 7.394 GiB.
- Final logit relative L2 difference ~0.73%.
- Selected-layer Q/K/V gradient relative L2 difference ~4.89%, ~4.54%, ~5.43%.

**This is still a training-equivalence blocker.** Small local BF16 differences
are amplified by subsequent model layers. The tiled P2-C kernel is more
efficient, but is NOT automatically more accurate end-to-end than the
rowwise V2 version. Do not silently promote P2-C as the default training
kernel until quality/gradient gates are addressed.

## 5. Implementation and measured evidence

- [Profiler stage breakdown](../experiments/profile_native_kernel.py)
- [Original rowwise BF16 VJP](../kernels/triton_native_backward_v2.py)
- [Tensor Core tiled factual](../kernels/triton_native_tile.py)
- [Tensor Core K/V gradient](../kernels/triton_native_gemm_kv.py)
- [Tensor Core Query gradient and VJP](../kernels/triton_native_qtile.py)
- [Real pretrained fixture parity script](../experiments/p2_real_gemm_parity.py)
- [Benchmark driver](../experiments/benchmark_native_tiled.py)
- [Forced Flash factual control](../reference/native_sdpa_cmve.py)
- [Sanitized CUDA benchmark](../experiments/results/tang_2026-11-21_p2_tensorcore_flash_baselines.json)
- [Real model QKV parity](../experiments/results/tang_2026-11-21_p2_real_qwen_tensorcore.json)
- [Frozen 3B gradient parity](../experiments/results/tang_2026-11-21_p2_frozen_qwen_model_gradient.json)

Raw QKV and model weights remain on Tang; only aggregate metrics and code
have been committed. Python syntax CI, Torch CUDA/Triton regression tests
were rerun after changes.

## 6. Required next research gates

1. **Controlled throughput runs:** multiple independent seeds and alternating
   method order, p50/p90 and peak allocations, with SM clock/temperature
   recorded, L=256..4096, varied visual-token percentages and image count.
2. **Strong performance baseline:** compare against official CFPO's exact
   legacy FlashAttention2 / verl execution rather than only a PyTorch
   FlashAttention-factual surrogate.
3. **Model-level numerical parity:** quantify downstream logits/gradient
   effects in more than one layer and multiple real prompts; investigate
   BF16 rounding order, proper original factual FlashAttention path and
   mixed-precision training loss scaling.
4. **Further optimization:** CUDA profiler guided tile tuning and reducing
   duplicate QK across factual/statistic/correction and VJP passes. The
   CMVE region/global threshold is *nonlocal*; statistics cannot simply be
   moved into one uncoordinated local tile.
5. **Generalizable systems novelty:** a special-case CMVE kernel alone is
   insufficient proof of A-conference contribution. Need an abstraction for
   statistics-conditioned Attention interventions, clear IO bounds,
   multi-family support, and genuine full training benefit.

**Policy:** GitHub is canonical source. Tang is the GPU verification host.
No A100/H100 rental yet; no synthetic microbenchmark should be marketed as
a completed academic result.
