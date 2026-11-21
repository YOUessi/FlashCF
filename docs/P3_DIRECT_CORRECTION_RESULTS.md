# P3 — Direct Correction-Only CMVE + Factual FlashAttention

**Executed on Tang (RTX 4090 Laptop, 16 GB) on 2026-11-21.**  
**Canonical code:** https://github.com/YOUessi/FlashCF  
**State:** research implementation with real pretrained Qwen2.5-VL validation, **not** end-to-end official CFPO RL training.

## 1. Why P3 was necessary

P2 all-Triton Q/K/V Attention already saved memory and accelerated longer
sequences, but it **recomputed the factual Attention output using different
BF16 reduction/rounding orders from the original CFPO FlashAttention path**.

When inserted in one layer of the full pretrained Qwen2.5-VL-3B and compared
against a forced FlashAttention factual + original explicit CMVE surrogate,
P2 showed large *downstream model* differences:

| Layer 0, full frozen Qwen model | P2 all-Triton | P3 direct |
|---|---:|---:|
| Final logit relative L2 | 0.01568 | **0.0** |
| Q-gradient relative L2 | 0.2420 | **0.0** |
| Native K-gradient relative L2 | 0.3765 | **0.002685** |
| Native V-gradient relative L2 | 0.1709 | **0.000141** |

This motivated keeping the exact factual FlashAttention kernel and
computing **only CMVE Value correction** with a custom tiled Triton kernel.

## 2. New execution architecture

Ordinary FlashAttention (separate factual branch):

```text
BF16 Q/K/V (native GQA)
    |
    +--> one PyTorch forced FlashAttention factual kernel --> factual output
    |
    +--> Triton LSE / global span statistics / selected Value delta --> delta output
                                                                 |
                 BF16 factual + BF16 delta ---------------------+
                                  |
                      same model interface
```

The old `hybrid_flash_factual.py` diagnostic computed two entire
counterfactual/factual Triton graphs and subtracted their outputs, plus a
third Flash factual path. P3 **eliminates this redundant factual computation**.

Core implementation:

- [triton_cmve_delta.py](../kernels/triton_cmve_delta.py): direct delta-only
  forward/backward and composition with one factual FlashAttention.
- [triton_native_bf16.py](../kernels/triton_native_bf16.py):
  `forward_native(delta_only=True)` calculates LSE, global visual moments,
  threshold and Value correction, **skipping the full factual P@V**.
- [triton_native_qtile.py](../kernels/triton_native_qtile.py):
  correction-only pre-cast BF16 Softmax VJP and Query gradient.
- [triton_native_gemm_kv.py](../kernels/triton_native_gemm_kv.py):
  correction-only dK/dV with negative direct selected-Value gradients,
  group-wise KV aggregation, plus scalar replacement-mean gradients.
- [test_triton_cmve_delta.py](../tests/test_triton_cmve_delta.py):
  native BF16 GQA, multi-region, no-region and 4D padding mask tests.

No extra full `[B,Hq,L,L]` Attention matrix is saved by the Triton
correction. **The user-supplied 4D additive mask itself is still quadratic**.
The current causal factual branch repeats native K/V groups using PyTorch
to call its pinned Flash backend; the Triton correction itself reads
native GQA head storage without that repeat.

### Mathematical differential scope

With fixed hard binary gate M, original CMVE correction is

```text
Delta_i = Sum_j P_ij * M_ij * (c - V_j)
```

where c is the scalar per-(batch,head) masked source Value mean. The
Triton backward calculates

```text
dL/dP_ij = M_ij * dot(dL/dDelta_i, c - V_j)
dL/dV_j (direct) = - Sum_i P_ij*M_ij*(dL/dDelta_i)
dL/dc = Sum_i,j P_ij*M_ij * Sum_d dL/dDelta_i,d
```

and propagates dL/dc back to the source V tokens. The hard comparator
has no gradient, matching the original PyTorch CMVE convention. This
is **not** claiming that value-delta algebra (already in CFPO) is new;
the contribution is its bounded-memory, GPU-efficient execution and
reverse-mode differentiation.

## 3. Verified correctness on Tang

**46/46 tests passed** (in commit `97315991fdcf29ca83a025ee3f4e7d84a372b6b4`).
P3 regression cases cover short real-Qwen-shaped BF16 GQA, two disjoint
image intervals, key padding with 4D additive mask and empty intervention.

On actual pretrained Qwen2.5-VL-3B captured BF16 activations
(`[1,16,97,128]`, native KV 2 heads):

| Against forced Flash + explicit CMVE | Layer 0 | Layer 8 |
|---|---:|---:|
| **BF16 forward max absolute error** | **0** | **0** |
| Q-gradient relative L2 | **0** | **0** |
| K-gradient relative L2 | 0.002498 | 0.002599 |
| V-gradient relative L2 | 0.000467 | 0.000246 |

The original **double-run hybrid diagnostic** still deviated from this
same real fixture baseline, whereas P3 forward exactly matched this
particular BF16 case. This does **not** prove exactness for arbitrary
input, length, mask or multiple images.

### One-layer frozen full model

At Qwen2.5-VL-3B **layer 0**, the original CMVE helper + forced Flash
and Direct Delta produced identical final next-token logits on this
single synthetic image. Loss was identical; selected-layer
Q-gradient was also identical. K/V gradient relative errors were
0.002685 / 0.000141.

At **layer 8** the final model logits were also identical (max abs 0);
selected-layer Q relative error 0, K/V error ~0.00294 / 0.000545.

This is substantially better than P2 all-Triton factual accuracy.

## 4. Long-sequence performance on Tang

Repeated independent synthetic BF16 Qwen-shaped workloads:
B=1, Hq=16, Hkv=2, D=128, 4D causal mask, single image interval.
Workload is **complete forward+backward of isolated Attention operator**.
Each length has two independent seeds, alternating method order,
3 rounds x 8 CUDA-event timings per method per seed (24 per seed).
Clocks and temperatures are preserved in the JSON.

| Sequence length | Forced Flash factual + explicit CMVE (p50) | P3 Direct Delta (p50) | Baseline / Direct speed ratio |
|---:|---:|---:|---:|
| 512 | 1.10–1.18 ms | 1.44–1.45 ms | 0.77–0.81x (Direct slower) |
| 1024 | 4.19–4.28 ms | 3.47–3.83 ms | 1.12–1.21x |
| 2048 | 16.58–16.59 ms | 12.79–12.80 ms | **1.296x** |

| Sequence length | Forced Flash + CMVE peak additional allocation | Direct peak additional allocation |
|---:|---:|---:|
| 512 | 76.033 MiB | 26.564 MiB |
| 1024 | 283.941 MiB | 53.127 MiB |
| 2048 | 1100.129 MiB | 106.252 MiB |

At 2048 the correction-only design reduces **incremental allocated peak
memory ~90.3%** against materialized CMVE and is ~29.6% more throughput
per isolated step under these conditions.

There is **no universal speedup at short sequence lengths**. P3 retains
LSE/CMVE moment computation and multiple GPU kernel launches; this
overhead dominates at 512 and below.

Data: [P3 robust repetitions](../experiments/results/tang_2026-11-21_p3_repeated_robust.json).

## 5. Full 36-layer pretrained Qwen3B test

We used a synthetic picture containing colored rectangles, 93 input tokens,
and replaced the factual+CMVE call in **all 36 language Attention layers**.
The full 3B model weights were frozen. This is a native Transformers model
integration, NOT the original older CFPO/verl training framework.

### Full-model forward only (one pass each)

- Materialized forced-Flash+CMVE: **381.63 ms**
- P3 direct: **293.01 ms**
- Model allocated peak: ~7.131 GiB each; at short lengths model weights
  dominate peak allocation.
- Predicted next-token Top1 ID: **3862** for both.
- Final logits relative L2: **0.005778 (~0.58%)**, maximum absolute
  difference 0.25 (BF16).

These timings are **one input / one run per method**; not a reliable
statistical claim about real deployments.

### Full 36-layer forward + backward to selected Q/K/V (one pass each)

Model weights were frozen; a deterministic scalar loss was backpropagated
through all 36 CMVE layers to layer 0, 8 and 35 Q/K/V only.

- Explicit forced-Flash+CMVE: **416.06 ms**; peak allocated **7.503 GiB**.
- P3 direct: **331.72 ms**; peak allocated **7.456 GiB**.
- Logit relative L2: **0.005777**, loss 74.307 vs 74.060.

Relative Q/K/V gradient differences:

| Layer | Q | K | V |
|---:|---:|---:|---:|
| 0 | 6.87% | 7.67% | 4.56% |
| 8 | 4.98% | 3.81% | 6.54% |
| 35 | 3.73% | 3.93% | 3.39% |

**Model-level numerical parity is still an open research gate:** repeatedly
applying BF16 differences across 36 layers amplifies them even though
isolated layers can appear nearly identical. We must not claim
training-equivalent full CFPO or unchanged reward optimization.

## 6. Reproducibility, data, and source constraints

- [P3 isolated real-QKV result](../experiments/results/tang_2026-11-21_p3_real_direct_qkv.json)
- [P3 layer0 full-model result](../experiments/results/tang_2026-11-21_p3_frozen_layer0.json)
- [P3 layer8 full-model result](../experiments/results/tang_2026-11-21_p3_frozen_layer8.json)
- [P3 repeated p10/p50/p90](../experiments/results/tang_2026-11-21_p3_repeated_robust.json)
- [P3 all-36-layer forward](../experiments/results/tang_2026-11-21_p3_all36_layers_forward.json)
- [P3 all-36-layer backward](../experiments/results/tang_2026-11-21_p3_all36_layers_backward.json)
- [Original P2 all-Triton results](P2_TENSORCORE_RESULTS.md)
- All raw pretrained weights and Q/K/V activations are stored on Tang and
  not submitted to Github.

Exact published CFPO CMVE source still pinned at commit
`98ca5a4123be2e85e94855fead46c9a385611136`. Original Qwen
model weights pinned at `66285546d2b821cf421d4f5eb2576359d3770cd3`.

## 7. Required next research phase

- **Numerical fidelity:** Multi-prompt, multi-image, longer histories and
  padding/packing; measure hard-mask flips, final-logit KL or top-k agreement,
  layer-by-layer error amplification, and gradient cosine similarities.
- **Full CFPO training integration:** Port injection into the official
  CFPO training graph (not only Qwen native eager) and test genuine
  counterfactual loss gradients and a short controlled optimizer run.
- **Systems optimization:** Eliminate unnecessary KV head repeat on the
  factual path (native GQA Flash), reduce global statistics kernel launches,
  and autotune by sequence length. Current 512-token step is slower.
- **Strong baselines:** Original CFPO pinned FlashAttention2+verl training,
  compatible torch compile/cudagraph options, and multiple GPU architectures.
- **Research novelty:** A specialized optimization of one CMVE branch is not
  automatically enough for MLSys/ASPLOS; formalize statistics-conditioned
  attention intervention as a reusable operator family and derive IO lower
  bounds and backward recomputation efficiency.

**No rented GPU server is required for the current Tang experiments.**
