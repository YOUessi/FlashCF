# FlashCF GPU Kernel Design and Verified Research State

Status: **experimental FP32 causal forward + custom backward**, tested on Tang's
RTX 4090 Laptop GPU (Ada SM 8.9). Official CFPO multi-modal model equivalence
has **NOT** been established.

## Relation to CFPO

CFPO already uses the value-difference identity below. This identity is **not**
our novelty. The proposed contribution is a bounded-intermediate, tiled GPU
execution algorithm which preserves intervention semantics and gradients.

For head h and query i:

- P_ij = softmax_j(Q_i K_j^T/sqrt(d)); 0 for future keys.
- For each image span (start,end), compute a *single threshold over all relevant
  heads and query/key positions*, excluding attention probabilities <= 1e-8.
- M_ij = 1 when key j is in the image span, query i >= end, P_ij>1e-8 and
  P_ij>threshold(span).
- c_bh is a **scalar** average of selected image (or text) V tokens and channels,
  separately for each batch/head, as the supported subset of CFPO defines.
- W_ij = V_j + M_ij(c_bh - V_j).
- Y_i = sum_j P_ij * W_ij.

## Forward: kernels/triton_forward.py

A. Row-wise online-softmax normal output + LSE, one program per query/head.
B. Recompute probabilities only for (query after image span, image-key columns)
   and write partial statistics (sum p, sum p^2, count). Reduce those
   across all heads/queries to get span-specific threshold. Uses sample-variance
   denominator count-1, with protection for count<=1.
C. Recompute the same image portions, evaluate M and accumulate Value correction.

The CUDA path avoids an explicit [B,H,L,L] probability tensor. Peak *extra*
memory is O(B*H*L*D + S*H*L*D) for S image spans plus small per-span statistics,
rather than an explicit O(B*H*L^2) attention allocation. S is not necessarily
constant for arbitrary workloads. q/k/v themselves are pre-existing.

Current forward is **not** a Tensor Core optimized tiled attention implementation:
each program handles one query row and loops over key tiles. This is a correct
starting point for performance optimization, not a final architecture claim.

## Backward v1: kernels/triton_backward.py

One program per query row re-computes each causal attention entry and directly
writes dQ; dK/dV are accumulated using tl.atomic_add. Correct on synthetic
FP32 but has serious contention for long sequences.

## Backward v2: kernels/triton_backward_gather.py

Two race-free passes:
- Query-centric pass: each query i gathers dQ_i over keys j<=i.
- Key-centric pass: each key j gathers dK_j and dV_j over future queries i>=j.
- Key-centric pass also emits partial scalar gradient of c_bh.
- A final linear reduction of c_bh's gradients distributes it to selected
  source V tokens through the actual replacement-value mean.

For hard M, holding the comparator's decisions fixed as PyTorch autograd does:

dlogit_ij = P_ij * [dY_i dot (W_ij - Y_i)]

dQ_i = sum_j dlogit_ij K_j/sqrt(d);
dK_j = sum_i dlogit_ij Q_i/sqrt(d);

(dV_j)_direct = sum_i P_ij(1-M_ij)dY_i;

g_c(b,h) = sum_i sum_j P_ij M_ij sum_d dY_id;

(dV_j)_mean = 1[j in replacement source] g_c/(source_token_count*d).

This version removes atomic contention and changes the parallel reduction
orientation. Both versions still cost O(L^2*d) attention arithmetic in the dense
causal case; their value is dramatically lower peak intermediate storage.

## Tang GPU results

See experiments/results/ for sanitized JSON generated from real GPU tests.
- All 13 baseline forward/backward and unit tests passed on Tang at commit
  5bb6565; additional gather tests 4/4 passed at d226bfd.
- FP32 [B=1,H=4,D=64], one image span:
  At L=1024, dense reference ~1.489 ms / 91.257 MiB extra,
  atomic ~4.096 ms / 6.036 MiB extra,
  gather ~1.707 ms / 6.051 MiB extra.
- Times are medians of a *small synthetic benchmark*, not statistically
  established cross-run results and NOT a comparison with optimized CFPO/
  FlashAttention fused reference.
- Gather v2 at 1024 is approximately 2.4x as fast as atomic v1; still slower
  than dense PyTorch reference in that measurement.

## Remaining blockers before paper-level claims

1. **Official CFPO fixture:** Record real rotated Q/K/V, per-sample image spans,
   additive attention mask, thresholds, output and grads using Qwen2.5-VL.
2. **Semantics:** Handle real 4D additive attention masks / packed sequences,
   nontrivial GQA, sliding-window variants, BF16/FP16 softmax casting and
   training dropout or explicitly delimit scope.
3. **Performance:** Compare against an optimized dense/materialized baseline,
   eager CFPO official + FlashAttention, FA2/FA4 where supported, and
   profile DRAM/tensor cores/occupancy. Use more lengths/head dims/ratios.
4. **Generalization:** Explore tile-row parallelism, fused on-device moment
   reduction and reduced recomputation. Verify any quality/accuracy difference.
5. **Research contribution:** A single special-case optimization may not be
   enough for MLSys/ASPLOS. Establish a reusable abstraction for a class of
   statistics-conditioned Attention interventions and derive IO/compute bounds.

## Reproduce

On Tang (only GitHub checkout is modified; no independent source edits):

    git pull --ff-only
    python3 -m unittest discover -s tests -v
    python3 experiments/benchmark_triton_forward.py --lengths 128 256 512
    python3 experiments/compare_backward_algorithms.py --lengths 128 256 512 1024

The above scripts use synthetic data and do not download models.

