# P0 — Real Qwen2.5-VL-3B validation on Tang

**Date:** 2026-11-21  
**Canonical source:** https://github.com/YOUessi/FlashCF  
**GPU:** Tang RTX 4090 Laptop (16GB); no cloud server, no full RL training.

## Exact source and model versions

- Model: Qwen/Qwen2.5-VL-3B-Instruct, pinned revision: 66285546d2b821cf421d4f5eb2576359d3770cd3
- Original CFPO code: Raven-July/CFPO commit 98ca5a4123be2e85e94855fead46c9a385611136
- Original CMVE Git blob SHA-1: 2b9e77ea947f3b07801c9e6a863b774ac2521a5a
- Runtime: torch 2.10.0+cu128, transformers 4.57.6, Triton 3.6.0
- Qwen model weights total 6.994 GiB, downloaded via ModelScope HTTP Ranges.
- Both weight shards were independently SHA-256 checked against pinned Hugging Face LFS metadata and ModelScope X-Linked-Etag.
- No model weights, images, or raw QKV activation tensors are uploaded to GitHub.

## A. Native pretrained model forward / capture

- Synthetic input: 224x224 picture containing 3 colored rectangles + one short question.
- Image processing: slow processor, avoiding Transformers 4.57 default fast-mode change.
- Input length 97, image token interval [15,79) (64 image tokens).
- Native eager Q/K/V captured **after M-RoPE** from decoder layers 0 and 8.
- Each expanded Q/K/V has shape [1,16,97,128] and BF16 dtype; native GQA repeat = 8.
- Actual attention mask: [1,1,97,97], normal causal, unpadded.
- Peak model forward allocation: **7.131 GiB** on Tang.
- Unmodified native eager factual output and independently recomputed BF16 normal Attention: max absolute error **0.0**.

Capture code: [p0_capture_native_qwen.py](../experiments/p0_capture_native_qwen.py)

## B. Original CFPO CMVE operator on *real pretrained QKV*

Instead of silently reimplementing the source, the original CFPO add_diffusion_noise, GMM_mask and values_noise_multiply functions were AST-extracted unchanged from the pinned original code after Git blob verification.

- Original BF16 CMVE mask vs independent reference: **exactly equal** in both captured layers.
- Original BF16 CMVE output vs independent BF16 replay: **max abs error 0.0**.
- Original FP32 operator vs independent FP32 dense replay on real QKV cast to FP32: **max abs error 0.0**.
- Current FP32 Triton gather kernel versus dense reference on real QKV:

| Decoder layer | FP32 output max abs | FP32 output RMSE | FP32 QKV gradient max abs | Fixed gate |
|---|---:|---:|---:|---|
| 0 | 8.0228e-5 | 1.2257e-6 | 4.7858e-4 | PASS |
| 8 | 7.1526e-7 | 3.3672e-8 | 9.5367e-6 | PASS |

The Triton output **is FP32 applied to Q/K/V cast from BF16**, not a native BF16 Triton path. Gradient checks use detached Q/K/V as independent leaves, **not whole-model gradients**.

Replay: [p0_replay_native_qwen.py](../experiments/p0_replay_native_qwen.py) and [p0_official_cmve_oracle.py](../tools/p0_official_cmve_oracle.py).

## C. Original CMVE operator inserted into a real Qwen2.5 layer

Using the original SHA-verified CFPO CMVE helper functions inside the **native** Transformers Qwen2.5-VL layer-8 eager Attention:

- Selected mask entries: **380 / 150,544**
- Intervention Attention output max absolute delta: **1.8828125**
- Full model next-token logit max absolute change: **0.8125**
- Mean absolute next-token logit change: **0.1273368**

Therefore, the original value intervention can execute inside the pretrained Qwen2.5-VL call path and influence model logits. **This is NOT the published legacy CFPO/verl multi-layer model forward or a training result.**

Injection source: [p0_inject_native_cmve.py](../experiments/p0_inject_native_cmve.py).

## Local paths on Tang

- Verified model: /home/you/.cache/flashcf_models/Qwen2.5-VL-3B-Instruct
- GitHub-pinned immutable source execution snapshots: /home/you/.cache/flashcf_runs/<commit-sha>
- Raw QKV fixtures: /home/you/.cache/flashcf_fixtures/qwen2_5_vl_3b/
- Sanitized output: /home/you/.cache/flashcf_results/
- Original SHA-validated CFPO source cache: /home/you/.cache/flashcf_upstream/qwen2_vl_cmve_98ca5a.py

GitHub fetch from Tang was intermittently unavailable; the execution runner downloads an **immutable codeload.github.com commit archive**, not an independently edited source tree. See [p0_run_tang_snapshot.sh](../experiments/p0_run_tang_snapshot.sh).

## Remaining P0 limitations / next gate

1. Native BF16, additive 4D masks, native GQA and gradient parity in a *production* FlashCF kernel: not yet done.
2. Padded and packed sequences, sliding window, nonzero dropout, and multiple true model images: not yet done.
3. Full patched original CFPO multi-layer model / verl RL training: not yet run; its published Qwen2VL model adapter does not directly share the same Transformers 4.57.6 native interface.
4. 4096-token FP32 strict max-error mask flip boundary: remains open; recorded separately, not silently relaxed.
5. Optimized FlashAttention/CFPO end-to-end training performance: not yet benchmarked.

**Next concrete task**: build a BF16-aware mixed-precision/causal/4D-mask wrapper that accepts the locally captured real Qwen2.5-VL fixtures, reproduces original BF16 CMVE choices, and falls back explicitly on unsupported mask shapes. Then implement backward parity and native multi-layer integration. Do not rent a GPU server yet.

## Sanitized experiment results

- [Real QKV and original BF16 operator parity](../experiments/results/tang_2026-11-21_p0_real_qwen_cfpo_operator.json)
- [Native model single-layer CMVE injection](../experiments/results/tang_2026-11-21_p0_native_model_cmve_injection.json)
- [Previous Triton synthetic benchmark](../experiments/results/tang_2026-11-21_sdpa_baseline_comparison.json)
