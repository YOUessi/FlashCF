# M1: CFPO 官方真实 Q/K/V 捕获与回放（2026-11-21）

这一步的目标是从官方 **apply_cmve=True** 分支捕获**已经执行完 M-RoPE、repeat_kv 的** Q/K/V 和真实 attention mask、图像区间、dropout / sliding window / mean mode / threshold mode；暂时不修改 CFPO 算法，也不运行新 GPU Kernel。

## 1. 固定基线

- 官方仓库：https://github.com/Raven-July/CFPO
- 此轮审阅基准 commit：`98ca5a4123be2e85e94855fead46c9a385611136`
- 目标文件：`verl/models/transformers/qwen2_vl_cmve.py`（审阅 blob SHA `2b9e77ea947f3b07801c9e6a863b774ac2521a5a`）。
- **使用真实 CFPO 的一小批输入，不需要立刻开始完整训练。** 示例最短的 debug forward 应包含多图视觉 token、后续查询 token，并触发 apply_cmve 路径；无真实模型/GPU时无法完成该项。

## 2. 接入方式（保证在 Ray/DDP worker 内执行）

把 `capture_official_qkv.py` 复制到官方仓库根目录，确保该模块可从 Python path 导入。**在官方 `qwen2_vl_cmve.py` 文件末尾**添加诊断代码（只在环境变量开启时生效）：

```python
# Diagnostic only; add at bottom of verl/models/transformers/qwen2_vl_cmve.py
import os as _flashcf_os
if _flashcf_os.environ.get('FLASHCF_CAPTURE') == '1':
    import sys as _flashcf_sys
    from capture_official_qkv import CFPOCapture as _CFPOCapture
    _flashcf_capture = _CFPOCapture(
        _flashcf_sys.modules[__name__],
        _flashcf_os.environ.get('FLASHCF_FIXTURES_DIR', './flashcf_fixtures'),
        max_examples=2, max_sequence_length=2048,
    ).install()
```

**注意**：这段插桩应当出现在对应模型模块被所有分布式 worker 导入时；不要只在 Ray driver 进程调用 `install()`。先以一个 worker 的短 forward 检验能否产生文件，再扩大。可通过 `RANK`/`LOCAL_RANK` 自动分目录防止不同 rank 覆盖文件。

在官方运行环境中：

```bash
export FLASHCF_CAPTURE=1
export FLASHCF_FIXTURES_DIR="$PWD/flashcf_fixtures"
# 用仓库现有的最小 CFPO debug forward/训练启动命令运行一批数据。
# 必须保证 apply_cmve=True，且至少有一个可被统计的图像 token 区间。
find flashcf_fixtures -type f -name 'cfpo_cmve_*.pt'
```

成功文件位于 `flashcf_fixtures/rank0/cfpo_cmve_000.pt`（不同 rank 目录不同）。捕获数据包括：Q/K/V、FlashAttention factual 输出、4-D mask（如有）、图像区间、位置 ID 和参数；不存原始用户图像/文本，但这些激活仍可能包含敏感信息，请勿分享未经审查的 fixture。

### 为什么采用两处函数 Hook

- `values_noise_multiply()` 抓取 visual token interval、threshold mode、noise 和 mean mode，但**不保存完整 `[B,H,L,L]` Attention**。
- `flash_attention_forward()` 抓取已经 M-RoPE 和 `repeat_kv` 之后的 Q/K/V，以及 Mask 和普通分支输出；返回原函数结果，不修改梯度。

### 当前 Hook 限制

- 要求官方代码确实在同一 Python module 中按 `values_noise_multiply → flash_attention_forward` 调用；如引入并发/异步重入，需要进一步加强调用关联。
- 如果 `attention_mask is None`，**当前 replay 会拒绝声称对齐**：官方显式 Dense 路径未加 causal mask，而 FlashAttention 的 causal 语义可能不同。必须在真实输入下明确实际 Mask 与位置处理后才能继续。
- `noise=True`（随机扩散）、nonzero dropout、sliding_window、q_len != k_len 不在此轮 deterministic replay 支持范围内；需要单独定义支持策略。

## 3. 离线回放与验证

```bash
python replay_official_fixture.py /path/to/flashcf_fixtures/rank0/cfpo_cmve_000.pt
python test_capture_official_qkv.py
python test_replay_official_fixture.py
bash run_tests.sh
```

`replay_official_fixture.py` 首先构造**官方风格 Dense Attention＋阈值干预**；它会报告普通 FlashAttention factual 输出与 PyTorch Dense factual 输出误差，以及被干预概率元素比例。此工具**尚未**把完整 CFPO projected attention 输出与自定义 backward 对比，需要后续把 fixture 接入 `flashcf_custom()` 的官方 4D mask 路径并核对 `dQ,dK,dV`。

**FP32/FP16/BF16 容差**：原始 CFPO 在部分路径将 Softmax 在 FP32 执行、再 cast 回输入 dtype；数值参考必须遵照这个执行语义。FP64 理论算子与这种计算会有预期的小差异。遇到阈值附近 Mask flip 应单独计数和标记，而不能简单放宽阈值将错误掩盖。

## 4. M1 → M2 进入条件

- 对至少一种官方真实 Qwen2.5-VL Attention fixture，独立密集计算的**输出、Mask 以及梯度**与原版符合数值容差。
- 已明确完整的 4D Mask / causal / packing / position IDs / head grouping 语义；先锁定支持子集，未覆盖的配置明确 fallback。
- 建立可重复的结果文件及显式的精度分层报告；不可只报告合成 FP64 理论等价。

进入 M2 后优先实现 `stats_reduce.triton`（跨 query/head、单个 image interval 的 Welford 聚合），然后是 `value_correction.triton`，再做自定义 backward 与完整 CFPO 训练集成。

## 5. 目前已验证的内容

- 本地工具在 CPU PyTorch 2.10.0 下测试通过：Hook 不改变函数输出/梯度；多区间 additive-mask Dense 回放在 8 个阈值/均值组合下与独立参考输出、梯度在容差内一致。
- 原先的 5 组 Dense/Tiled 前向梯度测试、6 组自定义反向对比及 Padding/GQA/FP32 测试通过。
- **尚未使用真实官方模型生成 fixture**；当前工具只是为这一步提供插桩与独立数值基线。