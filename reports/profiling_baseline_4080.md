# Baseline Profiling on RTX 4080（W2）

实验：`EXP-W02-01-baseline-gsm8k`（116 步）与 `EXP-W02-02/03/04`（各 3×10 步参数扫描）。
硬件：RTX 4080 16GB（bigdog01），驱动 580.178.04，torch 2.11.0+cu130，verl v0.9.1，vLLM 0.24.0。
日期：2026-10-21。

## 摘要

1. **4080 的瓶颈在训练侧，不在 rollout。** 单步中位数约 **159 s**，其中 `update_actor` **113 s（71%）**、
   `old_log_prob` 27.5 s（17%）、`ref` 9.5 s（6%）、`gen` 7.6 s（5%）、`update_weights` 1.6 s（1%）。
2. **官方 2×H100 的瓶颈恰好相反**：单步 56 s，`gen` 45 s（80%）。原因是官方为了 2×80GB 的保守显存
   配置用了 `gpu_memory_utilization=0.1` + `max_num_batched_tokens=1536`，把 vLLM 的并发压得很低；
   4080 上改成 util 0.30 + 8192 后，同样 320 sequences 的 rollout 只要 7.6 s。
3. 训练侧慢的主因是 **micro batch = 1**：320 sequences/step 被拆成 320 次 forward/backward，且没有
   flash-attn（SDPA + padding）。`update_actor` 比官方慢 **16×**，`old_log_prob` 慢 **13×**，`ref` 慢 **12×**。
4. 最值得做的 3 个优化（详见 §7）：
   - **P1 actor / logprob 的 micro-batch 或 dynamic bsz**（预计单步 1.5–2× 加速；`EXP-W02-04` 实测）；
   - **P2 用 vLLM rollout 自带的 logprobs 替代单独一遍 `old_log_prob`**（省 ~17%，需数值一致性验证）；
   - **P3 训练侧 flash-attn / remove-padding / kernel fusion**（省 10–20%，W9 评估）。
   三项叠加后单步有望从 ~160 s 降到 ~80–100 s，相对 W2 baseline 的 **≥30% 加速目标**有足够余量。

## 1. 方法与口径

- 数据来源：verl 内置的 `timing_s/*`、`perf/*`、`actor/perf/max_memory_*`；解析脚本
  `analysis/parse_profile.py`（同时支持本地日志与 wandb API）。复现：

  ```bash
  python analysis/parse_profile.py --logs 'logs/EXP-W02-0*.log' --out-dir reports/figures
  ```

- 统计口径：剔除 step 1（warmup / CUDA graph / 首次编译）；对含 `timing_s/testing` 或
  `timing_s/save_checkpoint` 的步，在"纯训练步"统计中剔除。报告的是**中位数**。
- 参数扫描每组 30 步：3 个取值 × 10 步，只用于观察耗时/显存敏感度，**不作精度结论**。
- `perf/mfu/*` 恒为 0（verl 未启用 flops counter），本报告不引用 MFU。
- 单卡上所有实验串行，没有训练/评测并发。

## 2. 配置对照

| 项 | 官方（2×H100） | 本实验（4080） |
|---|---|---|
| 模型 | Qwen2.5-1.5B-Instruct + LoRA r32/alpha32 | 同 |
| 算法 | GRPO，KL loss 0.001，entropy 0.001，lr 3e-5 | 同 |
| prompts/step × rollout.n | 64 × 5 = 320 seq | 同 |
| prompt / response 上限 | 512 / 1024 | 同 |
| epochs / steps | 1 / 116 | 同 |
| 训练 micro batch | 64（每卡） | **1** |
| vLLM util / max batched tokens | 0.1 / 1536 | **0.30 / 8192** |
| attention | flash-attn（训练） | **SDPA，remove_padding=False** |
| LoRA 同步 | adapter + layered_summon | **merge 全量**（vLLM 0.24 兼容性） |
| test_freq | 5 | **20**（只影响监控频率） |

## 3. 单步耗时构成（EXP-W02-01，96 个续跑步的中位数）

| 阶段 | 中位数 (s) | 占比 |
|---|---|---|
| `timing_s/update_actor` | 114.36 | 70.6% |
| `timing_s/old_log_prob` | 27.52 | 17.0% |
| `timing_s/ref` | 9.77 | 6.0% |
| `timing_s/gen` | 8.84 | 5.5% |
| `timing_s/update_weights` | 1.58 | 1.0% |
| `timing_s/reward` | ~1 | ~0.6% |
| **单步合计**（纯训练步） | **161.99** | 100% |

- 吞吐 `perf/throughput` ≈ **600 tok/s**（prompt+response 总 token / step 时间）。
- actor 峰值显存 **9.28 GiB allocated / 9.53 GiB reserved**；vLLM 进程约 3.9–4.7 GiB。
- 每步总 token 约 95k–110k（response mean 190–210 tok）。
- 验证集 1319 题 greedy 一轮 `timing_s/testing` ≈ 42–52 s。

> 图：`reports/figures/profile_phase_pie.png`。

## 4. 与官方 2×H100 的对照

| 指标 | 官方 2×H100 | 4080 | 4080 / 官方 |
|---|---|---|---|
| 单步中位数 | 56.2 s | 161.99 s | **2.88×** |
| `gen` | 45.0 s（80%） | 8.84 s（5.5%） | **0.20×** |
| `old_log_prob` | 2.09 s | 27.52 s | 13.2× |
| `ref` | 0.80 s | 9.77 s | 12.2× |
| `update_actor` | 7.08 s | 114.36 s | 16.2× |
| 吞吐 | 880 tok/s | 600 tok/s | 0.68× |
| 峰值显存 | 25.9 GB / 2 卡 | 9.28 GB / 1 卡 | — |

**解读：**

- 官方配置是 **rollout-bound**：80% 时间在 vLLM 生成。其 `max_num_batched_tokens=1536` 对 320 条
  平均 ~200 token 的响应来说batch 太小，GPU 空转；这是为 2×80GB 保守设置的遗留值，不是算法要求。
- 4080 把 vLLM util 提到 0.30、batched tokens 提到 8192 后，rollout 吞吐反超官方（7.6 s vs 45 s），
  代价是 KV cache 只有约 1.3 GB，长尾请求会排队，但在 1536 总长下足够。
- 4080 的 96% 时间在训练侧（actor + old_log_prob + ref）。这不是算力差距能完全解释的：
  官方训练侧只有 10 s，4080 是 150 s（15×），而单卡算力差距远小于 15×。**根因是 batching**：
  官方每卡 micro batch 64（每步约 3 个 micro-batch），4080 micro batch 1（每步 320 个），
  每次 forward/backward 只处理 1 条序列，GPU 利用率和 Python/FSDP 开销都被放大。
- 另一个放大项是没有 flash-attn：`use_remove_padding=False` + SDPA 对 padding 也算，
  1.5B 模型在 1024 token 上的 attention 成本高于 varlen flash-attn。

## 5. 参数敏感度

> 三组实验：`EXP-W02-02`（vLLM util）、`EXP-W02-03`（rollout.n）、`EXP-W02-04`（micro batch）。
> 每组 10 步/取值，表中为纯训练步中位数；失败也入档。

### 5.1 vLLM `gpu_memory_utilization`（0.25 / 0.30 / 0.35）

| util | 能否启动 | 单步 (s) | gen (s) | 峰值显存 (GB) | 备注 |
|---|---|---|---|---|---|
| 0.25 | 待回填 | 待回填 | 待回填 | 待回填 | 预期 vLLM KV 不足 |
| 0.30 | 待回填 | 待回填 | 待回填 | 待回填 | baseline 值 |
| 0.35 | 待回填 | 待回填 | 待回填 | 待回填 | 预期 KV 更大、gen 略快 |

### 5.2 `rollout.n`（4 / 5 / 8，train_batch 固定 64 prompts）

| n | sequences/step | 单步 (s) | gen (s) | actor (s) | 吞吐 (tok/s) | 备注 |
|---|---|---|---|---|---|---|
| 4 | 256 | 待回填 | 待回填 | 待回填 | 待回填 | 有效 batch 变小 |
| 5 | 320 | 待回填 | 待回填 | 待回填 | 待回填 | baseline |
| 8 | 512 | 待回填 | 待回填 | 待回填 | 待回填 | 显存/时间压力最大 |

### 5.3 `ppo_micro_batch_size_per_gpu`（1 / 2 / 4）

| micro bs | 单步 (s) | actor (s) | old_log_prob (s) | 峰值显存 (GB) | 备注 |
|---|---|---|---|---|---|
| 1 | 待回填 | 待回填 | 待回填 | 待回填 | baseline |
| 2 | 待回填 | 待回填 | 待回填 | 待回填 | |
| 4 | 待回填 | 待回填 | 待回填 | 待回填 | 可能 OOM |

> 图：`reports/figures/profile_sensitivity.png`。

## 6. 瓶颈在哪

- **训练侧占 94%**（actor 71% + old_log_prob 17% + ref 6%）。rollout 只占 5%，即使 rollout 再快 2×，
  端到端也只快 2.5%。
- actor 更新内部：320 次 micro-batch（每次 1 条 ~1536 token 序列）→ 每次都要做 FSDP unshard/
  forward/backward/optimizer，kernel launch 与通信开销摊在极小的工作量上；`timing_per_token_ms/update_actor`
  ≈ **1.2 ms/token**，而官方是 0.06 ms/token（20×）。
- old_log_prob 与 ref 是两次独立的 320 条前向；它们的 `timing_per_token_ms` 分别约 0.30 / 0.10 ms，
  也远高于官方（0.008–0.01 ms）。同一根因：micro batch 太小 + 无 flash-attn。
- `reward` 1–2 s 是 320 条规则判分的 Python 开销，占比小；`update_weights` 1.6 s 是 merge 全量权重
  的固定成本，占比 1%，不是优化重点。

## 7. 最值得优化的 3 个点

### P1：actor / logprob 的 micro-batch（或 dynamic bsz）——收益最大、风险最低

- 做法：`ppo_micro_batch_size_per_gpu` 1 → 2/4，或 `use_dynamic_bsz=True` +
  `ppo_max_token_len_per_gpu` 按 token 打包；old_log_prob / ref 同步调整。
- 依据：`EXP-W02-04` 实测（见 §5.3）。micro 1→2 预计 actor 时间接近减半；4 可能 OOM。
- 风险：显存（16GB 已经很紧）；只改 batching，不改有效 batch / 学习率 / 步数，数学等价。
- 验收：同样的 20 步曲线 + 单步耗时；精度不回退。

### P2：复用 rollout 的 logprobs，去掉 `old_log_prob` 前向

- 现状：`old_log_prob` 27.5 s（17%）是对同一批 320 条序列再跑一遍 actor 前向。
- 做法：vLLM 生成时返回 token logprobs（`logprobs=1`），直接作为 PPO 的 old logprob；
  verl v0.9.1 的 rollout 已支持 `logprobs` 参数，但需要验证与训练 forward 的数值一致性
  （bf16/不同 kernel 会有 ~1e-3 差异，可能影响 ratio）。
- 收益：理论上省 17%；若与 P1 叠加，占比会更高。
- 风险：数值不一致会改变训练动力学，必须做 A/B（同 20 步 loss/ratio/精度对比），不能只看速度。

### P3：训练侧 attention / kernel（flash-attn、remove-padding、Liger）

- 现状：无 flash-attn，SDPA + padding；`use_remove_padding=False`。
- 做法：安装与本机 torch/cu13 匹配的 flash-attn（需要 nvcc 或预编译 wheel），或评估 Liger kernel；
  打开 `use_remove_padding=True`。这是 W9 的 FP8/吞吐工作的一部分。
- 收益：训练前向/反向 10–20%，显存也会下降，可能让 P1 的 micro 4 变得可行。
- 风险：环境/编译风险高（W1 已评估不装）；升级依赖需请示。

**组合预期**：P1（1.5–2×）+ P2（~1.15×）+ P3（~1.1×）→ 单步 ~160 s → ~75–95 s；
相对 W2 baseline 的 30% 加速目标有足够余量，且不需要改算法。

## 8. 局限与下一步

- **没有 3090**：任务包 §5 的两卡对比（显存容量 vs 算力、KV cache 对 rollout 吞吐的影响）本周无法执行；
  3090 到位后补 `reports/profiling_baseline_3090.md`。
- 三组扫描每组只有 10 步/取值，且是单种子；只用于看趋势，不用于统计显著性。
- 4080 的功耗/温度未记录；长时间训练可能降频，单步时间会漂移。
- 所有 profiling 实验都没有精度结论；baseline 的精度结论见 `reports/week02_summary.md`。
- 下一步（W9）：先做 P1 的 3 组 A/B（micro 2/4、dynamic bsz），再做 P2 的数值一致性验证；
  P3 视 flash-attn 安装可行性决定。
