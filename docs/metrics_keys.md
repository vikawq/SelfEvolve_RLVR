# wandb / 训练指标键名（W1 实测，verl v0.9.1 legacy trainer）

实验：`EXP-W01-01-smoke`（Qwen2.5-1.5B-Instruct + LoRA r32 + GRPO，GSM8K，单卡 4080）。
采集自 `logs/EXP-W01-01-smoke_20261020-203913.log`（step 1–10）与 `..._204519.log`（step 11–20）。

## 1. 奖励与准确率

| key | 含义 | W1 实测 |
|---|---|---|
| `critic/score/mean` | 规则判分器给出的得分均值（GSM8K 0/1 准确率） | step 1–10：0.047–0.422（逐步波动大，冒烟样本少） |
| `critic/score/max` / `min` | 一个 batch 内的最高/最低分 | 1.0 / 0.0 |
| `critic/rewards/mean` | 加 KL 惩罚前的 token-level reward 统计（本实验 `use_kl_in_reward=False`，与 score 同值） | 同 score |
| `critic/advantages/mean` | **GRPO 的组内相对优势**（无 critic；沿用了历史 `critic/` 前缀，不要被名字误导） | ±1.5 区间 |
| `critic/returns/mean` | 与 advantages 同值（GRPO 下 return=advantage） | ±1.5 区间 |

> 注意：GRPO 没有 value function，`critic/*` 前缀是 legacy trainer 的历史命名；指标本身有效，但不要把 `critic/rewards` 当成 critic 网络的输出。

## 2. 响应长度 / prompt 长度

| key | 含义 |
|---|---|
| `response_length/mean` / `max` / `min` | 生成响应长度（token） |
| `response_length/clip_ratio` | 触到 `max_response_length` 上限的比例 |
| `response_length_non_aborted/*` | 排除异常中止后的长度统计 |
| `response/aborted_ratio` | 生成异常中止比例（W1 = 0） |
| `prompt_length/mean` / `max` / `min` / `clip_ratio` | prompt 长度统计 |
| `num_turns/min|max|mean` | agent loop 的轮数（W1 恒为 2：user + 生成） |

## 3. KL / 熵 / loss

| key | 含义 | W1 实测 |
|---|---|---|
| `actor/kl_loss` | actor 对 ref 的 KL（`low_var_kl`，乘 `kl_loss_coef` 前的原始值） | phase1 均值 0.0025（0–0.0073）；phase2 均值 0.020（0.011–0.029），仅冒烟，不作结论 |
| `actor/kl_coef` | KL 系数 | 0.001 |
| `actor/entropy` | 策略熵 | 0.38 → 0.34 |
| `actor/pg_loss` / `actor/loss` | policy gradient loss / 总 loss | — |
| `actor/grad_norm` | 梯度范数 | 0.03–0.04 |
| `actor/lr` | 学习率（constant） | 3e-5 |
| `actor/pg_clipfrac` / `actor/pg_clipfrac_lower` | PPO clip 触发比例 | 0.0 |
| `actor/ppo_kl` | 本步策略更新前后的 KL（PPO 内） | 0.0 |
| `rollout_corr/kl` / `rollout_corr/k3_kl` | rollout 策略与训练策略的 KL（rollout correction 诊断） | ~2e-4 |
| `rollout_corr/ppl_ratio` / `chi2_*` | 重要性采样诊断 | — |
| `training/rollout_probs_diff_mean` | rollout 与 actor 重新计算的 logprob 差（训练/推理一致性） | ~0.0035 |
| `training/rollout_actor_probs_pearson_corr` | 上述差值的相关系数（>0.999 说明权重同步正确） | 0.9995 |

## 4. 分阶段计时（`timing_s/*`）——验收要求的阶段计时已内置，无需补丁

| key | 含义 | W1 均值 |
|---|---|---|
| `timing_s/step` | 单步总耗时 | **23.2s**（phase1）/ 23.1s（phase2） |
| `timing_s/gen` | vLLM rollout 生成 | 3.71s |
| `timing_s/old_log_prob` | 旧策略 logprob 重算 | 3.54s |
| `timing_s/ref` | ref 模型 logprob | 1.97s |
| `timing_s/update_actor` | actor 反传 + 更新（16 个 micro-batch 累积） | **12.28s（占 53%）** |
| `timing_s/update_weights` | 权重同步到 vLLM（LoRA merge 全量） | 1.56s |
| `timing_s/adv` | 优势计算 | ~0.001s |
| `timing_s/testing` | 验证集评测（仅 `test_freq` 命中的步） | 54.2s / 48.0s |
| `timing_s/save_checkpoint` | 存 checkpoint（仅 `save_freq` 命中的步） | 1.67s |
| `timing_s/agent_loop/generate_sequences` | 单条轨迹生成耗时（min/mean/max） | mean 1.96s |
| `timing_s/agent_loop/compute_score` | 规则判分耗时 | ~0.0024s |

## 5. 吞吐与显存（`perf/*`）

| key | 含义 | W1 实测 |
|---|---|---|
| `perf/throughput` | 每秒训练 token 数 | **982 tok/s**（phase1）/ 950（phase2） |
| `perf/time_per_step` | 与 `timing_s/step` 相同 | 23.2s |
| `perf/total_num_tokens` | 本步 token 总数（prompt+response） | ~22k–24k |
| `actor/perf/max_memory_allocated_gb` | actor 进程 PyTorch 峰值显存 | **9.28 GiB** |
| `actor/perf/max_memory_reserved_gb` | 预留峰值 | 9.53 GiB |
| `actor/perf/cpu_memory_used_gb` | 进程 CPU 内存 | ~25.6 GB |
| `perf/mfu/actor` / `perf/mfu/actor_infer` | MFU | **恒为 0.0（flops counter 未启用，不要引用）** |

> 显存没有全局 key：vLLM 是独立进程，需配合 `nvidia-smi` 一起看。W1 训练稳态整卡占用约 8.5–13 GiB（vLLM 进程约 3.9 GiB）。详见 `docs/troubleshooting.md` §3。

## 6. 验证指标（`test_freq` 命中时）

| key | 含义 | W1 |
|---|---|---|
| `val-core/openai/gsm8k/acc/mean@1` | 验证集 greedy/mean@1 准确率 | step 10 = **0.5550**，step 20 = **0.7081** |
| `val-aux/openai/gsm8k/reward/mean@1` | 同一批样本的 reward 均值（与 acc 同值） | 同上 |
| `val-aux/num_turns/*` | 验证时轮数 | 2 |

## 7. 元信息

| key | 含义 |
|---|---|
| `training/global_step` | 全局步数（续跑后从 11 继续） |
| `training/epoch` | epoch 序号（20 步远小于 1 个 epoch） |
| `global_seqlen/*` | 全局序列长度（padding 前后 / batch balance 后） |

## 8. wandb

- 实例：`https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/`
- project：`selfevolve-rlvr`
- 成功运行：phase1 `runs/kv34uf1v`、phase2 `runs/16fmimie`
- 失败尝试（v1 trainer bug）：`runs/ktrsbih8`

## 9. W2 baseline 实测补充（320 sequences/step，96 步中位数）

- `perf/max_memory_allocated_gb` 在 v0.9.1 日志里以 `np.float64(9.28)` 形式打印，
  `analysis/parse_profile.py` 已兼容；解析时不要用只匹配裸数字的正则。
- 单步 **161.99 s**：`update_actor` 114.36 s（70.6%）、`old_log_prob` 27.52 s（17.0%）、
  `ref` 9.77 s（6.0%）、`gen` 8.84 s（5.5%）、`update_weights` 1.58 s（1.0%）、`reward` ~1 s。
- 吞吐 `perf/throughput` ~600 tok/s（4080，vLLM util 0.3）。
- 验证集 1319 题 greedy 一轮 `timing_s/testing` ≈ 42–52 s（比训练步快，因为只生成不反传）。
- `perf/mfu/*` 仍恒为 0；不要引用。
- 与官方 2×H100 对照、参数扫描见 `reports/profiling_baseline_4080.md`。

