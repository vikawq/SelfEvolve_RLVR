# Week 01 Summary

周期：2026-10-20。硬件：RTX 4080 16GB（bigdog01），verl v0.9.1 + vLLM 0.24.0 + torch 2.11.0+cu130。
实验：`EXP-W01-01-smoke`（Qwen2.5-1.5B-Instruct + LoRA r32 + GRPO，GSM8K，单卡）。

## 完成情况（对照验收标准逐条打勾）

- [x] 冒烟实验在 4080 上跑完 20 步，没有 OOM，保存了 checkpoint，并能从 step 10 断点续跑。
  - phase1 = step 1–10（`global_step_10`），phase2 = 用 `resume_mode=auto` 从 `global_step_10` 续跑到 20（`global_step_20`）；两次 launch 均 exit 0。
- [x] `configs/hw/4080.env` 已用实测值更新（含两处实测结论：vLLM util 0.3 是下限、训练必须 BF16 参数）。
- [x] `env/requirements.lock`（256 个包）和 `env/versions.md` 已提交。
- [x] `docs/data_format.md`（3 条真实样本 + schema）和 `docs/metrics_keys.md`（按 7 组分类，含实测值）已写好。
- [x] registry 中有 `EXP-W01-01-smoke` 的完整记录（状态、wandb、指标、GPU 时长、结论）。
- [x] `reports/week01_summary.md` 已写好（本文件）。
- [x] 额外完成：`docs/troubleshooting.md` 记录了 **7 类** 环境/版本/上游 bug 差异；`scripts/prepare_data.sh`、`scripts/run_exp.sh` 可重入；`configs/hw/3090.env` 已建（未实测）。

## 关键结果（附 wandb 链接）

训练本身只用于验证链路，**不构成任何精度结论**；下表数字仅用于 W2 的优化基线。

| 指标 | phase1（step 1–10） | phase2（step 11–20，续跑） |
|---|---|---|
| 单步耗时 `timing_s/step` | 23.23s（22.50–25.73） | 23.08s（21.89–25.18） |
| 其中 `update_actor` | 12.28s（53%） | 12.32s（53%） |
| `gen` / `old_log_prob` / `ref` / `update_weights` | 3.71 / 3.54 / 1.97 / 1.56s | 3.59 / 3.55 / 1.87 / 1.57s |
| 吞吐 `perf/throughput` | 982 tok/s | 950 tok/s |
| 峰值显存（actor 进程） | 9.28 GiB allocated / 9.53 GiB reserved | 同 |
| 验证集 `val-core/openai/gsm8k/acc/mean@1` | **0.5550**（step 10） | **0.7081**（step 20） |
| 响应长度均值 | 255.6 tok | 239.6 tok |
| 策略熵 `actor/entropy` | 0.378 | 0.339 |

- wandb（自建实例 `forge.coreweave.com/wandb/vika-liaowq-nanjing-university`，project `selfevolve-rlvr`）：
  - phase1：https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/kv34uf1v
  - phase2（续跑）：https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/16fmimie
  - 失败尝试（v1 trainer bug）：https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/ktrsbih8
- checkpoint：`checkpoints/EXP-W01-01-smoke/global_step_{10,20}`，各 3.6GB；`latest_checkpointed_iteration.txt` 续跑后为 `20`。
- 续跑证据：phase2 日志含 `Resuming from .../global_step_10`，并加载了 model / optimizer / rng / lr_scheduler；`training/global_step` 从 11 开始。

## 与计划的偏差及原因

1. **verl 版本**：任务包写法假设较老版本；实际安装最新稳定版 v0.9.1（2026-09-20，commit `1876b06`）。配套 torch 2.11.0 / vLLM 0.24.0 / transformers 5.9.0 由 verl 的 `pyproject.toml` 锁定。逐项差异表见 `docs/troubleshooting.md` §2。
2. **LoRA 模板**：`examples/grpo_trainer/` 下没有 LoRA 示例；改用 `examples/tuning/lora/run_qwen3_8b_fsdp.sh`（官方 README 指定的 canonical LoRA 脚本）。
3. **LoRA 同步走 merge 路径**：vLLM 0.24 的 `add_lora` 与 verl 0.9.1 的 adapter 增量同步不兼容（`StopIteration`），改为 `model.lora.merge=True`（官方同样支持的路径，每步同步全量权重，1.56s）。
4. **trainer**：v0.9.1 默认 `use_v1=True`，但 v1 在「无 critic 的 GRPO」下 `_compute_advantage` 硬编码读取 `values` 字段必崩（上游 bug）。改用 legacy trainer `use_v1=False`（两者共用同一 worker 实现）。
5. **attention**：不装 flash-attn，actor/ref 用 SDPA + `use_remove_padding=False`；vLLM rollout 仍用内置 FLASH_ATTN。原因：系统无 nvcc、未源码编译 flash-attn。
6. **显存**：文档 OOM 阶梯第 1 步（util 0.25）实际会让 vLLM 起不来（1.5B 权重 3.09GB）；真正的 OOM 根因是 verl FSDP 默认 `model_dtype=fp32`（参数 6.2GB），改为 `bf16` 后 micro batch 保持文档初值 2 即可跑通。实测值已回写 `configs/hw/4080.env`。
7. **一次实验两次 launch**：为了同时满足「跑完 20 步」和「从 step 10 续跑」，用 phase1(10 步) + phase2(resume→20) 组成一次 20 步的完整轨迹；这比先跑满 20 步再单独做一次续跑更省 GPU 时间，且续跑证据是一手日志。
8. **数据脚本参数**：`gsm8k.py` 的 `--local_dir` 已废弃，实际用 `--local_save_dir`。

## 失败与问题（按发生顺序，含错误信息与解法）

| # | 现象（关键错误） | 原因 | 解法 | 耗时 |
|---|---|---|---|---|
| 1 | `OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes` | `RAY_TMPDIR` 放在项目深层目录 | `RAY_TMPDIR=$HOME/raytmp` | ~3min |
| 2 | `ImportError: FlashAttention2 has been toggled on... not installed` | verl 默认 `attn_implementation=flash_attention_2` | SDPA + `use_remove_padding=False` | ~4min |
| 3 | `ValueError: ... max seq len (32768), 0.88GiB KV cache needed > 0.09GiB available` | `rollout.max_model_len` 默认 null → 用模型上限 32768 | 显式设 `max_model_len=1024` | ~4min |
| 4 | `NotImplementedError: offload_to_cpu=True and NO_SHARD is not supported yet` | 单卡 FSDP1 NO_SHARD 不支持 `layered_summon=True` | 关掉 `layered_summon` | ~5min |
| 5 | `StopIteration` in `vllm/lora/model_manager.py:_create_merged_loras_inplace` | verl adapter 增量同步与 vLLM 0.24 不兼容 | 改走 `model.lora.merge=True` | ~5min |
| 6 | `torch.OutOfMemoryError: ... Tried to allocate 892.00 MiB`（merge 恢复权重时） | FSDP 默认 `model_dtype=fp32`；util 0.25 又起不来 | `actor/ref.fsdp_config.model_dtype=bf16` | ~8min |
| 7 | `ValueError: Fields ['values'] are not ready...` | v1 trainer 对 critic-free GRPO 的 bug | `trainer.use_v1=False` | ~5min |

全部错误的现象/原因/解法已写进 `docs/troubleshooting.md`。**没有因为失败而降低验收标准**；也没有改动 verl / vLLM 源码，没有 patch 评测逻辑。

## GPU 时长 / 花费

- 成功轨迹：phase1 5.5min + phase2 5.0min ≈ **0.18 GPU·h**。
- 含 6 次调试尝试（20:17–20:36）总计约 **0.5 GPU·h**。
- 花费：**$0**（本地机器，未租云、未调用付费 API）。

## 给下周的交接

- **W2 baseline 起点**：直接复用 `configs/EXP-W01-01-smoke.sh` 的稳定组合（`use_v1=False`、`lora.merge=True`、`max_model_len=1024`、`model_dtype=bf16`、`ROLLOUT_GPU_MEM=0.3`、micro bs 2），把步数/数据换成 baseline 规模即可，不要再从零调参。
- **性能起点**：单步 23.2s，其中 `update_actor` 占 53%、`gen` 16%——W9 优化的第一优先级是训练侧（micro-batch/offload/kernel），不是 rollout。
- **W2 profiling**：`timing_s/*` 已内置（无需补丁）；注意 `perf/mfu/*` 恒为 0，如果要 MFU 需要自己实现 flops 统计。
- **风险提示**：v1 trainer 的 `values` bug 如果在下周仍未修，所有 GRPO 实验都要带 `use_v1=False`；若 W2 要用 async rollout / 新 v1 特性，需要先和负责人确认是否允许 patch 上游。
- **checkpoint**：每个 3.6GB（全量 actor + optimizer），做 baseline 时要控制 `max_actor_ckpt_to_keep`，否则 `/home` 会被吃满（857G，暂时够）。
- **3090**：`configs/hw/3090.env` 只有文档初值，到位后必须跑 `EXP-W01-02-smoke-3090` 回写；注意 3090 不支持 FP8（影响 W9）。

## 需要人决策的问题

1. **是否接受 `trainer.use_v1=False`（legacy trainer）作为后续基线路径**？W2–W11 都会受影响。若不接受，需要允许 patch `verl/trainer/ppo/v1/trainer_base.py` 的 `values` 字段逻辑，或升级 verl/vLLM。
2. **是否允许升级核心依赖**（vLLM ≥0.24.1 或 verl main）以修复 LoRA adapter 路径与 v1 GRPO bug？协议要求升级前请示，且升级后要重跑 baseline 冒烟。
3. W6 的硬件路线（3090 / 4080+0.6B / 云）需要在 W6 前定，本周不阻塞。
