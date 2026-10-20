# EXP-W01-01-smoke 实验记录

- 日期：2026-10-20
- 目的：W1 冒烟。验证 20 步 GRPO-LoRA 链路（rollout → 规则奖励 → 训练 → checkpoint → 断点续跑），不追求精度。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置文件：`configs/EXP-W01-01-smoke.sh` + `configs/hw/4080.env`
- 模型：`models/Qwen2.5-1.5B-Instruct`（LoRA rank 32 / alpha 32 / all-linear）
- 数据：`data/gsm8k/{train,test}.parquet`（7473 / 1319）
- seed：42
- 关键超参：`train_batch_size=16`、`rollout.n=4`、`max_prompt_length=max_response_length=512`、`lr=3e-5`、`kl_loss_coef=0.001`、`total_training_steps=20`、`save_freq=10`、`test_freq=10`
- 环境：verl v0.9.1 / vLLM 0.24.0 / torch 2.11.0+cu130 / transformers 5.9.0

## 运行方式

```bash
# phase 1：跑 10 步并落 checkpoint
TOTAL_STEPS=10 bash scripts/run_exp.sh EXP-W01-01-smoke
# phase 2：resume_mode=auto 从 global_step_10 续跑到 20
TOTAL_STEPS=20 bash scripts/run_exp.sh EXP-W01-01-smoke
```

## 结果

| 指标 | phase1 (1–10) | phase2 (11–20) |
|---|---|---|
| 单步耗时 | 23.23s | 23.08s |
| update_actor | 12.28s | 12.32s |
| gen | 3.71s | 3.59s |
| old_log_prob | 3.54s | 3.55s |
| ref | 1.97s | 1.87s |
| update_weights | 1.56s | 1.57s |
| throughput | 982 tok/s | 950 tok/s |
| response_length/mean | 255.6 | 239.6 |
| actor/entropy | 0.378 | 0.339 |
| actor/kl_loss | 0.0025 | 0.0200 |
| val acc@1 | 0.5550 (step 10) | 0.7081 (step 20) |

- wandb phase1：https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/kv34uf1v
- wandb phase2：https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/16fmimie
- checkpoint：`checkpoints/EXP-W01-01-smoke/global_step_10`、`global_step_20`（各 3.6GB）

## 调试过程中被否掉的组合（详见 docs/troubleshooting.md）

| 组合 | 结果 |
|---|---|
| v1 trainer（默认） | `Fields ['values'] are not ready` 崩溃 |
| adapter 增量同步（`layered_summon=True`） | FSDP1 NO_SHARD 不支持 offload_to_cpu |
| adapter 增量同步（`layered_summon=False`） | vLLM `add_lora` StopIteration |
| LoRA merge + fp32 参数 | OOM（892MB 分配失败） |
| `ROLLOUT_GPU_MEM=0.25` | vLLM KV cache 无法初始化 |
| 无 `max_model_len` | vLLM 按 32768 预留 KV cache，初始化失败 |
| `flash_attention_2` 默认 | 未安装 flash-attn，ImportError |

## 结论

链路可用。W2 直接复用本配置，不要从零调参。注意 v1 trainer 的上游 bug 与 LoRA adapter 路径的上游不兼容，已在 `reports/week01_summary.md` 列为需要人决策的问题。
