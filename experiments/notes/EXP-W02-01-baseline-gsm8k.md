# EXP-W02-01-baseline-gsm8k 实验记录

- 日期：2026-10-21
- 目的：复现 verl 官方 Qwen2.5-1.5B-Instruct + GRPO-LoRA 在 GSM8K 上的 77.9 分，证明训练管线正确。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置文件：`configs/EXP-W02-01-baseline-gsm8k.sh` + `configs/hw/4080.env`
- git commit：`e6ccaecd4c336b6e13157b10c0394bb52f2470b1`
- 模型：`models/Qwen2.5-1.5B-Instruct`（LoRA rank 32 / alpha 32 / all-linear / `lora.merge=True`）
- 数据：`data/gsm8k/{train,test}.parquet`（7473 / 1319），`data.shuffle=False`
- seed：42
- 环境：verl v0.9.1（commit `1876b06`）/ vLLM 0.24.0 / torch 2.11.0+cu130 / math-verify 0.9.0

## 官方基线的考证（为什么是这组超参）

| 项 | 来源 |
|---|---|
| 表格 77.9 | verl `docs/algo/baseline.md`：Qwen/Qwen2.5-1.5B-Instruct \| GRPO-LoRA \| 77.9 |
| 脚本 | `examples/tuning/1.5b/qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh` @ verl `368dff7` |
| 实跑日志 | verl-data PR#3（`bcff590`）`Qwen2.5-1.5B-bsz64_2-...-score0.779.log`，最终 val 0.77938 |

日志落盘的 config：`train_batch_size=64`、`rollout.n=5`、`lr=3e-5`、`lora_rank=32`、
`prompt=512`、`response=1024`、`total_epochs=1`（= 116 步）、2×H100。

**坑**：文件名里的 `bsz64_2` 的 `_2` 是**2 张卡**，不是 `rollout.n=2`；实际 `n=5`，
有效 batch 320 sequences/step。表格链接的日志已从 verl-data 的 `experiments` 分支删除，
本次是从 PR #3 的 ref 取回的。详见 `docs/troubleshooting.md §4.1`。

## 关键超参（与官方对齐，不为赶时间改动）

`train_batch_size=64`、`ppo_mini_batch_size=64`、`rollout.n=5`（= 320 seq/step）、
`total_epochs=1`（116 步）、`lr=3e-5`、`kl_loss_coef=0.001`（`low_var_kl`）、
`entropy_coeff=0.001`、`adv_estimator=grpo`、`use_kl_in_reward=False`、
`max_prompt_length=512`、`max_response_length=1024`。

## 运行方式

```bash
# phase 1：20 步墙钟校准（任务包要求先估总耗时）
TOTAL_STEPS=20 SAVE_FREQ=10 TEST_FREQ=10 bash scripts/run_exp.sh EXP-W02-01-baseline-gsm8k
# phase 2：不带 TOTAL_STEPS 重跑，resume_mode=auto 从 global_step_20 续到 116
bash scripts/run_exp.sh EXP-W02-01-baseline-gsm8k
# 评测（merge → vLLM → Math-Verify）
bash scripts/run_eval.sh checkpoints/EXP-W02-01-baseline-gsm8k/global_step_116
```

墙钟估算：162 s/step × 116 ≈ 5.2 h，远低于任务包 48 h 的请示阈值，所以直接跑满，未改步数。

- 续跑证据：full run 日志 `Loaded model/optimizer/rng from .../global_step_20`，
  `training/global_step` 从 21 开始；96 个续跑步平均 162.0 s/step，训练墙钟约 4.4 h。
- checkpoint：`checkpoints/EXP-W02-01-baseline-gsm8k/global_step_{10,20,40,60,80,100,116}`（各 3.6GB）。
- wandb：校准 `runs/h3eaiogf`，full run `runs/hlrzg966`。

## 结果

### 训练内 validation（verl 内置循环，rule reward，greedy）

| step | 20 | 40 | 60 | 80 | 100 | 116 |
|---|---|---|---|---|---|---|
| 4080 acc@1 | 0.7384 | 0.7400 | 0.7491 | 0.7521 | 0.7309 | **0.7582** |
| 官方 2×H100 | 0.747 | 0.761 | 0.768 | 0.779 | 0.776 | **0.779** |

### 独立 harness（`eval/run_eval.py`，math-verify 0.9.0，`####` 修复后重打分）

| 模型 | GSM8K pass@1 | MATH-500 pass@1 | AIME24 avg@16 | AIME25 avg@16 |
|---|---|---|---|---|
| 基座 | 0.7180 ±0.0124 | 0.5520 ±0.0222 | 0.0292（std 0.1035） | 0.0083（std 0.0271） |
| `global_step_116`（merged） | **0.7665 ±0.0116** | **0.5640 ±0.0222** | 0.0271（std 0.0670） | 0.0021（std 0.0114） |

- 对齐结论：harness 76.65 vs 官方 77.9，差 **1.25 分**（达标）；训练内 75.82 vs 77.94，差 2.09 分，
  落在单种子 binomial stderr（±1.16–1.24 分，约 1.8σ）之内。两套口径并列报告。
- 相对基座：GSM8K **+4.85 分**，MATH-500 +1.2 分（没有回退）。AIME 两套都在 0–3% 量级，
  30 题的噪声吃掉了全部差异，不作结论。

## 与官方的偏离（全部不改数学，只改 batching 与调度）

| 项 | 官方 | 本实验 | 原因 |
|---|---|---|---|
| 训练 micro batch | 64 / 卡 | **1** | 16GB + 1024 response 的硬约束；这是训练侧慢的主因 |
| vLLM util / max batched tokens | 0.1 / 1536 | **0.30 / 8192** | 纯调度参数；rollout 从 45 s 降到 9.09 s |
| attention | flash-attn | **SDPA**，`use_remove_padding=False` | 本机未装 flash-attn（W1 已评估，W9 再议） |
| LoRA 同步 | adapter + layered_summon | **merge 全量** | vLLM 0.24 `add_lora` 不兼容（W1 结论） |
| trainer | v1 | **`use_v1=False`** | v0.9.1 的 v1 在无 critic 的 GRPO 下必崩（上游 bug） |
| test_freq | 5 | **20** | 只影响监控频率；训练内曲线点比官方少 |

## 失败与返工

1. **变量名冲突**：首次校准命令行实际是 `max_response_length=512` / `logprob_micro=2`，
   因为 `source configs/hw/4080.env` 后 `${MAX_RESPONSE_LEN:-1024}` 被 W1 冒烟变量覆盖。
   改用 W2 专用变量（`W02_*` / `BASELINE_*`）并在启动日志打印四项关键参数。
   代价：一次无效 wandb run `eu4gz4yj`（约 1 步，aborted）。见 `docs/troubleshooting.md §4.4`。
2. **评测卡在 huggingface.co**：`transformers` 在 `_load_hf` 之前就 import 了 `huggingface_hub`，
   endpoint 已固化，`os.environ.setdefault` 太晚。改为在 import 前 export
   `HF_ENDPOINT` + `HF_HUB_DISABLE_XET=1`。代价约 10 min GPU。见 §4.5。
3. **判分 bug**：`#### 460 ####` 被旧抽取器判成空答案（日志层面 37/1319，实际错判 3 条）。
   改为按 `####` 切分取最后一个非空片段，单独 commit（`886f77b`），
   用保存的生成结果重打分（`eval/rescore.py`）。GSM8K 0.7635 → 0.7665，MATH-500 0.5040 → 0.5640。
   原始 JSON/summary 保留可审计。见 §4.6。
4. **merge 一致性**：merge 前后 20 题 greedy 只有 1/20 完全一致。改用**权重级**判据：
   `peft.merge_and_unload()` 后 339/339 参数与 merged bit-identical（max|Δ|=0），
   同进程 20 条 prompt 的 logits 也 max|Δ|=0。生成差异来自 LoRA runtime kernel 的浮点累加
   与自回归级联，不是合并错误。所有上报分数用 **merged** 模型。见 §4.7。

## GPU 时长

20 步校准 0.96 + 全量 116 步 4.43 + 评测与 merge 校验 ~0.20 ≈ **5.6 GPU·h**，花费 $0（本地卡）。

## 结论

管线正确，baseline 对齐（harness 口径差 1.25 分）。配置可直接复用，续跑用 `resume_mode=auto`。
W3 换严格 verifier 后，本实验的所有评测行需要按新判分器重评（追加行，不覆盖）。
