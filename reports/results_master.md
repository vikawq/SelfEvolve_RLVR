# Results Master

所有评测结果汇总。每行一个模型/checkpoint，**必须**标注评测时的判分器版本与评测口径。
评测代码：`eval/run_eval.py`（W2）；判分器：`math-verify==0.9.0` + 本项目封装（`eval/harness/scoring.py`）。

| 日期 | 实验 ID | 模型 / checkpoint | 数据集 | 样本数 | 采样参数 | 指标定义 | 分数 | 判分器版本 | GPU | wandb |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | EXP-W01-01-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_10` | GSM8K test | 1319 | temperature=1.0, top_p=1.0, n=1 | mean@1（verl 内置验证循环） | 0.5550 | verl v0.9.1 内置 GSM8K rule reward（**非** W3 严格 verifier） | RTX 4080 16GB | [kv34uf1v](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/kv34uf1v) |
| 2026-10-20 | EXP-W01-01-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_20` | GSM8K test | 1319 | temperature=1.0, top_p=1.0, n=1 | mean@1（verl 内置验证循环） | 0.7081 | 同上 | RTX 4080 16GB | [16fmimie](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/16fmimie) |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7180** | math-verify==0.9.0（fix `####` 重复标记后重评） | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5520** | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | AIME 2024 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0292** ± 0.1035 | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | AIME 2025 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0083** ± 0.0271 | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7665** | math-verify==0.9.0（fix `####` 重复标记后重评） | RTX 4080 16GB | [hlrzg966](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/hlrzg966) |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5640** | 同上 | RTX 4080 16GB | 同上 |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | AIME 2024 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0271** ± 0.0670 | 同上 | RTX 4080 16GB | 同上 |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | AIME 2025 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0021** ± 0.0114 | 同上 | RTX 4080 16GB | 同上 |

## 说明

- **W2 baseline 对齐**：官方表格 Qwen2.5-1.5B-Instruct + GRPO-LoRA GSM8K = **77.9**（verl 内置 rule reward）。
  本实验的独立 harness 分数 76.65，差 **1.25 分**；训练内 validation（同官方口径）0.7582，差 2.09 分。
  两套口径并列报告，差异来源见 `reports/week02_summary.md`。
- **判分器修复（重要）**：初版 harness 把 `#### 460 ####` 这类重复结尾标记误判为空答案（日志层面 37/1319，
  实际错判 3 条）；MATH-500 上影响更大（0.504 → 0.564）。修复 commit 单独提交，并用保存的生成结果
  重打分（`eval/rescore.py`），原始 `summary.json` 保留可审计。上表为修复后的分数。
- GSM8K/MATH-500 的 `std` 为 0/1 样本标准差（≈0.42–0.50），更合适的误差是 binomial stderr
  （GSM8K ±0.0117、MATH-500 ±0.0222）；AIME 的 std 是 30 题逐题 accuracy 的样本标准差。
- **merge 一致性**：merged 权重与 base+adapter 逐参数 bit-identical（339/339，max|Δ|=0，missing=0），
  同进程 20 条 prompt 的 logits 也 max|Δ|=0；vLLM runtime adapter 与 merged 的 greedy 生成 1/20 完全一致、
  共同前缀中位数 235 字符、准确率 14/20 vs 15/20，差异来自 LoRA runtime kernel 与自回归级联，不是权重错误。
  详见 `reports/week02_summary.md`。
- `指标定义` 必须写清楚：greedy pass@1 / avg@k / maj@k。
- `判分器版本` 对应 `src/selfevolve/verifier/` 的 git commit 或版本号；W3 之后训练奖励与评测共用同一严格 verifier，
  届时本表需要按新 verifier 重评。
- W1 两行是冒烟结果，判分器是 verl 内置 rule reward，**不是** W2 harness；保留用于审计，不与上表直接比较。
