# Results Master

所有评测结果汇总。每行一个模型/checkpoint，**必须**标注评测时的判分器版本与评测口径。
评测代码：`eval/run_eval.py`（W2）；判分器：`math-verify==0.9.0` + 本项目封装（`eval/harness/scoring.py`）。
新增行用 `python analysis/emit_results_row.py --summary <run>/summary.json ...` 生成，不要手抄数字。

| 日期 | 实验 ID | 模型 / checkpoint | 数据集 | 样本数 | 采样参数 | 指标定义 | 分数 | stderr | std | 截断率 | 空答案率 | 判分器版本 | GPU | wandb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | EXP-W01-01-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_10` | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1（verl `val_kwargs` 默认 `do_sample=False`） | mean@1（verl 内置验证循环） | 0.5550 | ±0.0137 | n/a | — | — | verl v0.9.1 内置 GSM8K rule reward（**非** W3 严格 verifier） | RTX 4080 16GB | [kv34uf1v](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/kv34uf1v) |
| 2026-10-20 | EXP-W01-01-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_20` | GSM8K test | 1319 | 同上 | mean@1（verl 内置验证循环） | 0.7081 | ±0.0125 | n/a | — | — | 同上 | RTX 4080 16GB | [16fmimie](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/16fmimie) |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7180** | ±0.0124 | n/a | 待回填 | 待回填 | math-verify==0.9.0（fix `####` 重复标记后重评） | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5520** | ±0.0222 | n/a | 待回填 | 待回填 | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | AIME 2024 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0292** | ±0.0189 | 0.1035 | 待回填 | 待回填 | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | AIME 2025 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0083** | ±0.0049 | 0.0271 | 待回填 | 待回填 | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7665** | ±0.0116 | n/a | 待回填 | 待回填 | math-verify==0.9.0（fix `####` 重复标记后重评） | RTX 4080 16GB | [hlrzg966](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/hlrzg966) |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5640** | ±0.0222 | n/a | 待回填 | 待回填 | 同上 | RTX 4080 16GB | 同上 |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | AIME 2024 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0271** | ±0.0122 | 0.0670 | 待回填 | 待回填 | 同上 | RTX 4080 16GB | 同上 |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | AIME 2025 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0021** | ±0.0021 | 0.0114 | 待回填 | 待回填 | 同上 | RTX 4080 16GB | 同上 |

> **AIME 两行待复核**：上表的 AIME 结果是在**旧的固定 `max_model_len=2560`** 下跑的，而 AIME 的
> `max_tokens=2048`，只给 prompt 留了 512 token。若有题干超过 512 token，vLLM 会静默压缩生成预算，
> 截断率与分数都会偏低。复核不需要 GPU：
>
> ```bash
> python eval/run_eval.py --model models/Qwen2.5-1.5B-Instruct --datasets aime2024,aime2025 --plan-only
> ```
>
> 输出的 `need` 若 ≤2560 则旧结果有效、原样保留；若 >2560，这两个数据集要用 auto 预算重跑
> （30 题 × 16 次 × 2 个模型，分钟级）。GSM8K/MATH-500 的 `max_tokens=1024`，prompt 预算有
> 1536 token，不受影响。
>
> **待回填的两列**：`截断率` / `空答案率` 已经在 W2 评测的 `summary.json` 里（字段
> `truncation_rate` / `empty_answer_rate`），但那批产物在 bigdog01 的 `outputs/`（不入 git），
> 本机无法读取，**不猜数字**。在 bigdog01 上执行即可回填，无需重跑 GPU：
>
> ```bash
> for d in outputs/eval/*/; do
>   python analysis/emit_results_row.py --summary "$d/summary_rescored.json" \
>     --date 2026-10-21 --exp-id EXP-W02-01-baseline-gsm8k --model '<照抄本表模型列>'
> done
> ```

## 说明

- **误差口径**：`pass@1` 行报 **binomial stderr** `sqrt(p(1-p)/n)`，`std` 列记 `n/a`——逐题 0/1 的
  样本标准差恒在 0.42–0.50，没有信息量。`avg@16` 行的 `std` 是 30 题逐题 accuracy 的样本标准差，
  `stderr = std/sqrt(30)`。两种口径在 `summary.json → reported_stderr_kind` 里标明，不可混用。
- **W2 baseline 对齐**：官方表格 Qwen2.5-1.5B-Instruct + GRPO-LoRA GSM8K = **77.9**（verl 内置 rule reward）。
  本实验的独立 harness 分数 76.65，差 **1.25 分**（达标）；训练内 validation（同官方口径）0.7582，差 2.09 分。
  后者**在单种子噪声内**：1319 题的 binomial stderr 为 ±1.16–1.24 个百分点，2.09 分约 1.8σ，
  且两次运行的 micro-batch / attention kernel / 调度参数都不同。没有第二种子，所以不下"显著不同"的结论；
  以 harness 为主口径，两套并列报告。差异来源见 `reports/week02_summary.md`。
- **采样口径更正**：W1 两行原记为 `temperature=1.0`，那是 verl **顶层** `rollout.temperature` 的默认值
  （训练 rollout 用），不是验证循环。本项目所有配置都未覆盖 `rollout.val_kwargs`，验证走默认的
  `do_sample=False, temperature=0`，即 **greedy**。证据与更正过程见 `docs/troubleshooting.md §4.9`。
- **判分器修复（重要）**：初版 harness 把 `#### 460 ####` 这类重复结尾标记误判为空答案（日志层面 37/1319，
  实际错判 3 条）；MATH-500 上影响更大（0.504 → 0.564）。修复 commit 单独提交，并用保存的生成结果
  重打分（`eval/rescore.py`），原始 `summary.json` 保留可审计。上表为修复后的分数。
- **MATH-500 / AIME 的绝对分数不可外比**：这两个数据集复用了 GSM8K 的 `####` 指令（口径对称，
  不做 per-dataset prompt 调优），但它们的标准答案常常不是纯数值，模型也没在这种分布上被训成用
  `####` 收尾。因此绝对分数被系统性压低，**不能与基于 `\boxed{}` 的公开榜单/论文数字比较**；
  只有本表内同一口径下的 base ↔ RL 差值是有效结论。见 `eval/datasets.md §6`。
- **merge 一致性**：merged 权重与 base+adapter 逐参数 bit-identical（339/339，max|Δ|=0，missing=0），
  同进程 20 条 prompt 的 logits 也 max|Δ|=0；vLLM runtime adapter 与 merged 的 greedy 生成 1/20 完全一致、
  共同前缀中位数 235 字符、准确率 14/20 vs 15/20，差异来自 LoRA runtime kernel 与自回归级联，不是权重错误。
  详见 `reports/week02_summary.md`。
- `指标定义` 必须写清楚：greedy pass@1 / avg@k / maj@k。
- `判分器版本` 对应 `src/selfevolve/verifier/` 的 git commit 或版本号；W3 之后训练奖励与评测共用同一严格 verifier，
  届时本表需要按新 verifier 重评：**追加新行、不覆盖旧行**，并保留 Math-Verify 口径的行用于对比审计。
- W1 两行是冒烟结果，判分器是 verl 内置 rule reward，**不是** W2 harness；保留用于审计，不与上表直接比较。
