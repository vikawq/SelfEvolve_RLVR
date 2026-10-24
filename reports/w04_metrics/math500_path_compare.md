# MATH-500 评测路径对齐（W4，2026-10-24）

## 背景

L1 的 MATH-500 评测用 `analysis/w04_eval_batched.py`（全量生成 + 每条样本一个进程 + 30 s 硬超时）；
其余 4 个实验与基座用 `eval/run_eval.py`（harness）。为了消掉"评分路径不同"的质疑，
在 bigdog01 上用 **`--rescore-only`** 把其余 4 个实验 + 基座的**同一批生成文本**重新过一遍
batched 路径（不重新生成、不换判分函数、不换模型），逐 checkpoint 对比。

## 结果：分数完全一致，唯一差异是 1 条硬超时

| 模型 / checkpoint | harness MATH-500 | batched rescore-only | Δ | timeouts |
|---|---|---|---|---|
| 基座 `Qwen2.5-0.5B-Instruct` | 0.3320 | 0.3320 | 0.0000 | 0 |
| 对照 step100 | 0.3260 | 0.3260 | 0.0000 | 0 |
| 对照 step200 | 0.3540 | 0.3540 | 0.0000 | 0 |
| L2 step100 | 0.3280 | 0.3280 | 0.0000 | 0 |
| L2 step200 | 0.2880 | 0.2880 | 0.0000 | 0 |
| L3 step100 | 0.0720 | 0.0720 | 0.0000 | 0 |
| L3 step200 | 0.0740 | 0.0740 | 0.0000 | 0 |
| L4 step100 | 0.2200 | 0.2200 | 0.0000 | 0 |
| L4 step200 | 0.0260 | 0.0260 | 0.0000 | 0 |
| L1 step100 | 未完成（生成后评分卡死被 kill） | 0.2880 | — | **1** |
| L1 step200 | 未运行 | 0.3020 | — | 0 |

结论：除 L1 外，所有 MATH-500 分数在两条路径下**完全相同（Δ=0.0000）**；L1 的两行本来就没有
可用的 harness 分数。因此"L1 用 batched、其余用 harness"不会带来口径差；报告里统一描述为
"**MATH-500 全部走 batched 路径**（其余 4 个实验与基座用 `--rescore-only` 复核）"。

## 唯一一条 timeout 样本

| 字段 | 值 |
|---|---|
| run | `EXP-W04-01-l1-format-weight-global_step_100` |
| sample_index | 65 |
| sample_id | `test/prealgebra/1747.json` |
| gold | `1` |
| reason / error | `timeout` / `timeout` |
| num_tokens / finish_reason | 215 / `stop` |
| text_chars | 467 |
| text_sha256 | `a5659d398f87e7c1480f73e7d3adbc89c9fa4c6238fe4013c795530b5967df0d` |
| 题面 | `What is $1^{(2^{235423523})}$?` |
| 模型 boxed | `\boxed{2^{2^{235423523}}}`（把 `1^x` 错写成 `2^{2^…}`） |

说明：这题 gold 是 `1`，模型的 `\boxed{2^{2^{235423523}}}` 明显是错的；sympy 在 `parse` 时
试图把这个 21 字符的"指数塔"求值，进入天文数字的大整数运算，C 层不响应 SIGALRM。
30 s 硬超时判错与人工判断一致。`MAX_PARSE_CHARS=2000` 拦不住这种"短但数值巨大"的表达式。

说明：harness 的卡死是**累积内存**问题（评分到第 63 条后 RSS 114 GB、整机换页，无法定位到单条）；
batched 路径在 500 条里只命中这 1 条 30 s 硬超时。两者既不是同一条样本，也不是同一个失效机制：
累积内存见 `docs/troubleshooting.md §5.15`，单条表达式在 C 层吞掉 SIGALRM 见 §5.16。两者的共同
根源都是 `math_verify` 在长/畸形 LaTeX 上的行为。

完整记录（含 sha256、题面与模型原始响应）：`reports/w04_metrics/math500_timeout_samples.json`（顶层 `generated_at` / `rule` / `count` / `samples`）。
每个 run 的 `scoring_errors` 在各自的 summary 里：9 个 rescore-only run 的 summary 入库在
`reports/w04_metrics/eval_batched_math500/*.summary.json`（**全为 `{}`**）；L1 两个 batched run 的
summary 在 bigdog01 的 `outputs/eval/<run>/summary.json`（不入 git），其中 step100 计入上面这 1 条
timeout、step200 为 0。

## 产物

- `reports/w04_metrics/math500_timeout_samples.json`
- `reports/w04_metrics/eval_batched_math500/*.summary.json`（9 个 rescore-only 的 summary）
- `outputs/eval_batched/`（bigdog01，不入 git）
- 复现命令：`bash outputs/w04/rescore_math500.sh`（脚本在 bigdog01 的 `outputs/w04/`，逻辑见下）

```bash
# 对每个 run：把 harness 的 math500.jsonl 拷到 outputs/eval_batched/<run>/
# 再用同一批文本 rescore（不生成）：
python analysis/w04_eval_batched.py \
  --model models/merged/<run> --datasets math500 --limit 500 \
  --output-dir outputs/eval_batched --run-name <run> \
  --rescore-only --overwrite --score-timeout-s 30
```
