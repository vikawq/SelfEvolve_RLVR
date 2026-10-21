# 评测数据集与口径（W2 定版，W3 换 verifier 时重评）

更新时间：2026-10-21。评测代码：`eval/`；判分器：`math-verify`（版本写入每个 `summary.json`）。

## 1. 数据集

| key | 数据集 | 来源 | 样本数 | 指标 | 采样 |
|---|---|---|---|---|---|
| `gsm8k` | GSM8K test | 本地 `data/gsm8k/test.parquet`（与训练同源，只读 test） | 1319 | pass@1 | greedy（temperature=0, top_p=1, n=1） |
| `math500` | MATH-500 | HF `HuggingFaceH4/MATH-500` test | 500 | pass@1 | greedy |
| `aime2024` | AIME 2024 | HF `Maxwell-Jia/AIME_2024` train（30 题） | 30 | avg@16 | temperature=0.6, top_p=0.95, n=16 |
| `aime2025` | AIME 2025 | HF `yentinglin/aime_2025` train（30 题） | 30 | avg@16 | temperature=0.6, top_p=0.95, n=16 |
| `mmlu_pro`（可选） | MMLU-Pro test 随机 1000 | HF `TIGER-Lab/MMLU-Pro`，seed=42 | 1000 | pass@1 | greedy |

- AIME 2024/2025 都是 30 题（两场各 15 题）。方差大，**必须**报 avg@16 与逐题标准差，不能只看均值。
- MMLU-Pro 是可选回归检查；W2 若时间不够可不跑，但代码与口径已固定。
- 数据下载走 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`（W1 排障记录 §1.2/1.2b）。

## 2. Prompt 模板

- **GSM8K**：逐字使用训练 parquet 的 `prompt` 字段。内容为题干 + 指令
  `Let's think step by step and output the final answer after "####".`，
  再套 Qwen2.5 的 chat template（`add_generation_prompt=True`）。
- **MATH-500 / AIME 2024 / AIME 2025**：题干 + 同一句 `####` 指令，再套 chat template。
  这样基座与 RL 模型面对相同的输出格式要求；不针对某个数据集做 prompt 调优。
- **MMLU-Pro**：选择题模板，选项按 A–J 列出，要求最后一行 `Answer: X`。

模板集中在 `eval/harness/data.py`，每个数据集在 `DatasetSpec.prompt_template` 里显式命名
（`gsm8k_training_verbatim` / `####_suffix` / `mc_answer_letter`），写入 `summary.json` 与
`config.json`；改模板必须同步改本文件并重评受影响结果。

## 3. 判分

- 数学：`math_verify.parse` + `math_verify.verify`。gold 会先包成 `\boxed{...}`
  （纯 LaTeX 直接 parse 会丢结构，已实测）；预测优先取最后一个 `####` 之后的内容，
  失败再退回全文抽取。
- 选择题：最后一行 `Answer: X` 优先，否则取最后一个独立选项字母。
- 每条结果保留 `extracted`、`gold_parsed`、`pred_parsed`、`error`，便于人工复核。
- 判分器版本写在 `summary.json` 的 `config.scorer` 与每个样本的 `scorer` 字段。

## 4. 指标定义

- `pass@1`：greedy 单次生成，正确率 = 正确样本数 / 总样本数。
- `avg@16`：每题采样 16 次，`accuracy_i = 正确次数/16`；报告 `mean_i(accuracy_i)` 与
  逐题样本标准差（`std`），并给出 `stderr = std/sqrt(30)`。
- `truncation_rate`：`finish_reason == "length"` 的比例（诊断是否被 `max_tokens` 截断）。
- `empty_answer_rate`：抽取不到答案的比例。
- 所有指标只对测试集计算；测试集不参与训练、过滤或出题（协议 §4 红线）。

## 5. 与训练内 validation 的差异（报告里必须解释）

训练日志里的 `val-core/openai/gsm8k/acc/mean@1` 是 verl 内置验证循环：
用训练时的 rule reward（只比对 `####` 后的数值）、`temperature=1.0`、`n=1`、1319 题。
本 harness 用 Math-Verify、greedy、chat template 与显式 `max_tokens`。
两套数字**不可直接混用**；结果表以 harness 为准，训练内 validation 只作训练过程监控。
