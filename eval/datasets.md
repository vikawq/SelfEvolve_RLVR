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

- `pass@1`：greedy 单次生成，正确率 = 正确样本数 / 总样本数。误差报
  **binomial stderr** `sqrt(p(1-p)/n)`（`summary.json → binomial_stderr`）；
  逐题 0/1 的样本标准差恒在 0.4–0.5，没有信息量，**不进报告**。
- `avg@16`：每题采样 16 次，`accuracy_i = 正确次数/16`；报告 `mean_i(accuracy_i)` 与
  逐题样本标准差（`std`），并给出 `stderr = std/sqrt(30)`。
- 每条结果的误差口径写在 `summary.json → reported_stderr_kind`
  （`binomial` / `per_question_sample`），避免两套误差被混用。
- `truncation_rate`：`finish_reason == "length"` 的比例（诊断是否被 `max_tokens` 截断）。
- `empty_answer_rate`：抽取不到答案的比例。
- 所有指标只对测试集计算；测试集不参与训练、过滤或出题（协议 §4 红线）。

## 5. 与训练内 validation 的差异（报告里必须解释）

训练日志里的 `val-core/openai/gsm8k/acc/mean@1` 是 verl 内置验证循环，同样是 **greedy**：
本项目的所有配置都没有覆盖 `actor_rollout_ref.rollout.val_kwargs`，走 verl 默认值
`do_sample=False, temperature=0, top_p=1.0, top_k=-1, n=1`（verl 已锁定的
commit `1876b06`，`verl/trainer/config/rollout/rollout.yaml`；注意同文件里
**顶层** `rollout.temperature=1.0` 是训练 rollout 的采样温度，不是验证的）。

因此两套口径的差异**只在判分，不在采样**：

| | 训练内 validation | 本 harness |
|---|---|---|
| 采样 | greedy（val_kwargs 默认） | greedy（`temperature=0, top_p=1, n=1`） |
| 生成上限 | `data.max_response_length=1024` | `max_tokens=1024`（GSM8K/MATH） |
| 判分 | verl 内置 GSM8K rule reward：只比对 `####` 后的数值 | Math-Verify + 本项目抽取器 |
| 题数 | 1319 | 1319 |

两套数字**不可直接混用**；结果表以 harness 为准，训练内 validation 只作训练过程监控。

> 留档：W2 初稿曾把训练内 validation 写成 `temperature=1.0`（误取了顶层 rollout 默认值），
> 已按上述证据更正，见 `docs/troubleshooting.md §4.9`。

## 6. 跨数据集可比性的边界（报告里必须声明）

- **`####` 指令对 MATH-500 / AIME 是外加格式**。这两个数据集的标准答案常常不是纯数值
  （表达式、分数、区间），而模型并没有在这种分布上被训练成用 `####` 收尾。所以这里的
  **绝对分数被系统性压低，不能与任何基于 `\boxed{}` 的公开榜单或论文数字比较**；
  只有本 harness 内同一口径下的 base ↔ RL 差值是有效结论。
- 选择同一指令的理由是对称性：基座与 RL 模型面对完全相同的格式要求，不做 per-dataset
  prompt 调优。代价就是上面这条，必须写在报告里而不是留给读者猜。
- `truncation_rate` 必须和分数一起上报。`max_tokens` 1024/2048 下，"没有提升"与"被截断"
  在分数上长得一样，只有截断率能把两者分开。
- `max_model_len` 由 `run_eval.py` 按**每个数据集**的 `最长 prompt + max_tokens` 推算
  （默认 `--max-model-len 0` = auto）。固定值不够时 vLLM **不报错**，只会静默压缩生成预算，
  截断率随之失真；现在预算不足会直接报错。每次运行的预算写入 `config.json.length_budgets`
  与每个数据集的 `summary.json → length_budget.budget_ok`。
