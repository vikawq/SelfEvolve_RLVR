# Week 03 计划：严格 Verifier

周期：2026-10-22 起。硬件：RTX 4080 16GB（bigdog01，单卡）。GPU 预算：≤ 1.5 GPU·h（20 步冒烟 + 零星测速），**$0**（本地机器，不租云、不调付费 API）。

## 0. 开工前的勘察（已完成，结论先行）

| 项 | 事实 | 影响 |
|---|---|---|
| 开发机 | 本机是 Windows，Python 3.14 / pytest 9.1.1，**无 GPU、无 torch/verl/math-verify** | 代码与单测在本机写；本地用 `uv` 建 `.venv`（Python 3.12 + `math-verify==0.9.0`）跑单测，与 bigdog01 的 0.9.0 对齐 |
| 训练机 | `bigdog01`（ssh `home`，RTX 4080 16GB）可达；`/home/angus/workspace/SelfEvolve_RLVR` 工作树 clean，落后本地 3 个 commit | 产物在 bigdog01：`outputs/eval/{Qwen2.5-1.5B-Instruct, EXP-W02-01-...-global_step_116}`，含 4 个数据集逐样本 JSONL 与 `summary_rescored.json` |
| 同步方式 | 本地 → bigdog01 用 `git archive` 打 tar + scp（工作树 line ending 混杂，不用 rsync 直传） | 长任务仍用 `setsid nohup`，日志落 `logs/` |
| 已安装版本 | verl **v0.9.1**（`1876b06`）、math-verify **0.9.0**、transformers 5.9.0、vLLM 0.24.0 | 任务包写的接口以源码为准（见 §4 差异表） |

W2 遗留的两个小尾巴本周顺手清掉（都零 GPU）：
1. AIME 两行的长度预算复核：`python eval/run_eval.py --plan-only`。
2. `results_master.md` 的 `截断率` / `空答案率` 两列回填（数据在 bigdog01 的 `summary_rescored.json`）。

## 1. 计划：7 个工作包

### W3-1 设计定稿 + 差异记录
- 依据：`weeks/week03_strict_verifier.md`、W2 的 `eval/harness/scoring.py`、verl v0.9.1 源码。
- 产出：本文件 §4 的"文档 vs 实际"差异表（后续并入 `docs/troubleshooting.md §5`）。

### W3-2 实现 `src/selfevolve/verifier/math_verifier.py`
- `VerifyResult(score, extracted, parse_ok, reason)`，`reason ∈ {correct, wrong, no_answer, multiple_answers, truncated, timeout}`。
- 抽取优先级：最后一个 `\boxed{...}`（嵌套大括号计数）→ `####` 后最后一个非空片段 → 结尾 "The answer is …" / "答案是 …"。
- 防投机：多候选/互斥 boxed → `multiple_answers`；只看最终答案；截断 → `truncated`。
- 归一化：千分位逗号、`$`、`\%`、`\left/\right`、`\dfrac→\frac`、`\text{}`、空白、科学计数法。
- 等价判断：归一化后用 Math-Verify `parse`/`verify`；解析失败时退化为严格字符串相等（覆盖 `\text{Evelyn}` 这类文本答案）。
- 超时：**不依赖** Math-Verify 自带的 signal 超时（见风险 R2），自己实现（POSIX 主线程 `SIGALRM` / 其余情况进程池 + `future.result(timeout)`）。
- 确定性：同输入同输出；所有随机性为零；进程池失败也只影响"超时"分支。

### W3-3 单元测试 `tests/test_math_verifier.py`（≥ 80 用例）
三段：等价形式（分数/小数/百分数/科学计数法/单位/千分位/负数/各种 LaTeX）、对抗样本（多答案、推理里出现正确数字、空 boxed、嵌套 boxed、互斥 boxed、截断、超长垃圾、自称 verified correct）、边界（空串、纯空白、非 UTF-8 字节、超长 100k 字符）。
验收：`pytest -q` 全绿（本机 .venv 与 bigdog01 各跑一次）。

### W3-4 三判分器对比 + 人工核对
- 从 bigdog01 的 W2 输出随机抽 300 条（seed 固定），三份打分：
  (a) verl 默认 GSM8K `compute_score`（strict 模式，`verl/utils/reward_score/gsm8k.py`）；
  (b) 原样 Math-Verify（复刻 W2 `scoring.score`）；
  (c) 严格 verifier。
- 输出：`reports/verifier_validation.md` + `analysis/verifier_disagreements.csv`（逐样本可审）+ 人工核对包（100 条，含题面摘要/金标/输出尾部/三个判分器的判定/agent 初判）。
- 准确率口径：以人工核对为金标准；agent 初判单独标注，**未人工确认前不写成定稿数字**。

### W3-5 接入训练与评测
- `src/selfevolve/rewards/strict_reward.py`：`compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`，返回 dict（`score` + 诊断字段），verl 的 `NaiveRewardManager` 已支持 dict。
- 配置：`reward.custom_reward_function.path=.../strict_reward.py`、`reward.custom_reward_function.name=compute_score`（任务包写的顶层 `custom_reward_function.*` 在 v0.9.1 是 `reward.custom_reward_function.*`，见 §4）。
- 评测：`eval/harness/scoring.py` 改为同一 verifier 的薄封装；`eval/rescore.py` 走同一路径。
- 测速：单样本 mean / p50 / p99；与单步 162 s 对比，占比 > 5% 才考虑并行（verl 自带 `reward.num_workers=8`，先实测再说）。

### W3-6 回归（零 GPU）
- `eval/rescore.py` 用严格 verifier 重打分 W2 baseline（greedy 部分零 GPU；AIME avg@16 同样是重打分）。
- `results_master.md` **追加新行**（保留 math-verify 0.9.0 的旧行）；`registry.csv` 新增/更新行，注明 verifier commit。

### W3-7 20 步冒烟 + 周报
- `EXP-W03-01-strict-reward-smoke`：在 W2 baseline 配置上只换 reward（其余逐字不变，micro=1 保持可比），20 步，SAVE_FREQ=10、TEST_FREQ=10。
- 观察：reward 均值/分布、`correct` 率与 W2 同 step 的差异、单步耗时、`reward_extra_info` 是否落进 wandb。
- 周报 `reports/week03_summary.md` 按协议模板。

## 2. 验收标准对照

| 验收项 | 计划证据 | 状态 |
|---|---|---|
| `pytest -q` 全绿，≥80 用例 | W3-3，两个环境各跑一次 | 待做 |
| `verifier_validation.md` 人工核对集准确率 ≥98% | W3-4，人工核对包 + 逐样本 CSV | 待做（**需人核对**，见 R1） |
| 训练与评测共用同一 verifier，20 步冒烟 | W3-5/W3-7 | 待做 |
| 周报完成 | W3-7 | 待做 |

## 3. 风险清单

| # | 风险 | 概率/影响 | 缓解 / 触发动作 |
|---|---|---|---|
| R1 | **验收要求"人工核对 ≥100 条"，我无法冒充人**。若人没核对就写"准确率 ≥98%"，等于伪造 | 高 / 高（诚信红线） | 产出可审核对包（CSV + Markdown 摘要）；报告里把我的初判标为"agent 初判"，人工确认前**不写定稿准确率**；在周报"需要人决策"里点名 |
| R2 | Math-Verify 的超时实现：POSIX 用 `signal.alarm`（非主线程直接 `ValueError`），Windows 用 `multiprocessing`（每次调用起进程）。训练侧 reward manager 跑在线程/异步里 | 高 / 高 | 永远以 `parse(..., parsing_timeout=None)` / `verify(..., timeout_seconds=None)` 调用，自己实现超时；单测覆盖"线程里调用"这一路径 |
| R3 | Math-Verify 解析缺陷（已实测）：裸 gold `460` 解析为空（必须 `\boxed{}`）；`\dfrac` 不识别；`3.5\times10^{3}` 只解析出 `10`；`3 or 5` 只取 `5`；`(3,5)` 被当成开区间；`\%` 当单位丢掉 | 高 / 中 | 归一化层兜住 dfrac/科学计数法/百分号；多答案/互斥 boxed 自己判；文本答案走严格字符串相等；逐条写进报告与 troubleshooting |
| R4 | 严格判分会让 W2 分数变化（可能下降），历史行必须保留 | 中 / 中 | 追加行不覆盖；`verifier_validation.md` 给出逐类差异来源；AIME 30 题噪声大，不作结论 |
| R5 | 训练奖励切换后接口不符（legacy trainer 走 `RewardLoopManager` + `naive` manager，8 workers） | 中 / 高 | 先看源码确认签名与 dict 返回值；冒烟前用 20 条真实样本直接调 `compute_score` 对拍；冒烟开 `trainer.val_before_train=False` 省时间 |
| R6 | 20 步冒烟 ~1 GPU·h（162 s/step）；根分区只剩 18G，checkpoint 3.6GB/个 | 中 / 低 | 冒烟只留 2 个 ckpt 后清理；跑前 `df -h` |
| R7 | 测试集污染红线：抽样的 300 条来自 test split | 低 / 高 | 抽样只用于**判分器审计**，不回流训练；W3-7 冒烟仍只用 `train.parquet`；报告里写明 |
| R8 | 严格 verifier 变慢（科学计数法/长表达式） | 低 / 中 | 实测 mean/p99；与 162 s/step 比，>5% 才上并行；已有 `reward.num_workers=8` |

## 4. 文档 vs 已安装版本（先记差异，实现时按实际）

| 文档写法（任务包） | 实际（verl v0.9.1 `1876b06` / math-verify 0.9.0） | 处理 |
|---|---|---|
| `custom_reward_function.path/name` | `reward.custom_reward_function.path/name`（`verl/trainer/config/reward/reward.yaml`）；顶层同名键只在 `legacy_reward_impl.yaml` | 用 `reward.custom_reward_function.*`，并保留顶层写法在注释里说明 |
| `compute_score(data_source, solution_str, ground_truth, extra_info=None)` | 实际以**关键字**调用（`naive.py:67`），且支持 `**extra_reward_kwargs` | 签名写成 `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`，位置/关键字都兼容 |
| reward 返回 float | `NaiveRewardManager` 支持 dict：`score = result["score"]`，其余键进 `reward_extra_info` | 返回 dict（含 reason/extracted），便于审计 |
| "看看有没有自带并行 reward manager" | 有：`reward.num_workers: 8`，`RewardLoopManager` 起 8 个 worker 分片 | 先用默认，不需自建 |
| 超时示例"进程池 + timeout" | math-verify 0.9 `parse(parsing_timeout=)` 自身用 signal/multiprocessing | 关掉它自带超时，外部统一实现 |
| `\boxed{}` 抽取 | math-verify 的 `LatexExtractionConfig` 也能抽，但会"任意匹配"（多答案时取任一） | 自己抽，抽到唯一候选再交给 verify |

## 5. 决策门

- **花钱**：本周 $0（本地 GPU），无需请示。
- **改评测代码**：W2 周报 §"人工决策门"第 3 条已批准，附 4 条执行要求（单独 commit / registry 记 hash / 追加不覆盖 / 逐样本分歧报告 / 用 rescore 重打分）。本计划逐条落实，不再重复请示。
- **改验收标准**：不做。若发现"人工核对"无法在无人参与下达标，按 R1 报告并停下。
- **升级依赖**：不做（math-verify 固定 0.9.0）。
- **结果与预期严重不符**：按协议 §6 先报告。

## 6. 交付物清单

- `src/selfevolve/verifier/math_verifier.py`、`src/selfevolve/rewards/strict_reward.py`
- `tests/test_math_verifier.py`、`tests/test_strict_reward.py`
- `reports/verifier_validation.md`、`analysis/verifier_disagreements.csv`、人工核对包
- 更新的 `eval/harness/scoring.py`、`eval/rescore.py`、`reports/results_master.md`、`experiments/registry.csv`
- `reports/week03_summary.md`、`docs/troubleshooting.md §5`
