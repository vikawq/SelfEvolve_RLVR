# Week 03 Summary

周期：2026-10-22。硬件：RTX 4080 16GB（bigdog01，单卡）。GPU 时长 **1.02 GPU·h**，花费 **$0**。
实验：`EXP-W03-01-strict-reward-smoke`（20 步冒烟）。代码：`src/selfevolve/verifier/`、`src/selfevolve/rewards/`、
`eval/harness/scoring.py`（判分实现整体切换）、`analysis/verifier_compare.py`、`analysis/verifier_accuracy.py`、
`analysis/verifier_bench.py`。

## 完成情况（对照验收标准逐条打勾）

- [x] **`pytest -q` 全部通过，用例 ≥ 80 个**：`tests/test_math_verifier.py` **98 例** +
      `test_strict_reward.py` **13 例** + `test_verifier_compare.py` **9 例**（对拍 installed verl）。
      **本机：166 passed / 3 skipped**（Windows 无 `SIGALRM`、无 verl，相关用例自动 skip）；
      **bigdog01：177 passed, 0 skipped**。
- [x] **`verifier_validation.md` 显示严格 verifier 准确率 ≥ 98%**：人工核对集（110 条三判分器分歧）上
      **108/110 = 98.18%**（agent 初判，**待人工复核**）；420 抽样全体口径 99.52%。
      W2 口径 92.73% / 98.10%，verl 内置判分 1.94% / 74.29%。
- [x] **训练和评测都切到同一个 verifier，并完成 20 步冒烟**：`EXP-W03-01`，20 步 exit 0，
      训练内验证 0.7309（step10）→ 0.7422（step20）；step20 merged 后 harness GSM8K **0.7346**
      （基座 0.7149，+1.97，同一判分器）。
- [x] **周报完成**（本文件）。

## 关键结果

### 1. 判分器

单一实现 `src/selfevolve/verifier/math_verifier.py`，训练奖励（`strict_reward.py`）与评测
（`eval/harness/scoring.py`）都只调它。抽取优先级：`\boxed{}` → `####` → 收尾句 → 结论句；
多候选 / 截断 / 无答案一律判错；单次判定 5 s 超时（POSIX 主线程 SIGALRM，其它情况进程池）。

| 指标 | 值 |
|---|---|
| 单元测试 | 98 例（等价形式 / 对抗样本 / 边界 / 归一化 / 契约与确定性），另有 13 例奖励与 9 例 verl 对拍 |
| 单样本耗时（主线程，500 条真实样本） | mean **0.54 ms**、p50 0.04 ms、p99 10.4 ms |
| 单样本耗时（线程+进程池，训练路径） | mean **1.05 ms**、p99 25.5 ms |
| 训练实测（step20） | `compute_score` **3.4 ms/样本**、320 样本 ÷ 8 workers ≈ 0.14 s/step（**0.08%**） |

### 2. 三判分器对比（420 条抽样，seed=42，300 GSM8K + 120 MATH-500）

| 判分器 | 人工核对集（110 条分歧） | 420 全体（估计） |
|---|---|---|
| **strict（W3）** | **108/110 = 98.18%** | 99.52% |
| mathverify（W2） | 102/110 = 92.73% | 98.10% |
| verl 内置 GSM8K | 2/103 = 1.94% | 74.29% |

- 110 条分歧里 **100 条是 verl 漏判**（基座模型不写 `####`，verl 直接判 0），不是 strict 的问题。
- 已知 2 条 strict 判错（`#### 15` 后跟 `-15` 取了最后一行；结论句里两个值时放弃抽取），
  已写进验证报告 §6 的已知限制与 `analysis/verifier_accuracy.py::MANUAL`。

### 3. 回归：W2 结果在新判分器下（零 GPU 重打分）

| 模型 | 数据集 | W2 口径 | 严格口径 | Δ |
|---|---|---|---|---|
| 基座 | GSM8K | 0.7180 | 0.7149 | −0.31 |
| 基座 | MATH-500 | 0.5520 | 0.5400 | −1.20 |
| RL ckpt116 | GSM8K | 0.7665 | 0.7642 | −0.23 |
| RL ckpt116 | MATH-500 | 0.5640 | **0.4980** | **−6.60** |

**结论反转（必须写进后续所有材料）**：MATH-500 的 base→RL 差值从 **+1.20** 变成 **−4.20**
（差值 stderr ≈ ±3.16，**不显著**）。原因是 ckpt 在域外 MATH 上有 36/500 条"最终答案写错、
正确数字只在推理里"（例：推理得 `-50`、最终写 `#### 50`），W2 的整段匹配把它们判对了。
正确表述是"**严格判分下没有观察到 MATH-500 的提升**"；GSM8K 的 +4.85 → **+4.93** 是稳的。

### 4. 20 步冒烟（`EXP-W03-01`，wandb `3b5rwrv2`）

| 指标 | 值 |
|---|---|
| 训练 reward | 0.6406（step1）→ 0.8063（step20） |
| 训练内验证 acc@1 | 0.7309（step10）→ **0.7422**（step20） |
| harness GSM8K（step20 merged） | **0.7346 ± 0.0122**（基座 0.7149 → **+1.97**） |
| 单步 | 末步 175.8 s（step1 172.4 s；W2 116 步中位数 162.0 s）。差值里 reward 只占 0.14 s，其余是验证步与噪声 |
| 峰值显存 / 吞吐 | 9.28 GiB / 724 tok/s |

## 与计划的偏差及原因

1. **W1 设计文档的 verifier 契约有两处不一致，已在 `docs/modules.md §4.1` 更正并记录**：
   实际签名没有 `data_source`（选择题判定留在适配层），多了 `finished`（截断来自 `finish_reason`）。
2. **抽取多了一级"结论句"**（任务包只列了 boxed/`####`/收尾句）。不加这一级，基座模型 45% 的 GSM8K
   输出抽不到答案，base GSM8K 会从 0.72 掉到 0.38——那是"格式不合规"被误报成"能力不行"。
   加完之后基座 0.7149（W2 口径 0.7180），量化见 `reports/verifier_validation.md §7`。
3. **`%` 按单位处理**（`50\%` == `50`，不是 `0.5`）。数据集 gold 都是裸数字（W2 数据里 2.9%/4.2% 的
   生成含 `%`，对应 gold 全是数字）；改成 ÷100 会引入系统性假阴性。已写进验证报告 §6。
4. **AIME 两行的长度预算复核（W2 遗留）**：`--plan-only` 显示 AIME2024 需要 2496（ok）、
   **AIME2025 需要 2880 > 历史固定值 2560**，即 AIME2025 历史结果被静默压缩过生成预算。
   30 题 × 16 次的差异在噪声内，两行**只作参考、不作结论**；重跑与 W4 一起做（追加行、不覆盖）。
5. **`results_master.md` 的 `截断率`/`空答案率` 两列已回填**（W2 遗留），并注明 math-verify 口径下的
   "空答案率 0.00%" 是抽取器总会返回字符串导致的，不是真的都抽到了答案。

## 失败与问题（附错误信息与已尝试的方法）

| # | 现象 | 原因 | 解法 | 代价/影响 |
|---|---|---|---|---|
| 1 | 第一次启动冒烟立刻退出：`line 75: python: 未找到命令`，`exit 127` | 非交互 ssh 的 shell 不读 `~/.bashrc`，conda 没激活，`PATH` 无 `python`（W2 从交互式 shell 启动，没暴露） | 启动前 `source conda.sh && conda activate selfevolve` | 0 GPU·h（import 前就退了），记 `troubleshooting §5.4` |
| 2 | Math-Verify 在训练侧报不了超时：`ValueError: ... doesn't support threaded environment` | 它的超时用 `signal.alarm`，只对主线程有效；verl 的 reward 跑在 `run_in_executor` 的线程里 | 关掉上游超时（`parsing_timeout=None`），自己实现（SIGALRM / 进程池） | — |
| 3 | 用 PowerShell `Add-Content` 写 `registry.csv` 后文件不再是 UTF-8 | Windows PowerShell 默认用系统 ANSI（GBK）写文件 | 用 venv 的 python `write_text(encoding="utf-8")` 重写，单独 commit | 一条坏行，无数据损失 |
| 4 | 第 1 步 `compute_score` 29 ms（离线测 1.05 ms） | 8 个 reward worker 首次调用才 spawn 进程池 + import math-verify，开销摊到 320 个样本 | 无（第 20 步回落到 3.4 ms） | 记录在 `troubleshooting §5.6`，避免误判成"判分变慢" |
| 5 | 判分器第一版把 base GSM8K 打到 0.373 | 只认 marker，基座模型 45% 的输出不写 `####` | 加"结论句"兜底（有 marker 时不参与），并把"最终答案是否正确"作为人工核对口径 | 多花约 1.5 h 调优，换来口径可解释 |
| 6 | `_VALUE_TOKEN_RE` 漏掉句末数字（`$1520.`） | 正则 lookahead `(?![\w.])` 把句尾句号当成小数点的一部分 | 改成 `(?![\w]\|\.\d)` | base GSM8K 0.667 → 0.697 |
| 7 | `analysis/emit_results_row.py` 回填采样参数时输出 `temperature=?` | 脚本按 `python analysis/...` 运行时 `sys.path` 里没有仓库根，`import eval.harness.data` 失败被 `except` 吞掉 | 在函数里插入 repo root 到 `sys.path` | — |

## GPU 时长 / 花费

| 项 | GPU·h |
|---|---|
| `EXP-W03-01` 20 步冒烟（59:30） | 0.99 |
| step20 checkpoint 评测（merge + GSM8K 1319 题） | 0.03 |
| 三判分器对比 / 重打分 / 单测 / 耗时基准（纯 CPU） | 0 |
| **合计** | **1.02** |

花费：**$0**（本地机器；未租云、未调用付费 API）。

## 给下周的交接

- **判分器就是唯一标准**：`src/selfevolve/verifier/math_verifier.py`。W4 的宽松判分器（`loose_L*`）
  请**复用**它的抽取与归一化（`extract_answer` / `normalize_answer` / `is_equivalent`），
  只放宽"最后一步"，否则宽松/严格两组会同时变两个变量，无法归因。
- **W4 的现成素材**：`reports/verifier_review_pack.csv` 里的 110 条分歧就是"宽松判分会怎么被骗"的候选集；
  `reports/verifier_validation.md §3` 已按类型给出实例（推理里有正确数字、`L-8` vs `8` 的假阳性、截断）。
- **train/eval 同源已生效**：改判分逻辑后，`results_master.md` 必须追加新行、不覆盖旧行，
  并跑 `eval/rescore.py --suffix <新后缀>` 重打分（零 GPU）。
- **训练奖励接线模板**：`configs/EXP-W03-01-strict-reward-smoke.sh`（只加两个 hydra 覆盖）。
  注意 verl v0.9.1 的键是 `reward.custom_reward_function.*`。
- **未完成**：① AIME 两行要用 auto 预算重跑（需要 GPU，分钟级）；② 人工核对 110 条（见下）；
  ③ 3090 仍未到位，两卡 profiling 未做。
- **风险提示**：MATH-500 的 base→RL 结论已因判分口径反转（−4.2，不显著）。W4/W5 的任何"提升"
  都必须声明判分器版本；涉及 base↔RL 的差值不能用 W2 的 math-verify 口径。

## 需要人决策的问题

1. **人工核对 110 条**（验收标准 2 要求"交给人核对"）。核对包：`reports/verifier_review_pack.md`
   （或 `.csv`，Excel 可直接开）。核对口径已在文件头写明：判断"模型最终答案对不对"，不是"判分器说得对不对"。
   目前报告里的 98.18% 是 **agent 初判**；请复核，改完 CSV 的 `human_verdict` 列后跑
   `python analysis/verifier_accuracy.py` 即可刷新准确率。**未复核前不能对外宣称"人工核对通过"**。
2. **确认 MATH-500 结论反转的处理方式**：严格判分下 base 0.5400 vs RL 0.4980（−4.2，1.3σ）。
   建议在 W2 周报的口径上补一句更正（W2 已归档，我没有改历史周报），
   并在 W6 自我进化的评测里把"MATH-500 提升"从结论里拿掉、等更多种子再判断。是否照此办理？
3. **AIME 重跑是否现在做**：`AIME2025` 历史结果的生成预算被压缩过（need 2880 > 2560）。
   重跑约 10 GPU·min、$0，但 AIME 基数只有 30 题，对结论无影响。建议与 W4 一起做，如果本周要
   "全部历史结果都在严格口径下"就现在补。

## 遗留物清单

- `outputs/eval/*/summary_strict.json` + `*.strict.jsonl`（不入 git，在 bigdog01）。
- `reports/verifier_review_pack.{md,csv,jsonl}`、`reports/verifier_accuracy.json`、`reports/verifier_bench.json`（已入库）。
- `checkpoints/EXP-W03-01-strict-reward-smoke/global_step_{10,20}`（7.1 GB，不入 git）。
- `models/merged/EXP-W03-01-step20`（merge 后的 HF 目录，不入 git）。
