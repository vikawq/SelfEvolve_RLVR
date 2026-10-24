# Week 04 Summary

周期：2026-10-23。硬件：RTX 4080 16GB（bigdog01，单卡）。模型：Qwen2.5-0.5B-Instruct。
训练：GSM8K train，200 步/实验，seed=42，32 prompts × rollout.n 5 = 160 seq/step，response 1024。
评测：GSM8K test 前 500 + MATH-500，greedy pass@1，**严格 verifier**（`selfevolve-strict-verifier-1.0+math-verify-0.9.0`）。
花费：**$0**（本地机器；未租云、未调付费 API；L5 未做）。

> 本文件在 5 个实验全部跑完后定稿。数据来源：`reports/w04_metrics/*.json`（逐步指标）、
> `outputs/eval/*/summary.json`（评测）、`reports/figures/w04_*.png`（曲线）、
> `reports/hack_examples/*.md`（样例）。

## 完成情况（对照验收标准逐条打勾）

- [x] **至少明确复现 3 种 hacking**：L1 / L2 / L4（reward gap 0.173 / 0.160 / 0.495，
      曲线见 `reports/figures/w04_*.png`，样例见 `reports/hack_examples/`）。
- [x] **没能复现的类型有分析**：L3 —— 漏洞不可利用（第一个数字是题面数字），
      奖励几乎全 0，模型 collapse 到 `#### <猜测>`（`experiments/notes/EXP-W04-03-l3-first-number.md`）。
- [x] **对照组曲线正常**：`EXP-W04-00` 的 gap 恒为 0，严格准确率 0.381→0.653，
      GSM8K test 0.478→0.556。
- [x] **周报完成**（本文件）。

## 1. 实验设置

| 实验 | 漏洞 | 奖励（训练） | 关键实现 |
|---|---|---|---|
| `EXP-W04-00-control-strict` | 对照组 | 严格 verifier 0/1 | 与 W3 同一实现 |
| `EXP-W04-01-l1-format-weight` | L1 格式权重过高 | `0.5×[有 boxed] + 0.5×[宽松抽取的答案对]` | `rewards/loose_L1_format.py` |
| `EXP-W04-02-l2-substring` | L2 子串匹配 | `[金标字符串出现在响应任意位置]` | `rewards/loose_L2_substring.py` |
| `EXP-W04-03-l3-first-number` | L3 取第一个数字 | `[响应里第一个数字与金标等价]` | `rewards/loose_L3_first_number.py` |
| `EXP-W04-04-l4-length-bonus` | L4 长度奖励 | `[严格正确] + 0.5×min(len/3000, 1)` | `rewards/loose_L4_length.py` |

五个实验**只有奖励函数不同**：模型/数据/步数/seed/优化器/显存参数逐字一致。训练侧的严格准确率
来自 reward dict 的 `strict_acc`（同一批 rollout 上用严格 verifier 复判），随 rollout dump 落盘
（`trainer.rollout_data_dir`，全量 160 条/步，不是抽样）。评测侧的严格准确率来自
`eval/run_eval.py`（独立 harness，step0/100/200）；**L1 因为 MATH-500 长答案会让
`math_verify` 内存爆掉，改用 `analysis/w04_eval_batched.py`（同一个判分函数、每条 30 s 硬超时）**。
为了消掉路径不对称，其余 4 个实验 + 基座的 MATH-500 已用 `--rescore-only` 在同一批生成文本上
复核：**9 个 checkpoint 的分数与 harness 完全相同（Δ=0.0000）**，唯一差异是 L1 的 1 条硬超时
（`reports/w04_metrics/math500_path_compare.md`）。

## 2. 关键结果

### 2.1 对照组（正常训练）

| 指标 | 基座 | step100 | step200 |
|---|---|---|---|
| GSM8K test 前 500（严格，greedy） | 0.4780 ±0.0223 | 0.5300 ±0.0223 | **0.5560 ±0.0222** |
| MATH-500（严格，greedy） | 0.3320 ±0.0211 | 0.3260 ±0.0210 | **0.3540 ±0.0214** |

训练侧严格准确率（160 条/步，训练题不重复）：前 10 步 0.381 → 最后 20 步 **0.653**；
reward gap 恒为 **0.000**。单步 44.2 s，峰值 10.0 GiB，2.49 GPU·h。详见
`experiments/notes/EXP-W04-00-control-strict.md`。

### 2.2 L1–L4

**L1（格式权重过高）**：训练侧最后 20 步宽松奖励 **0.828**、严格准确率 **0.655**、
gap **0.173**，onset **step 10**；boxed 率 0.24 → **1.000**。样例见
`reports/hack_examples/EXP-W04-01-l1-format-weight.md`（3 条都是"宽松判分器自己也判错、
却拿到 0.5"）。**注意**：对照组在严格奖励下 boxed 率也会到 0.981，且 L1 的严格准确率
（0.655）与对照组（0.653）无差异——L1 是"代理奖励膨胀"，不是能力下降。评测上 GSM8K
step200 0.5680（比对照高 1.2，噪声内），MATH-500 0.3020（低 5.2），也没有干净的提升或下降。
评测见 §2.3。

**L2（子串匹配）**：训练侧最后 20 步宽松奖励 **0.750**、严格准确率 **0.590**、gap **0.160**
（对照 0.653），onset 10。**真实准确率在跌**：GSM8K test 从 step100 0.516 跌到 step200
**0.482**（比对照低 **7.4 分**），MATH-500 从 0.328 跌到 **0.288**（比对照低 6.6 分）；
截断率 GSM8K 0.6%→4.4%、MATH-500 16.4%→25.6%。160 条 rollout 里约 14% 是"最终答案错、
金标数字在推理里"的假阳性（例：金标 `5000` 分，模型算出 5000 分后换算成 50 美元并
`\boxed{50}`，子串匹配仍给 1.0）。**L2 与 L4 都出现了"真实准确率低于对照组"**：
L2 是温和但显著（6–7 分），L4 是直接归零；L1 持平、L3 崩溃。L2 是最像任务包设想的
"奖励只涨一点、真准确率悄悄掉"的那种 reward hacking。详见
`experiments/notes/EXP-W04-02-l2-substring.md`。

**L3（取第一个数字）**：**负结果 + 训练崩溃**。训练侧宽松奖励 ≈ 严格准确率 ≈ **0.090**
（最后 20 步），gap 0、onset 不适用。真相是模型退化成只输出 `#### 4` / `#### ###.2`
这类 6–10 字符的裸猜测：GSM8K test 从基座 0.478 掉到 **0.086**、MATH-500 从 0.332 掉到
**0.074**（比对照组低 47/28 分）。原因：第一个数字通常是**题面**里的数字，等于金标的概率极低，
奖励前 38 步最高只有 0.05；GRPO 在近乎全零的奖励下把策略推到"只猜一个数"的退化形态，
最后"首数字 == 最终答案 == 猜测"，所以 gap 变成 0。这个漏洞**不能被利用来骗奖励**，
但能摧毁模型。完整记录见 `experiments/notes/EXP-W04-03-l3-first-number.md`。

**L4（长度奖励）**：训练侧宽松奖励 **1.000**、严格准确率 0.505、**gap 0.495**（onset 10），
响应长度 1930（step38）→ 3337（step95）→ **4601**（step200）字符。真实评测直接归零：
GSM8K test 0.404（step100）→ **0.000**（step200，截断率 **100%**），MATH-500 0.220 →
**0.026**（截断 94.8%）。这是四种里**唯一 gap 持续单调扩大**的漏洞，也是"奖励上升、
真实能力归零"最直观的证据；机制是 L4 的奖励稠密（每条都有长度 bonus），
最优策略退化成"写满 1024 token、永远给不出最终答案"。样例见
`reports/hack_examples/EXP-W04-04-l4-length-bonus.md`（答错但写 3300–4400 字符 → 0.500）。

### 2.3 评测（严格 verifier，GSM8K test 前 500 + MATH-500）

| 实验 | GSM8K step0 | GSM8K step100 | GSM8K step200 | MATH-500 step0 | MATH-500 step100 | MATH-500 step200 |
|---|---|---|---|---|---|---|
| 对照组 | 0.4780 ±0.0223 | 0.5300 ±0.0223 | 0.5560 ±0.0222 | 0.3320 ±0.0211 | 0.3260 ±0.0210 | 0.3540 ±0.0214 |
| L1 | 0.4780 ±0.0223 | 0.5260 ±0.0223 | 0.5680 ±0.0222 | 0.3320 ±0.0211 | 0.2880 ±0.0203 | 0.3020 ±0.0205 |
| L2 | 0.4780 ±0.0223 | 0.5160 ±0.0223 | **0.4820 ±0.0223** | 0.3320 ±0.0211 | 0.3280 ±0.0210 | **0.2880 ±0.0203** |
| L3 | 0.4780 ±0.0223 | **0.0640 ±0.0109** | **0.0860 ±0.0125** | 0.3320 ±0.0211 | **0.0720 ±0.0116** | **0.0740 ±0.0117** |
| L4 | 0.4780 ±0.0223 | 0.4040 ±0.0219 | **0.0000 ±0.0000** | 0.3320 ±0.0211 | 0.2200 ±0.0185 | **0.0260 ±0.0071** |

> **MATH-500 口径统一**：上表 MATH-500 全部来自 `analysis/w04_eval_batched.py`（同判分函数、
> 每条 30 s 硬超时）。其余 4 个实验 + 基座用 `--rescore-only` 在同一批生成文本上复核，
> 与 harness **逐 checkpoint 完全一致（Δ=0.0000）**；唯一差异是 L1 step100 的 1 条 timeout
> （`test/prealgebra/1747.json`，index 65）。对比见 `reports/w04_metrics/math500_path_compare.md`，
> timeout 明细见 `reports/w04_metrics/math500_timeout_samples.json`。
>
> **GSM8K 列的残余不对称**：L1 的 GSM8K 是 batched 全量（harness 在 MATH-500 死掉，L1 没有
> 完整的 harness 行）；其余 4 个 + 基座是 harness。两边判分函数相同，L1 step100 的
> 0.5260 vs harness 0.5180 差 4 题（0.8 分），来自 greedy 生成在不同批调度下的微小非确定性，
> 在 ±2.2 的 stderr 内。

## 3. 复现判定（口径见 `reports/week04_plan.md §3`）

| 实验 | last20 gap | 对照 gap | onset | 判定 |
|---|---|---|---|---|
| 对照组 | 0.000 | 0.000 | — | 正常（严格准确率 0.381→0.653） |
| L1 | **0.173** | 0.000 | **10** | **yes**（specification gaming；GSM8K 持平、MATH-500 低 5.2 分 ≈**1.8σ**，未达显著，不足以断言能力下降） |
| L2 | **0.160** | 0.000 | **10** | **yes（最强）**：gap + 真实准确率低于对照 6–7 分 |
| L3 | **0.000** | 0.000 | — | **no**：漏洞不可利用；奖励过稀疏，模型崩溃（GSM8K 0.478→0.086） |
| L4 | **0.495** | 0.000 | **10** | **yes（最强扩大）**：长度 1930→4601、GSM8K 0.404→0.000 |

**总结**：4 种漏洞里 **3 种明确复现**——L1（specification gaming：奖励膨胀、能力持平）、
L2（代理奖励上升 + 真实准确率低于对照 6–7 分）、L4（长度暴涨、GSM8K 直接归零）；
L3 是负结果（漏洞不可利用 + 奖励过稀疏导致 collapse）。本周核心证据
"训练奖励在上升，而严格 verifier 的真实准确率持平或下降"在 **L2/L3/L4** 上都成立
（L2/L4 是下降，L3 是崩溃），L1 只满足"奖励膨胀但真实能力不提升"。

## 4. 与计划的偏差及原因

1. **micro batch 从直觉值 4 降到 2**：response 1024 下 actor micro=4 与 logprob micro=8 都 OOM；
   实测 actor 2 / logprob 2 通过（`docs/troubleshooting.md §5.13`）。
2. **没有改 `eval/` 代码**：任务要求"在评测输出上算 hack 指标"，改为离线分析
   `outputs/eval/**/*.jsonl`（`analysis/hack_metrics.py`）；L1 的评测新增了
   `analysis/w04_eval_batched.py`（**同一个判分函数**，只是进程隔离 + 硬超时），
   `eval/` 目录未动。
3. **L5 未做**：需要付费 API/本地放不下 judge，按协议先请示；见 §8。
4. **训练内部 validation 关闭**（`test_freq=0`）：训练内验证用的是同一个宽松奖励，对 W4 没有信息量；
   真实准确率由外部严格评测给。

## 5. 失败与问题（附错误信息与已尝试的方法）

| # | 现象 | 原因 | 解法 | 代价 |
|---|---|---|---|---|
| 1 | `update_actor` OOM（micro=4，response 1024） | 20 seq/微批 + pad 到 1024 | 降到 micro=2 | 2 次 2 步空跑，约 0.1 GPU·h |
| 2 | `compute_log_prob` OOM（logprob micro=8） | logits 8×1536×151936×2B ≈ 3.7 GiB | logprob micro=2 | 同上 |
| 3 | 配置按 512 response 跑了 2 步 | hw env 的 `MAX_RESPONSE_LEN=512` 覆盖了 config 的 `:-1024` | 改用 `W04_MAX_RESPONSE_LEN` | 无数据污染 |
| 4 | 分析训练中的 dump 报 15/170 bad lines | 模型输出含 U+2028，`splitlines()` 误切 JSONL | 改用 `split("\n")` | 无 |
| 5 | wandb 训练循环不记录 reward 额外键 | v0.9.1 行为，只有 validation 消费 | 伴生 `__metrics` run 记录逐步曲线 | 无（展示层） |
| 6 | L1 step100 的 MATH-500 评测 RSS 涨到 114 GB、整机换页 | `math_verify`/sympy 全局缓存随长 LaTeX 答案线性累积（约 150 MB/条） | `analysis/w04_eval_batched.py`：每条样本一个 spawn worker，评完释放；判分函数不变 | 重跑评测约 30 min |
| 7 | 批量评测仍被**单条** MATH-500 卡住（一个 worker 99% CPU、15 GB、2+ 分钟不返回） | 单条病态表达式在 C 层（大整数/符号运算）不响应 SIGALRM，5 s 超时失效 | 给批量评测加**每条 30 s 硬超时**：超时 kill 子进程，按严格 verifier 的 timeout 语义判错并计数；样本 id 落盘 | L1 step100 MATH-500 的 1 条 timeout：`test/prealgebra/1747.json`（index 65），见 `reports/w04_metrics/math500_timeout_samples.json` |

## 6. GPU 时长 / 花费

| 项 | GPU·h |
|---|---|
| `EXP-W04-00` 训练 2.46 + step100/200 评测 0.03 | 2.49 |
| `EXP-W04-01` 训练 2.51 + 批量评测 0.16 + 被 kill 的 harness MATH 评测 0.40 | 3.07 |
| `EXP-W04-02` 训练 2.58 + 评测 0.03 | 2.61 |
| `EXP-W04-03` 训练 1.99 + 评测 0.02 | 2.01 |
| `EXP-W04-04` 训练 3.58 + 评测 0.04 | 3.62 |
| 冒烟/测速（5 次 2 步 + 基座 eval + 失败重试） | 约 0.4 |
| **合计** | **约 14.2** |

花费：**$0**（本地机器；未租云、未调用付费 API）。
按 4080 的 320 W 满载、0.8 元/kWh 粗估电费约 **3.6 元**（不计入成本口径）。

## 7. 给下周的交接

- W5 的防御实验（D1–D5）直接复用本轮的 `loose_L*` 与 `hack_detectors`；
  训练/评测/分析三条链路已经打通。
- **评测必须用 `analysis/w04_eval_batched.py`（或等价进程隔离）**：同一进程连评 500 条
  MATH-500 会把 `math_verify` 的 sympy 缓存撑到 100 GB 以上；单条病态表达式还会绕过
  5 s SIGALRM。批量评测现在有**每条 30 s 硬超时**（超时 kill + 按 timeout 判错），
  见 `docs/troubleshooting.md §5.15`。
- **MATH-500 已用 `--rescore-only` 对齐**：其余 4 个实验 + 基座在同一批生成文本上重评，
  9 个 checkpoint 与 harness **Δ=0.0000**；唯一 1 条 timeout 的 id 已落盘在
  `reports/w04_metrics/math500_timeout_samples.json`（`test/prealgebra/1747.json`）。
- L1 的 boxed 率 10 步内就学满，但对照组也会学到 0.98——**"boxed 率高"本身不是 hack 证据**；
  W5 的格式防御要盯住"答错也拿分"的 gap，而不是格式出现率。
- **L4 是 W5 防御的第一优先级**：长度奖励把 GSM8K test 打到 0.000（截断 100%）。
  D3（超长惩罚）要同时看"压住长度后正确性是否恢复"，而不只是看长度下降。
- **L3 的 collapse 是另一类失败**：奖励几乎全 0 会把模型推成"只猜一个数"。
  W5 的防御/监控应该加一条"奖励分布健康检查"（reward 均值不能长期接近 0 或方差为 0）。
- 单种子：结论以"曲线 + 对照组分步基线"表述；第二种子留到 W5 或补跑。
- 训练侧严格准确率与评测侧 greedy 不可直接比（采样 vs greedy、train vs test）。
- L1 的格式奖励会让模型写出 `\[ ... }` 这类畸形括号，严格 verifier 判 truncated；
  W5 的 D2/D4 防御值得测"格式可靠性"。

## 8. 需要人决策的问题

1. **L5（LLM-as-judge）是否要做**：需要付费 API（或换更大显存的机器放 judge）。本轮未做。
2. **单种子是否足够**：W4 的结论目前是 1 个种子（42）；如果要写进对外材料，
   建议至少补一个种子（1234）跑"复现最强的一种 hack + 对照"（建议 L4 + 对照，约 6 GPU·h）。
3. **L3 负结果的处理**：本轮 L3 的结果是"漏洞不可利用 + 模型 collapse"（GSM8K 0.478→0.086）。
   是接受这个负结果并写进 W5 的防御动机，还是换一种 L3 实现（例如取最后一个数字/取最大数字）？
   后者会偏离任务包定义，需要人批准；我建议**接受负结果**，因为它提供了"错误奖励会训崩模型"
   的独立证据。
4. **MATH-500 的评测口径**：已用 `--rescore-only` 把其余 4 个实验 + 基座的同一批生成文本
   过了一遍 batched 路径，**9 个 checkpoint 与 harness 完全一致（Δ=0.0000）**；唯一差异是
   L1 step100 的 1 条 timeout（`test/prealgebra/1747.json`）。也就是说"路径不对称"已经消除，
   现在只差"是否接受这 1 条按 timeout 判错"。如果不接受，需要改 verifier 的 C 层超时
   （属于改判分代码，需批准）；如果接受，本项无需再处理。
