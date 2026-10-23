# Week 04 Summary

周期：2026-10-23。硬件：RTX 4080 16GB（bigdog01，单卡）。模型：Qwen2.5-0.5B-Instruct。
训练：GSM8K train，200 步/实验，seed=42，32 prompts × rollout.n 5 = 160 seq/step，response 1024。
评测：GSM8K test 前 500 + MATH-500，greedy pass@1，**严格 verifier**（`selfevolve-strict-verifier-1.0+math-verify-0.9.0`）。
花费：**$0**（本地机器；未租云、未调付费 API；L5 未做）。

> 本文件在 5 个实验全部跑完后定稿。数据来源：`reports/w04_metrics/*.json`（逐步指标）、
> `outputs/eval/*/summary.json`（评测）、`reports/figures/w04_*.png`（曲线）、
> `reports/hack_examples/*.md`（样例）。

## 完成情况（对照验收标准逐条打勾）

- [ ] **至少明确复现 3 种 hacking**：reward gap 显著扩大 + 曲线 + 样例。结果见 §3。
- [ ] **没能复现的类型有分析**：是步数不够、模型太小，还是漏洞无法被利用。
- [ ] **对照组曲线正常**：`EXP-W04-00` 的 gap 恒为 0，严格准确率随训练上升。
- [ ] **周报完成**（本文件）。

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
`eval/run_eval.py`（独立 harness，step0/100/200）。

## 2. 关键结果

### 2.1 对照组（正常训练）

| 指标 | 基座 | step100 | step200 |
|---|---|---|---|
| GSM8K test 前 500（严格，greedy） | 0.4780 ±0.0223 | 0.5300 ±0.0223 | **0.5560 ±0.0222** |
| MATH-500（严格，greedy） | 0.3320 ±0.0211 | 0.3260 ±0.0210 | **0.3540 ±0.0214** |

训练侧严格准确率（160 条/步，训练题不重复）：前 10 步 0.381 → 最后 20 步 **0.653**；
reward gap 恒为 **0.000**。单步 44.2 s，峰值 10.0 GiB，2.49 GPU·h。详见
`experiments/notes/EXP-W04-00-control-strict.md`。

### 2.2 L1–L4（待填）

见 `reports/w04_metrics/` 与 `analysis/w04_summary.py` 生成的表。

## 3. 复现判定（口径见 `reports/week04_plan.md §3`）

待填。

## 4. 与计划的偏差及原因

1. **micro batch 从直觉值 4 降到 2**：response 1024 下 actor micro=4 与 logprob micro=8 都 OOM；
   实测 actor 2 / logprob 2 通过（`docs/troubleshooting.md §5.13`）。
2. **没有改评测代码**：任务要求"在评测输出上算 hack 指标"，改为离线分析
   `outputs/eval/**/*.jsonl`（`analysis/hack_metrics.py`），`eval/` 未动。
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
| 5 | wandb 训练循环不记录 reward 额外键 | v0.9.1 行为，只有 validation 消费 | 训练后回填同一 run | 无（展示层） |

## 6. GPU 时长 / 花费

待填（5 个实验的和）。

## 7. 给下周的交接

- W5 的防御实验（D1–D5）直接复用本轮的 `loose_L*` 与 `hack_detectors`；
  训练/评测/分析三条链路已经打通。
- 单种子：结论以"曲线 + 对照组分步基线"表述；第二种子留到 W5 或补跑。
- L1 的 boxed 率 ~1 步就学会（0.24→1.0 在 10 步内），说明 0.5B 对该格式漏洞极敏感；
  W5 的格式防御（D2 答案唯一性）应优先测。
- 训练侧严格准确率与评测侧 greedy 不可直接比（采样 vs greedy、train vs test）。

## 8. 需要人决策的问题

1. **L5（LLM-as-judge）是否要做**：需要付费 API（或换更大显存的机器放 judge）。本轮未做。
2. **单种子是否足够**：W4 的结论目前是 1 个种子（42）；如果要写进对外材料，
   建议至少补一个种子（1234）跑"复现最强的一种 hack + 对照"，约 5 GPU·h。
3. **L3 若为负结果**：是否接受"该漏洞不可利用"作为结论，还是换一种 L3 实现
   （例如取最后一个数字/取最大数字）——后者会偏离任务包定义，需要人批准。
