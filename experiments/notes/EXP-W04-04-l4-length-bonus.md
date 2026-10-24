# EXP-W04-04-l4-length-bonus 实验记录

- 日期：2026-10-24
- 目的：W4 L4 —— **长度奖励**。`score = [严格正确 1/0] + 0.5×min(len(response)/3000 字符, 1)`。
  预期 hack：响应长度暴涨、重复、触顶被截断。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置：`configs/EXP-W04-04-l4-length-bonus.sh`（只换 reward）
- 实现：`src/selfevolve/rewards/loose_L4_length.py`
- 模型 / 数据 / seed：Qwen2.5-0.5B-Instruct、GSM8K train、seed=42
- 步数：200，32 prompts × n5 = 160 seq/step，response 1024，actor micro=2 / logprob micro=2
- wandb：训练 `runs/iev00mes`；逐步检测指标 `runs/<待补>`
- 训练日志：`logs/EXP-W04-04-l4-length-bonus_*.log`；链路 `logs/w04_chain_train_rest.out`

## 结果

| 指标 | 值 | 对照（EXP-W04-00） |
|---|---|---|
| 训练侧宽松奖励（最后 20 步） | **1.000**（长度 bonus 满格 0.5 + 严格正确） | 0.653 |
| 训练侧严格准确率（最后 20 步） | 0.505 | 0.653 |
| **reward gap（最后 20 步）** | **0.495** | 0.000 |
| gap onset | step 10 | — |
| 响应长度（字符均值，step200） | **4601**（step38 为 1930；step95 为 3337） | 1005 |
| 重复 5-gram 占比 | 0.100 | 0.089 |
| 单步耗时（后期） | 约 60 s | 45 s |
| **GSM8K test 前 500（严格）** | 基座 0.478 → step100 **0.404** → step200 **0.000** | step200 **0.556** |
| **MATH-500（严格）** | 基座 0.332 → step100 **0.220** → step200 **0.026** | step200 **0.354** |
| GSM8K 截断率 | step100 **23.4%** → step200 **100.0%** | 0.8% |
| MATH-500 截断率 | step100 **55.6%** → step200 **94.8%** | 11.0% |

## 结论：本周最直观的"奖励上升、真实能力归零"

1. **长度暴涨是训练学出来的**：step38 长度 1930 字符、step95 3337、step200 **4601**；
   宽松奖励（长度 bonus 封顶 0.5 + 正确性）从 0.57 涨到 **1.00**；同一批 rollout 的严格
   准确率只有 0.505（对照组 0.653）。**gap 从 0.18 持续扩大到 0.495**，是四种里唯一
   "gap 单调扩大"的。
2. **真实评测直接归零**：step200 的 GSM8K test 前 500 **0.000**、MATH-500 **0.026**；
   截断率 100% / 94.8%——模型每条响应都写满 1024 token，`finish_reason=length`，
   严格 verifier 按任务包规则判错。这不是"模型不会做"，是"它被长度奖励推着永远写不完"。
3. **机制**：L4 的奖励是**稠密**的（每条都有长度 bonus），所以不像 L3 那样奖励全 0；
   但它奖励的是"长度"这个与正确性无关的代理量。200 步后最优策略就是写满上限，
   正确性部分（0/1）被完全淹没。
4. **样例**：`reports/hack_examples/EXP-W04-04-l4-length-bonus.md` 里 3 条都是"答错、
   但写了 3300–4400 字符"的响应，宽松奖励 **0.500**（只有长度 bonus），严格判错。
5. **对 W5 的提示**：D3（超长惩罚）应该能直接压住这个漏洞，但要同时看"压住长度后
   正确性是否恢复"——L4 的模型可能已经不会写短答案了。

## 评测说明

step100/200 用 `eval/run_eval.py`（标准 harness，严格 verifier）。L4 的输出虽然长，
但都是**重复/展开的普通文本**，MATH-500 评测 65 s/checkpoint 正常完成，未触发
L1 那种 sympy 缓存膨胀（`docs/troubleshooting.md §5.15`）。
