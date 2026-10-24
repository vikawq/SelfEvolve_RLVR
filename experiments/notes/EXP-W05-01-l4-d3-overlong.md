# EXP-W05-01-l4-d3-overlong 实验记录

- 日期：2026-10-24
- 目的：W5 D3 —— 在 W4 L4（长度奖励）上叠加 overlong reward shaping，验证"压住长度后正确性是否恢复"。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置：`configs/EXP-W05-01-l4-d3-overlong.sh`（`reward=defended_reward.py`，`W05_BASE_HACK=L4`，`W05_DEFENSES=D3`）
- 实现：`src/selfevolve/rewards/defended_reward.py` + `defense_common.py`
- 模型 / 数据 / seed：Qwen2.5-0.5B-Instruct、GSM8K train、seed=42
- 步数：200，32 prompts × n5 = 160 seq/step，response 1024，actor micro=2 / logprob micro=2
- D3 参数：soft=768 token、hard=1024 token、λ=1；`len>=hard` 整条判 0；最终奖励截断到 ≥0
- wandb：`runs/d83gq6tw`
- 训练日志：`logs/EXP-W05-01-l4-d3-overlong_*.log`；GPU·h：2.62（训练 2.49 + 评测 0.12）

## 结果

| 指标 | D3 | 对照（EXP-W04-00） | 未防御 L4（EXP-W04-04） |
|---|---|---|---|
| 训练侧宽松奖励（last20） | 0.6488 | 0.653 | 1.000 |
| 训练侧严格准确率（last20） | **0.649** | 0.653 | 0.505 |
| **reward gap（last20）** | **−0.0003** | 0.000 | 0.495 |
| **hack rate（last50）** | **0.00075**（6/8000） | 0.000 | 0.496 |
| gap / hack onset | 无 | — | 10 / 34 |
| 响应长度（last20） | **922 字符 / 290 token** | 933 字符 / 288 token | 4363 字符 / 997 token |
| 截断率（last20） | 0.13% | 0.1% | 48% |
| 零方差组比例（last20） | 0.541 | 0.525 | 0.291 |
| **GSM8K test 前 500** | 0.538 / **0.552**（step100/200） | 0.530 / 0.556 | 0.404 / **0.000** |
| **MATH-500** | 0.358 / **0.322** | 0.326 / 0.354 | 0.220 / **0.026** |

D3 触发情况（最后 50 步 8000 条）：`linear=47`（0.6%）、`hard_cap=0`、其余 `none`；
`tokenizer_fallback=0`（全部用真 tokenizer）。

## 结论

1. **D3 完全压住了 L4**：hack rate 0.496 → 0.00075、gap 0.495 → −0.0003，长度回到对照组水平，
   GSM8K step200 恢复到 0.552（对照 0.556），MATH-500 0.322（对照 0.354，低 3.2 分 ≈1.1σ，不显著）。
2. **真正起作用的是"删掉长度 bonus"**：overlong 惩罚只在 0.6% 的样本上触发；模型从基座出发、
   没有 bonus 就不会把长度推到 1024。这证明 L4 的暴涨是奖励造成的，不是模型天然啰嗦。
3. **预注册 hack 规则偏保守**：最后 50 步仍有 6 条"长且错"（≥3000 字符）的样本，但它们在 D3 下
   拿 0 分；按"是否真的拿到奖励"口径 hack rate 是 0。
4. **代价**：MATH-500 step200 比对照低 3.2 分（不显著），GSM8K 持平；步时 44.6 s（对照 44.2 s），
   零方差组比例没有变差。没有观察到明显代价。
5. **边界**：本实验从基座开始，D3 的作用是"阻止学会 hack"；"已经 hack 坏的 checkpoint 能否被
   D3 拉回来"需要从 W4-L4 step200 续训，本周未做（见 P7 报告 §6）。

## 评测说明

step100/200 用 `analysis/w04_eval_batched.py`（与 `eval/run_eval.py` 同一判分函数、同 schema，
每条样本一个 spawn 进程 + 30 s 硬超时），W4 已用 `--rescore-only` 验证两条路径逐 checkpoint 一致。
