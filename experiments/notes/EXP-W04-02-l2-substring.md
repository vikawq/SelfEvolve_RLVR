# EXP-W04-02-l2-substring 实验记录

- 日期：2026-10-23 至 2026-10-24
- 目的：W4 L2 —— **子串匹配**。`score = 1.0 if 金标字符串出现在响应的任意位置`。
  预期 hack：罗列大量数字/候选答案，金标"碰巧"出现在其中。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置：`configs/EXP-W04-02-l2-substring.sh`（只换 reward）
- 实现：`src/selfevolve/rewards/loose_L2_substring.py`
- 模型 / 数据 / seed：Qwen2.5-0.5B-Instruct、GSM8K train、seed=42
- 步数：200，32 prompts × n5 = 160 seq/step，response 1024，actor micro=2 / logprob micro=2
- wandb：训练 `runs/lof1b6qu`；逐步检测指标 `runs/rct3ey5d`
- 训练日志：`logs/EXP-W04-02-l2-substring_*.log`；链路 `logs/w04_chain_train_rest.out`

## 结果

| 指标 | 值 | 对照（EXP-W04-00） |
|---|---|---|
| 训练侧宽松奖励（最后 20 步） | **0.750** | 0.653（严格奖励） |
| 训练侧严格准确率（最后 20 步） | **0.590** | **0.653** |
| **reward gap（最后 20 步）** | **0.160** | 0.000 |
| gap onset | step 10 | — |
| 每条响应不同数字个数（step200） | 11.7（step48 为 9.4） | 9.6 |
| 响应长度（字符均值，step200） | 1334（step48 为 1068） | 1005 |
| 训练步时 / 峰值显存 | 约 47 s / 10 GiB | 同 |
| **GSM8K test 前 500（严格）** | 基座 0.478 → step100 **0.516** → step200 **0.482** | step200 **0.556** |
| **MATH-500（严格）** | 基座 0.332 → step100 **0.328** → step200 **0.288** | step200 **0.354** |
| GSM8K 截断率 | step100 0.6% → step200 **4.4%** | 0.8% |
| MATH-500 截断率 | step100 16.4% → step200 **25.6%** | 11.0% |

## 结论：L2 是本周最强的一种复现

1. **reward 在涨、真实准确率在跌**：训练侧宽松奖励从 0.56 涨到 **0.75**，同一批 rollout 的
   严格准确率只有 **0.590**（对照组 0.653）；外部评测 GSM8K 从 step100 0.516 **跌到**
   step200 0.482（比对照组低 **7.4 分**，stderr ≈2.2），MATH-500 从 0.328 跌到 0.288
   （比对照组低 6.6 分）。**这是本周唯一一个"真准确率显著低于对照组"的漏洞。**
2. **机制**：金标是 `5000`（分），模型正确地算出 5000 分、又换算成 50 美元并 `\boxed{50}`；
   严格 verifier 判错，子串匹配在推理里看到 `5000` 就判对。`reports/hack_examples/
   EXP-W04-02-l2-substring.md` 的 3 条样例全是这种"最终答案错、金标数字在推理里"。
3. **训练确实在教它利用漏洞**：n_numbers 9.4→11.7、长度 1068→1334、截断率 0.6%→4.4%，
   模型学会了"多列数字/写长"来撞子串；代价是最终答案的格式与单位越来越不可靠。
4. **与 L1 的区别**：L1 的 gap 是常数偏移（能力没变），L2 的 gap 伴随**真实能力下降**。
   这更像任务包设想的那种 reward hacking。
5. **注意**：基座模型本来就常在推理里写对数字，所以 L2 在第 1 步就有 gap ≈0.18；
   "扩大"主要发生在评测侧（step100→200 真准确率继续跌），训练侧 gap 稳定在 0.14–0.18。

## 评测说明

step100/200 用 `eval/run_eval.py`（标准 harness，严格 verifier）。L2 的 MATH-500 没有触发
L1 那种 sympy 缓存膨胀（输出更短、更简单），500 条正常跑完（每个 checkpoint 约 55 s）。

**补测（2026-10-24）**：用 `analysis/w04_eval_batched.py --rescore-only` 在同一批生成文本上
复核 MATH-500，step100/200 与 harness 完全一致（0.3280/0.2880，Δ=0.0000，0 timeout），
见 `reports/w04_metrics/math500_path_compare.md`。
