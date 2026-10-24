# EXP-W04-01-l1-format-weight 实验记录

- 日期：2026-10-23 至 2026-10-24
- 目的：W4 L1 —— **格式权重过高**。`score = 0.5×[有 \boxed{}] + 0.5×[宽松抽取的答案对]`。
  预期 hack：永远输出 boxed，但答案质量不提升。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置：`configs/EXP-W04-01-l1-format-weight.sh`（只换 reward；其余与 `EXP-W04-00` 逐字一致）
- 实现：`src/selfevolve/rewards/loose_L1_format.py`
- 模型 / 数据 / seed：Qwen2.5-0.5B-Instruct、GSM8K train、seed=42
- 步数：200，32 prompts × n5 = 160 seq/step，response 1024，actor micro=2 / logprob micro=2
- wandb：训练 `runs/7kfhv9ng`；逐步检测指标 `runs/0ug4axwh`
- 训练日志：`logs/EXP-W04-01-l1-format-weight_*.log`；链路 `logs/w04_chain_rest.out`

## 训练结果（200 步，exit 0）

| 指标 | 值 | 对照（EXP-W04-00） |
|---|---|---|
| 训练侧宽松奖励（最后 20 步） | **0.828** | 0.653（严格奖励） |
| 训练侧严格准确率（最后 20 步） | **0.655** | **0.653** |
| **reward gap（最后 20 步）** | **0.173** | 0.000 |
| gap 首次持续 ≥0.10 的步数（onset） | **step 10** | — |
| boxed 率 | **1.000**（step 1 为 0.24） | 0.981（step 200） |
| 每条响应不同数字个数 | 10.1 | 9.6 |
| 响应长度（字符均值） | 1032 | 1005 |
| 重复 5-gram 占比 | 0.141 | 0.089 |
| 峰值显存 / 步时 | 10.0 GiB / 约 45 s | 同 |

## 结论（重要，别过度解读）

1. **L1 确实复现了"格式换奖励"**：一个答错的响应，只要写了 `\boxed{16}`（金标 12），
   宽松奖励就有 **0.500**；同样的响应严格判定是错。`reports/hack_examples/EXP-W04-01-l1-format-weight.md`
   里的 3 条样例都是"宽松判分器自己也认为答错、但拿到 0.5"的样本。
2. **但真实能力没有因此下降**：最后 20 步严格准确率 0.655 vs 对照组 0.653（差 0.2 分，
   远小于 ±3.9 的逐步噪声）。这个漏洞是**奖励信号膨胀**，不是"模型学会了错误行为"。
3. **boxed 率不是 L1 独有的**：对照组在严格奖励下也会从 0.24 涨到 **0.981**——base 模型在
   GSM8K 上本来就会收敛到 boxed 格式（严格 verifier 也优先抽 boxed）。L1 只是把这条本来就会
   发生的格式变化额外奖励了 0.5。
4. **副作用（值得 W5 注意）**：L1 的格式奖励让模型在结尾同时写 `\boxed{}` 和畸形的 `####`，
   还出现 `\[ ... }` 这种括号写坏的情况；严格 verifier 按"宁严勿松"判 `truncated`/`wrong`。
   也就是说，宽松奖励会把**格式可靠性**换成**格式存在性**，严格指标会因此变差一点，
   但这条在 200 步、0.5B 上不足以把严格准确率拉下来。
5. 因此 L1 的定性是：**specification gaming（代理奖励被满足）成立，capability degradation 不成立**。

## 评测

step100/200 用 `analysis/w04_eval_batched.py`（与 `eval/run_eval.py` **同一个**判分函数；
每条样本一个 spawn 进程 + 30 s 硬超时，见 `docs/troubleshooting.md §5.15`）。

| 数据集 | step100 | step200 | 对照 step200 |
|---|---|---|---|
| GSM8K test 前 500 | 0.5260 ±0.0223 | **0.5680 ±0.0222** | 0.5560 ±0.0222 |
| MATH-500 | 0.2880 ±0.0203 | 0.3020 ±0.0205 | 0.3540 ±0.0214 |

- L1 的 GSM8K 比对照高 1.2 分（噪声内），MATH-500 比对照低 5.2 分——**不是干净的能力提升或下降**，
  与"L1 是奖励膨胀、不是能力变化"的结论一致。
- 评测侧的 reward gap 没有重算（`--skip-eval-loose`）：L1 的 MATH-500 长 LaTeX 答案会让
  `math_verify` 在离线复算时卡死。训练侧 gap 不受影响。
- 评测用批量评测后，同一 checkpoint 的 GSM8K 与之前 harness 的 0.518 差 0.8 分（4 题）：
  这是 greedy 生成在不同批调度下的微小非确定性，不是判分差异；两个数都在 ±2.2 的 stderr 内。
