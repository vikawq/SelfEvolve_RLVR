# EXP-W03-01-strict-reward-smoke 实验记录

- 日期：2026-10-22
- 目的：W3 验收标准 3 —— 证明**训练奖励与评测共用同一个严格 verifier**，并能接上 verl 的
  reward 链路（`reward.custom_reward_function.*` → `RewardLoopManager` → `NaiveRewardManager`）。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置文件：`configs/EXP-W03-01-strict-reward-smoke.sh`（除奖励函数外**逐字复用** W2 baseline 配置）
- 代码：`src/selfevolve/rewards/strict_reward.py` + `src/selfevolve/verifier/math_verifier.py`
- 模型 / 数据 / seed：Qwen2.5-1.5B-Instruct、GSM8K train、seed=42（与 W2 同）
- 步数：20（save_freq=10、test_freq=10），micro batch=1（与 W2 一致，保证可对照）
- wandb：`runs/3b5rwrv2`
- 日志：`logs/EXP-W03-01-strict-reward-smoke_20261022-112816.log`（第一次启动失败，见下）

## 目的与设计

W2 的训练内验证用的是 verl 内置 GSM8K rule reward（要求 `#### 数字`）。W3 把训练奖励换成严格
verifier 之后，**同一份生成在训练侧和评测侧必须得到同一个分数**。这次冒烟验证三件事：

1. verl 能不能加载 `src/selfevolve/rewards/strict_reward.py`（importlib 从文件路径加载，不是包 import）；
2. `compute_score` 返回的 dict 能不能被 `NaiveRewardManager` 接受（`score` 用于训练，其余键进 wandb）；
3. 训练链路不被判分器拖慢（reward 耗时占单步 < 5%）。

## 结果（20 步，59:30 墙钟，exit 0）

| 指标 | 值 | 对照 |
|---|---|---|
| 单步耗时 | 末步 **175.8 s**（step1 172.4 s） | W2 baseline 116 步中位数 162.0 s；reward 只占 0.14 s，其余差异是验证步与噪声 |
| 训练 reward（`critic/rewards/mean`） | 0.6406（step1）→ **0.8063**（step20） | — |
| 训练内验证 `val-core/openai/gsm8k/acc/mean@1` | step10 **0.7309** → step20 **0.7422** | 同一判分器（严格 verifier）、greedy |
| harness GSM8K pass@1（step20 merged） | **0.7346 ± 0.0122**（截断 0.53%、空答案 1.21%） | 基座 **0.7149**（同判分器）→ **+1.97 分** |
| reward 耗时（`compute_score`） | step20 **3.4 ms/样本**（max 13 ms） | 320 样本 ÷ 8 workers ≈ **0.14 s/step**，占 0.08% |
| 峰值显存 | 9.28 GiB | 与 W2 相同 |
| 吞吐 | 724 tok/s | W2 697 tok/s（micro=1） |
| checkpoint | `global_step_10`、`global_step_20`（7.1 GB） | — |
| GPU 时长 | 训练 0.99 + 评测 0.03 = **1.02 GPU·h** | — |

结论：**接线正确，链路可训练**。20 步从 0.7149 涨到 0.7346（+1.97），与 W2 的 116 步 +4.85 趋势一致；
只有 20 步，**不作精度结论**。

## 第一次启动失败（保留记录，0 GPU·h）

第一次 `setsid nohup bash scripts/run_exp.sh EXP-W03-01-...` 立刻退出：

```
configs/EXP-W02-01-baseline-gsm8k.sh: 行 75: python: 未找到命令
== exit code: 127 ==
```

原因：conda 在 `~/.bashrc` 里初始化，非交互 ssh 的 shell 不读它，`PATH` 里没有 `python`
（W2 是从交互式 shell 启动的，没暴露）。解法：先
`source /home/angus/miniconda3/etc/profile.d/conda.sh && conda activate selfevolve`。
已记入 `docs/troubleshooting.md §5.4`。

## 与 W2 的可比性

- 只有奖励函数不同；其余超参与系统参数（micro=1、util=0.30、max_num_batched_tokens=8192）与 W2 相同，
  **不一致的只有 `test_freq`（W2 是 20，W3 是 10）**，这只影响监控频率。
- 训练内 `critic/score/mean` 与 W2 同 step 的数**不可直接比**：W2 是 verl 内置 rule reward，
  W3 是严格 verifier（多了结论句兜底）。可比的是评测口径：
  基座 0.7149 与 step20 0.7346 都是同一天、同一个 verifier 通过 `eval/run_eval.py` 出来的。
- 训练内验证（0.7422）与 harness（0.7346）差 0.76 分：前者走 verl 的验证循环、后者走独立 harness，
  两者 prompt 相同但引擎参数/批调度不同（与 W2 的 0.7582 vs 0.7665 是同类差异），不作为结论。
