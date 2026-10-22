# EXP-W03-01-strict-reward-smoke 实验记录

- 日期：2026-10-22
- 目的：W3 验收标准 3 —— 证明**训练奖励与评测共用同一个严格 verifier**，并能接上 verl 的
  reward 链路（`reward.custom_reward_function.*` → `RewardLoopManager` → `NaiveRewardManager`）。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置文件：`configs/EXP-W03-01-strict-reward-smoke.sh`（除奖励函数外**逐字复用** W2 baseline 配置）
- 代码：`src/selfevolve/rewards/strict_reward.py` + `src/selfevolve/verifier/math_verifier.py`
- 模型 / 数据 / seed：Qwen2.5-1.5B-Instruct、GSM8K train、seed=42（与 W2 同）
- 步数：20（save_freq=10、test_freq=10），micro batch=1（与 W2 一致，保证可对照）

## 目的与设计

W2 的训练内验证用的是 verl 内置 GSM8K rule reward（要求 `#### 数字`）。W3 把训练奖励换成严格
verifier 之后，**同一份生成在训练侧和评测侧必须得到同一个分数**。这次冒烟要验证三件事：

1. verl 能不能加载 `src/selfevolve/rewards/strict_reward.py`（importlib 从文件路径加载，不是包 import）；
2. `compute_score` 的返回 dict 能不能被 `NaiveRewardManager` 接受并写进 wandb（`acc`/`reason`/`extracted`）；
3. 训练链路不被判分器拖慢（reward 耗时占单步 < 5%）。

## 结果

（训练完成后回填：单步耗时、reward 均值、val 指标、GPU 时长、wandb run）

## 与 W2 的可比性

- 只有奖励函数不同，其余超参与系统参数（micro=1、util=0.30、max_num_batched_tokens=8192、
  test_freq=10）与 W2 **不一致的地方只有 test_freq（W2 是 20）**，这只影响监控频率。
- 训练内 `critic/score/mean` 与 W2 同 step 的数**不可直接比**：W2 是 verl 内置 rule reward，
  W3 是严格 verifier（多了结论句兜底）。可比的是评测口径（`outputs/eval/*` 的重打分结果）。
