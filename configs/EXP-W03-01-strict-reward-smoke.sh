#!/usr/bin/env bash
# EXP-W03-01-strict-reward-smoke | 20 步冒烟：训练奖励换成严格 verifier
#
# 目的（`weeks/week03_strict_verifier.md` 验收标准 3）：
#   证明训练和评测用的是**同一个**严格 verifier，且接得上 verl 的 reward 链路。
#
# 做法：除奖励函数外，**逐字复用** W2 baseline 配置（`EXP-W02-01-baseline-gsm8k.sh`），
# 这样 step 10/20 的指标可以和 W2 的同 step 直接对照。只通过位置参数追加两个 hydra 覆盖：
#   reward.custom_reward_function.path / .name
#
# 注意（与任务包写法不同，已记入 docs/troubleshooting.md §5）：
#   verl v0.9.1 的键是 `reward.custom_reward_function.*`，不是顶层 `custom_reward_function.*`。
#
# 用法：
#   bash configs/EXP-W03-01-strict-reward-smoke.sh
#   # 只看前 2 步是否有 reward 数值（快速验证接线，不占满 20 步）：
#   TOTAL_STEPS=2 SAVE_FREQ=99 bash configs/EXP-W03-01-strict-reward-smoke.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=${EXP_ID:-EXP-W03-01-strict-reward-smoke}
export TOTAL_STEPS=${TOTAL_STEPS:-20}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TEST_FREQ=${TEST_FREQ:-10}

echo "[$EXP_ID] 20 步冒烟：奖励 = 严格 verifier（src/selfevolve/rewards/strict_reward.py）"
echo "[$EXP_ID] steps=$TOTAL_STEPS save_freq=$SAVE_FREQ test_freq=$TEST_FREQ"

exec bash "$REPO_ROOT/configs/EXP-W02-01-baseline-gsm8k.sh" \
  reward.custom_reward_function.path="$REPO_ROOT/src/selfevolve/rewards/strict_reward.py" \
  reward.custom_reward_function.name=compute_score \
  "$@"
