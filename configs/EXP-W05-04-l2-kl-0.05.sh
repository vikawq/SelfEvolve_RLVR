#!/usr/bin/env bash
# EXP-W05-04-l2-kl-0.05 | L2 子串匹配 + D5 KL 强度 0.05
#
# 同 EXP-W05-03，KL 系数 0.05。预期：hack onset 更晚/更弱，但策略更新被压住、严格准确率代价更大。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W05-04-l2-kl-0.05
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/loose_L2_substring.py"
export KL_COEF=0.05
export ROLLOUT_DIR=${ROLLOUT_DIR:-$REPO_ROOT/outputs/w05/rollouts/$EXP_ID}

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
