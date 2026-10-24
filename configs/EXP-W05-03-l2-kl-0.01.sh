#!/usr/bin/env bash
# EXP-W05-03-l2-kl-0.01 | L2 子串匹配 + D5 KL 强度 0.01
#
# 奖励与 EXP-W04-02 完全相同（loose_L2_substring.py），只把 actor 的 KL loss 系数从 0.001 提到 0.01。
# 目的：观察更强的 KL 约束是否推迟 hack onset、压缩 reward gap。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W05-03-l2-kl-0.01
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/loose_L2_substring.py"
export KL_COEF=0.01
export ROLLOUT_DIR=${ROLLOUT_DIR:-$REPO_ROOT/outputs/w05/rollouts/$EXP_ID}

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
