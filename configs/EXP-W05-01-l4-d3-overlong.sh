#!/usr/bin/env bash
# EXP-W05-01-l4-d3-overlong | L4 长度奖励 + D3 overlong shaping
#
# 底层的宽松判分器仍是 W4 的 L4（loose_L4_length.py），但 D3 会用 overlong 惩罚**替换**长度 bonus：
#   len <= 768 token        : score = 严格正确 0/1
#   768 < len < 1024 token  : score = max(0, 正确 - (len-768)/(1024-768))
#   len >= 1024 token       : score = 0（按任务文档；DAPO 原文是 -1，差异见 defense_common.py）
#
# 对照：EXP-W04-04-l4-length-bonus（同模型/数据/步数/seed，只差 D3）。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W05-01-l4-d3-overlong
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/defended_reward.py"
export W05_BASE_HACK=L4
export W05_DEFENSES=D3
export W05_D3_SOFT_TOKENS=${W05_D3_SOFT_TOKENS:-768}
export W05_D3_HARD_TOKENS=${W05_D3_HARD_TOKENS:-1024}
export W05_D3_WEIGHT=${W05_D3_WEIGHT:-1.0}
export W05_TOKENIZER_PATH=${W05_TOKENIZER_PATH:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}
export ROLLOUT_DIR=${ROLLOUT_DIR:-$REPO_ROOT/outputs/w05/rollouts/$EXP_ID}

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
