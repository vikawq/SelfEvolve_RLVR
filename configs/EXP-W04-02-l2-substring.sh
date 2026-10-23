#!/usr/bin/env bash
# EXP-W04-02-l2-substring | L2：子串匹配
#
# 只把奖励换成 `src/selfevolve/rewards/loose_L2_substring.py`，其余逐字复用对照组配置。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W04-02-l2-substring
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/loose_L2_substring.py"

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
