#!/usr/bin/env bash
# EXP-W04-03-l3-first-number | L3：取第一个数字
#
# 只把奖励换成 `src/selfevolve/rewards/loose_L3_first_number.py`，其余逐字复用对照组配置。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W04-03-l3-first-number
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/loose_L3_first_number.py"

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
