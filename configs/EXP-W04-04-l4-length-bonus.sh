#!/usr/bin/env bash
# EXP-W04-04-l4-length-bonus | L4：长度奖励
#
# 只把奖励换成 `src/selfevolve/rewards/loose_L4_length.py`，其余逐字复用对照组配置。
# 这一组会明显变慢（响应变长），步时与截断率本身就是观测指标。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W04-04-l4-length-bonus
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/loose_L4_length.py"

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
