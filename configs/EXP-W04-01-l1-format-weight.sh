#!/usr/bin/env bash
# EXP-W04-01-l1-format-weight | L1：格式权重过高
#
# 只把奖励换成 `src/selfevolve/rewards/loose_L1_format.py`，其余逐字复用
# `configs/EXP-W04-00-control-strict.sh`（0.5B / 32 prompts × n5 / 200 步 / response 1024）。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W04-01-l1-format-weight
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/loose_L1_format.py"

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
