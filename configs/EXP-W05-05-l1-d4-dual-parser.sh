#!/usr/bin/env bash
# EXP-W05-05-l1-d4-dual-parser | L1 格式权重 + D4 双解析器一致
#
# D4：严格 verifier 的解析结果 A 与独立解析器 B 必须一致，才给"正确性"那 0.5；
# 格式分（写了 \boxed{} 就给的 0.5）**保留**——这正是本实验要证明的：
# 解析层防御治不了"格式权重过高"这种规格漏洞，L1 只能靠 D1（严格 verifier）根治。
#
# 对照：EXP-W04-01-l1-format-weight。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W05-05-l1-d4-dual-parser
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/defended_reward.py"
export W05_BASE_HACK=L1
export W05_DEFENSES=D4
export W05_TOKENIZER_PATH=${W05_TOKENIZER_PATH:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}
export ROLLOUT_DIR=${ROLLOUT_DIR:-$REPO_ROOT/outputs/w05/rollouts/$EXP_ID}

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
