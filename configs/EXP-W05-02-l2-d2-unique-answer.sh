#!/usr/bin/env bash
# EXP-W05-02-l2-d2-unique-answer | L2 子串匹配 + D2 答案唯一性
#
# D2：最终答案区必须唯一（多个不同 boxed / 多个不同 #### 候选 / 多个不同收尾句 → 0）；
# 只有唯一候选且与金标等价才给 1。它把"金标出现在推理任意位置"的漏洞关掉，但代价是
# 输出没有显式唯一答案的样本一律 0（奖励变稀疏）。
#
# 对照：EXP-W04-02-l2-substring。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W05-02-l2-d2-unique-answer
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/defended_reward.py"
export W05_BASE_HACK=L2
export W05_DEFENSES=D2
export W05_TOKENIZER_PATH=${W05_TOKENIZER_PATH:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}
export ROLLOUT_DIR=${ROLLOUT_DIR:-$REPO_ROOT/outputs/w05/rollouts/$EXP_ID}

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
