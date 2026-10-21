#!/usr/bin/env bash
# EXP-W02-03-prof-rollout-n | rollout.n 敏感度 | baseline 配置 10 步 × {4, 5, 8}
#
# 注意：不改 train_batch_size（64 prompts），因此每步样本数 = 64×n；
# 这不是公平的"有效 batch"对比，只用于测生成/训练耗时随 n 的变化。
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

for n in 4 5 8; do
  echo "==================== EXP-W02-03 n=$n ===================="
  EXP_ID="EXP-W02-03-prof-rollout-n${n}" \
  CKPT_DIR="$REPO_ROOT/checkpoints/EXP-W02-03/n${n}" \
  TOTAL_STEPS=10 SAVE_FREQ=1000 TEST_FREQ=1000 \
  bash "$REPO_ROOT/configs/EXP-W02-01-baseline-gsm8k.sh" \
    actor_rollout_ref.rollout.n="$n"
  rc=$?
  echo "==================== EXP-W02-03 n=$n exit=$rc ===================="
  sleep 5
done
