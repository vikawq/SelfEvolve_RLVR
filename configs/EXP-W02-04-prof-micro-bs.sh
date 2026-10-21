#!/usr/bin/env bash
# EXP-W02-04-prof-micro-bs | micro batch 敏感度 | baseline 配置 10 步 × {1, 2, 4}
#
# n=5、64 prompts/step 固定；只改 actor 与 logprob 的 micro batch。
# 4 很可能 OOM（1024 response、16GB），失败也保留记录，不静默跳过。
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

for m in 1 2 4; do
  echo "==================== EXP-W02-04 micro_bs=$m ===================="
  EXP_ID="EXP-W02-04-prof-micro-bs${m}" \
  CKPT_DIR="$REPO_ROOT/checkpoints/EXP-W02-04/m${m}" \
  TOTAL_STEPS=10 SAVE_FREQ=1000 TEST_FREQ=1000 \
  bash "$REPO_ROOT/configs/EXP-W02-01-baseline-gsm8k.sh" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$m" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$m" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$m"
  rc=$?
  echo "==================== EXP-W02-04 micro_bs=$m exit=$rc ===================="
  sleep 5
done
