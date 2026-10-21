#!/usr/bin/env bash
# EXP-W02-02-prof-gpu-mem | 显存敏感度 | baseline 配置 10 步 × {0.25, 0.30, 0.35}
#
# 任务包要求 3 组短实验（各 30 步）观察参数敏感度。本组解释为：每个取值跑 10 步，
# 三个取值合计 30 步；只用于耗时/显存，不作精度结论。
# 0.25 预期在 16GB 上会让 vLLM 起不来（W1 实测），保留失败记录。
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

for v in 0.25 0.30 0.35; do
  tag="${v/./}"
  echo "==================== EXP-W02-02 util=$v ===================="
  EXP_ID="EXP-W02-02-prof-gpu-mem-u${tag}" \
  CKPT_DIR="$REPO_ROOT/checkpoints/EXP-W02-02/u${tag}" \
  TOTAL_STEPS=10 SAVE_FREQ=1000 TEST_FREQ=1000 \
  bash "$REPO_ROOT/configs/EXP-W02-01-baseline-gsm8k.sh" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$v"
  rc=$?
  echo "==================== EXP-W02-02 util=$v exit=$rc ===================="
  sleep 5
done
