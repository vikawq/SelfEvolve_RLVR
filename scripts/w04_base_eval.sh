#!/usr/bin/env bash
# W4 基座评测：Qwen2.5-0.5B-Instruct 在 GSM8K 前 500 题 + MATH-500 上的严格 verifier 分数。
#
# 这是 L1–L4 与对照组的 step0 参照。评测口径与 checkpoint 评测完全相同
# （eval/run_eval.py，greedy pass@1，严格 verifier）。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

MODEL=${MODEL:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}
OUT="${OUT:-$REPO_ROOT/outputs/eval/Qwen2.5-0.5B-Instruct}"

if [[ -f "$OUT/summary.json" ]]; then
  echo "[w04_base_eval] summary exists, skip: $OUT/summary.json"
  exit 0
fi

bash scripts/run_eval.sh "$MODEL" --limit 500 --datasets gsm8k,math500
