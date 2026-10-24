#!/usr/bin/env bash
# W4 MATH-500 路径对齐：把 harness 已评的生成文本用 batched 路径（--rescore-only）重评。
#
# 用途：L1 的 MATH-500 因 math_verify 内存问题用了 `analysis/w04_eval_batched.py`；
# 其余 4 个实验 + 基座原本走 `eval/run_eval.py`。本脚本把**同一批生成文本**（不重新生成、
# 不换判分函数）过一遍 batched 路径，证明两条路径逐 checkpoint 一致
# （W4 实测：9 个 checkpoint Δ=0.0000，唯一 1 条 timeout 见 reports/w04_metrics/）。
#
# 在 bigdog01 上运行：
#   bash scripts/w04_rescore_math500.sh
# 产物：outputs/eval_batched/<run>/summary.json + math500.jsonl（不入 git）
#       reports/w04_metrics/math500_timeout_samples.json（timeout 样本 id + 全文 + sha256）
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

OUT=outputs/eval_batched
mkdir -p "$OUT"

runs="Qwen2.5-0.5B-Instruct \
EXP-W04-00-control-strict-global_step_100 EXP-W04-00-control-strict-global_step_200 \
EXP-W04-02-l2-substring-global_step_100 EXP-W04-02-l2-substring-global_step_200 \
EXP-W04-03-l3-first-number-global_step_100 EXP-W04-03-l3-first-number-global_step_200 \
EXP-W04-04-l4-length-bonus-global_step_100 EXP-W04-04-l4-length-bonus-global_step_200"

for name in $runs; do
  case "$name" in
    Qwen2.5-0.5B-Instruct) model="models/Qwen2.5-0.5B-Instruct" ;;
    *) model="models/merged/$name" ;;
  esac
  src="outputs/eval/$name/math500.jsonl"
  dst="$OUT/$name"
  mkdir -p "$dst"
  cp "$src" "$dst/math500.jsonl"
  if [[ -f "$dst/summary.json" ]]; then
    echo "skip $name (summary exists)"
    continue
  fi
  echo "=== rescore $name $(date +%H:%M:%S) ==="
  python analysis/w04_eval_batched.py \
    --model "$model" \
    --datasets math500 \
    --limit 500 \
    --output-dir "$OUT" \
    --run-name "$name" \
    --rescore-only \
    --overwrite \
    --score-timeout-s 30
done

echo "ALL DONE $(date)"

# 汇总 timeout 样本 id（含 harness + batched 两个目录；batched 优先）
python analysis/collect_timeouts.py
