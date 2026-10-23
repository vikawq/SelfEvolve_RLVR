#!/usr/bin/env bash
# W4 训练后统一评测：L1-L4 × step100/200，先 merge 再跑批量评测。
#
# 为什么不用 `scripts/run_eval.sh`：MATH-500 的 500 条长 LaTeX 答案会让 math_verify/sympy 的
# 全局缓存线性膨胀（实测 RSS 114 GB、机器换页），所以用 `analysis/w04_eval_batched.py`
# （判分函数不变，只是每条样本一个 spawn worker，评完释放缓存）。
# 详见 `docs/troubleshooting.md §5.15`。
#
# 用法（训练链跑完之后）：
#   setsid nohup bash scripts/w04_eval_all.sh > logs/w04_eval_all.out 2>&1 < /dev/null &
#
# 可重入：已存在的 `models/merged/<name>/merge_report.json` 和
# `outputs/eval/<name>/summary.json` 都会跳过。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

BASE_MODEL=${BASE_MODEL:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}
EVAL_STEPS=${EVAL_STEPS:-"100 200"}
WORKERS=${WORKERS:-4}
EXPS=(
  EXP-W04-01-l1-format-weight
  EXP-W04-02-l2-substring
  EXP-W04-03-l3-first-number
  EXP-W04-04-l4-length-bonus
)

log() { printf '[w04_eval_all %s] %s\n' "$(date '+%F %T')" "$*"; }

for exp in "${EXPS[@]}"; do
  for step in $EVAL_STEPS; do
    ckpt="$REPO_ROOT/checkpoints/$exp/global_step_$step"
    name="$exp-global_step_$step"
    merged="$REPO_ROOT/models/merged/$name"
    if [[ ! -d "$ckpt" ]]; then
      log "ERROR missing checkpoint: $ckpt"
      exit 1
    fi
    if [[ ! -f "$merged/merge_report.json" ]]; then
      log "merge START $name"
      python eval/merge_lora.py --checkpoint "$ckpt" --base-model "$BASE_MODEL" --output "$merged"
      log "merge DONE $name"
    else
      log "skip merge $name (merge_report exists)"
    fi
    if [[ -f "$REPO_ROOT/outputs/eval/$name/summary.json" ]]; then
      log "skip eval $name (summary exists)"
      continue
    fi
    log "eval START $name"
    python analysis/w04_eval_batched.py \
      --model "$merged" \
      --datasets gsm8k,math500 \
      --limit 500 \
      --run-name "$name" \
      --workers "$WORKERS" \
      --overwrite
    log "eval DONE $name"
  done
done

log "ALL DONE"
