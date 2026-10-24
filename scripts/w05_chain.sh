#!/usr/bin/env bash
# W5 实验链：依次"训练 200 步 → merge LoRA → step100/200 严格评测"，阶段级 `.done` 可重入。
#
# 用法（在 bigdog01 上）：
#   source /home/angus/miniconda3/etc/profile.d/conda.sh && conda activate selfevolve
#   setsid nohup bash scripts/w05_chain.sh EXP-W05-01-l4-d3-overlong ... \
#       > logs/w05_chain_console.log 2>&1 < /dev/null &
#
# 评测用 `analysis/w04_eval_batched.py`（与 `eval/run_eval.py` 同一判分函数、同 schema，
# 只是每条样本一个 spawn 子进程 + 30 s 硬超时），原因见 docs/troubleshooting.md §5.15/5.16；
# W4 已用 `--rescore-only` 验证两者逐 checkpoint 一致（Δ=0.0000）。
#
# - 失败即退出（set -e）；训练用 verl 的 resume_mode=auto 续跑。
# - 已存在的 checkpoint / merged / summary 会跳过。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export BASE_MODEL=${BASE_MODEL:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}
STAGE_DIR=${STAGE_DIR:-$REPO_ROOT/outputs/w05/stages}
EVAL_LIMIT=${EVAL_LIMIT:-500}
EVAL_DATASETS=${EVAL_DATASETS:-gsm8k,math500}
EVAL_STEPS=${EVAL_STEPS:-"100 200"}
WORKERS=${WORKERS:-4}
mkdir -p "$STAGE_DIR" logs outputs/eval models/merged outputs/w05/rollouts

log() { printf '[w05_chain %s] %s\n' "$(date '+%F %T')" "$*"; }

run_train() {
  local exp="$1"
  if [[ -f "$STAGE_DIR/$exp.train.done" ]]; then
    log "skip train $exp (done)"
    return 0
  fi
  log "train START $exp"
  bash scripts/run_exp.sh "$exp"
  touch "$STAGE_DIR/$exp.train.done"
  log "train DONE $exp"
}

run_eval_step() {
  local exp="$1" step="$2"
  local ckpt="$REPO_ROOT/checkpoints/$exp/global_step_$step"
  local name="$exp-global_step_$step"
  local merged="$REPO_ROOT/models/merged/$name"
  if [[ -f "$REPO_ROOT/outputs/eval/$name/summary.json" ]]; then
    log "skip eval $name (summary exists)"
    return 0
  fi
  if [[ ! -d "$ckpt" ]]; then
    log "ERROR missing checkpoint: $ckpt"
    return 1
  fi
  if [[ ! -f "$merged/merge_report.json" ]]; then
    log "merge START $name"
    python eval/merge_lora.py --checkpoint "$ckpt" --base-model "$BASE_MODEL" --output "$merged"
    log "merge DONE $name"
  else
    log "skip merge $name (merge_report exists)"
  fi
  log "eval START $name"
  python analysis/w04_eval_batched.py \
    --model "$merged" \
    --datasets "$EVAL_DATASETS" \
    --limit "$EVAL_LIMIT" \
    --run-name "$name" \
    --workers "$WORKERS" \
    --overwrite
  log "eval DONE $name"
}

if [[ $# -eq 0 ]]; then
  echo "usage: bash scripts/w05_chain.sh <EXP-ID> [<EXP-ID> ...]" >&2
  exit 2
fi

for exp in "$@"; do
  run_train "$exp"
  for step in $EVAL_STEPS; do
    run_eval_step "$exp" "$step"
  done
done

log "ALL DONE: $*"
