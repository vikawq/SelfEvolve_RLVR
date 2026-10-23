#!/usr/bin/env bash
# W4 实验链：依次"训练 200 步 → step100/200 严格评测"，阶段级 `.done` 可重入。
#
# 用法（在 bigdog01 上）：
#   source /home/angus/miniconda3/etc/profile.d/conda.sh && conda activate selfevolve
#   setsid nohup bash scripts/w04_chain.sh EXP-W04-01-l1-format-weight ... \
#       > logs/w04_chain_console.log 2>&1 < /dev/null &
#
# - 每个实验的训练日志/评测日志仍由 run_exp.sh / run_eval.sh 落到 logs/。
# - 训练完成后若 GPU 上还有别的进程，本脚本不会自己停：一次只跑一个实验。
# - 失败即退出（set -e）；重跑会跳过已完成的阶段，训练用 verl 的 resume_mode=auto 续跑。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

export BASE_MODEL=${BASE_MODEL:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}
STAGE_DIR=${STAGE_DIR:-$REPO_ROOT/outputs/w04/stages}
mkdir -p "$STAGE_DIR" logs

EVAL_LIMIT=${EVAL_LIMIT:-500}
EVAL_DATASETS=${EVAL_DATASETS:-gsm8k,math500}
EVAL_STEPS=${EVAL_STEPS:-"100 200"}

log() { printf '[w04_chain %s] %s\n' "$(date '+%F %T')" "$*"; }

train_done() {
  [[ -f "$STAGE_DIR/$1.train.done" ]]
}

eval_done() {
  local exp="$1" step="$2"
  local name
  name="$(basename "$exp")-global_step_${step}"
  [[ -f "$REPO_ROOT/outputs/eval/$name/summary.json" ]]
}

run_train() {
  local exp="$1"
  if train_done "$exp"; then
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
  if eval_done "$exp" "$step"; then
    log "skip eval $exp @$step (summary exists)"
    return 0
  fi
  if [[ ! -d "$ckpt" ]]; then
    log "ERROR missing checkpoint: $ckpt"
    return 1
  fi
  log "eval START $exp @$step"
  bash scripts/run_eval.sh "$ckpt" --limit "$EVAL_LIMIT" --datasets "$EVAL_DATASETS"
  touch "$STAGE_DIR/$exp.eval$step.done"
  log "eval DONE $exp @$step"
}

if [[ $# -eq 0 ]]; then
  echo "usage: bash scripts/w04_chain.sh <EXP-ID> [<EXP-ID> ...]" >&2
  exit 2
fi

for exp in "$@"; do
  run_train "$exp"
  for step in $EVAL_STEPS; do
    run_eval_step "$exp" "$step"
  done
done

log "ALL DONE: $*"
