#!/usr/bin/env bash
# 一条命令评任意模型 / LoRA checkpoint：
#
#   bash scripts/run_eval.sh models/Qwen2.5-1.5B-Instruct
#   bash scripts/run_eval.sh checkpoints/EXP-W02-01-baseline-gsm8k/global_step_116
#   bash scripts/run_eval.sh <ckpt> --limit 20          # 冒烟
#   bash scripts/run_eval.sh <ckpt> --datasets gsm8k    # 只评 GSM8K
#
# - 输入是 FSDP/LoRA checkpoint 时，先调 eval/merge_lora.py 合并（带 logit 一致性校验），
#   合并产物在 models/merged/<ckpt 名>/；已合并则跳过（可重入）。
# - 其余情况直接当普通 HF 模型评。
# - 评测日志写 logs/eval_<run_name>_<时间戳>.log。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

# 必须在 python 启动前 export：huggingface_hub import 时会固化 endpoint。
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/run_eval.sh <model-or-checkpoint> [eval/run_eval.py args...]" >&2
  exit 2
fi

SRC=$1
shift || true
SRC_ABS=$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")
NAME=$(basename "$SRC_ABS")
BASE_MODEL=${BASE_MODEL:-$REPO_ROOT/models/Qwen2.5-1.5B-Instruct}

is_checkpoint=0
if [[ -f "$SRC_ABS/actor/fsdp_config.json" || -f "$SRC_ABS/fsdp_config.json" ]]; then
  is_checkpoint=1
  # 避免不同实验的 global_step_N 重名
  NAME="$(basename "$(dirname "$SRC_ABS")")-$(basename "$SRC_ABS")"
fi

mkdir -p logs outputs/eval models/merged
LOG="logs/eval_${NAME}_$(date +%Y%m%d-%H%M%S).log"

if [[ "$is_checkpoint" == "1" ]]; then
  MERGED="$REPO_ROOT/models/merged/$NAME"
  if [[ -f "$MERGED/merge_report.json" ]]; then
    echo "[eval] reuse merged model: $MERGED"
  else
    echo "[eval] merging checkpoint $SRC_ABS -> $MERGED"
    python eval/merge_lora.py --checkpoint "$SRC_ABS" --base-model "$BASE_MODEL" --output "$MERGED"
  fi
  MODEL=$MERGED
else
  MODEL=$SRC_ABS
fi

echo "[eval] model=$MODEL log=$LOG"
set +e
python eval/run_eval.py --model "$MODEL" --run-name "$NAME" "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[eval] exit code: $rc"
exit "$rc"
