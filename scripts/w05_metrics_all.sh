#!/usr/bin/env bash
# W5 训练后统一分析：逐实验生成 reports/w05_metrics/*.json 与 reports/hack_examples/EXP-W05-*.md。
#
# 用法（在 bigdog01 上，训练链跑完后）：
#   bash scripts/w05_metrics_all.sh
#
# 可重入：每次覆盖同名 JSON/Markdown（分析是只读的，不碰训练产物）。
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

TOKENIZER=${TOKENIZER:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}
OUT_DIR=${OUT_DIR:-$REPO_ROOT/reports/w05_metrics}
EX_DIR=${EX_DIR:-$REPO_ROOT/reports/hack_examples}
mkdir -p "$OUT_DIR" "$EX_DIR"

# 实验 -> (hack id, defenses)；bash 3.2 兼容：不用关联数组。
EXPERIMENTS=(
  "EXP-W05-01-l4-d3-overlong:L4:D3"
  "EXP-W05-02-l2-d2-unique-answer:L2:D2"
  "EXP-W05-03-l2-kl-0.01:L2:none"
  "EXP-W05-04-l2-kl-0.05:L2:none"
  "EXP-W05-05-l1-d4-dual-parser:L1:D4"
)

log() { printf '[w05_metrics_all %s] %s\n' "$(date '+%F %T')" "$*"; }

for item in "${EXPERIMENTS[@]}"; do
  IFS=":" read -r exp hack defenses <<< "$item"
  rollout_dir="$REPO_ROOT/outputs/w05/rollouts/$exp"
  if [[ ! -d "$rollout_dir" ]]; then
    log "SKIP $exp (no rollout dir)"
    continue
  fi
  log "metrics $exp (hack=$hack)"
  python analysis/w05_metrics.py \
    --experiment "$exp" \
    --hack-id "$hack" \
    --rollout-dir "$rollout_dir" \
    --tokenizer "$TOKENIZER" \
    --output "$OUT_DIR/$exp.json"
  if [[ "$defenses" != "none" ]]; then
    log "examples $exp (base=$hack defenses=$defenses)"
    python analysis/w05_select_examples.py \
      --experiment "$exp" \
      --base-hack "$hack" \
      --defenses "$defenses" \
      --rollout-dir "$rollout_dir" \
      --steps 50 \
      --top-k 3 \
      --output "$EX_DIR/$exp.md"
  fi
done

log "ALL DONE"
