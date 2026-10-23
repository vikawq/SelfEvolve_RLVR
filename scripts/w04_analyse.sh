#!/usr/bin/env bash
# W4 训练后分析：逐步指标 + 典型样例 + 双曲线图（在 bigdog01 上跑，产物在 outputs/w04/）。
#
# 用法：
#   python -m pytest -q                      # 先保证单测全绿
#   bash scripts/w04_analyse.sh              # 5 个实验全跑
#   W04_NO_WANDB=1 bash scripts/w04_analyse.sh   # 不联网回填 wandb（调试用）
#   ONLY=EXP-W04-02-l2-substring bash scripts/w04_analyse.sh   # 只分析一个
#
# 产物（随后 scp 回本地 reports/）：
#   outputs/w04/metrics/<EXP>.json        逐步曲线 + eval 指标 + onset
#   outputs/w04/hack_examples/<EXP>.md    每种 hack 3 条典型输出
#   outputs/w04/figures/w04_*.png         每个实验一张图 + 汇总 gap 图
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/models}
TOKENIZER=${TOKENIZER:-$MODEL_DIR/Qwen2.5-0.5B-Instruct}
METRICS_DIR=${METRICS_DIR:-$REPO_ROOT/outputs/w04/metrics}
EXAMPLES_DIR=${EXAMPLES_DIR:-$REPO_ROOT/outputs/w04/hack_examples}
FIGURES_DIR=${FIGURES_DIR:-$REPO_ROOT/outputs/w04/figures}
mkdir -p "$METRICS_DIR" "$EXAMPLES_DIR" "$FIGURES_DIR"

WANDB_ARGS=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
NO_WANDB_ARGS=()
if [[ "${W04_NO_WANDB:-0}" == "1" ]]; then
  NO_WANDB_ARGS+=(--no-wandb)
fi

SPECS=(
  "EXP-W04-00-control-strict strict"
  "EXP-W04-01-l1-format-weight L1"
  "EXP-W04-02-l2-substring L2"
  "EXP-W04-03-l3-first-number L3"
  "EXP-W04-04-l4-length-bonus L4"
)

for spec in "${SPECS[@]}"; do
  read -r exp hack <<<"$spec"
  if [[ -n "${ONLY:-}" && "$exp" != "$ONLY" ]]; then
    continue
  fi
  rollout_dir="$REPO_ROOT/outputs/w04/rollouts/$exp"
  if [[ ! -d "$rollout_dir" ]]; then
    echo "[w04_analyse] SKIP $exp (no rollout dump)"
    continue
  fi
  echo "[w04_analyse] ==== $exp ($hack) ===="
  python analysis/hack_metrics.py \
    --experiment "$exp" --hack-id "$hack" \
    --rollout-dir "$rollout_dir" \
    --tokenizer "$TOKENIZER" \
    --output "$METRICS_DIR/$exp.json" \
    "${NO_WANDB_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
  if [[ "$hack" != "strict" ]]; then
    python analysis/select_hack_examples.py \
      --experiment "$exp" --hack-id "$hack" \
      --rollout-dir "$rollout_dir" \
      --out "$EXAMPLES_DIR/$exp.md" || true
  fi
done

METRICS=()
for exp in EXP-W04-00-control-strict EXP-W04-01-l1-format-weight EXP-W04-02-l2-substring \
           EXP-W04-03-l3-first-number EXP-W04-04-l4-length-bonus; do
  if [[ -f "$METRICS_DIR/$exp.json" ]]; then
    METRICS+=("$METRICS_DIR/$exp.json")
  fi
done

if [[ ${#METRICS[@]} -gt 0 ]]; then
  python analysis/plot_hacking.py \
    --metrics "${METRICS[@]}" \
    --control "$METRICS_DIR/EXP-W04-00-control-strict.json" \
    --out-dir "$FIGURES_DIR"
fi

echo "[w04_analyse] done. metrics=$METRICS_DIR examples=$EXAMPLES_DIR figures=$FIGURES_DIR"
