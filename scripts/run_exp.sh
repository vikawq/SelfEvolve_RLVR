#!/usr/bin/env bash
# 统一实验入口：bash scripts/run_exp.sh <EXP-ID> [额外的 hydra 覆盖 ...]
# - 读取 configs/<EXP-ID>.sh
# - 日志同时写 logs/<EXP-ID>_<时间戳>.log
# - 退出码透传（pipefail），失败不会被 tee 吞掉
set -euo pipefail

EXP_ID=${1:-}
if [[ -z "$EXP_ID" ]]; then
  echo "usage: bash scripts/run_exp.sh <EXP-ID> [hydra overrides...]" >&2
  echo "available:" >&2
  ls "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs"/*.sh 2>/dev/null | sed 's/.*\///; s/\.sh$//' >&2
  exit 2
fi
shift || true

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$REPO_ROOT/configs/${EXP_ID}.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 2
fi

mkdir -p "$REPO_ROOT/logs"
LOG="$REPO_ROOT/logs/${EXP_ID}_$(date +%Y%m%d-%H%M%S).log"
echo "== run $EXP_ID =="
echo "== log: $LOG =="

set +e
bash "$CONFIG" "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "== exit code: $rc =="
exit "$rc"
