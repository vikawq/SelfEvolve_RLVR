#!/usr/bin/env bash
# 一次性准备数据与模型，可重入：每步完成后写 .done 标记，重跑自动跳过。
# 用法：bash scripts/prepare_data.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VERL_SRC=${VERL_SRC:-$REPO_ROOT/third_party/verl}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/gsm8k}
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/models}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # 本机直连 huggingface.co 超时
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}         # 镜像不代理 Xet CAS，否则 401

mkdir -p "$DATA_DIR" "$MODEL_DIR" "$REPO_ROOT/logs"

if [[ ! -f "$DATA_DIR/.done" ]]; then
  echo "== preprocess GSM8K -> $DATA_DIR =="
  # 注意：v0.9.1 的参数是 --local_save_dir；--local_dir 已废弃（见 docs/troubleshooting.md）
  python "$VERL_SRC/examples/data_preprocess/gsm8k.py" --local_save_dir "$DATA_DIR"
  touch "$DATA_DIR/.done"
else
  echo "== GSM8K already prepared =="
fi

for repo in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct; do
  name=${repo##*/}
  marker="$MODEL_DIR/.${name}.done"
  if [[ ! -f "$marker" ]]; then
    echo "== download $repo -> $MODEL_DIR/$name =="
    python - "$repo" "$MODEL_DIR/$name" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo_id, local_dir=local_dir)
print("downloaded", repo_id, "->", local_dir)
PY
    touch "$marker"
  else
    echo "== $name already downloaded =="
  fi
done
echo "== prepare_data done =="
