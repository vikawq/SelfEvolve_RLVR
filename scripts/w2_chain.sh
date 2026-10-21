#!/usr/bin/env bash
# W2 收尾流水线（训练结束后自动执行；不产生费用）：
#   等 baseline 训练进程退出 -> 等显存释放 -> 评基座 -> 评 step116（自动 merge）
#   -> merge 前后 20 题一致性 -> 3 组 profiling
#
# 用法（detached）：
#   nohup bash scripts/w2_chain.sh > logs/w2_chain_console.log 2>&1 &
# 进度：logs/w2_chain.log；完成标记：logs/w2_chain.done
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
mkdir -p logs
exec >>"$REPO_ROOT/logs/w2_chain.log" 2>&1

# 评测阶段要下载/读取 HF 数据集；必须在 python 启动前 export（huggingface_hub 会固化 endpoint）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

EXP=EXP-W02-01-baseline-gsm8k
TRAIN_LOG="$REPO_ROOT/logs/EXP-W02-01-baseline-gsm8k_20261021-134342.log"
CKPT="$REPO_ROOT/checkpoints/$EXP/global_step_116"

echo "=== chain start $(date) ==="
echo "[chain] waiting for training process to exit (max 6h)"

for _ in $(seq 1 720); do
  if ! pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; then
    break
  fi
  sleep 30
done
echo "[chain] training process gone $(date)"

# 等 GPU 显存释放（Ray/vLLM 退出会有延迟）
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  used=${used:-99999}
  if [ "$used" -lt 500 ] 2>/dev/null; then
    echo "[chain] gpu released (${used} MiB) $(date)"
    break
  fi
  sleep 10
done

if [ ! -d "$CKPT" ]; then
  echo "[chain] FATAL: missing checkpoint $CKPT; training tail:"
  tail -20 "$TRAIN_LOG" || true
  exit 1
fi
echo "[chain] checkpoint 116 present $(date)"
if grep -q "== exit code: 0 ==" "$TRAIN_LOG"; then
  echo "[chain] trainer exited 0"
else
  echo "[chain] WARNING: no 'exit code: 0' marker in $TRAIN_LOG (continue, checkpoint exists)"
fi

# --- 1. 基座全量评测 ---
echo "[chain] base eval start $(date)"
bash scripts/run_eval.sh models/Qwen2.5-1.5B-Instruct --run-name Qwen2.5-1.5B-Instruct
echo "[chain] base eval rc=$? $(date)"

# --- 2. baseline checkpoint 全量评测（run_eval.sh 会先 merge）---
echo "[chain] ckpt eval start $(date)"
bash scripts/run_eval.sh "$CKPT" --run-name "$EXP-global_step_116"
echo "[chain] ckpt eval rc=$? $(date)"

# --- 3. merge 前后 20 题一致性（greedy）---
ADAPTER="$REPO_ROOT/models/merged/$EXP-global_step_116_work/extracted/lora_adapter"
MERGED="$REPO_ROOT/models/merged/$EXP-global_step_116"
if [ -d "$ADAPTER" ] && [ -d "$MERGED" ]; then
  echo "[chain] merge consistency start $(date)"
  python eval/run_eval.py --model models/Qwen2.5-1.5B-Instruct --lora-adapter "$ADAPTER" \
    --datasets gsm8k --limit 20 --run-name mergecheck-adapter --overwrite
  echo "[chain] adapter eval rc=$?"
  python eval/run_eval.py --model "$MERGED" \
    --datasets gsm8k --limit 20 --run-name mergecheck-merged --overwrite
  echo "[chain] merged eval rc=$?"
  python eval/compare_outputs.py --left outputs/eval/mergecheck-adapter --right outputs/eval/mergecheck-merged \
    --datasets gsm8k --limit 20
  echo "[chain] merge compare rc=$? $(date)"
else
  echo "[chain] SKIP merge consistency: adapter or merged model missing"
fi

# --- 4. profiling：三组参数扫描 ---
for cfg in EXP-W02-02-prof-gpu-mem EXP-W02-03-prof-rollout-n EXP-W02-04-prof-micro-bs; do
  echo "[chain] profiling $cfg start $(date)"
  bash scripts/run_exp.sh "$cfg"
  echo "[chain] profiling $cfg rc=$? $(date)"
done

touch "$REPO_ROOT/logs/w2_chain.done"
echo "=== chain done $(date) ==="
