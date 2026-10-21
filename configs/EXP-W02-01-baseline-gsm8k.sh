#!/usr/bin/env bash
# EXP-W02-01-baseline-gsm8k | GRPO + LoRA | Qwen2.5-1.5B-Instruct | GSM8K | 4080 单卡
#
# 目的：复现 verl 官方 baseline（GSM8K 77.9 分），证明训练管线正确。
#
# 官方依据（考据过程见 reports/week02_plan.md §0）：
#   - 表格：docs/algo/baseline.md "Qwen/Qwen2.5-1.5B-Instruct | GRPO-LoRA | 77.9"
#   - 脚本：examples/tuning/1.5b/qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh @ verl 368dff7
#   - 实跑日志：verl-data PR#3（bcff590）Qwen2.5-1.5B-bsz64_2-...-score0.779.log
#     日志落盘 config：train_batch_size=64, rollout.n=5, lr=3e-5, rank=32,
#     prompt=512, response=1024, total_epochs=1（=116 步）, 2×H100, 最终 val 0.77938
#   注意："bsz64_2" 的 2 指 2 张卡，不是 rollout.n=2；实际 n=5。
#
# 4080 单卡的偏离（全部记录在 notes 与周报）：
#   - micro batch 1（16GB + 1024 response；官方在 80GB 上跑 micro=64）
#   - test_freq 20（官方 5；只影响监控频率，不影响训练）
#   - max_num_batched_tokens 8192（官方 1536；纯调度参数）
#   - 无 flash-attn：SDPA；LoRA 同步走 merge 路径；legacy trainer（W1 结论）
#
# 用法：
#   bash scripts/run_exp.sh EXP-W02-01-baseline-gsm8k                  # 全量 116 步（1 epoch）
#   TOTAL_STEPS=20 SAVE_FREQ=10 TEST_FREQ=10 \
#     bash scripts/run_exp.sh EXP-W02-01-baseline-gsm8k                # 20 步校准
#   # 之后不带 TOTAL_STEPS 重跑即从 step 20 续跑到 116（resume_mode=auto）
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HW=${HW:-4080}
# shellcheck source=/dev/null
source "$REPO_ROOT/configs/hw/${HW}.env"

EXP_ID=${EXP_ID:-EXP-W02-01-baseline-gsm8k}
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/models}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/gsm8k}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/checkpoints/$EXP_ID}

# --- 与官方对齐的训练规模（不要为了赶时间改这些）---
TRAIN_BS=${TRAIN_BS:-64}          # prompts / step
PPO_MINI_BS=${PPO_MINI_BS:-64}    # prompts / mini-batch（= 320 sequences）
ROLLOUT_N=${ROLLOUT_N:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}   # 7473/64 -> 116 步
LR=${LR:-3e-5}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
KL_COEF=${KL_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024}
SEED=${SEED:-42}

# --- 4080 系统参数 ---
MICRO_BS=${MICRO_BS:-$PPO_MICRO_BS_LONG}
LOGPROB_MICRO_BS=${LOGPROB_MICRO_BS:-$LOGPROB_MICRO_BS_LONG}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}

# 可选：TOTAL_STEPS=20 只跑前 20 步做墙钟校准；不设置时按 total_epochs 跑满
STEPS_ARGS=()
if [[ -n "${TOTAL_STEPS:-}" ]]; then
  STEPS_ARGS+=(trainer.total_training_steps="${TOTAL_STEPS}")
fi

cd "$REPO_ROOT"
mkdir -p "$CKPT_DIR"

echo "[$EXP_ID] HW=$HW total_epochs=$TOTAL_EPOCHS total_steps=${TOTAL_STEPS:-auto}"
echo "[$EXP_ID] bs=$TRAIN_BS x n=$ROLLOUT_N = $((TRAIN_BS * ROLLOUT_N)) seq/step; micro=$MICRO_BS; util=$ROLLOUT_GPU_MEM"
echo "[$EXP_ID] ckpt=$CKPT_DIR"

# 单卡：关掉官方示例默认的 uv runtime_env，直接用当前 conda 环境的 python。
# lora.merge=True -> 绕开 vLLM 0.24 add_lora 不兼容；use_v1=False -> 绕开 v1 GRPO values bug。
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_batch_size="$TRAIN_BS" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESPONSE_LEN" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  data.seed="$SEED" \
  actor_rollout_ref.model.path="$MODEL_DIR/Qwen2.5-1.5B-Instruct" \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing="$GRAD_CKPT" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BS" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BS" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM" \
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BS" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BS" \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.use_v1=False \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=selfevolve-rlvr \
  trainer.experiment_name="$EXP_ID" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "${STEPS_ARGS[@]}" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train=False \
  trainer.resume_mode=auto \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@"
