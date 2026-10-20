#!/usr/bin/env bash
# EXP-W01-01-smoke | GRPO + LoRA | Qwen2.5-1.5B-Instruct | GSM8K | 单卡 20 步冒烟
#
# 目的：只验证链路（rollout -> reward -> 训练 -> checkpoint -> 断点续跑），不追求精度。
# 参数名以已安装的 verl v0.9.1 为准；与 weeks/week01 文档的差异见 docs/troubleshooting.md。
#
# 用法：
#   bash scripts/run_exp.sh EXP-W01-01-smoke                 # 跑到 TOTAL_STEPS（默认 20）
#   TOTAL_STEPS=10 bash scripts/run_exp.sh EXP-W01-01-smoke   # 第一阶段：只到 step 10，落 checkpoint
#   bash scripts/run_exp.sh EXP-W01-01-smoke                  # 第二阶段：auto 续跑 11..20
#
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HW=${HW:-4080}
# shellcheck source=/dev/null
source "$REPO_ROOT/configs/hw/${HW}.env"

EXP_ID=${EXP_ID:-EXP-W01-01-smoke}
MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/models}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/gsm8k}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/checkpoints/$EXP_ID}

TOTAL_STEPS=${TOTAL_STEPS:-20}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
SEED=${SEED:-42}
TRAIN_BS=${TRAIN_BS:-16}
PPO_MINI_BS=${PPO_MINI_BS:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
LR=${LR:-3e-5}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
KL_COEF=${KL_COEF:-0.001}

cd "$REPO_ROOT"
mkdir -p "$CKPT_DIR"

echo "[$EXP_ID] HW=$HW model=$MODEL_DIR/Qwen2.5-1.5B-Instruct total_steps=$TOTAL_STEPS ckpt=$CKPT_DIR"

# 单卡：关掉官方示例默认的 uv runtime_env，直接用当前 conda 环境的 python
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
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BS" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="$KL_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEM" \
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
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train=False \
  trainer.resume_mode=auto \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@"
