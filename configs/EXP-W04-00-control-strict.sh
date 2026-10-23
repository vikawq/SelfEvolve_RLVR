#!/usr/bin/env bash
# EXP-W04-00-control-strict | W4 对照组：0.5B + GSM8K + 严格 verifier 奖励
#
# 目的（`weeks/week04_reward_hacking_reproduce.md`）：同一套训练配置下，只把奖励换成严格
# verifier，作为 L1–L4 的参照曲线。它的 reward gap 按定义恒为 0；它的严格准确率曲线就是
# "正常训练"长什么样。
#
# 关键配置（与 W2 baseline 的差异都记在 experiments/notes/EXP-W04-00-control-strict.md）：
#   - 模型换成 Qwen2.5-0.5B-Instruct（任务包要求；它跑得快、hacking 出现得早）
#   - 32 prompts × rollout.n 5 = 160 sequences/step（0.5B 够快；200 步 ≈ 6400 条不重复 train 题）
#   - response 仍为 1024，与 W2/W3 同口径；L4 的长度暴涨才不会撞到不同的上限
#   - test_freq=0：训练内验证用的是**同一个（宽松）奖励**，对 W4 没有信息量；
#     真实准确率由外部 eval/run_eval.py（严格 verifier）在 step100/200 上评
#   - rollout_data_dir：verl v0.9.1 原生落盘，每步一个 <step>.jsonl，含 output/gts/score
#     和 reward dict 的全部诊断键（任务包说"不支持就 1% 抽样"，本版本支持）
#
# 用法：
#   bash scripts/run_exp.sh EXP-W04-00-control-strict
#   TOTAL_STEPS=2 SAVE_FREQ=99 TEST_FREQ=0 bash scripts/run_exp.sh EXP-W04-00-control-strict
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HW=${HW:-4080}
# shellcheck source=/dev/null
source "$REPO_ROOT/configs/hw/${HW}.env"

export EXP_ID=${EXP_ID:-EXP-W04-00-control-strict}
# 01–04 的 config 会覆盖这两个变量；默认是严格 verifier（对照组）。
export REWARD_PATH=${REWARD_PATH:-$REPO_ROOT/src/selfevolve/rewards/strict_reward.py}
export TOTAL_STEPS=${TOTAL_STEPS:-200}
export SAVE_FREQ=${SAVE_FREQ:-100}
export TEST_FREQ=${TEST_FREQ:-0}

MODEL_DIR=${MODEL_DIR:-$REPO_ROOT/models}
DATA_DIR=${DATA_DIR:-$REPO_ROOT/data/gsm8k}
CKPT_DIR=${CKPT_DIR:-$REPO_ROOT/checkpoints/$EXP_ID}
ROLLOUT_DIR=${ROLLOUT_DIR:-$REPO_ROOT/outputs/w04/rollouts/$EXP_ID}

TRAIN_BS=${TRAIN_BS:-32}
PPO_MINI_BS=${PPO_MINI_BS:-32}
ROLLOUT_N=${ROLLOUT_N:-5}
LR=${LR:-3e-5}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
KL_COEF=${KL_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
# 用独立变量名：`MAX_PROMPT_LEN` / `MAX_RESPONSE_LEN` 会被 configs/hw/*.env 里给 W1 冒烟用的
# 512/512 覆盖（W2 baseline 踩过同样的坑，见 configs/EXP-W02-01-baseline-gsm8k.sh）。
MAX_PROMPT_LEN=${W04_MAX_PROMPT_LEN:-512}
MAX_RESPONSE_LEN=${W04_MAX_RESPONSE_LEN:-1024}
SEED=${SEED:-42}
MICRO_BS=${MICRO_BS:-$W04_PPO_MICRO_BS}
LOGPROB_MICRO_BS=${LOGPROB_MICRO_BS:-$W04_LOGPROB_MICRO_BS}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}

STEPS_ARGS=()
if [[ -n "${TOTAL_STEPS:-}" ]]; then
  STEPS_ARGS+=(trainer.total_training_steps="${TOTAL_STEPS}")
fi

cd "$REPO_ROOT"
mkdir -p "$CKPT_DIR" "$ROLLOUT_DIR"

echo "[$EXP_ID] HW=$HW reward=$REWARD_PATH"
echo "[$EXP_ID] prompt=$MAX_PROMPT_LEN response=$MAX_RESPONSE_LEN bs=$TRAIN_BS x n=$ROLLOUT_N = $((TRAIN_BS * ROLLOUT_N)) seq/step"
echo "[$EXP_ID] steps=$TOTAL_STEPS save_freq=$SAVE_FREQ test_freq=$TEST_FREQ micro=$MICRO_BS logprob_micro=$LOGPROB_MICRO_BS"
echo "[$EXP_ID] ckpt=$CKPT_DIR rollout_dump=$ROLLOUT_DIR util=$ROLLOUT_GPU_MEM"

# 单卡：直接 conda python（不开 uv runtime_env）；lora.merge=True 与 use_v1=False 是 W1 的结论。
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
  actor_rollout_ref.model.path="$MODEL_DIR/Qwen2.5-0.5B-Instruct" \
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
  reward.custom_reward_function.path="$REWARD_PATH" \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.use_v1=False \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=selfevolve-rlvr \
  trainer.experiment_name="$EXP_ID" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DIR" \
  trainer.total_epochs=1 \
  "${STEPS_ARGS[@]}" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train=False \
  trainer.resume_mode=auto \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  "$@"
