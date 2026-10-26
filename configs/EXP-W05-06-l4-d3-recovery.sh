#!/usr/bin/env bash
# EXP-W05-06-l4-d3-recovery | D3 能不能救回一个已经 hack 坏的策略？
#
# W5 的五个防御组都是**从基座开始**训练，回答的是"能不能阻止模型学会 hack"。
# 本实验回答另一半，也是 P7 §6 明确列为未验证的那一条：
#
#   从 EXP-W04-04（L4 长度奖励，step200 时 hack rate 0.496、长度 4363 字符、GSM8K 归零）
#   的 checkpoint 续训，把奖励换成 L4+D3，再跑 100 步，看长度/hack rate/GSM8K 回不回来。
#
# 奖励与 EXP-W05-01 逐字相同（L4 底层 + D3 overlong shaping，soft 768 / hard 1024 token），
# 唯一的差别是**起点**：W05-01 从基座起跑，本实验从被污染的策略起跑。两条曲线并排看，
# 才能把"预防"和"补救"分开说。
#
# ── 续训的三个关键点（verl 0.9.1，third_party/verl 实测）────────────────────────
#
# 1. global_step 从来源 checkpoint 的编号继续：
#    `self.global_steps = int(folder.split("global_step_")[-1])`（ray_trainer.py:1083）
#    → 从 global_step_200 起跑，所以 TOTAL_STEPS=300 才是"再跑 100 步"，
#      落盘的 checkpoint 是 global_step_250 / global_step_300。
#
# 2. **必须 TOTAL_EPOCHS=2**。GSM8K train 7473 题 / bs 32 = 233 步/epoch，
#    `current_epoch = 200 // 233 = 0`，而循环是 `for epoch in range(current_epoch, total_epochs)`
#    （ray_trainer.py:1432、1465）。若沿用默认的 total_epochs=1，epoch 0 里只剩 33 步，
#    训练会在 step 233 静默结束，拿不到 100 步 —— 不会报错，只会少跑，很容易被当成跑完了。
#
# 3. 写入只去 `trainer.default_local_dir`（ray_trainer.py:987），即本实验自己的
#    checkpoints/$EXP_ID；**来源目录 EXP-W04-04 只读**。P7 报告依赖 W04-04 的产物，不能被覆盖。
#
# ── 一个必须在报告里写明的口径 ─────────────────────────────────────────────────
#
# data.shuffle=False 且 dataloader 状态会从 data.pt 恢复（ray_trainer.py:1102，200 不在 epoch
# 边界上所以确实恢复）。于是 step 201–233 接着看 epoch 0 没看过的尾部数据，step 234–300 进入
# 第二个 epoch、**从数据集开头重看**，这 67 步里的题目 W04-04 在它自己的 step 1–67 见过。
# 不改成 shuffle=True 是为了和 W05-01 的曲线保持同一数据顺序口径；这个重叠要写进 §4.4.1。
#
# 用法（bigdog01）：
#   source /home/angus/miniconda3/etc/profile.d/conda.sh && conda activate selfevolve
#   bash scripts/run_exp.sh EXP-W05-06-l4-d3-recovery
#   # 冒烟（2 步，确认能加载 + 奖励接上，跑完记得删 global_step_202 和冒烟 rollout）：
#   TOTAL_STEPS=202 SAVE_FREQ=999 bash scripts/run_exp.sh EXP-W05-06-l4-d3-recovery
#
# 评测（与 W5 其余实验同一判分函数/同 schema）：
#   python analysis/w04_eval_batched.py --experiment EXP-W05-06-l4-d3-recovery --steps 250,300
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export EXP_ID=EXP-W05-06-l4-d3-recovery

# 奖励：与 EXP-W05-01 完全一致（L4 + D3）。
export REWARD_PATH="$REPO_ROOT/src/selfevolve/rewards/defended_reward.py"
export W05_BASE_HACK=L4
export W05_DEFENSES=D3
export W05_D3_SOFT_TOKENS=${W05_D3_SOFT_TOKENS:-768}
export W05_D3_HARD_TOKENS=${W05_D3_HARD_TOKENS:-1024}
export W05_D3_WEIGHT=${W05_D3_WEIGHT:-1.0}
export W05_TOKENIZER_PATH=${W05_TOKENIZER_PATH:-$REPO_ROOT/models/Qwen2.5-0.5B-Instruct}

# 起点：被 L4 污染的 step200 策略（只读）。
export RESUME_MODE=resume_path
export RESUME_FROM_PATH=${RESUME_FROM_PATH:-$REPO_ROOT/checkpoints/EXP-W04-04-l4-length-bonus/global_step_200}

# 200（来源）+ 100 = 300；epochs=2 见文件头第 2 点；save_freq=50 → 250 / 300。
export TOTAL_STEPS=${TOTAL_STEPS:-300}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
export SAVE_FREQ=${SAVE_FREQ:-50}

export ROLLOUT_DIR=${ROLLOUT_DIR:-$REPO_ROOT/outputs/w05/rollouts/$EXP_ID}

exec bash "$REPO_ROOT/configs/EXP-W04-00-control-strict.sh" "$@"
