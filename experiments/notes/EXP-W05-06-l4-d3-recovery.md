# EXP-W05-06-l4-d3-recovery 实验记录

> 状态：**已登记，未跑**（配置与 registry 条目就绪，等开跑）。

- 日期：2026-10-26（登记）
- 目的：补上 P7 §6 明确列为未验证的那一条 —— D3 只验证了"阻止模型学会 hack"，没有验证
  "从一个已经 hack 坏的 checkpoint 续训能否恢复"。本实验从 `EXP-W04-04`（L4 长度奖励）
  step200 的策略出发，换成 L4+D3 再跑 100 步，看长度 / hack rate / GSM8K 回不回来。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置：`configs/EXP-W05-06-l4-d3-recovery.sh`
- 奖励：与 `EXP-W05-01` **逐字相同**（`W05_BASE_HACK=L4`、`W05_DEFENSES=D3`、soft 768 / hard 1024 token、λ=1）。
  唯一的差别是起点 —— W05-01 从基座起跑，本实验从被污染的策略起跑。
- 起点 checkpoint：`checkpoints/EXP-W04-04-l4-length-bonus/global_step_200`
  （2026-10-26 在 bigdog01 上确认存在、完整：`actor/{model,optim,extra_state}_world_size_1_rank_0.pt`
  + `fsdp_config.json` + `lora_train_meta.json` + `huggingface/` + `data.pt`，1.3 GB；
  `latest_checkpointed_iteration.txt` = 200；`/home` 余 690 GB）
- 步数：global_step 201 → 300（再跑 100 步），checkpoint 落在 250 / 300
- 模型 / 数据 / seed：Qwen2.5-0.5B-Instruct、GSM8K train、seed=42（与 W4/W5 全系一致）
- 预估：约 2.5 GPU·h 训练 + 0.13 评测

## 对照关系

| 曲线 | 实验 | 问题 |
|---|---|---|
| 从基座 + 无防御 | `EXP-W04-04` | hack 长什么样（hack rate 0.496、长度 4363 字符、GSM8K 0.000） |
| 从基座 + D3 | `EXP-W05-01` | D3 能不能**预防**（0.00075、922 字符、0.552） |
| **从污染点 + D3** | **本实验** | D3 能不能**补救** |
| 从基座 + 严格 verifier | `EXP-W04-00` | 正常训练的参照线（0.000、933 字符、0.556） |

读法：把本实验 step 201–300 的曲线叠在 `EXP-W05-01` 的 step 1–100 上。若长度和 hack rate 在
可比的步数内回落到同一水平，说明 D3 既能预防也能补救；若回落明显更慢或停在高位，
说明"已经学会的 hack 需要比阻止它更多的代价"—— 这两种结果都值得写进报告。

## verl 续训的三个坑（0.9.1，`third_party/verl` 实测，不是文档推断）

1. **global_step 从来源编号继续**：`self.global_steps = int(folder.split("global_step_")[-1])`
   （`trainer/ppo/ray_trainer.py:1083`）→ 从 200 起算，所以 `TOTAL_STEPS=300` 才是"再跑 100 步"。
2. **必须 `TOTAL_EPOCHS=2`**：GSM8K train 7473 题 / bs 32 = 233 步/epoch，
   `current_epoch = 200 // 233 = 0`，循环是 `for epoch in range(current_epoch, total_epochs)`
   （`:1432`、`:1465`）。沿用默认 `total_epochs=1` 的话 epoch 0 里只剩 33 步，训练会在
   **step 233 静默结束**，不报错、只少跑 67 步 —— 这是本实验最容易被当成"跑完了"的失败模式。
   跑完务必核对 `global_step_300` 真的存在。
3. **写入只去 `trainer.default_local_dir`**（`:987`）= `checkpoints/EXP-W05-06-l4-d3-recovery`。
   来源目录 `EXP-W04-04` 全程只读。P7 报告依赖它的产物，被覆盖就得重跑 W4-04 才能复现。
   config 里还加了三道前置校验（路径必须绝对、必须含 `global_step_`、必须有 `actor/`），
   缺一个就 exit 2，不会跑到一半才发现。

## 数据顺序的口径（必须写进报告 §4.4.1）

`data.shuffle=False`，且 dataloader 状态会从 `data.pt` 恢复（`:1102`；200 不在 epoch 边界上，
所以确实恢复）。于是：

- step 201–233：接着看 epoch 0 剩下的、W04-04 没看过的尾部数据；
- step 234–300：进入第二个 epoch，**从数据集开头重看**，这 67 步的题目 W04-04 在它自己的
  step 1–67 见过。

没有改成 `shuffle=True`，是为了和 `EXP-W05-01` 保持同一数据顺序口径（否则两条曲线不可叠）。
代价是最后 67 步有数据重叠，解读"恢复速度"时要把这一点说明白。

## 跑法

```bash
# bigdog01
source /home/angus/miniconda3/etc/profile.d/conda.sh && conda activate selfevolve
cd ~/workspace/SelfEvolve_RLVR

# 1) 冒烟 2 步：确认能加载污染 checkpoint、D3 奖励接上、global_step 从 201 开始
TOTAL_STEPS=202 SAVE_FREQ=999 bash scripts/run_exp.sh EXP-W05-06-l4-d3-recovery
#    跑完删掉冒烟产物（参考 W05-01 的教训，见 reports/week05_summary.md §3.5）：
#    rm -rf checkpoints/EXP-W05-06-l4-d3-recovery outputs/w05/rollouts/EXP-W05-06-l4-d3-recovery

# 2) 正式 100 步
bash scripts/run_exp.sh EXP-W05-06-l4-d3-recovery
ls checkpoints/EXP-W05-06-l4-d3-recovery/   # 必须看到 global_step_250 和 global_step_300

# 3) 评测（与 W5 其余实验同一判分函数、同 schema）
python analysis/w04_eval_batched.py --experiment EXP-W05-06-l4-d3-recovery --steps 250,300

# 4) 指标
python analysis/w05_metrics.py --experiment EXP-W05-06-l4-d3-recovery --hack-id L4 \
  --rollout-dir outputs/w05/rollouts/EXP-W05-06-l4-d3-recovery \
  --tokenizer models/Qwen2.5-0.5B-Instruct \
  --output reports/w05_metrics/EXP-W05-06-l4-d3-recovery.json
```

注意 `analysis/w05_metrics.py` 的 last20 / last50 窗口是按"最后 N 步"算的，对本实验就是
step 281–300 / 251–300；和其他实验比的时候要说清是"续训后的最后 N 步"，不是"训练的最后 N 步"。

## 结果

待跑。

## 结论

待跑。
