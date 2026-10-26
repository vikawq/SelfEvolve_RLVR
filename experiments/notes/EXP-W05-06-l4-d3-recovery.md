# EXP-W05-06-l4-d3-recovery 实验记录

> 状态：**done**（2026-10-26 跑完，100 步，exit 0）。

- 日期：2026-10-26
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
- 步数：global_step 201 → 300（再跑 100 步），checkpoint 落在 250 / 300（均已生成，`latest=300`）
- 模型 / 数据 / seed：Qwen2.5-0.5B-Instruct、GSM8K train、seed=42（与 W4/W5 全系一致）
- wandb：`runs/531wsuju`
- 训练日志：`logs/EXP-W05-06-l4-d3-recovery_20261026-155303.log`
- GPU·h：训练 **1.54**（100 步 × 55.5 s 实测）+ 评测 0.13 ≈ **1.67**；预估是 2.6，实际省了四成，
  原因见下面结论第 4 条（长度塌下来之后步时从 ~64 s 降到 ~49 s）

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

# 3) 评测：走 W5 实验链，与其余五个防御组同一条 merge + 判分路径
#    训练已经单独跑过了，所以先占位，否则 run_train 会因为缺 .done 而重跑 100 步
touch outputs/w05/stages/EXP-W05-06-l4-d3-recovery.train.done
EVAL_STEPS="250 300" bash scripts/w05_chain.sh EXP-W05-06-l4-d3-recovery
#    注意 analysis/w04_eval_batched.py 收的是 --model <merged 目录>，没有 --experiment/--steps；
#    merge 由 eval/merge_lora.py 做，链条脚本已经把两步串好。

# 4) 指标
python analysis/w05_metrics.py --experiment EXP-W05-06-l4-d3-recovery --hack-id L4 \
  --rollout-dir outputs/w05/rollouts/EXP-W05-06-l4-d3-recovery \
  --tokenizer models/Qwen2.5-0.5B-Instruct \
  --output reports/w05_metrics/EXP-W05-06-l4-d3-recovery.json
```

注意 `analysis/w05_metrics.py` 的 last20 / last50 窗口是按"最后 N 步"算的，对本实验就是
step 281–300 / 251–300；和其他实验比的时候要说清是"续训后的最后 N 步"，不是"训练的最后 N 步"。

## 结果

**D3 把已经 hack 坏的策略救回来了，用了约 50 步。**

逐段轨迹（每段内全部 rollout 汇总；gap = 代理奖励 − 严格准确率）：

| 步段 | 代理奖励 | 严格 acc | gap | hack rate | 长度(token) | D3 hard / linear / 不触发 | 零方差组 |
|---|---|---|---|---|---|---|---|
| 201–205 | 0.020 | 0.481 | −0.461 | 0.511 | 1015 | 43.4% / 55.4% / **1.2%** | 0.338 |
| 206–215 | 0.051 | 0.507 | −0.455 | 0.471 | 996 | 40.3% / 54.1% / 5.6% | 0.319 |
| 216–235 | 0.422 | 0.516 | −0.094 | 0.270 | 664 | 13.0% / 24.8% / 62.3% | 0.394 |
| 236–265 | 0.600 | 0.601 | −0.001 | 0.048 | 425 | 1.9% / 3.9% / 94.2% | 0.511 |
| **281–300（last20）** | **0.640** | **0.640** | **−0.00003** | **0.0131** | **372** | 0.2% / 1.7% / **98.0%** | 0.550 |

与三条参照曲线对比（last20 为训练侧，评测为严格 verifier greedy pass@1）：

| | 起点 | 严格 acc | 长度(token) | hack rate | GSM8K | MATH-500 | 零方差组 |
|---|---|---|---|---|---|---|---|
| `EXP-W04-04` 未防御 | 基座 | 0.505 | 997 | 0.496 | **0.0000** | 0.0260 | 0.291 |
| `EXP-W05-01` 从基座 + D3 | 基座 | 0.649 | 290 | 0.00075 | 0.5520 | 0.3220 | 0.541 |
| **本实验 从污染点 + D3** | **W04-04 step200** | **0.640** | **372** | 0.0131 | **0.5480** | **0.3280** | 0.550 |
| `EXP-W04-00` 严格 verifier | 基座 | 0.653 | 288 | 0.000 | 0.5560 | 0.3540 | 0.525 |

评测逐 step：GSM8K **0.5160 / 0.5480**（step250 / 300），MATH-500 **0.3300 / 0.3280**，
stderr 约 ±0.022 / ±0.021。

恢复速度：hack rate 的 10 步滑动均值在 **step 253** 首次降到 ≤0.05 并此后保持，
即续训约 **50 步**完成恢复。

## 结论

1. **D3 既能预防也能补救，终点几乎相同**：GSM8K 从 L4 的 0.0000 回到 **0.5480**，与从基座跑 D3
   的 0.5520 差 0.4 分（stderr ±0.022，远在噪声内），与对照 0.5560 差 0.8 分；MATH-500 0.3280
   vs 从基座 0.3220 基本相同。训练侧 gap 回到 −0.00003，hack rate 0.511 → 0.0131。
   **P7 §6 原先"D3 只验证了阻止学会"的局限，由本实验补上。**
2. **防御的工作量完全由起点决定**：D3 的"不触发"比例从 **1.2% 爬到 98.0%**。同一个防御，在被
   污染的策略上开头对 98.8% 的样本生效，在干净策略上（W05-01）只对 0.6% 生效。这把
   §4.4.1 原来的"D3 几乎是 no-op"修正成了"**在预防场景下是 no-op，在补救场景下是主力**"——
   no-op 是起点的性质，不是防御的性质。
3. **没有 collapse，而且原因和直觉相反**：开头奖励被压到 0.02，我一度担心会像 L3 那样因奖励
   长期接近 0 而崩。实际零方差组比例是从 0.338 **上升**到 0.550（向对照的 0.525 收敛），
   即早期组内区分度反而更强。GRPO 的 advantage 是组内 `(r−mean)/std`，奖励绝对值小不要紧，
   要紧的是组内有没有差异。**"奖励均值接近 0"不等于"没有学习信号"，该监控的是零方差组比例。**
4. **防御把算力也省回来了**：步时从 W04-04 的 64.4 s 降到实测均值 55.5 s（末段约 49 s），
   100 步只花 1.54 GPU·h，比按 2.5 GPU·h 的预估少四成。与 §4.5 第 4 条同一现象。
5. **残留代价：长度没有完全回去**。last20 长度 372 token，比对照的 288 和从基座 D3 的 290
   高约 29%；MATH-500 上也看得到（输出 609 token vs 对照 530、从基座 D3 570；截断率 0.154
   vs 0.110 / 0.134）。也就是说"救回来的策略"和"从没坏过的策略"在行为上不完全等价，
   准确率追平了，长度分布留了疤。这是本实验最该写进报告的那句话。
6. **onset 指标在本实验里没有意义**：`hack_onset_step=210` 只是因为这个 run **一开始就是**
   hack 状态，滑动均值从第一个窗口起就 ≥0.10。对恢复实验要看的是"offset"——上面的 step 253。
   指标脚本没有 offset 概念，这个数字是我在 notes 里单独算的；若 W6 要做更多恢复实验，
   值得把 offset 加进 `analysis/w05_metrics.py`。
