# EXP-W02-04-prof-micro-bs 实验记录

- 日期：2026-10-21
- 目的：W2 profiling 之三。测 `ppo_micro_batch_size_per_gpu` / `log_prob_micro_batch_size_per_gpu`
  对单步耗时/显存/吞吐的敏感度。这是 profiling 里最重要的一组——它直接决定 W9 的优化顺序。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置文件：`configs/EXP-W02-04-prof-micro-bs.sh`
- git commit：`ac34ccd714bd36018099f65123fea42b828b1c59`
- 模型 / 数据 / seed：同 `EXP-W02-01`（seed=42）
- 取值：`micro = 1 / 2 / 4`，每个取值 10 步，**不作精度结论**

## 结果（纯训练步中位数，来源 `analysis/profile_summary.json`）

| micro | 单步 (s) | update_actor (s) | old_log_prob (s) | ref (s) | 峰值显存 (GB) | 吞吐 (tok/s) |
|---|---|---|---|---|---|---|
| 1 | 169.68 | 115.05 | 28.38 | 11.12 | 9.28 | 697.2 |
| 2 | **107.87** | **64.86** | **17.94** | 9.81 | **11.91** | **1092.3** |
| 4 | — | — | — | — | — | **OOM** |

## 失败点（保留记录）

`micro=4`：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.39 GiB.
GPU 0 has a total capacity of 15.57 GiB of which 839.88 MiB is free.`
1024 response × 4 条序列的激活放不下。**micro=2 是 16GB 上的实际上限**。
见 `docs/troubleshooting.md §4.8`。

## 结论（W9 的第一优先级 = P1）

- micro 1→2：actor **-43.6%**、old_log_prob **-36.8%**、单步 **-36.4%（1.57×）**、吞吐 **+57%**；
  代价是显存 9.28 → 11.91 GiB（仍余约 3.6 GiB）。
- 116 步墙钟预计从 ~5.2 h 降到 ~3.5 h。
- 只改 batching，不改有效 batch / 学习率 / 步数，**数学等价**；但 W9 落地时仍要跑 20 步的
  loss / ratio / 精度 A/B 确认。
- **仅 P1 一项就超过 W9 的"单步下降 ≥30%"目标**，且风险最低。
- micro=4 要先做 P3（flash-attn / `use_remove_padding=True`）腾出显存才可能可用。

GPU 时长：**0.86 GPU·h**，花费 $0。
