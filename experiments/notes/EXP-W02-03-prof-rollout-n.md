# EXP-W02-03-prof-rollout-n 实验记录

- 日期：2026-10-21
- 目的：W2 profiling 之二。测 `actor_rollout_ref.rollout.n` 对单步耗时/显存/吞吐的敏感度。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置文件：`configs/EXP-W02-03-prof-rollout-n.sh`
- git commit：`ac34ccd714bd36018099f65123fea42b828b1c59`
- 模型 / 数据 / seed：同 `EXP-W02-01`（seed=42）
- 取值：`n = 4 / 5 / 8`，每个取值 10 步；`train_batch_size` 固定 64 prompts，**不作精度结论**

## 结果（纯训练步中位数，来源 `analysis/profile_summary.json`）

| n | sequences/step | 单步 (s) | gen (s) | update_actor (s) | 吞吐 (tok/s) | 峰值显存 (GB) |
|---|---|---|---|---|---|---|
| 4 | 256 | 137.04 | 10.85 | 92.44 | 688.1 | 9.28 |
| 5 | 320 | 171.05 | 13.27 | 116.04 | 683.5 | 9.28 |
| 8 | 512 | 269.27 | 19.25 | 184.12 | 708.8 | 9.28 |

## 结论

- 单步时间与 n **近似线性**（n=8 是 n=4 的 1.97×），各阶段按比例增长，吞吐基本不变
  （688 → 709 tok/s）。
- **`n` 不是效率杠杆**，只是"每步用多少样本"的算法选择：改 n 就改了有效 batch，
  不能当成加速手段。
- 峰值显存不随 n 变化（9.28 GiB）：序列是分 micro-batch 串行处理的，n 只增加 micro-batch 个数。
- n=8 在 16GB 上不 OOM，但每步 269 s，116 步要 8.7 h。
- baseline 选 **n=5 是为了对齐官方**，不是为了速度。

GPU 时长：**1.69 GPU·h**，花费 $0。
