# EXP-W02-02-prof-gpu-mem 实验记录

- 日期：2026-10-21
- 目的：W2 profiling 之一。测 vLLM `rollout.gpu_memory_utilization` 对单步耗时/显存/吞吐的敏感度。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置文件：`configs/EXP-W02-02-prof-gpu-mem.sh`
- git commit：`ac34ccd714bd36018099f65123fea42b828b1c59`
- 模型 / 数据 / seed：同 `EXP-W02-01`（Qwen2.5-1.5B-Instruct + LoRA r32，GSM8K，seed=42）
- 取值：`0.25 / 0.30 / 0.35`，每个取值 10 步（共 30 步），**不作精度结论**

## 结果（纯训练步中位数，来源 `analysis/profile_summary.json`）

| util | 能否启动 | 单步 (s) | gen (s) | 峰值显存 (GB) | 吞吐 (tok/s) |
|---|---|---|---|---|---|
| 0.25 | **否** | — | — | — | — |
| 0.30 | 是 | 170.85 | 13.72 | 9.28 | 679.7 |
| 0.35 | 是 | 167.62 | 9.66 | 9.28 | 712.2 |

## 失败点（保留记录）

`util=0.25` 下 vLLM 初始化失败：16GB 里 0.25 不足以同时放 1.5B 权重（~3.1GB）与 1536 总长的
KV cache。**0.30 是可用下限**。见 `docs/troubleshooting.md §3.3 / §4.8`。

## 结论

- util 0.30 → 0.35 只有约 **2%** 的单步收益（-1.9%），吞吐 +4.8%。
- 峰值显存不随 util 变化（9.28 GiB 由 actor 侧主导，不是 vLLM）。
- 原因：93.5% 的时间在训练侧，rollout 的 KV cache 大小**不是当前瓶颈**。
  所以 baseline 保留 util=0.30，不把显存让给 vLLM——留给训练侧的 micro-batch（见 `EXP-W02-04`）更值。

GPU 时长：**1.02 GPU·h**，花费 $0。
