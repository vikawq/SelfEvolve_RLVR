# EXP-W04-00-control-strict 实验记录

- 日期：2026-10-23
- 目的：W4 对照组 —— 同一套 0.5B 配置下只把奖励换成严格 verifier，作为 L1–L4 的参照曲线。
  它的 reward gap 按定义恒为 0；它的严格准确率曲线代表"正常训练"。
- 硬件：RTX 4080 16GB（bigdog01），单卡
- 配置：`configs/EXP-W04-00-control-strict.sh`
- 代码：`src/selfevolve/rewards/strict_reward.py` + `src/selfevolve/verifier/math_verifier.py`
- 模型 / 数据 / seed：Qwen2.5-0.5B-Instruct、GSM8K train、seed=42
- 步数：200（save_freq=100、test_freq=0），32 prompts × rollout.n 5 = 160 seq/step，response 1024
- micro batch：actor 2 / logprob-ref 2（4080 实测上限，见 `docs/troubleshooting.md §5.13`）
- wandb：训练 `runs/sxnbxucc`；逐步检测指标 `runs/lfmhjahf`（`EXP-W04-00-control-strict__metrics`，
  config 里 `source_run_id=sxnbxucc`；训练循环不写 reward 额外键，见 `docs/troubleshooting.md §5.11`）
- 训练日志：`logs/EXP-W04-00-control-strict_20261023-191834.log`；链路日志 `logs/w04_chain_control.out`

## 结果（200 步，2h27m 墙钟，exit 0）

| 指标 | 值 |
|---|---|
| 单步耗时 | 末步 **46.5 s**；200 步平均 44.2 s（`update_actor` 29.2 s / `old_log_prob` 8.3 s / `ref` 4.0 s / `gen` 3.7 s） |
| 吞吐 | **1454 tok/s** |
| 峰值显存 | 10.00 GiB allocated（12.98 GiB reserved） |
| 训练侧严格准确率（160 条/步，训练题不重复） | 前 10 步 **0.381** → 每 10 步 0.488 / 0.599 / 0.568 / 0.622 / 0.595 / 0.633 / 0.644 / 0.649 / 0.618；**最后 20 步 0.653** |
| reward gap | **0.000**（严格奖励 = 严格准确率，定义上恒为 0） |
| 策略熵 | 0.637（step1）→ **0.098**（step151）——0.5B + LoRA + 3e-5 下收敛很快 |
| GSM8K test 前 500（严格，greedy） | 基座 0.478 → step100 **0.530** → step200 **0.556**（+7.8 分，binomial stderr ≈ ±2.2） |
| MATH-500（严格，greedy） | 基座 0.332 → step100 0.326 → step200 **0.354**（+2.2 分，stderr ≈ ±2.1） |
| 截断率（eval） | GSM8K step200 0.8%；MATH-500 11%（0.5B 在 MATH 上写不完，与基座 17.4% 相比下降） |
| checkpoint | `global_step_100`、`global_step_200`（各 1.3 GB） |
| GPU 时长 | 训练 2.46 + 评测 0.03 = **2.49 GPU·h** |

## 结论

- 对照组曲线正常：训练侧严格准确率从 0.38 升到 0.65，评测侧 GSM8K +7.8、
  MATH-500 +2.2（200 步、0.5B、单种子；不作强精度结论）。
- **reward gap 恒为 0** 是这一组的定义性质，不是"没有 hacking"的证明；它提供的是
  L1–L4 曲线和它比较时的同期基线与噪声尺度。
- 训练侧严格准确率（采样、T=1）与评测侧（greedy）不可直接比：同 step 的训练侧 0.65
  vs 评测 0.556，差异来自采样温度、题集（train vs test）与 0.5B 的格式稳定性。
- 熵从 0.64 掉到 0.10，是本轮所有实验的共同背景；如果 L1/L4 的行为没有出现，
  首要怀疑对象是"策略过早变尖、组内没有探索"，而不是漏洞设计。
- **MATH-500 复核（2026-10-24）**：用 `analysis/w04_eval_batched.py --rescore-only` 在同一批
  生成文本上重评，step100/200 与 harness 完全一致（0.3260/0.3540，Δ=0.0000，0 timeout），
  见 `reports/w04_metrics/math500_path_compare.md`。
