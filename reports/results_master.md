# Results Master

所有评测结果汇总。每行一个模型/checkpoint，**必须**标注评测时的判分器版本与评测口径。
评测代码：`eval/run_eval.py`；判分器：**W3 起**统一走 `src/selfevolve/verifier/math_verifier.py`
（`selfevolve-strict-verifier-1.0+math-verify-0.9.0`）。W2 的 `math-verify==0.9.0`（整段输出参与解析）
作为对照组**保留**，不覆盖——按 `判分器版本` 列区分。
新增行用 `python analysis/emit_results_row.py --summary <run>/summary.json ...` 生成，不要手抄数字。

| 日期 | 实验 ID | 模型 / checkpoint | 数据集 | 样本数 | 采样参数 | 指标定义 | 分数 | stderr | std | 截断率 | 空答案率 | 判分器版本 | GPU | wandb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | EXP-W01-01-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_10` | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1（verl `val_kwargs` 默认 `do_sample=False`） | mean@1（verl 内置验证循环） | 0.5550 | ±0.0137 | n/a | — | — | verl v0.9.1 内置 GSM8K rule reward（**非** W3 严格 verifier） | RTX 4080 16GB | [kv34uf1v](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/kv34uf1v) |
| 2026-10-20 | EXP-W01-01-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_20` | GSM8K test | 1319 | 同上 | mean@1（verl 内置验证循环） | 0.7081 | ±0.0125 | n/a | — | — | 同上 | RTX 4080 16GB | [16fmimie](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/16fmimie) |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7180** | ±0.0124 | n/a | 0.53% | 0.00% | math-verify==0.9.0（fix `####` 重复标记后重评） | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5520** | ±0.0222 | n/a | 9.80% | 0.00% | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | AIME 2024 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0292** | ±0.0189 | 0.1035 | 15.62% | 0.00% | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | AIME 2025 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0083** | ±0.0049 | 0.0271 | 7.92% | 0.00% | 同上 | RTX 4080 16GB | — |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7665** | ±0.0116 | n/a | 0.08% | 0.00% | math-verify==0.9.0（fix `####` 重复标记后重评） | RTX 4080 16GB | [hlrzg966](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/hlrzg966) |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5640** | ±0.0222 | n/a | 9.00% | 0.00% | 同上 | RTX 4080 16GB | 同上 |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | AIME 2024 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0271** | ±0.0122 | 0.0670 | 11.46% | 0.00% | 同上 | RTX 4080 16GB | 同上 |
| 2026-10-21 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | AIME 2025 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0021** | ±0.0021 | 0.0114 | 4.38% | 0.00% | 同上 | RTX 4080 16GB | 同上 |
| 2026-10-22 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7149** | ±0.0124 | n/a | 0.53% | 3.87% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | — |
| 2026-10-22 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5400** | ±0.0223 | n/a | 9.80% | 8.40% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | — |
| 2026-10-22 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | AIME 2024 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0271** | ±0.0189 | 0.1034 | 15.62% | 11.67% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | — |
| 2026-10-22 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct（基座，无 RL） | AIME 2025 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0063** | ±0.0046 | 0.0252 | 7.92% | 5.42% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | — |
| 2026-10-22 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7642** | ±0.0117 | n/a | 0.08% | 0.15% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [hlrzg966](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/hlrzg966) |
| 2026-10-22 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.4980** | ±0.0224 | n/a | 9.00% | 8.20% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | 同上 |
| 2026-10-22 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | AIME 2024 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0271** | ±0.0122 | 0.0670 | 11.46% | 8.12% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | 同上 |
| 2026-10-22 | EXP-W02-01-baseline-gsm8k | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_116`（merged） | AIME 2025 | 30 | temperature=0.6, top_p=0.95, n=16 | avg@16 | **0.0000** | ±0.0000 | 0.0000 | 4.38% | 3.96% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | 同上 |
| 2026-10-22 | EXP-W03-01-strict-reward-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_20`（20 步冒烟，严格奖励） | GSM8K test | 1319 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.7346** | ±0.0122 | n/a | 0.53% | 1.21% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [3b5rwrv2](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/3b5rwrv2) |
| 2026-10-24 | EXP-W04-00-control-strict | Qwen2.5-0.5B-Instruct（基座，无 RL；W4 参照） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.4780** | ±0.0223 | n/a | 2.60% | 3.80% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | — |
| 2026-10-24 | EXP-W04-00-control-strict | Qwen2.5-0.5B-Instruct（基座，无 RL；W4 参照） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3320** | ±0.0211 | n/a | 17.40% | 12.60% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | — |
| 2026-10-24 | EXP-W04-00-control-strict | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（严格奖励） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5300** | ±0.0223 | n/a | 0.00% | 0.60% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [sxnbxucc](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/sxnbxucc) |
| 2026-10-24 | EXP-W04-00-control-strict | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（严格奖励） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3260** | ±0.0210 | n/a | 11.00% | 9.60% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [sxnbxucc](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/sxnbxucc) |
| 2026-10-24 | EXP-W04-00-control-strict | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（严格奖励） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5560** | ±0.0222 | n/a | 0.80% | 1.80% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [sxnbxucc](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/sxnbxucc) |
| 2026-10-24 | EXP-W04-00-control-strict | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（严格奖励） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3540** | ±0.0214 | n/a | 11.00% | 10.20% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [sxnbxucc](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/sxnbxucc) |
| 2026-10-24 | EXP-W04-01-l1-format-weight | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L1 格式奖励训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5260** | ±0.0223 | n/a | 0.40% | 0.40% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [7kfhv9ng](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/7kfhv9ng) |
| 2026-10-24 | EXP-W04-01-l1-format-weight | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L1 格式奖励训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.2880** | ±0.0203 | n/a | 12.40% | 10.00% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [7kfhv9ng](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/7kfhv9ng) |
| 2026-10-24 | EXP-W04-01-l1-format-weight | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L1 格式奖励训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5680** | ±0.0222 | n/a | 0.40% | 1.40% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [7kfhv9ng](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/7kfhv9ng) |
| 2026-10-24 | EXP-W04-01-l1-format-weight | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L1 格式奖励训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3020** | ±0.0205 | n/a | 8.60% | 7.80% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [7kfhv9ng](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/7kfhv9ng) |
| 2026-10-24 | EXP-W04-02-l2-substring | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L2 子串奖励训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5160** | ±0.0223 | n/a | 0.60% | 1.60% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [lof1b6qu](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/lof1b6qu) |
| 2026-10-24 | EXP-W04-02-l2-substring | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L2 子串奖励训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3280** | ±0.0210 | n/a | 16.40% | 14.00% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [lof1b6qu](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/lof1b6qu) |
| 2026-10-24 | EXP-W04-02-l2-substring | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L2 子串奖励训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.4820** | ±0.0223 | n/a | 4.40% | 5.20% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [lof1b6qu](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/lof1b6qu) |
| 2026-10-24 | EXP-W04-02-l2-substring | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L2 子串奖励训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.2880** | ±0.0203 | n/a | 25.60% | 21.20% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [lof1b6qu](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/lof1b6qu) |
| 2026-10-24 | EXP-W04-03-l3-first-number | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L3 首数字奖励训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.0640** | ±0.0109 | n/a | 0.00% | 0.00% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [mingymc1](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/mingymc1) |
| 2026-10-24 | EXP-W04-03-l3-first-number | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L3 首数字奖励训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.0720** | ±0.0116 | n/a | 0.00% | 1.40% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [mingymc1](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/mingymc1) |
| 2026-10-24 | EXP-W04-03-l3-first-number | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L3 首数字奖励训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.0860** | ±0.0125 | n/a | 0.00% | 0.40% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [mingymc1](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/mingymc1) |
| 2026-10-24 | EXP-W04-03-l3-first-number | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L3 首数字奖励训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.0740** | ±0.0117 | n/a | 0.00% | 1.80% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [mingymc1](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/mingymc1) |
| 2026-10-24 | EXP-W04-04-l4-length-bonus | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L4 长度奖励训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.4040** | ±0.0219 | n/a | 23.40% | 20.20% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [iev00mes](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/iev00mes) |
| 2026-10-24 | EXP-W04-04-l4-length-bonus | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L4 长度奖励训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.2200** | ±0.0185 | n/a | 55.60% | 42.60% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [iev00mes](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/iev00mes) |
| 2026-10-24 | EXP-W04-04-l4-length-bonus | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L4 长度奖励训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.0000** | ±0.0000 | n/a | 100.00% | 18.40% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [iev00mes](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/iev00mes) |
| 2026-10-24 | EXP-W04-04-l4-length-bonus | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L4 长度奖励训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.0260** | ±0.0071 | n/a | 94.80% | 41.80% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [iev00mes](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/iev00mes) || 2026-10-24 | EXP-W05-01-l4-d3-overlong | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L4+D3 训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5380** | ±0.0223 | n/a | 1.00% | 1.80% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [d83gq6tw](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/d83gq6tw) |
| 2026-10-24 | EXP-W05-01-l4-d3-overlong | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L4+D3 训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3580** | ±0.0214 | n/a | 11.80% | 9.60% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [d83gq6tw](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/d83gq6tw) |
| 2026-10-24 | EXP-W05-01-l4-d3-overlong | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L4+D3 训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5520** | ±0.0222 | n/a | 0.60% | 0.40% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [d83gq6tw](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/d83gq6tw) |
| 2026-10-24 | EXP-W05-01-l4-d3-overlong | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L4+D3 训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3220** | ±0.0209 | n/a | 13.40% | 10.60% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [d83gq6tw](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/d83gq6tw) |
| 2026-10-24 | EXP-W05-02-l2-d2-unique-answer | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L2+D2 训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5020** | ±0.0224 | n/a | 1.40% | 1.20% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [bwc7jszb](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/bwc7jszb) |
| 2026-10-24 | EXP-W05-02-l2-d2-unique-answer | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_100`（L2+D2 训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3560** | ±0.0214 | n/a | 12.80% | 9.00% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [bwc7jszb](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/bwc7jszb) |
| 2026-10-24 | EXP-W05-02-l2-d2-unique-answer | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L2+D2 训练） | GSM8K test | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.5560** | ±0.0222 | n/a | 1.00% | 1.00% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [bwc7jszb](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/bwc7jszb) |
| 2026-10-24 | EXP-W05-02-l2-d2-unique-answer | Qwen2.5-0.5B-Instruct + LoRA r32 @ `global_step_200`（L2+D2 训练） | MATH-500 | 500 | temperature=0, top_p=1.0, n=1 | greedy pass@1 | **0.3440** | ±0.0212 | n/a | 16.00% | 11.60% | selfevolve-strict-verifier-1.0+math-verify-0.9.0 | RTX 4080 16GB | [bwc7jszb](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/bwc7jszb) |

## 说明

- **W4 的 22 行（2026-10-24）是 reward hacking 实验**：`训练奖励` 各不相同（L1 格式 / L2 子串 /
  L3 首数字 / L4 长度 / 对照组严格），但**评测一律用严格 verifier**，所以同一列可以横向比较
  "训练奖励把模型带到了哪里"。L1 的评测用 `analysis/w04_eval_batched.py`（同判分函数、
  每条 30 s 硬超时，见 `docs/troubleshooting.md §5.15–§5.16`），其余 4 个实验 + 基座的 MATH-500
  已用 `--rescore-only` 在同一批生成文本上复核：**9 个 checkpoint 与 harness 完全一致
  （Δ=0.0000）**，唯一 1 条 timeout 的 id 落盘在 `reports/w04_metrics/math500_timeout_samples.json`
  （对比表 `reports/w04_metrics/math500_path_compare.md`）。
  - L4 step200 的 **0.0000 / 0.0260** 是**截断率 100% / 94.8%**：模型每条都写满 1024 token、
    给不出最终答案，严格 verifier 按规则判错。这不是评测故障，是长度奖励的直接后果。
  - L3 的 0.064/0.086（GSM8K）是**训练崩溃**：奖励几乎全 0，模型退化成 `#### <猜测>`。
  - 逐 step 的 reward gap、boxed 率、长度、重复 n-gram 在 `reports/w04_metrics/*.json`；
    对应 wandb 伴生 run 见各 `experiments/notes/EXP-W04-0*.md`。
  - 总分析见 `reports/week04_summary.md` 与 `reports/figures/w04_*.png`。
- **两套判分器的关系**：W2 的 `math-verify` 行（2026-10-21）与 W3 的 `strict` 行（2026-10-22）是**同一批生成结果**
  的两种打分，不是两次运行——W3 用 `eval/rescore.py --suffix strict` 从落盘 JSONL 重打分，**零 GPU**。
  对比分析见 `reports/verifier_validation.md` §5。
- **20 步冒烟（EXP-W03-01）**：训练与评测共用严格 verifier，harness GSM8K 从基座 0.7149 涨到 **0.7346**
  （+1.97，同一判分器、同一天评估），训练内验证 0.7309（step10）→ 0.7422（step20）。
  只有 20 步，**不作精度结论**，作用是证明奖励接线正确、链路可训练。详见 `experiments/notes/EXP-W03-01-strict-reward-smoke.md`。
- **严格判分带来的一处结论反转（重要）**：MATH-500 的 base→RL 差值从 W2 的 **+1.20** 变成 **−4.20**
  （差值的 stderr ≈ ±3.16，统计上不显著）。原因不是判分器 bug：RL checkpoint 在域外的 MATH-500 上有
  36/500 条"最终答案写错、正确数字只出现在推理里"（例如推理得 `-50` 却写 `#### 50`），
  W2 的整段匹配把这 36 条判对了。**正确表述**：严格判分下没有观察到 MATH-500 的提升，W2 的 +1.2 不可作为结论。
- **GSM8K 的结论是稳的**：基座 0.7149 → RL 0.7642（+4.93），W2 口径 +4.85，两套判分器一致。
- **`空答案率` 两列口径不同**：`math-verify` 行的 0.00% 是因为 W2 的抽取器**总会**返回一个字符串
  （哪怕只是把最后 300 字符里的最后一个数字拿来兜底），不代表"都抽到了答案"；
  `strict` 行的空答案率才是真正的"抽不到最终答案"的比例（基座 GSM8K 3.87%，其它见各行）。
- **AIME 两行的长度预算不足（W2 遗留，W3 复核确认）**：历史运行用的是固定 `max_model_len=2560`，
  而 `--plan-only` 显示 AIME2024 需要 2496（ok）、**AIME2025 需要 2880（超出）**，
  即 AIME2025 里 prompt 超过 512 token 的题目被 vLLM 静默压缩了生成预算（截断率偏低、分数偏低）。
  这两行只作参考，**不用于任何结论**（30 题 × 16 次，基数太小）。重跑需要 GPU（分钟级），
  已列入 W4 的待办；重跑后追加新行、不覆盖。
- **误差口径**：`pass@1` 行报 **binomial stderr** `sqrt(p(1-p)/n)`，`std` 列记 `n/a`——逐题 0/1 的
  样本标准差恒在 0.42–0.50，没有信息量。`avg@16` 行的 `std` 是 30 题逐题 accuracy 的样本标准差，
  `stderr = std/sqrt(30)`。两种口径在 `summary.json → reported_stderr_kind` 里标明，不可混用。
- **W2 baseline 对齐**：官方表格 Qwen2.5-1.5B-Instruct + GRPO-LoRA GSM8K = **77.9**（verl 内置 rule reward）。
  本实验的独立 harness 分数 76.65（W2 口径）/ 76.42（严格口径），差 **1.25 / 1.48 分**（均达标）；
  训练内 validation（同官方口径）0.7582，差 2.09 分，在单种子噪声内（见 `reports/week02_summary.md`）。
- **采样口径更正**：W1 两行原记为 `temperature=1.0`，那是 verl **顶层** `rollout.temperature` 的默认值
  （训练 rollout 用），不是验证循环。本项目所有配置都未覆盖 `rollout.val_kwargs`，验证走默认的
  `do_sample=False, temperature=0`，即 **greedy**。证据与更正过程见 `docs/troubleshooting.md §4.9`。
- **MATH-500 / AIME 的绝对分数不可外比**：这两个数据集复用了 GSM8K 的 `####` 指令（口径对称，
  不做 per-dataset prompt 调优），但它们的标准答案常常不是纯数值，模型也没在这种分布上被训成用
  `####` 收尾。因此绝对分数被系统性压低，**不能与基于 `\boxed{}` 的公开榜单/论文数字比较**；
  只有本表内同一口径下的 base ↔ RL 差值是有效结论。见 `eval/datasets.md §6`。
- **merge 一致性**：merged 权重与 base+adapter 逐参数 bit-identical（339/339，max|Δ|=0，missing=0），
  同进程 20 条 prompt 的 logits 也 max|Δ|=0；vLLM runtime adapter 与 merged 的 greedy 生成 1/20 完全一致。
  详见 `reports/week02_summary.md`。
- `指标定义` 必须写清楚：greedy pass@1 / avg@k / maj@k。
- **判分器版本**对应 `src/selfevolve/verifier/` 的 git commit：W3 的两行对应 `bda2499`
  （`strict_reward.py` 接线）/ `7803e46`（verifier 本体）。W1 两行是冒烟结果，判分器是 verl 内置
  rule reward，**不是** W2 harness，保留用于审计，不与上表直接比较。
