# Results Master

所有评测结果汇总。每行一个模型/checkpoint，**必须**标注评测时的判分器版本。

| 日期 | 实验 ID | 模型 / checkpoint | 数据集 | 样本数 | 采样参数 | 指标定义 | 分数 | 判分器版本 | GPU | wandb |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | EXP-W01-01-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_10` | GSM8K test | 1319 | temperature=1.0, top_p=1.0, n=1 | mean@1（verl 内置验证循环） | 0.5550 | verl v0.9.1 内置 GSM8K rule reward（**非** W3 严格 verifier） | RTX 4080 16GB | [kv34uf1v](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/kv34uf1v) |
| 2026-10-20 | EXP-W01-01-smoke | Qwen2.5-1.5B-Instruct + LoRA r32 @ `global_step_20` | GSM8K test | 1319 | temperature=1.0, top_p=1.0, n=1 | mean@1（verl 内置验证循环） | 0.7081 | 同上 | RTX 4080 16GB | [16fmimie](https://forge.coreweave.com/wandb/vika-liaowq-nanjing-university/selfevolve-rlvr/runs/16fmimie) |

## 说明

- **W1 的两行不是精度结论**：只跑 20 步、单种子、无对照组，且判分器是 verl 内置的规则 reward（只比对 `####` 后的数值），不是 W3 的严格 verifier。保留在表里是为了满足「所有评测结果都要入档」，也给 W2 的 baseline 对齐一个起点。
- `指标定义` 必须写清楚：greedy pass@1 / avg@k / maj@k。
- `判分器版本` 对应 `src/selfevolve/verifier/` 的 git commit 或版本号；W3 之后训练奖励与评测共用同一严格 verifier，届时本表需要按新 verifier 重评。
- 采样参数来自本实验配置：训练 `rollout.n=4`；验证循环为 n=1、temperature=1.0、top_p=1.0。
