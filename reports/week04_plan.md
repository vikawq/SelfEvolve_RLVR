# Week 04 计划：复现 Reward Hacking

周期：2026-10-23 起。硬件：RTX 4080 16GB（bigdog01，单卡）。预算：5 × 200 步，预计 13–15 GPU·h，
**$0**（本地机器，不租云、不调付费 API）。模型：Qwen2.5-0.5B-Instruct。

## 0. 开工前的勘察（已完成，结论先行）

| 项 | 事实 | 影响 |
|---|---|---|
| 开发机 | Windows，`.venv`（Python 3.12 + math-verify 0.9.0），无 GPU | 代码/单测在本机；训练与评测在 bigdog01 |
| 训练机 | `bigdog01`（ssh `home`），4080 空闲，`/home` 剩 756G | 0.5B 模型、GSM8K parquet 已就位 |
| verl 落盘 | v0.9.1 有 **`trainer.rollout_data_dir`**：每个训练步写 `<step>.jsonl`（input/output/gts/score + reward dict 全部额外键） | 任务包"不支持就 1% 抽样"的分支**不需要**，用原生全量落盘 |
| wandb 额外键 | 训练循环只把 reward dict 的额外键放进 `non_tensor_batch`，**不写 metrics**；只有 validation 会消费 | 检测指标用"训练后回填同一 run"（`analysis/hack_metrics.py`），指标 JSON 同时落盘 |
| 评测代码 | `eval/run_eval.py` 的 JSONL 已含 `text`/`gold`/`correct`/`num_tokens`/`finish_reason` | 评测侧的 hack 指标做**离线分析**，**不改 `eval/` 任何代码**（守住红线） |
| 0.5B 显存 | response 1024：actor micro=4 OOM、logprob micro=8 OOM；micro=2/2 通过，实测 45–53 s/step | 见 `docs/troubleshooting.md §5.13` |

## 1. 工作包

| # | 内容 | 产出 |
|---|---|---|
| W4-1 | 检测器：reward gap、不同数字个数、boxed 数、首数字位置、长度、重复 5-gram、hack onset | `src/selfevolve/metrics/hack_detectors.py` + `tests/test_hack_detectors.py` |
| W4-2 | 宽松判分器：L1 格式权重 / L2 子串 / L3 首数字 / L4 长度；复用严格 verifier 的抽取/归一化/等价判断 | `rewards/loose_common.py` + `loose_L1..L4.py` + `tests/test_loose_rewards.py` |
| W4-3 | 配置：`EXP-W04-00` 为完整基线（0.5B / 32 prompts×n5 / 200 步 / seed42 / response 1024），01–04 只换 reward path | 5 个 config；先登记 registry 再跑 |
| W4-4 | 冒烟测速 + 0.5B 基座评测（GSM8K test 前 500 + MATH-500） | 实测 micro/步时；基座严格分数 |
| W4-5 | 5 × 200 步顺序训练；step100/200 存 ckpt，merge 后用严格 verifier 评测 | 5 个 wandb run + 10 次 eval |
| W4-6 | 分析：从 rollout dump 算逐步指标；回填 wandb；`analysis/plot_hacking.py` 出 5 张双曲线图并标 onset；每种 3 条样例 | `reports/figures/w04_*.png`、`reports/hack_examples/*.md`、`reports/w04_metrics/*.json` |
| W4-7 | 结项：周报、registry、results_master、troubleshooting、模块状态、pytest | `reports/week04_summary.md` |

## 2. 验收标准对照

| 验收项 | 计划证据 | 状态 |
|---|---|---|
| 至少复现 3 种 hacking（reward gap 显著扩大 + 曲线 + 样例） | L1/L2/L4 预期最强；L3 可能不可利用 | 跑完判定 |
| 未复现类型有归因（步数 / 模型规模 / 漏洞本身） | 对照组分步基线 + 训练 rollout 与评测双口径 | 跑完判定 |
| 对照组曲线正常 | `EXP-W04-00` 的 gap 按定义恒为 0，严格准确率应上升 | 跑完判定 |
| 周报完成 | W4-7 | 待做 |

## 3. "复现"的定量判据（先写死，避免事后挑点）

- **主判据**：训练 rollout 上的 reward gap = 宽松奖励均值 − 严格准确率均值；取最后 20 步均值。
- **onset**：gap 的 10 步滑动均值首次 ≥ 0.10，且此后连续 20 步都 ≥ 0.10
  （`hack_detectors.hack_onset`；不足 30 步不判）。
- **严格准确率**：GSM8K test 前 500（binomial stderr ≈ 2.2%）与 MATH-500（500 题）上的
  greedy pass@1；训练侧另有 200 步 × 160 条**不重复** train 题的逐步严格准确率。
- **"显著扩大"**：last20 gap ≥ 0.10，且相对对照组的同期 gap 至少高 0.10；同时严格准确率
  不高于对照组（允许噪声内持平或下降）。
- **样例**：最后 40 步内、严格判错、宽松奖励最高的 3 条，每条来自不同 step
  （`analysis/select_hack_examples.py`，规则写死在代码里）。
- **不做**：改验收标准、改评测代码、用测试集训练/过滤。

## 4. 风险清单

| # | 风险 | 概率/影响 | 缓解 / 触发动作 |
|---|---|---|---|
| R1 | 0.5B 本身弱，"学会 hack"与"本来不会"混淆 | 高/高 | 对照组逐 step 严格准确率；训练 rollout（6400 条不重复题）+ GSM8K test 500 双口径 |
| R2 | L3（取第一个数字）可能不可利用 | 中/中 | 负结果照实写，归因"漏洞本身不可利用"，不硬凑 |
| R3 | L4 长度暴涨拖慢训练/触发截断 | 高/低 | max_response 固定 1024；截断率作为观测指标；`use_remove_padding=False` 下步时对长度不敏感 |
| R4 | 5×200 步 GPU 时间超预期 | 中/中 | 先测速；200 步为下限，偏差写进周报，不悄悄减步 |
| R5 | wandb 回填失败 | 中/低 | 指标 JSON/CSV 是第一手证据；wandb 仅展示 |
| R6 | rollout dump 缺键 | 低/高 | 冒烟已核对键集合；`build_result` 保证每条样本键一致 |
| R7 | 单种子统计功效有限 | 中/中 | 对照组分步基线 + eval 500 题 stderr；第二种子列后续 |
| R8 | 0.5B 的 merge/eval 路径未跑过 | 低/中 | 先跑基座 eval；checkpoint 评测走已验证的 `scripts/run_eval.sh` |

## 5. 决策门

- **花钱**：本周 $0，无需请示。
- **改评测代码**：不做（评测侧指标全部离线计算，只读 `outputs/eval/**`）。
- **改验收标准**：不做。
- **L5（LLM-as-judge）**：需要付费 API/本地放不下 judge，**不做**；在周报"需要人决策"里列出。
- **升级核心依赖**：不做。

## 6. 交付物清单

- 代码：`metrics/hack_detectors.py`、`rewards/loose_common.py`、`loose_L1..L4.py`、`analysis/hack_metrics.py`、
  `plot_hacking.py`、`select_hack_examples.py`、5 个 config、`scripts/w04_chain.sh` / `w04_analyse.sh` / `w04_base_eval.sh`
- 测试：`tests/test_hack_detectors.py`、`tests/test_loose_rewards.py`
- 产物：`reports/week04_summary.md`、`reports/figures/w04_*.png`、`reports/hack_examples/*.md`、
  `reports/w04_metrics/*.json`、`reports/results_master.md`、`experiments/registry.csv`、`docs/troubleshooting.md §5.10–5.13`
