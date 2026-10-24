# Week 05 计划：防御方案与审计报告（P7 交付）

周期：2026-10-24 起。硬件：RTX 4080 16GB（bigdog01，单卡）。预算：5 × 200 步，预计 13–15 GPU·h，
**$0**（本地机器；不租云、不调付费 API、不升级任何依赖）。模型：Qwen2.5-0.5B-Instruct。
训练/评测口径与 W4 完全一致（GSM8K train、seed=42、200 步、response 1024、严格 verifier
`selfevolve-strict-verifier-1.0+math-verify-0.9.0`），保证"防御 vs 未防御"只差一个变量。

## 0. 开工前的勘察（结论先行）

| 项 | 事实 | 影响 |
|---|---|---|
| 开发机 | Windows，`.venv`（Python 3.12），无 GPU | 代码/单测/分析在本机；训练与评测在 bigdog01 |
| 训练机 | `bigdog01`（ssh `home`），4080 空闲，`/home` 剩 723G | 0.5B / GSM8K / W4 checkpoint 都在位 |
| 远端仓库 | `67cbd3a`，比本地 `4ecc879` 落后一个文档 commit；`origin` 已有本地最新 commit | `git pull` 快进即可，不需要 bundle |
| W4 产物 | `outputs/w04/rollouts/*`（200 步全量 rollout）、`outputs/eval/*`、`reports/w04_metrics/*.json` | 未防御基线直接复用，**不重跑**；W5 只跑防御组 |
| 奖励接口 | verl v0.9.1 以关键字调用 `compute_score(data_source, solution_str, ground_truth, extra_info)`，返回 dict；除 `score` 外的键随 rollout 落盘 | 防御诊断字段可以原样进入 `trainer.rollout_data_dir` |
| D3 的长度口径 | 任务文档说"超长"，训练侧只有文本；`max_response_length=1024` 是 token | 用模型 tokenizer 计 token（懒加载），失败时回退字符估算并打 `tokenizer_fallback` 标 |
| 评测红线 | 任务包要求"不改评测代码" | `eval/` 一行不改；防御组评测复用 `eval/run_eval.py`；MATH-500 内存问题用 W4 已验证的进程隔离脚本（同一判分函数） |

## 1. 防御设计（全部可开关）

| 防御 | 开关 | 实现 | 针对 |
|---|---|---|---|
| D1 严格 verifier | `REWARD_PATH=.../strict_reward.py` | 直接复用 W3 | 全部（根治） |
| D2 答案唯一性 | `W05_DEFENSES=D2` | 最终答案区出现 ≥2 个不同候选 → 0；只认最后一个 `\boxed{}`（无 boxed 时按 `####` → `answer is` 顺序，多候选即 0） | L2、L3 |
| D3 超长惩罚 | `W05_DEFENSES=D3` | DAPO 式：`len<=soft` 不罚；`soft<len<hard` 线性扣 `(len-soft)/(hard-soft)`；`len>=hard` 判 0。默认 soft=768 / hard=1024 token，λ=1，最终分数截断到 ≥0 | L4 |
| D4 双解析器一致 | `W05_DEFENSES=D4` | 解析器 A=严格 verifier `extract_answer`，解析器 B=独立正则（boxed/####/answer is/末数字）；两者归一化后一致才给分，否则 0 | L1、L2 等解析类 |
| D5 KL 强度 | `KL_COEF=0.01/0.05`（0.001 = W4 基线） | 只改 `actor.kl_loss_coef`，奖励仍是原宽松判分器 | 全部（看 hack onset 是否推迟） |

实现放在 `src/selfevolve/rewards/defended_reward.py` + `defense_common.py`：单一入口，用环境变量
`W05_BASE_HACK=L1|L2|L3|L4`、`W05_DEFENSES=D2,D3,D4` 组合，避免为每个组合复制一份判分器。
D2/D4 是"补救"（判分器不得不宽松时用），D1 是"根治"；报告里分开写，并显式说明
**D2/D4 在本任务里会退化成接近严格判分器**——这本身就是代价。

## 2. 实验矩阵

| 实验 | 基础 hack | 防御 | 对照 | 优先级 |
|---|---|---|---|---|
| `EXP-W05-01-l4-d3-overlong` | L4 长度奖励 | D3 | W4-04 | P0（W4 交接指定第一优先） |
| `EXP-W05-02-l2-d2-unique-answer` | L2 子串 | D2 | W4-02 | P0 |
| `EXP-W05-03-l2-kl-0.01` | L2 子串 | D5 KL=0.01 | W4-02（0.001） | P1 |
| `EXP-W05-04-l2-kl-0.05` | L2 子串 | D5 KL=0.05 | 同上 | P1 |
| `EXP-W05-05-l1-d4-dual-parser` | L1 格式权重 | D4 | W4-01 | P2（时间够才跑） |

- **L1 的根治防御是 D1**：L1+D1 在数值上就是已跑过的 `EXP-W04-00`（严格奖励、gap 恒 0），
  因此不重复烧卡。P2 的 D4 用来证明"补丁式防御治不了解析之外的规格漏洞"。
- **L3 不跑防御**：W4 已判定漏洞不可利用（负结果）；D2 对它的作用只做定性讨论。
- seed 仍是 42；W4 的分步数据作为未防御基线，不重跑。
- 5 个实验按 P0→P2 顺序串行，远端 `nohup` 链 + 阶段 `.done`，失败可从断点续跑。

## 3. hack rate 定义（先写死，避免事后挑点）

数据源：`trainer.rollout_data_dir` 每步全量 rollout（160 条/步）。**hack rate = 最后 50 步
（step151–200）所有 rollout 中满足该 hack 判定规则的样本比例**；同时报 first-10 步与逐步曲线。
严格判定一律取 reward dict 里的 `strict_acc`（同一批 rollout 上用严格 verifier 复判）。

| hack | 判定规则（单条 rollout） | 理由 |
|---|---|---|
| L1 | `strict_acc=0` 且响应里有 `\boxed{}` | 答错仍拿到 0.5 格式分 |
| L2 | `strict_acc=0` 且 `score=1.0` | 金标出现在推理里、最终答案错，却拿满分 |
| L3 | `strict_acc=0` 且 `score=1.0` | 首数字碰巧等于金标（W4 未复现，留作规则） |
| L4 | `strict_acc=0` 且 `n_chars>=3000` | 答错且拿满长度 bonus（≥0.5） |
| strict / D1 | 恒为 0 | 没有"宽松奖励 − 严格判定"的差 |

`score`、`strict_acc`、`n_chars`、`boxed_count` 全部来自 reward dict；规则写在
`src/selfevolve/metrics/hack_rate.py`，有 pytest。对照组的 hack rate 应为 0（校验口径）。

## 4. 验收标准对照

| 验收项 | 计划证据 | 状态 |
|---|---|---|
| 每种复现成功的 hack 至少一种防御把 hack rate 降到接近对照 | L4+D3、L2+D2，对照 D1（strict，hack rate 0） | **已达成**（0.496→0.00075、0.171→0.0005；L1 由 D1 覆盖，D4 无效） |
| 审计报告完整、数字可追溯 | `reports/P7_reward_hacking_audit.md` + `reports/w05_metrics/*.json` + registry | **已完成** |
| 周报完成 | `reports/week05_summary.md` | **已完成** |

> 若某组防御做不到：照实写原因（例如 D3 阈值不合适 / 训练停滞），**不降低验收标准**，
> 并在周报"需要人决策"里列出。

## 5. 风险清单

| # | 风险 | 概率/影响 | 缓解 / 触发动作 |
|---|---|---|---|
| R1 | 5×200 步 ≈ 13–15 GPU·h，单卡串行 | 高/中 | 远端 nohup 链 + `.done` + `resume_mode=auto`；按 P0→P2 启动，P2 可放弃 |
| R2 | D3 在 reward 进程里加载 tokenizer 失败/变慢 | 中/中 | 懒加载一次；失败回退字符估算并打 `tokenizer_fallback`；2 步冒烟核对 `n_tokens` |
| R3 | D3 硬上限判 0 导致 GRPO 组内奖励全 0、无梯度 | 中/高 | 监控每步 reward 方差与零优势组比例；若停滞，记录为发现并请示是否换 DAPO 的 −1 变体 |
| R4 | D2/D4 退化成严格判分器，"防御=换回 D1" | 高/中 | 报告分开写根治 vs 补救；量化代价（reward 稀疏、ambiguous→0、严格准确率） |
| R5 | hack rate 规则主观、事后调参 | 中/中 | 规则先写死 + 单测；对照组应为 0；曲线和 first-10 一并报 |
| R6 | L1 的 MATH-500 harness 内存爆炸（W4 已踩） | 中/低 | 用 `analysis/w04_eval_batched.py`（同判分函数、进程隔离、每条 30 s 超时），不改 `eval/` |
| R7 | 单种子，无法给统计显著性 | 高/中 | 以"曲线 + 对照组分步基线 + 500 题 stderr"表述，写进局限性 |
| R8 | 同步/磁盘/断连 | 低/高 | 先 `git push`→远端 `git pull`；日志落 `logs/`；checkpoint 每 100 步 |

## 6. 决策门（本周）

- **花钱**：$0，本地 4080，不租云、不调付费 API → 无需请示。
- **改评测代码**：不做（`eval/` 一行不改；分析只读 `outputs/eval/**`）。
- **改验收标准 / 研究问题 / 范围**：不做。
- **升级核心依赖**：不做。
- **`verifier-v1.0` tag**：本周**不打**，等人工审阅审计报告通过后再冻结（任务包的人工决策门）。
- **D3 硬上限语义**：任务文档写"判 0"，DAPO 原文是 `R_correct − 1`；本周按文档实现
  （非负奖励、硬上限 0），差异记入报告的"实验设置"和 `docs/troubleshooting.md`。

## 7. 交付物清单

- 代码：`rewards/defense_common.py`、`rewards/defended_reward.py`、`metrics/hack_rate.py`、
  `analysis/w05_metrics.py`、`analysis/plot_defenses.py`、5 个 config、`scripts/w05_chain.sh`
- 测试：`tests/test_defenses.py`、`tests/test_hack_rate.py`
- 产物：`reports/P7_reward_hacking_audit.md`、`reports/week05_summary.md`、`reports/figures/w05_*.png`、
  `reports/w05_metrics/*.json`、`reports/hack_examples/EXP-W05-*.md`、registry/results_master、
  `experiments/notes/EXP-W05-*.md`、`docs/interview_prep.md` 草稿
