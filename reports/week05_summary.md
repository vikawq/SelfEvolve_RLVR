# Week 05 Summary

周期：2026-10-24。硬件：RTX 4080 16GB（bigdog01，单卡）。模型：Qwen2.5-0.5B-Instruct。
训练：GSM8K train，200 步/实验，seed=42，32 prompts × rollout.n 5 = 160 seq/step，response 1024。
评测：GSM8K test 前 500 + MATH-500，greedy pass@1，严格 verifier（`selfevolve-strict-verifier-1.0+math-verify-0.9.0`）。
花费：**$0**（本地机器；未租云、未调付费 API）。本周交付：`reports/P7_reward_hacking_audit.md`。

## 完成情况（对照验收标准逐条打勾）

- [ ] **每种复现成功的 hack 至少一种防御把 hack rate 降到接近对照**：
      L4+D3、L2+D2（`@@W5_ACCEPTANCE@@`）；L1 由 D1（严格 verifier，即 W4 对照）覆盖，
      D4 是"补丁无效"的对照证据。
- [ ] **审计报告完整，所有数字可追溯到 registry**：`reports/P7_reward_hacking_audit.md` +
      `reports/w05_metrics/*.json` + `experiments/registry.csv`。
- [ ] **周报完成**（本文件）。

## 1. 实验设置

本周只跑"防御组"，未防御基线全部复用 W4（同模型/数据/步数/seed，只差奖励或 KL）：

| 实验 | 基础 hack | 防御 | 对照 | 状态 |
|---|---|---|---|---|
| `EXP-W05-01-l4-d3-overlong` | L4 长度奖励 | D3 soft=768 / hard=1024 token，λ=1 | `EXP-W04-04` | @@STATUS_01@@ |
| `EXP-W05-02-l2-d2-unique-answer` | L2 子串 | D2 答案唯一性 | `EXP-W04-02` | @@STATUS_02@@ |
| `EXP-W05-03-l2-kl-0.01` | L2 子串 | D5 KL=0.01 | `EXP-W04-02`（0.001） | @@STATUS_03@@ |
| `EXP-W05-04-l2-kl-0.05` | L2 子串 | D5 KL=0.05 | 同上 | @@STATUS_04@@ |
| `EXP-W05-05-l1-d4-dual-parser` | L1 格式权重 | D4 双解析器一致 | `EXP-W04-01` | @@STATUS_05@@ |

D1（严格 verifier）作为全漏洞的根治防御，数值上等于 `EXP-W04-00`，不重复烧卡。L3 是 W4 的负结果
（漏洞不可利用 + 奖励过稀疏导致 collapse），本周不跑防御，只在报告里讨论。

## 2. 关键结果

@@W5_RESULTS@@

## 3. 与计划的偏差及原因

1. **D3 硬上限语义**：任务文档写"超过硬上限直接判 0"，DAPO 原文是 `R_correct − 1`。按文档实现
   （非负奖励、硬上限 0），差异记入报告 §4.1 与 `docs/troubleshooting.md §5.17`。
2. **D3 的 token 口径**：训练侧只有文本，用 `W05_TOKENIZER_PATH` 的 tokenizer 计 token；
   失败回退 `chars/3.5` 并打 `tokenizer_fallback=1`。冒烟 160/160 条走真 tokenizer。
3. **L1 不重跑 D1**：L1 的根治防御就是严格 verifier，与 W4 对照同配置，重跑只会得到同一条曲线；
   P2 的 L1+D4 用来证明补丁无效。
4. **评测路径**：5 个防御组统一用 `analysis/w04_eval_batched.py`（与 `eval/run_eval.py` 同一判分函数、
   同 schema，进程隔离 + 每条 30 s 硬超时），W4 已用 `--rescore-only` 验证两者逐 checkpoint 一致。
   `eval/` 目录本周**一行未改**。
5. **冒烟后的清理**：`EXP-W05-01` 的 2 步冒烟产生了 `global_step_2` checkpoint；正式跑之前删掉了
   该 checkpoint 与冒烟 rollout，保证正式训练从 step 0 开始、步数与其他实验一致。

## 4. 失败与问题（附错误信息与已尝试的方法）

| # | 现象 | 原因 | 解法 | 代价 |
|---|---|---|---|---|
| 1 | 本地 `ssh` 启动链后不返回 | 远端链是 `setsid nohup` 后台进程，外层 `bash -c` 仍持有 ssh 通道 | 远端链不受影响；本地 job 手动 kill | 无 |
| 2 | 远端 `git pull` 被未跟踪的 `reports/w05_metrics/` 挡住 | 指标文件先在远端生成、后被本地 commit | 移到 `/tmp`、快进、`diff -r` 确认一致后删除备份 | 无 |
| 3 | 冒烟 log 出现 wandb `ConnectionResetError` | 训练结束时 wandb 后台线程断连，属展示层 | 不影响训练退出码（exit 0）与指标落盘 | 无 |
| 4 | @@W5_ISSUE_D2@@ | @@W5_ISSUE_D2_CAUSE@@ | @@W5_ISSUE_D2_FIX@@ | @@W5_ISSUE_D2_COST@@ |

## 5. GPU 时长 / 花费

@@W5_GPU@@

花费：**$0**（本地机器；未租云、未调用付费 API）。
按 4080 的 320 W 满载、0.8 元/kWh 粗估电费约 @@W5_ELEC@@ 元（不计入成本口径）。

## 6. 给下周的交接

- **W6 的自我进化系统**：奖励审计的结论直接复用——训练侧严格 verifier 是主奖励；
  `hack_rate`/`reward_gap`/`zero_variance_group_rate` 三个监控指标已经脚本化
  （`analysis/w05_metrics.py`），W6/W8 的监控面板直接接这三个量。
- **D2/D3 的取舍**：@@W5_HANDOFF_DEFENSE@@
- **D5 的取舍**：@@W5_HANDOFF_KL@@
- **L1 的教训**：格式分必须与正确性绑定；不要给"出现某个 token"单独发奖励。
- **L3 的教训**：奖励长期接近 0 会 collapse；监控 reward 均值与零方差组比例。
- **单种子**：本报告的防御代价以"曲线 + 对照组分步基线"表述；第二种子（1234）留到 W6 或补跑。

## 7. 需要人决策的问题

1. **审计报告审阅**：通过后把 W3 严格 verifier 冻结为 `verifier-v1.0`（打 tag）。本周**未打 tag**，
   等人工决策门。
2. **D2 的定位**：如果 D2 的代价（奖励变稀疏、学习变慢）不可接受，是否把它从"推荐补丁"降级为
   "只在没有 verifier 时备选"？
3. **L5（LLM-as-judge）**：仍未做，需要付费 API 或更大显存；是否在 W6–W8 安排预算？
4. **第二种子**：是否补跑 L4+D3 与对照的 seed=1234（约 6 GPU·h）？
