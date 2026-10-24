# 面试材料草稿（持续更新；W12 定稿）

> 规则：本文件里的数字只能来自 `reports/P7_reward_hacking_audit.md` / `reports/results_master.md`
> 的真实结果；**占位符 `{{...}}` 未替换前不得用于简历或对外材料**。W12 统一核对。

## 1. 简历条目（P7：Reward Hacking 审计）

**草稿 A（研究向）**

> 在 Qwen2.5-0.5B + GSM8K 上搭建 RLVR 奖励审计流程：复现 {{N_HACKS}}/4 类 reward hacking
> （格式权重、子串匹配、长度奖励），训练奖励与严格 verifier 的 gap 最高达 {{MAX_GAP}}；
> 实现 {{N_DEFENSES}} 种可开关防御并量化代价，其中 {{BEST_DEFENSE}} 把 hack rate 从
> {{BEST_BEFORE}} 压到 {{BEST_AFTER}}，同时保持 GSM8K/MATH-500 严格准确率 {{EVAL_DELTA}}。

**草稿 B（工程向）**

> 为 RLVR 训练设计“奖励健康度”监控：逐 rollout 记录代理奖励与严格判定，定义 reward gap、
> hack rate、零方差组比例与 hack onset；在 {{N_EXPERIMENTS}} 个 200 步实验中定位到
> “奖励上升、真实准确率下降”的 {{N_HACKS}} 类机制，并给出“根治 vs 补救”的防御选型清单。

**草稿 C（系统向，若 W9 数字可用再合并）**

> 单卡 4080 上 0.5B GRPO 训练 {{STEP_TIME}} s/step、峰值显存 {{PEAK_MEM}} GiB；
> 200 步实验的端到端（训练 + merge + 双数据集评测）约 {{GPU_HOURS}} GPU·h，全部本地、$0。

## 2. 面试问答提纲（P7 部分）

### Q1：什么是 reward hacking？你怎么证明它发生了，而不是模型变差了？

- 定义：代理奖励被优化到很高，但真实目标没有相应改善，甚至变差。
- 证据链：同一批 rollout 上同时算代理奖励与严格 verifier；`reward gap = 代理均值 − 严格均值`；
  与严格奖励训练的对照组比同期曲线；外部评测（GSM8K test 500 / MATH-500）独立复核。
- 关键数字：L2 的 gap {{L2_GAP}}、GSM8K 比对照低 {{L2_GSM8K_DROP}} 分；L4 的 gap {{L4_GAP}}、
  GSM8K step200 = {{L4_GSM8K}}（截断 {{L4_TRUNC}}）。

### Q2：为什么 L1（格式奖励）不算能力下降？

- L1 的严格准确率（{{L1_STRICT}}）与对照（{{CTRL_STRICT}}）持平，gap 全部来自“答错也拿 0.5”；
  这是 specification gaming（代理奖励膨胀），不是能力退化。
- 反例是 L2/L4：代理奖励上升的同时严格准确率下降，才是真正的 hacking 危害。

### Q3：D2/D4 和 D1 有什么区别？为什么还需要补丁？

- D1 直接换严格 verifier，是根治；本任务里 D1 的 hack rate 恒为 0。
- 现实任务经常没有完美 verifier（开放式答案、LLM judge），只能打补丁：
  D2 要求最终答案唯一，D4 要求两套解析器一致，D3 惩罚超长。
- 代价：D2 会把“没有唯一答案”的正确样本判 0（离线回放里 loose mean 从 {{L2_LOOSE}} 降到
  {{L2_D2_LOOSE}}，低于严格准确率 {{L2_STRICT}}）；D4 对 L1 几乎无效（gap 只从 {{L1_GAP}}
  到 {{L1_D4_GAP}}）——补丁只能治它针对的那一层。

### Q4：D3 的阈值怎么定？为什么不是拍脑袋？

- 用 token 而不是字符：字符/token 比在不同实验里从 1.76 到 3.93 不等。
- soft=768 高于对照组响应长度的 p95（{{CTRL_P95}} token），正常训练不会被误罚；
  hard=1024 等于 `max_response_length`，超过就是截断，严格 verifier 本来就判错。
- 监控副作用：零方差组比例（D3 把奖励压到 0 时 GRPO 可能没梯度）、截断率、长度 p95。

### Q5：如果重做一次，你会怎么设计奖励？

- 先写“真实目标 vs 代理指标”的差，再写奖励；
- 训练奖励与评测共用严格 verifier；
- 每个 rollout 落盘代理奖励 + 严格判定，按步算 gap/hack rate/onset；
- 不给格式单独发奖励；长度 shaping 的 soft cap 必须高于正常响应 p95；
- 上线前用离线反事实回放（同一批文本重打分）估防御效果与代价，再决定是否烧 GPU。

### Q6：这个项目的局限是什么？

- 0.5B、GSM8K、200 步、单种子；hack 是设计出来的，真实任务可能更隐蔽；
- 防御实现各只有一种变体；D5 只扫了 L2 的两个 KL 值；
- 结论应表述为“在这个设置下复现并防御”，不外推成“所有 RLVR 都如此”。

## 3. 待填数字清单（W12 从审计报告回填）

| 占位符 | 来源 |
|---|---|
| `{{N_HACKS}}` / `{{N_DEFENSES}}` / `{{N_EXPERIMENTS}}` | P7 报告 §3/§4 |
| `{{MAX_GAP}}` / `{{BEST_DEFENSE}}` / `{{BEST_BEFORE}}` / `{{BEST_AFTER}}` / `{{EVAL_DELTA}}` | P7 报告 §4.3 |
| `{{L2_GAP}}` / `{{L2_GSM8K_DROP}}` / `{{L4_GAP}}` / `{{L4_GSM8K}}` / `{{L4_TRUNC}}` | P7 报告 §3 |
| `{{L1_STRICT}}` / `{{CTRL_STRICT}}` / `{{L1_GAP}}` / `{{L1_D4_GAP}}` | P7 报告 §3.1/§4 |
| `{{L2_LOOSE}}` / `{{L2_D2_LOOSE}}` / `{{L2_STRICT}}` | P7 报告 §4 / troubleshooting §5.18 |
| `{{CTRL_P95}}` | `reports/w05_metrics/EXP-W04-00-control-strict.json` |
| `{{STEP_TIME}}` / `{{PEAK_MEM}}` / `{{GPU_HOURS}}` | registry / 训练日志（W9 后更新） |
