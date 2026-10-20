# 流程设计

> 本文回答「**东西是怎么跑起来的**」：训练单步、自我进化单轮、多轮回路、评测、实验生命周期、二阶闭环。
> 分层见 [`docs/architecture.md`](architecture.md)；模块与接口见 [`docs/modules.md`](modules.md)。
>
> 状态：**[已实现]** / **[设计]** / **[计划]**。最后更新：2026-10-20（W1 收工）。

## 1. 项目总流程：三条主线

12 周不是线性推进，而是三条主线交汇。看懂这张图就知道每周的产物被谁消费。

```mermaid
flowchart LR
  W1["W1 环境与骨架<br/>冒烟跑通"] --> W2["W2 Baseline + 评测 harness + Profiling"]

  W2 --> W3["W3 严格 Verifier"]
  W3 --> W4["W4 复现 Reward Hacking<br/>L1-L4"]
  W4 --> W5["W5 防御与 P7 审计报告<br/>冻结 verifier-v1.0"]

  W3 --> W6["W6 自我进化设计与组件<br/>dry-run"]
  W6 --> W7["W7 iter1"]
  W7 --> W8["W8 iter2-3 + 对照组 + 面板"]

  W2 -->|"profiling 基线"| W9["W9 吞吐优化 + FP8 + 上云准备"]
  W8 --> W9
  W9 --> W10["W10 云端主实验 4B<br/>+ 同置 vs 分离对比"]
  W10 --> W11["W11 消融 A1-A5"]
  W8 --> W11
  W5 -->|"A5：闭环里换宽松判分器"| W11
  W11 --> W11B["W11b（可选）二阶闭环"]
  W11 --> W12["W12 技术报告 / 开源 / 面试材料"]
  W11B --> W12

  classDef done fill:#e8f5e9,stroke:#2e7d32,color:#14281a
  classDef gate fill:#fff8e1,stroke:#f9a825,color:#2b2203
  class W1 done
  class W2,W5,W6,W9,W11,W11B,W12 gate
```

黄色 = 带人工决策门（`00_README.md` 总览表）。**W3 是两条主线的共同地基**：P7 审计用它当对照基准，P1 自我进化用它当闭环奖励。W5 的 A5 消融把两条线接回一起 —— 闭环对判分器质量有多敏感，是全项目的亮点之一。

## 2. GRPO 单步训练流程 [已实现]

下图是 W1 实测跑通的链路，括号里是 4080 上的实测耗时（单步均值 23.2s，来源 `docs/metrics_keys.md` §4）。

```mermaid
sequenceDiagram
  autonumber
  participant T as Trainer 主控
  participant V as vLLM rollout 引擎
  participant R as Reward 判分
  participant A as Actor FSDP bf16
  participant F as Ref 模型
  participant C as Checkpoint + wandb

  T->>V: 取 batch 16 条 prompt，每条采样 n=4
  V-->>T: 64 条响应（gen 3.71s）
  T->>R: 逐条打分
  R-->>T: score 0/1（compute_score 约 2.5ms/条）
  Note over T,R: W1 用 verl 内置 GSM8K 规则奖励<br/>W3 起换成严格 verifier
  T->>A: 重算旧策略 logprob（old_log_prob 3.54s）
  T->>F: 参考策略 logprob（ref 1.97s）
  T->>T: 组内相对优势 GRPO，无 critic（adv 约 1ms）
  T->>A: 反传 + 更新，micro bs 2 梯度累积（update_actor 12.28s，占 53%）
  A->>V: LoRA merge 后全量权重同步（update_weights 1.56s）
  T->>C: 每 save_freq 存 ckpt / 每 test_freq 跑验证
  Note over T,C: 一致性哨兵：rollout 与 actor 的 logprob<br/>pearson 相关 0.9995 → 权重同步正确
```

**两个要点**：

- **瓶颈在训练侧，不在 rollout**：`update_actor` 占 53%，`gen` 只占 16%。W9 的优化顺序由此决定（micro-batch / 打包 / kernel 优先于 rollout 并发）。
- **`training/rollout_actor_probs_pearson_corr` 是权重同步的哨兵**：它掉下来就说明 rollout 用的策略和训练的策略脱钩了。W9 做 FP8 rollout 时，这个指标和 log-prob 差异是判断「轻度 off-policy 是否可接受」的主要依据。

## 3. 自我进化单轮迭代流程 [设计 W6 / 实现 W7]

`scripts/run_iteration.sh` 的 7 个阶段。每个阶段完成后写 `.done`，重跑时跳过 —— 这是协议 §8 的可重入要求，也是云端被回收后能接着跑的前提。

```mermaid
flowchart TB
  S0["输入：上一轮的 Challenger / Solver checkpoint<br/>首轮 = 同一基座 + 各自一个 LoRA"]

  S1["① challenger_train<br/>GRPO 训练出题能力 100-200 步"]
  S2["② generate<br/>出题池 约 4000 道"]
  S3["③ label<br/>Solver 采样 m=10 次 → 等价合并计票 → 伪标签 + p̂"]
  S4["④ filter<br/>合法性 + 去重 + 难度带 0.3-0.7"]
  S5["⑤ solver_train<br/>GRPO，伪标签当 ground truth，奖励由严格 verifier 给"]
  S6["⑥ eval<br/>评测 harness：GSM8K / MATH-500 / AIME24 / AIME25"]
  S7["⑦ monitor<br/>200 道题的伪标签准确率 + 多样性 + p̂ 分布"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
  S3 -->|"不确定性奖励 r_unc 减去重复惩罚"| S1
  S7 --> OUT["产物：results_master 一行 + 监控 JSON + 图表<br/>下一轮的起点"]

  CHK["健全性检查（W7 验收项）<br/>· 随机抽 30 道题由人阅读<br/>· 与评测集 13-gram 重叠 → 命中即剔除"]
  S4 --> CHK
  CHK -->|"剔除污染题"| S5

  REF["监控用参考答案<br/>强模型投票 16 次<br/>仅度量，绝不进训练/过滤"]
  REF --> S7

  classDef danger fill:#fdecea,stroke:#c62828,color:#3b1413
  classDef todo fill:#f5f5f5,stroke:#9e9e9e,color:#222,stroke-dasharray:4 3
  class CHK,REF danger
  class S1,S2,S3,S4,S5,S6,S7 todo
```

**阶段 ③ 是整条流程的脖子**：它既产出 Solver 的训练标签，又产出 Challenger 的奖励信号（p̂）。所以

- 它的显存拓扑最紧（见架构文档拓扑 B：Challenger 训练进程 + Solver 推理服务同卡）；
- 它的耗时可能超过训练本身（W7 常见坑），W9 会专门优化批量请求大小与并发；
- 它的口径必须与评测一致（投票复用 verifier 的等价判断），否则伪标签准确率会虚高。

**每个阶段都要记墙钟时间和 GPU 时长** —— 这是 W9 优化自我进化流程的唯一依据。

## 4. 多轮迭代与饱和判定 [计划 W8]

RQ2 的问题不是「能不能涨」，而是「**涨到第几轮停、为什么停**」。

```mermaid
flowchart LR
  I1["iter1"] --> I2["iter2"] --> I3["iter3"]
  I1 --> M1["监控：伪标签准确率 / p̂ 分布 / 多样性"]
  I2 --> M2["监控"]
  I3 --> M3["监控"]
  M1 & M2 & M3 --> ANA["reports/iteration_analysis.md<br/>· 增益曲线与饱和点<br/>· 伪标签准确率 vs 增益的相关性<br/>· 多样性是否塌缩<br/>· 与对照组的比较"]

  CTRL1["对照组 no-challenger-rl<br/>基座直接出题，其余相同"] --> ANA
  CTRL2["对照组 seed2<br/>换种子重跑 iter1，估波动"] --> ANA

  RULE["规则：每轮最多只调一处超参<br/>并在 registry 写明"]
  RULE -.-> I2
  RULE -.-> I3
```

两条判读纪律：

1. **轮次间波动可能小于换种子的波动** —— 没有 seed2 数据时，不对小于 1 分的差异下结论。
2. **饱和不等于失败**，但必须给出机制解释：伪标签噪声累积？多样性塌缩（Challenger 退化成反复出相似题）？还是 Solver 已经吃干了这个难度带？

## 5. 评测流程 [计划 W2]

所有对外数字都走这一条路，训练框架内的验证分数只作参考。

```mermaid
flowchart LR
  CK["checkpoint（LoRA）"] --> MG["合并权重<br/>verl model merger 或 peft merge_and_unload"]
  MG --> SAN["合并校验：先跑 20 题<br/>确认合并前后输出一致"]
  SAN --> GEN["vLLM 离线推理<br/>评测时 util 可到 0.85（无训练进程占卡）"]
  GEN --> SC["严格 verifier 判分"]
  SC --> J["samples.jsonl 逐样本<br/>id / output / extracted / correct / reason"]
  SC --> SUM["summary.json 汇总"]
  SUM --> RM["reports/results_master.md<br/>必须标注判分器版本 + GPU + 采样参数"]

  PT["prompt 模板显式传入<br/>与训练一致"] --> GEN
  OP["口径：GSM8K / MATH-500 pass@1 greedy<br/>AIME avg@16 + 标准差"] --> SUM

  classDef todo fill:#f5f5f5,stroke:#9e9e9e,color:#222,stroke-dasharray:4 3
  class CK,MG,SAN,GEN,SC,J,SUM todo
```

- **框架内分数与 harness 分数不一致是预期现象**（W2 常见坑）：以 harness 为准，但必须解释差异来源。
- **AIME 只有 30 题**，必须 avg@16 并报标准差；Qwen 系列可能有数据污染，所以只看「相对基座的提升」，并重点参考 AIME 2025。
- W1 已入档的两行是 `global_step_{10,20}` 的冒烟分数，用的是 verl 内置规则奖励，`results_master.md` 里已显式标注**不是**严格 verifier，W3 后需按新 verifier 重评。

## 6. 实验生命周期 [已实现]

协议 §3 的流程，现在由 `scripts/run_exp.sh` + registry 承载。**跑未登记的实验是红线。**

```mermaid
flowchart TB
  P1["① 写/改 configs/EXP-ID.sh<br/>超参与路径都在配置里"]
  P2["② registry 登记<br/>id date purpose config model gpu seed git_commit status=running"]
  P3["③ 先 commit 配置<br/>hash 写入 registry"]
  P4["④ bash scripts/run_exp.sh EXP-ID<br/>长任务用 nohup，日志落 logs/"]
  P5{"⑤ 跑通？"}
  P6["⑥ 补全 registry<br/>wandb_url key_metrics gpu_hours cost_usd conclusion"]
  P7["⑦ 结果入 results_master<br/>标注判分器版本与 GPU"]
  P8["⑧ 写 experiments/notes/EXP-ID.md 与周报"]
  F1["记录现象/原因/解法 → docs/troubleshooting.md"]
  F2{"同一阻塞<br/>试了 2 次以上？"}
  F3["停下来请示人<br/>附排查记录"]

  P1 --> P2 --> P3 --> P4 --> P5
  P5 -->|是| P6 --> P7 --> P8
  P5 -->|否| F1 --> F2
  F2 -->|否| P1
  F2 -->|是| F3

  classDef danger fill:#fdecea,stroke:#c62828,color:#3b1413
  class F3 danger
```

**失败的实验同样入档**（status=failed + 原因）。W1 的 7 次失败尝试全部记在 `docs/troubleshooting.md`，其中一次失败的 wandb run 也在周报里留了链接 —— 这不是瑕疵，而是这个项目对外最值钱的部分。

### 6.1 什么时候必须停下来请示人（协议 §6）

```mermaid
flowchart LR
  E1["花钱：租云卡 / 付费 API / 超预算"] --> ASK["停下来请示"]
  E2["升级核心依赖"] --> ASK
  E3["修改验收标准 / 研究问题 / 项目范围"] --> ASK
  E4["同一阻塞尝试 2 次以上未解"] --> ASK
  E5["结果与预期严重不符<br/>可能是 bug，也可能是发现"] --> ASK
  classDef danger fill:#fdecea,stroke:#c62828,color:#3b1413
  class ASK danger
```

W1 实际触发了 E2：撞上两个上游 bug 后，没有擅自升级 vLLM / verl，而是走官方支持的替代路径（`use_v1=False` + `lora.merge=True`）并把升级问题提交决策。

## 7. 二阶闭环流程 [可选 W11b]

主闭环改进的是「解题」和「出题」；二阶闭环改进的是「**怎么改进**」—— 由 LLM 控制器决定下一轮的流程超参。

```mermaid
flowchart TB
  IT["一轮迭代（§3 的 7 阶段）"] --> MON["monitor 输出 JSON<br/>伪标签准确率 / p̂ 分布 / 多样性 / 奖励曲线 / 各阶段耗时"]
  MON --> LLM["LLM 元控制器<br/>输入：本轮指标 + 历史决策及其结果"]
  LLM --> OUT["输出：下一轮配置 + 决策理由"]
  OUT --> CHK{"白名单校验<br/>代码层强制"}
  CHK -->|合法| NEXT["写入下一轮配置"]
  CHK -->|越界/非法| BACK["回退上一轮配置<br/>并记录这次回退"]
  NEXT --> IT
  BACK --> IT
  OUT --> LOG["experiments/meta_decisions.jsonl<br/>输入摘要 / 输出 / 理由 / token 花费"]

  HARD["硬约束：不得改 verifier、评测代码、数据与种子策略<br/>每轮 GPU 时长有上限<br/>测试集分数绝不可见"]
  HARD --> CHK

  CMP["同等 GPU 预算下三组对比<br/>固定超参 / 规则控制器 / LLM 控制器"]
  IT --> CMP
  CMP --> REP["reports/meta_loop.md<br/>含控制器的坏决策案例"]

  classDef danger fill:#fdecea,stroke:#c62828,color:#3b1413
  class HARD,CHK danger
```

**要盯的失效模式**：控制器可能把难度带调向「伪标签更一致但题目更简单」的一侧 —— 内部指标变好、测试分数不动。这是 RSI 场景下的代理目标问题，与 P7 的 reward hacking 遥相呼应，必须单独记录。
**判读纪律**：只有 LLM 控制器明显优于规则控制器时，才能说 LLM 带来了价值；否则如实写「简单规则就够了」。

## 8. 关联文档

- [`docs/architecture.md`](architecture.md) —— 分层、进程/显存拓扑、ADR、信任边界
- [`docs/modules.md`](modules.md) —— 各阶段调用的模块与接口契约
- `docs/metrics_keys.md` —— 图 §2 中所有计时与性能 key 的实测值
- `docs/data_format.md` —— 阶段 ④ 产出的 parquet 目标格式
- `reports/week01_summary.md` —— §2 实测耗时与 §6 流程的一手记录
