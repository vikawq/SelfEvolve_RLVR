# 严格 Verifier 验证报告（W3）

对应任务：`weeks/week03_strict_verifier.md` 任务 3（三判分器对比 + 人工核对）与验收标准 2。

> **人工核对已完成（2026-10-23）**：`reports/verifier_accuracy.json` 的 `human_review.status` =
> **`complete`**（`disputed_reviewed` 110/110），`results_basis = human`。下面 §1 的主表就是**人工口径**。
> 项目负责人用 WPS 填 `reports/verifier_review_pack.csv` 的 `human_verdict` 列（1=模型答对 / 0=模型答错），
> 110 条分歧 + 310 条"三家一致"抽查全部填满；WPS 存成了 GBK，入库前用
> `analysis/review_adjudication.py` 的口径重建为 UTF-8（保留原文件所有列 + 人工两列，见 §9）。
>
> **复核（2026-10-23）**：首轮回填后有 4 条与 agent 初判不一致，逐条回看题面后确认 3 条填反了，
> 已按复核值改回并在 `human_note` 里写明理由：
> `gsm8k/1221#0` 0→**1**（题目问 80 个肉丸的调料用量 = 10，模型结尾就是 10；首轮被 W2 抽取器给出的 `80` 带偏）、
> `gsm8k/23#0` 1→**0**（题目问"短了多少厘米"= 8，模型推理里算出 8 后改答剩余长度、最终给 `L-8`）、
> `prealgebra/1436.json#0` 0→**1**（题目问 56 Drinkets 值多少 Trinkets = 6，模型结尾就是 6）。
> `gsm8k/164#0`（`#### 15` 后跟 `-15`）保留人工判定"模型错"——模型自相矛盾，判错是口径选择。
>
> 复核后：110 条分歧里 **109 条与 agent 初判一致、1 条不同意**（`gsm8k/164#0`）；
> **strict 与人工只在 1 条上不一致**（`prealgebra/1436.json#0`，§1.1）。
> 310 条"三家一致"样本人工也全部填了、**0 条反例**，故全量口径是**实测值**，
> 不再需要"假设一致样本全对"这个假设。
>
> **agent 初判为什么不能直接交差**：初判里 96/110 条走的是"`strict_extracted` 归一化后 == gold"
> 这条自动规则 —— 用 strict 自己的抽取结果当参考答案，而 strict 这时判的就是 1，必然计为"判对"。
> 这对 strict 是**循环论证**；`results_agent` 只保留作过程记录，不作为验收依据。

## 1. 结论（人工核对口径，`results_human`）

| 判分器 | 核对集（110 条分歧） | 420 抽样全体（**实测**） | agent 初判（对照） |
|---|---|---|---|
| **strict（W3 严格 verifier）** | **109/110 = 99.09%** | **419/420 = 99.76%** | 108/110 = 98.18% |
| mathverify（W2 口径） | 101/110 = 91.82% | 411/420 = 97.86% | 102/110 = 92.73% |
| verl 内置 GSM8K 判分 | 1/103 = 0.97% | 198/300 = 66.00% | 2/103 = 1.94% |

- 严格 verifier 在**分歧富集**的核对集上 **99.09%**（只错 1 条），**达到验收标准 2（≥98%）**；
  全量 419/420 = 99.76%。即使按首轮未复核的回填算（108/110 = 98.18%）也达标 ——
  这条结论不依赖那 3 条的判定方向。
- verl 内置判分在 0.97% 是**抽样偏差**：核对集就是按"三家不一致"选的，而 verl 的分歧几乎全部是
  "模型没写 `####`、它直接判 0"。它的真实水平看全量实测（**66.00%**，分母 300）。
- **全量口径现在是实测，不是估计**：420 条全部人工核对过。每个判分器的分母是"它适用的样本数" ——
  verl 对 MATH-500 不适用（判定 `None`），分母是 300 条 GSM8K，不是 420。
  旧版 `estimated_accuracy_on_pool` 用 `(一致数 + 判对数) / 420` 估计，把 MATH 行算成了 verl "判对"，
  于是给出虚高的 74.05%；实测 66.00%。strict / mathverify 对 420 条全适用，两种算法一致。
- Math-Verify（W2 口径）在 420 条里错 9 条，比严格 verifier 多 8 条：
  **7 条假阳性**（把"最终答案写错、但推理里出现过正确数字"判对：`L-8` vs `8`、`#### 15` 后跟 `-15`、
  截断、`8` vs `-3/8`、`6` vs `(-1,6)`、`1` vs `2`、`3/3` vs `√3/3`），
  **2 条假阴性**（`\boxed{144}` 后跟一句解释导致解析失败；抽到句尾的题目数字 `80` 而非答案 `10`）。

### 1.1 人工与 strict 不一致的 1 条（复核后）

| sample_id | gold | strict 抽取 | strict reason | 人工 | 说明 |
|---|---|---|---|---|---|
| `Qwen2.5-1.5B-Instruct/test/prealgebra/1436.json#0` | 6 | （空） | `no_answer` | **1（模型对）** | 模型结尾 "Therefore, 56 Drinkets are equivalent to 6 Trinkets" 给出了 gold 6，但句内还有题面数字 56；strict 按"结论句里有多个值就不猜"放弃抽取 → 判错。这是 §6 限制 #2 的实例，**不是抽取 bug，是宁严勿松的代价** |

这是 420 条里 strict 唯一的一次判错，所以核对集是 109/110、全量 419/420。

> 首轮回填曾把 `gsm8k/1221#0` 和 `gsm8k/23#0` 也记为"strict 与人工不一致"，复核后两条都是人工填反
> （见顶部状态块），strict 在这两条上是对的。保留这段记录以便审计。

## 2. 方法

### 2.1 抽样

从 W2 的评测输出（`outputs/eval/{Qwen2.5-1.5B-Instruct, EXP-W02-01-...-step116}`）抽样，`seed=42`：

| 数据集 | 基座 | RL checkpoint | 合计 |
|---|---|---|---|
| GSM8K（pass@1，每条 1 个生成） | 150 | 150 | 300 |
| MATH-500（pass@1） | 60 | 60 | 120 |
| **合计** | 210 | 210 | **420** |

AIME 没有纳入三判分器对比：verl 的内置判分对 AIME 的 `data_source` 走另一条分支，无法与 GSM8K 规则直接比较；
AIME 的回归结果见 §5。

### 2.2 三个判分器

| 代号 | 实现 | 规则 |
|---|---|---|
| `verl` | `verl/utils/reward_score/gsm8k.py`（v0.9.1, `1876b06`） | 取最后 300 字符里最后一个 `#### (-?[0-9.,]+)`，去逗号后**字符串相等**才算对 |
| `mathverify` | W2 `eval/harness/scoring.py` @ `93b9682`（逐行复刻在 `analysis/verifier_compare.py`） | 整段输出 + `####` 片段 + `\boxed{}` 包装都交给 Math-Verify，**任一候选命中**即算对 |
| `strict` | `src/selfevolve/verifier/math_verifier.py` | 只认**最终答案**（boxed → `####` → 收尾句 → 结论句）；多候选/截断/无答案一律判错 |

`verl` 在装有 verl 的机器上直接 import，否则用逐行复刻版（`analysis/verifier_compare.py::_verl_extract_solution`）。

### 2.3 人工核对口径

对每条样本问一个问题：**"模型最终给出的答案对不对？"**（不是"这个判分器说得对不对"）

- 无 marker 但有一句明确结论（"Therefore, Janet makes $18 ..."）→ **算模型对**；
- 正确数字只出现在推理中、最终答案写错（"…= -50 … #### 50"）→ **算模型错**；
- 输出被 `max_tokens` 截断、没给出最终答案 → **算模型错**（任务包要求截断判错）；
- 判分器判对了 → 记为一致。

自动化部分（96/110）：strict 抽取的最终答案与 gold **归一化后字符串完全相等**，这类不需要人工判断；
其余 14 条逐条写在 `analysis/review_adjudication.py::MANUAL`。这套自动规则只用来生成初判草稿
（`results_agent`）和挑出需要人看的样本，**验收数字以人工核对的 `results_human` 为准**。

## 3. 判定模式分类（420 条抽样全体）

> 下表是**全体 420 条**的模式分布，不是 110 条分歧的细分：
> 其中 `(0,1,1)`、`(None,1,0)`、`(0,1,0)`、`(1,1,0)`、`(0,0,1)`、`(None,0,1)`
> 六行加起来 = **110 条分歧**（100+6+1+1+1+1）；`(1,1,1)` 131 / `(None,1,1)` 60 /
> `(0,0,0)` 66 / `(None,0,0)` 53 这四行是三家一致的 310 条。
> 另外注意：**strict 与 mathverify 在 420 条里只分歧 10 条**，剩下 100 条分歧全是 verl 漏判 ——
> 所以这 110 条里真正有判别力的只有那 10 条，加上 `MANUAL` 里需要人判断的 14 条。

| 模式 (verl, mathverify, strict) | 条数 | 含义 | 谁对（人工核对） |
|---|---|---|---|
| (0, 1, 1) | 100 | GSM8K 基座模型不写 `####`，verl 直接判 0；另两家抽取到了正确答案 | 模型对：verl 错，mv/strict 对 |
| (None, 1, 1) | 60 | MATH-500 同样情况（verl 不适用） | — |
| (0, 0, 0) / (None, 0, 0) | 66 / 53 | 三家都判错 | 都对 |
| (None, 1, 0) | 6 | mv 判对而 strict 判错 | 5 条是 **mv 假阳性**（模型最终答案确实错，strict 对）；1 条是 `1436`"结论句多值"→ **strict 错**（§1.1） |
| (0, 1, 0) | 1 | `\boxed{L - 8}` vs gold `8`，题目问"短了多少厘米"= 8 | 模型错：**mv 错，strict 对** |
| (1, 1, 0) | 1 | `#### 15` 后跟 `-15`；人工认为模型自相矛盾，判错 | **verl/mv 错，strict 对** |
| (0, 0, 1) | 1 | 结尾 "**10 tablespoons**"（gold 10）；mv 抽到句尾的题面数字 `80` | 模型对：**verl/mv 错，strict 对** |
| (None, 0, 1) | 1 | `#### 144` 后跟一句解释导致 mv 解析失败 | strict 对，mv 错 |

合计（人工口径，复核后）：**strict 错 1 / 对 109；mv 错 9 / 对 101；verl 在 103 条可比样本里只对 1 条。**

典型例子：

- **verl 漏判（100 条里最常见）**：基座模型结尾是 `Therefore, Kayla needs 12 more ...`，
  verl 只在最后 300 字符里找 `#### 数字`，找不到就 0 分；strict 从结论句里抽到 12。
- **W2 口径的"宽松救援"**：`…= -50 …\n#### 50`（模型把负号丢了）。W2 把整段交给 Math-Verify，
  在推理里匹配到 `-50` 判对；strict 只看 `#### 50`，判错。这类在 ckpt 的 MATH-500 上有 32 条。
  `(None,1,0)` 的 6 条里有 5 条属于同一类：**mv 把"推理里的数字"当成答案，是假阳性**。
- **Math-Verify 对结构化答案的误判**：`\boxed{6}` vs gold `(-1,6)`、`\boxed{3/3}` vs `\frac{\sqrt3}{3}`
  被 W2 口径判对，人工确认模型确实错，strict 判错是对的。
- **截断**：`finish_reason=length` 的样本 W2 仍按文本判分，strict 按任务包要求判错
  （AIME24 基座 15.6% 的生成都被截断）。
- **strict 唯一的一次判错**（§1.1）：结论句里同时出现题面数字 56 和答案 6，strict 不猜 →
  `no_answer`。这是 §6 限制 #2，是"宁严勿松"的代价，不是抽取 bug。
- **人工核对自己也会踩这个坑**：首轮回填把 `L-8`（推理里算出过 8）判成模型对、
  把结尾明写 `10 tablespoons` 的样本判成模型错 —— 恰好是"正确数字只在推理里"
  和"被抽取器误导"这两种形态。复核时逐条回看**题面**才纠正过来。
  这正说明判分口径必须写死在代码里，靠人逐条看同样会漂。

## 4. 耗时（任务包要求：单样本平均耗时与 p99）

`python analysis/verifier_bench.py --n 500`（bigdog01，500 条真实 GSM8K 生成）：

| 路径 | 平均 | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| 主线程（`signal.setitimer`，评测脚本走这条） | **0.54 ms** | 0.04 ms | 0.90 ms | 10.4 ms | 67.5 ms |
| 线程（进程池 + `future.result(timeout)`，训练 reward 走这条） | **1.05 ms** | 0.11 ms | 1.59 ms | 25.5 ms | 159 ms |

- 训练步 320 个序列：`320 × 1.05 ms ÷ 8 workers ≈ 0.04 s/step`，相对 W2 的 162 s/step 是 **0.02%**，
  **不需要额外的多进程并行**（verl 自带 `reward.num_workers=8` 已经够）。
- 实测训练日志里 `timing_s/agent_loop/compute_score/mean` 在**第 1 步是 29 ms**（max 0.53 s）：
  这是 8 个 reward worker 各自第一次调用时**进程池冷启动**（spawn + import math-verify）的摊销，后续步回落到毫秒级。
  结论不变，但记录在此，避免下次误判成"判分变慢"。

## 5. 回归：W2 结果在严格判分下的变化

用 `eval/rescore.py --suffix strict` 对 W2 的生成结果重打分（**零 GPU**，不重新生成）。
`results_master.md` 追加了新行，旧的 math-verify 行原样保留。

| 模型 | 数据集 | W2（math-verify 0.9.0） | 严格 verifier | Δ | W2对/严格错 | W2错/严格对 |
|---|---|---|---|---|---|---|
| 基座 | GSM8K | 0.7180 | **0.7149** | −0.31 | 15 | 11 |
| 基座 | MATH-500 | 0.5520 | **0.5400** | −1.20 | 7 | 1 |
| 基座 | AIME24 | 0.0292 | **0.0271** | −0.21 | 1 | 0 |
| 基座 | AIME25 | 0.0083 | **0.0063** | −0.20 | 1 | 0 |
| RL ckpt116 | GSM8K | 0.7665 | **0.7642** | −0.23 | 5 | 2 |
| RL ckpt116 | MATH-500 | 0.5640 | **0.4980** | **−6.60** | **36** | 3 |
| RL ckpt116 | AIME24 | 0.0271 | **0.0271** | 0 | 0 | 0 |
| RL ckpt116 | AIME25 | 0.0021 | **0.0000** | −0.21 | 0 | 0 |

### 5.1 最重要的一条：MATH-500 的 base→RL 差值方向反了

| 口径 | 基座 | RL ckpt116 | 差值 |
|---|---|---|---|
| W2（math-verify） | 0.5520 | 0.5640 | **+1.20** |
| 严格 verifier | 0.5400 | 0.4980 | **−4.20** |

- 差值 stderr = √(0.0223² + 0.0224²) = **±3.16 分**，−4.2 约 1.3σ，**统计上不显著**，
  所以正确表述是"**严格判分下没有观察到 MATH-500 的提升**"，而不是"RL 让 MATH-500 变差了"。
- W2 周报里的 "+1.2 分（没有回退）" 依赖宽松口径：把"最终答案写错、但推理里出现过正确数字"的样本判对。
  ckpt 在 MATH-500 上有 **36 条**这样的样本（32 条 `wrong` + 4 条截断），占总题数的 7.2%。
  典型形态：推理算出 `-50`，最终写 `#### 50`；算出 `3√13`，最终写 `#### 3`（模型把答案从 LaTeX 里截断了）。
- 这不是判分器的 bug，而是 **RL 在域外数学题上的格式退化**：模型学会了 GSM8K 的 `####` 形式，
  但在 MATH-500 上常只把答案的第一个 token 写进 `####`。这一点对 W6（自我进化）很重要——
  训练信号必须能看见这种退化，宽松判分会把它抹平。

### 5.2 GSM8K 的结论是稳的

基座 0.7149 → RL 0.7642，**+4.93 分**（W2 口径是 +4.85）。两种判分器给出一致的 +4.8～+4.9，
说明这个提升不是判分口径造出来的。

## 6. 已验证的边界与已知限制

| # | 限制 | 影响 | 处置 |
|---|---|---|---|
| 1 | `#### 15\n-15` 这类"最后一行是另一个值"的写法，strict 取最后一行 | 抽样里 1 条（模型自身矛盾） | **人工核对确认这条 strict 判对了**（模型自相矛盾 = 判错），verl/mv 在这条上才是错的。规则"最后一个声明优先"保留，与 verl 内置判分一致 |
| 2 | 结论句里有多个值时放弃抽取（`56 Drinkets are equivalent to 6 Trinkets`，gold 6） | 抽样里 1 条，**是 420 条里 strict 唯一的判错**（§1.1） | 宁严勿松：不猜。若改成"取结论引导词后的值"可救回这条，但会把"推理里的数字"也放进来，不改 |
| 3 | `\begin{pmatrix} ... \\ ... \end{pmatrix}` 多行矩阵 Math-Verify 解析不了 | 极少 | 记录；不为此改写判分逻辑 |
| 4 | gold 带面积/体积上标单位（`864 \mbox{ inches}^2`）时可能判错 | 极少 | 记录 |
| 5 | `%` 按**单位**处理（`50\%` == `50`，不是 `0.5`） | 影响所有百分数题 | 有意为之：数据集 gold 全是裸数字（W2 数据里 2.9%/4.2% 的生成含 `%`，对应 gold 都是数字）。若改成 ÷100，GSM8K 会出现系统性假阴性 |
| 6 | 没有最终答案标记的散文，用"结论句"兜底（`Therefore … is X` / 强调 `**X**` / 系动词后第一个值） | 基座 GSM8K 靠它从 0.382 拉回 0.715 | 见 §7 |
| 7 | 截断一律判错（任务包要求） | AIME 影响最大（15.6% 截断） | 敏感度：若改成"有完整答案就判分"，AIME24 基座 +0.4 分（1 条），不影响任何结论 |

## 7. 设计取舍：为什么要有"结论句"这层兜底

只认 `\boxed{}` / `####` / "The answer is" 时，**Qwen2.5-1.5B-Instruct 有 45% 的 GSM8K 输出抽不到答案**，
base GSM8K 会从 0.72 掉到 **0.38**——那不是模型变差，是"格式不合规"被误报成"能力不行"。

加了"结论句"兜底（必须是最后一句、必须有结论引导词或是短句、且句内只有一个可判定的值）之后：

| 版本 | 基座 GSM8K | 基座 MATH-500 | ckpt GSM8K | ckpt MATH-500 |
|---|---|---|---|---|
| 只认 marker（中途版本，未加归一化修补） | 0.373 | 0.482 | 0.7627 | 0.474 |
| + 归一化/多答案修补 | 0.382 | 0.502 | 0.7627 | 0.486 |
| + 结论句兜底（当前） | **0.7149** | **0.5400** | **0.7642** | **0.4980** |
| W2 口径（参照） | 0.7180 | 0.5520 | 0.7665 | 0.5640 |

兜底没有把"推理里的数字"变成分数：`We compute 42 in the middle. #### 7` 仍然判错（有 marker 时结论句不参与），
`The total is 42 dollars. The answer is \boxed{7}` 也判错。这一点在 `tests/test_math_verifier.py`
的对抗样本里有专门用例。

## 8. 产物与复现

| 产物 | 说明 |
|---|---|
| `reports/verifier_review_pack.md` | 人工核对包（110 条分歧，含题面/金标/输出尾部/三家判定/agent 初判与理由/**已填好的人工结论**） |
| `reports/verifier_review_pack.csv` | 420 行 × 17 列，可直接在 Excel 里核对（`utf-8-sig`，打开不乱码）。`verl`/`mathverify`/`strict` 三家判定（verl 对非 GSM8K 写 `n/a`）、`agent_verdict`/`agent_note` 初判与理由、`human_verdict`/`human_note` 留给人填 |
| `analysis/review_adjudication.py` | 判定口径的单一实现：agent 初判、CSV 列定义、`human_verdict` 解析（两个脚本共用） |
| `reports/verifier_review_pack.jsonl` | 含完整响应，供回查 |
| `reports/verifier_accuracy.json` | 本报告 §1/§3 的机器可读版本 |
| `reports/verifier_bench.json` | §4 的耗时数据 |

```bash
# 重新生成对比 + 核对包（读 W2 的 outputs/eval，需要 bigdog01 上的生成结果）
python analysis/verifier_compare.py --n 300 --math-n 120 --seed 42 --out reports/verifier_review_pack
# 只重建核对包的 CSV/MD（不重新打分、不需要 outputs/eval，也不需要装 math-verify）
python analysis/verifier_compare.py --from-jsonl reports/verifier_review_pack.jsonl     --out reports/verifier_review_pack
# 人工核对：填 reports/verifier_review_pack.csv 的 human_verdict 列，然后刷新准确率
# （默认就会去读那个 CSV；--no-human 只看 agent 初判）
python analysis/verifier_accuracy.py
# 耗时
python analysis/verifier_bench.py --n 500
# 用严格 verifier 重打分（零 GPU）
python eval/rescore.py --run-dir outputs/eval/<run> --suffix strict --out outputs/eval/<run>/summary_strict.json
```

## 9. 人工核对包入库时的编码处理（WPS / Excel）

核对包是 CSV，但桌面表格软件会做两件破坏性的事，入库前必须处理：

| 现象 | 原因 | 处理 |
|---|---|---|
| 存成 GBK，`read_human_verdicts` 报 `UnicodeDecodeError` | WPS 表格"另存为 CSV"用系统 ANSI（GBK）；Excel 选错格式同理 | 重建成 **`utf-8-sig`** |
| 数值单元格被改写：`2180.`→`2180`、`$1430`→`$1,430 `、`5.00`→`5`、`000` 等 | 表格软件把看起来像数字的单元格按自己的格式重写 | **只取人工填的两列**（`human_verdict` / `human_note`），其余列全部用 `git show HEAD:reports/verifier_review_pack.csv` 的原始内容按 `sample_id` 拼回 |
| 行序被排序过 | 人工操作 | 以 `sample_id` 为键合并，不按行号；合并后**行序恢复为原文件的顺序**（Excel 行号 = 原文件行号，便于回溯） |
| 出现 `000` 这种值 | 手输/格式残留 | 按 `0` 处理并在合并时打印出来（本次仅 1 处） |

本次入库的具体做法（可复核，未入库的一次性脚本）：

1. `git show HEAD:reports/verifier_review_pack.csv` 取原始 420 行（UTF-8）；
2. 读人工文件（先试 `utf-8-sig`、失败再试 `gbk`），建立 `sample_id -> (human_verdict, human_note)`；
3. 用原始行 + 人工两列写回 `utf-8-sig`，断言 420 行、`sample_id` 集合完全一致；
4. 跑 `python analysis/verifier_accuracy.py`，`human_review.status` 应为 `complete`。

`human_verdict` 的合法值：`1`/`0`（也接受 `对`/`错`、`true`/`false`、`yes`/`no`），空 = 未核对；
无法识别的值会**报错并列出全部坏行**（`HumanVerdictError`），不会静默跳过 —— 避免"填了但没被统计"看起来像通过了。
