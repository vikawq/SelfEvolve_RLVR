# Week 02 Summary

周期：2026-10-21。硬件：RTX 4080 16GB（bigdog01，单卡；3090 本周未到位）。
实验：`EXP-W02-01-baseline-gsm8k`、`EXP-W02-02/03/04`（profiling）。
代码：`configs/`、`eval/`、`analysis/parse_profile.py`、`scripts/run_eval.sh`。

## 完成情况（对照验收标准逐条打勾）

- [x] **baseline 与官方差距 ≤ 2 分**：官方 77.9（verl rule reward），本实验独立 harness
  GSM8K **76.65**，差 **1.25 分**。训练内 validation（同官方口径）75.82，差 2.09 分；两套口径都如实报告。
- [x] 用一条命令对任意 checkpoint 跑完全套评测：`bash scripts/run_eval.sh <model-or-ckpt>`
  （自动 merge LoRA → vLLM 离线 → Math-Verify → JSONL + summary）。
- [x] `results_master.md` 中有基座和 baseline 两行（实际 8 行：2 模型 × 4 数据集）。
- [x] `profiling_baseline_4080.md` 已完成（三组参数扫描结果见该报告 §5）。
- [x] 周报已完成（本文件）。

## 关键结果

### 训练（verl 内置验证循环，rule reward，**非** harness）

| step | 20 | 40 | 60 | 80 | 100 | 116 |
|---|---|---|---|---|---|---|
| 4080 baseline acc@1 | 0.7384 | 0.7400 | 0.7491 | 0.7521 | 0.7309 | **0.7582** |
| 官方 2×H100 acc@1 | 0.747 | 0.761 | 0.768 | 0.779 | 0.776 | **0.779** |

- 训练 116 步（1 epoch），96 个续跑步平均 **162.0 s/step**；总训练墙钟约 **4.4 h**。
- wandb：calibration（20 步）`runs/h3eaiogf`，full run `runs/hlrzg966`。
- checkpoint：`checkpoints/EXP-W02-01-baseline-gsm8k/global_step_{10,20,40,60,80,100,116}`（每个 3.6GB）。
- 续跑证据：full run 日志 `Loaded model/optimizer/rng from .../global_step_20`，`training/global_step` 从 21 开始。

### 独立评测 harness（Math-Verify 0.9.0；修复 `####` 重复标记后重打分）

| 模型 | GSM8K pass@1 | MATH-500 pass@1 | AIME24 avg@16 | AIME25 avg@16 |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct（基座） | 0.7180 | 0.5520 | 0.0292 ± 0.1035 | 0.0083 ± 0.0271 |
| baseline `global_step_116`（merged） | **0.7665** | **0.5640** | 0.0271 ± 0.0670 | 0.0021 ± 0.0114 |

- GSM8K 相对官方 77.9 差 **1.25 分**；MATH-500 相对基座 **+1.2 分**（没有回退）。
- AIME 只有 30 题、每题 16 次采样；base→RL 的变化（-0.2 / -0.6 分）在噪声范围内，不能作为结论。
- 采样：GSM8K/MATH greedy（temperature=0, top_p=1, n=1）；AIME temperature=0.6, top_p=0.95, n=16。
- 判分器：`math-verify==0.9.0`；每行结果标了判分器版本与修复 commit。

### Profiling（4080，116 步中位数）

| 阶段 | 秒 | 占比 |
|---|---|---|
| update_actor | 114.46 | 70.5% |
| old_log_prob | 27.65 | 17.0% |
| ref | 9.78 | 6.0% |
| gen | 9.09 | 5.6% |
| update_weights | 1.58 | 1.0% |

- 单步 **162.39 s**，吞吐 600 tok/s，actor 峰值 9.28 GiB。
- 官方 2×H100 单步 56.2 s，其中 gen 45 s（80%）；4080 瓶颈在训练侧（93.5%）。
- **P1 已验证**：micro batch 1→2，单步 169.68 → **107.87 s（1.57×）**，吞吐 697 → 1092 tok/s，
  显存 9.28 → 11.91 GiB；micro=4 OOM（差 2.39 GiB）。
- util 0.30→0.35 只有 ~2% 收益；rollout.n 与单步近似线性、吞吐不变。详见 `reports/profiling_baseline_4080.md` §5–§7。

## 与计划的偏差及原因

1. **官方 baseline 的脚本/日志考证**：表格链接的日志已从 verl-data 的 `experiments` 分支删除，
   本次从 PR #3 的 ref 取回；文件名 `bsz64_2` 的 `_2` 是 **2 张卡**，不是 `rollout.n=2`。
   实际 `n=5`（有效 batch 320 seq/step）、1 epoch=116 步。见 `docs/troubleshooting.md` §4.1。
2. **test_freq=20**（官方 5）：只影响监控频率；训练内曲线点比官方少（20 步一个点）。
3. **max_num_batched_tokens=8192**（官方 1536）：纯调度参数，rollout 显著加速（45 s → 8.8 s）。
4. **micro batch=1**（官方 64/卡）：16GB + 1024 response 的硬件约束；这是训练侧慢的主因，不改数学。
5. **无 flash-attn**：SDPA + remove_padding=False；W1 已记录，W9 再评估。
6. **LoRA 同步走 merge**：vLLM 0.24 的 add_lora 兼容性，W1 已记录。
7. **3090 未到位**：任务包 §5 的两卡对比未执行，不编造。

## 失败与问题（附错误信息与已尝试的方法）

| # | 现象 | 原因 | 解法 | 代价/影响 |
|---|---|---|---|---|
| 1 | 首次 20 步校准命令行是 `max_response_length=512`、`logprob_micro=2` | baseline 配置 `source configs/hw/4080.env` 后，`${MAX_RESPONSE_LEN:-1024}` 被 W1 冒烟变量覆盖 | W2 专用变量（`W02_*`、`BASELINE_*`）+ 启动日志打印四项参数 | 一次无效 wandb run `eu4gz4yj`（约 1 步，aborted） |
| 2 | 评测卡在 `huggingface.co` HEAD 超时（重试 5 次） | `transformers` 在 `_load_hf` 之前就 import 了 `huggingface_hub`，endpoint 已固化；`os.environ.setdefault` 太晚 | `run_eval.py`/`run_eval.sh`/`data.py` 在 import 前 export `HF_ENDPOINT` + `HF_HUB_DISABLE_XET=1` | 浪费约 10 min GPU + 一次手动重启 |
| 3 | **判分 bug**：`#### 460 ####` 被抽成空答案；初版 GSM8K 0.7635、MATH-500 0.5040 | 旧抽取器取"最后一个 `####` 之后"的文本，重复标记导致空串 | 按 `####` 切分取最后一个非空片段；单独 commit；用保存的生成结果重打分（`eval/rescore.py`） | GSM8K +0.3 → 0.7665；MATH-500 +6.0 → 0.5640；原始 JSON/summary 保留可审计 |
| 4 | merge 前后 20 题 greedy 只有 1/20 完全一致 | vLLM 的 LoRA runtime kernel 与 merged 权重 kernel 不同，长链推理因浮点差异级联分叉 | 用权重级检查证明 339/339 参数 bit-identical；记录共同前缀/准确率 | 见下方"merge 一致性" |
| 5 | `EXP-W02-02` util=0.25 vLLM 起不来 | 16GB + 1536 总长，KV cache 不足 | 保留失败记录，继续 0.30/0.35 | 约 2 min GPU；数据点已入报告 |
| 6 | `EXP-W02-04` micro=4 OOM | 1024 response × 4 seq，显存差 2.39 GiB（仅剩 840 MiB） | 记录并停止该取值；micro=2 已足够 | 约 3 min GPU |

### merge 一致性（重要）

- **权重级（权威）**：`peft.merge_and_unload()` 后 **339/339 参数与 merged 模型 bit-identical**
  （max|Δ|=0，missing=0）；同一进程内 `unloaded` vs `merged` 的 20 条 GSM8K prompt logits 也
  **max|Δ|=0**。
- **生成级（vLLM，仅供参考）**：PEFT runtime adapter 与 merged 的 20 题 greedy 输出 1/20 完全一致，
  共同前缀中位数 235 字符、15/20 ≥100 字符；准确率 14/20 vs 15/20（1 题之差）。
- 结论：merge 权重正确；生成差异来自 LoRA runtime kernel（不同浮点累加）与自回归级联，不是合并错误。
  所有上报分数都用 **merged** 模型，避免 runtime 差异。
- 注意：CPU bf16 的 PEFT runtime forward 本身跨进程不确定（同一权重 logit 差 0–10），
  所以判据必须是"unloaded vs merged"的权重/同进程 logits，而不是 runtime adapter 的生成。

## GPU 时长 / 花费

| 项 | GPU·h |
|---|---|
| 20 步校准 | 0.96 |
| 全量 116 步（96 步续跑，162 s/step） | 4.43 |
| 评测（基座 + checkpoint + merge 校验） | ~0.20 |
| Profiling（`EXP-W02-02` 1.02 + `03` 1.69 + `04` 0.86） | 3.57 |
| **合计** | **9.16** |

花费：**$0**（本地机器；未租云、未调用付费 API）。

## 给下周的交接

- **baseline 路径已稳定**：`configs/EXP-W02-01-baseline-gsm8k.sh` 可直接复用；续跑用 `resume_mode=auto`。
- **评测统一入口** `bash scripts/run_eval.sh <ckpt>`：自动 merge LoRA → vLLM 离线 → Math-Verify。
  口径见 `eval/datasets.md`；W3 换严格 verifier 后要整体重评。
- **Profiling 结论**：W9 的第一优先级是训练侧 micro-batch（P1，**实测 micro=2 单步 1.57×**，
  169.68 → 107.87 s），其次是复用 rollout logprobs（P2），最后是 flash-attn/kernel（P3）。
  micro=4 已 OOM，需先做 P3。详见 `reports/profiling_baseline_4080.md` §7。
- **风险提示**：
  - 训练内 validation 与 harness 口径不同：**差异只在判分器与答案抽取**（verl 内置 GSM8K rule
    reward 只比 `####` 后的数值 vs Math-Verify + 本项目抽取器），**不在采样**——两者都是 greedy。
    证据：本项目所有配置都未覆盖 `rollout.val_kwargs`，verl `1876b06` 的默认值是
    `do_sample=False, temperature=0`（顶层 `rollout.temperature=1.0` 是训练 rollout 的，不是验证的）。
    W2 初稿曾把验证记成 `temperature=1.0`，已更正，见 `docs/troubleshooting.md §4.9`。
  - GSM8K 训练内 75.82 vs 官方 77.94（-2.1）略超 2 分，harness 76.65 vs 77.9（-1.25）达标；
    差异可由单种子噪声（1319 题 binomial stderr ±1.16–1.24 分，-2.09 约 1.8σ）与系统差异
    （micro-batch / attention kernel / 调度参数）解释，但没有第二种子，不能下定论。
  - AIME 只有 30 题，base/RL 都在 0–3% 量级，暂不能用于区分。
  - MATH-500 没有回退（+1.2），但 GSM8K RL 对更难的 MATH 的提升有限，W6 需要关注通用能力。
- **3090**：到位后跑 `HW=3090` 的 baseline 冒烟 + profiling，回写 `configs/hw/3090.env`。

## 人工决策门：已批准（2026-10-22）

计划里的"人审阅两件事：baseline 是否对齐、评测口径是否合理"已完成，三问全部获批，附带条件如下。
本节即为协议要求的批准记录。

1. **baseline 对齐 —— 接受，不追加种子。** 结论口径：以 harness 为主（76.65，差 1.25，达标），
   训练内 75.82（差 2.09）两套并列报告，并在 `results_master.md` 明确标注 -2.09 落在单种子
   binomial stderr（±1.16–1.24 分）之内、约 1.8σ，不下"显著不同"的结论。
   理由：验收标准问的是 harness 分数；项目的研究主张是"相对基座的提升"（71.80 → 76.65，+4.85），
   不是与 verl 表格的绝对对齐；4.5 GPU·h/种子只换来 ±0.8 分误差棒。
   **替代方案（零关键路径成本）**：3090 到位后用 seed=1234 跑 116 步，与 4080 上的 W3 工作并行。
   多种子要求保留给 W4/W7 的处理效应结论（协议 §3 的"下统计结论"指那里）。
2. **评测口径 —— 批准，附 4 条修正（已落地）**：
   - GSM8K prompt 逐字复用训练 parquet；MATH-500/AIME 用同一 `####` 指令（口径对称，不做
     per-dataset prompt 调优）。**代价必须声明**：MATH/AIME 的绝对分数被系统性压低，
     不可与基于 `\boxed{}` 的公开榜单比较，只有本 harness 内 base↔RL 的差值有效
     （已写入 `eval/datasets.md §6` 与 `results_master.md`）。
   - GSM8K/MATH greedy pass@1；AIME avg@16（temp 0.6 / top_p 0.95）—— 原样批准。
   - `max_tokens` 1024/2048 批准，但 **`truncation_rate` 必须随分数一起上报**；
     `results_master.md` 已加 `截断率` / `空答案率` 两列。
   - **`max_model_len` 不再固定 2560**：vLLM 在 `prompt + max_tokens > max_model_len` 时不报错，
     只会静默压缩生成预算（截断率失真）。`run_eval.py` 改为按每个数据集的
     `最长 prompt + max_tokens` 推算（`--max-model-len 0` = auto），显式传值不足时直接报错。
   - 误差口径统一：`pass@1` 报 binomial stderr，`avg@k` 报 `std/sqrt(30)`，
     `summary.json → reported_stderr_kind` 标明，不混用。vLLM util 0.85 原样批准（纯吞吐参数）。
3. **W3 修改 `eval/` 判分实现 —— 批准**（协议 §4 本来就要求训练奖励与评测共用同一严格 verifier），
   附 4 条执行要求：
   - 严格 verifier 单独 commit，registry 注明 commit hash（W2 的 `886f77b` 是范例）；
   - `results_master.md` **追加新行、不覆盖旧行**，保留 Math-Verify 0.9.0 的行用于对比审计；
   - 必须产出**逐样本分歧报告**（Math-Verify 判对/严格判错，及反向）。这批分歧样本同时是 W4
     复现 reward hacking 的现成候选集；
   - 重评走 `eval/rescore.py` 从保存的生成重打分，greedy 部分零 GPU 成本。

## 遗留给 W3 的小尾巴

- **AIME 两行要先复核长度预算**：W2 的 AIME 结果是在旧的固定 `max_model_len=2560` 下跑的，
  而 AIME `max_tokens=2048`，只给 prompt 留了 512 token。超过就会被 vLLM 静默压缩生成预算。
  复核零 GPU：`python eval/run_eval.py --model <m> --datasets aime2024,aime2025 --plan-only`；
  `need > 2560` 时用 auto 预算重跑这两个数据集（分钟级）。GSM8K/MATH-500（`max_tokens=1024`，
  prompt 预算 1536）不受影响。
- `results_master.md` 的 `截断率` / `空答案率` 两列为 `待回填`：数据在 bigdog01 的
  `outputs/eval/*/summary.json`（`outputs/` 不入 git），用 `analysis/emit_results_row.py`
  一条命令回填，不需要 GPU。本机读不到那批产物，所以没有填数字。
- MMLU-Pro（可选回归检查）代码与口径已固定，W2 未跑。
- 3090 未到位，两卡 profiling 对比未执行。

---

## 后续更正（2026-10-22，W3 追加）

> 本节是**勘误**，W2 正文一字未改（保留审计链）。以下更正优先于正文中被点名的那几句。

### 更正 1：MATH-500 的 base→RL 结论反转

W3 的严格 verifier（`src/selfevolve/verifier/math_verifier.py`，
`selfevolve-strict-verifier-1.0+math-verify-0.9.0`）对**同一批生成结果**重打分（零 GPU，未重新生成）后：

| 模型 | W2 口径（math-verify 0.9.0） | 严格口径 | Δ |
|---|---|---|---|
| 基座 | 0.5520 | **0.5400** | −1.20 |
| RL `global_step_116` | 0.5640 | **0.4980** | **−6.60** |
| **base→RL 差值** | **+1.20** | **−4.20** | 方向反转 |

**作废**：

- 第 38 行 "MATH-500 相对基座 **+1.2 分**（没有回退）"；
- 第 124 行 "MATH-500 没有回退（+1.2），但 GSM8K RL 对更难的 MATH 的提升有限"。

**正确表述**：**严格判分下没有观察到 MATH-500 的提升**。差值 −4.20 的 stderr ≈ ±3.16 分
（√(0.0223² + 0.0224²)），约 1.3σ，**统计上不显著** —— 所以也**不能**反过来说"RL 让 MATH-500 变差了"。
单种子，等更多种子才能判断。

**原因不是判分器 bug**，而是 RL checkpoint 在域外 MATH-500 上的**格式退化**：有 **36/500 条**
（7.2%；32 条判错 + 4 条截断）属于"最终答案写错、正确数字只出现在推理里"，
典型形态是推理算出 `-50` 却写 `#### 50`、算出 3√13 却只把第一个 token 写进 `####`。
W2 把整段输出交给 Math-Verify、任一候选命中即算对，于是在推理文本里匹配到正确数字，把这 36 条判成了对。

**GSM8K 的结论不受影响**：基座 0.7149 → RL 0.7642 = **+4.93**（W2 口径 +4.85），两套判分器一致。

**出处**：`reports/verifier_validation.md` §5.1（逐条差异与典型样本）、
`reports/results_master.md`（严格口径的新行与说明，旧行保留）、`reports/week03_summary.md` §3。

### 对后续各周的约束

- 任何涉及 base ↔ RL 差值的结论，**不得**使用 W2 的 math-verify 口径；
- 报告"提升"时必须声明判分器版本；
- W6 及之后的材料里**不要**把"MATH-500 提升"写进结论。
