# Week 02 执行计划与风险

周期：2026-10-21 起。硬件：RTX 4080 16GB（bigdog01，单卡，本轮无 3090）。
上游依据：`weeks/week02_baseline_eval_profiling.md`；协议 `02_AGENT_PROTOCOL.md`；W1 交接 `reports/week01_summary.md`。

## 0. 开工前已核实的上游事实（与任务包的差异先记录）

任务包要求"在 verl 文档 Algorithm Baselines 找到 77.9 分对应的脚本和超参"。已从官方仓库考据到：

| 项 | 官方值 | 来源 |
|---|---|---|
| 表格分数 | Qwen2.5-1.5B-Instruct + GRPO-LoRA，GSM8K **77.9** | `docs/algo/baseline.md`（v0.9.1 内置同表；该表 2025-06-18 由 PR #2083 加入） |
| 参考脚本 | `examples/tuning/1.5b/qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh` | verl commit `368dff7`（表格加入时的版本） |
| 实际运行日志 | `Qwen2.5-1.5B-bsz64_2-prompt512-resp1024-lorarank32-score0.779.log` | `eric-haibin-lin/verl-data` PR #3（commit `bcff590`）；当前 `experiments` 分支已删除该文件，本次从 PR ref 取回，存于 `third_party/refs/` |
| 实际 batch | `train_batch_size=64` prompts/step × `rollout.n=5` = **320 sequences/step** | 日志开头 `script:` + 落盘 config（`'train_batch_size': 64` / `'n': 5`） |
| 步数 | `total_epochs=1`，7473/64 → **116 步**（drop_last=True） | 日志 + `ray_trainer.py:409` |
| 超参 | lr=3e-5，LoRA r=32/alpha=32/all-linear，KL loss 0.001（low_var_kl），entropy_coeff=0.001，prompt 512 / response 1024，shuffle=False | 日志 |
| 硬件 | 2×H100，vLLM util 0.1，TP=2，layered_summon=True | 日志 |
| 最终分数 | step 116 validation `0.77938`（verl 内置 GSM8K rule reward，greedy n=1） | 日志末尾 |

需要记录的差异与判断：

1. **文件名里的 `_2` 不是 rollout.n=2，而是 2 张卡**（1.5B/3B 用 2 卡，7B 以上用 8 卡，见其余日志名）。实际 `rollout.n=5`。本实验按日志配置复现，避免把有效 batch 从 320 误改成 128。
2. 官方脚本是 **1 卡 H100 的 tuning 版**（`nproc_per_gpu=128`、`n=5`、1 epoch），与日志里的 2 卡版超参一致（batch 总数仍 64）。单卡 4080 上按日志的 64 prompts × n=5 对齐。
3. 官方没有单独维护 1.5B 的 GSM8K 脚本；`examples/tuning/1.5b/` 的脚本是唯一的官方实现依据。已存入 `docs/troubleshooting.md` §4。
4. v0.9.1 不再支持官方日志里的 `ppo_micro_batch_size`、`data.val_batch_size` 等旧字段；改用 `_per_gpu` 字段（W1 已记录）。

## 1. 本周目标（对应验收标准）

- [ ] `EXP-W02-01-baseline-gsm8k`：4080 单卡复现，GSM8K test 与 77.9 差距 ≤ 2 分（或完整排查记录）。
- [ ] `eval/` harness：一条命令评任意模型/checkpoint；逐样本 JSONL + 汇总 JSON；Math-Verify 判分；LoRA merge 前后 20 题一致性校验。
- [ ] `results_master.md` 增加基座与 baseline 两行。
- [ ] `reports/profiling_baseline_4080.md`：瓶颈、参数敏感度、3 个最值得优化的点。
- [ ] `reports/week02_summary.md`；停在人工决策门（baseline 是否对齐、评测口径是否合理）。

## 2. 执行顺序（单卡，串行）

| 阶段 | 内容 | 预计 GPU 时长 |
|---|---|---|
| P0 | 计划、配置、registry、评测 harness 代码与单测（不占 GPU） | 0 |
| P1 | 20 步校准（`TOTAL_STEPS=20`），测单步耗时并外推 116 步 | 0.7–1.5 h |
| P2 | 全量训练：从 step 20 续跑到 116（1 epoch） | 4–8 h |
| P3 | 评测：merge LoRA → 基座 + checkpoint 跑 GSM8K/MATH-500/AIME 2024/2025 | 1–1.5 h |
| P4 | 3 组 profiling：vLLM util（0.25/0.30/0.35）、rollout.n（4/8）、micro-bs（1/2/4），每组 30 步（每个取值 10 步） | 3–5 h |
| P5 | 解析 profiling、写报告、更新 registry/results_master、提交 | 0 |

时间估算依据：官方 2×H100 单步 55–68 s（320 seq、~110k tok/step）。4080 按 3–5× 慢估：120–300 s/step → 116 步 3.9–9.7 h。若 20 步实测外推 > 48 h，按任务包要求停下请示，不擅自减步数。

## 3. 关键实现决定

### 3.1 Baseline 配置（`configs/EXP-W02-01-baseline-gsm8k.sh`）

沿用 W1 已验证的稳定组合（`use_v1=False`、`lora.merge=True`、SDPA、bf16、`max_model_len=1536`），只改与 baseline 有关的部分：

| 参数 | 值 | 理由 |
|---|---|---|
| `data.train_batch_size` | 64 | 官方 |
| `rollout.n` | 5 | 官方（有效 batch 320 seq） |
| `ppo_mini_batch_size` | 64 | 官方（prompts 口径） |
| `ppo_micro_batch_size_per_gpu` | 1 | 16GB + 1024 response 的唯一安全值；profiling 再测 2/4 |
| `lr` / `entropy_coeff` | 3e-5 / 0.001 | 官方 |
| `total_epochs` | 1（=116 步） | 官方 |
| `test_freq` | 20 | **偏离**：官方 5。只影响监控频率，不影响训练结果；省约 1 h |
| `save_freq` | 20 | 6 个 checkpoint ≈ 22GB，磁盘足够 |
| `max_num_batched_tokens` | 8192 | **偏离**：官方 1536（2×80GB 保守值）。仅调度参数，不改数学 |
| 其余 | 同 W1 | 16GB 约束 |

### 3.2 评测口径（先定死，见 `eval/datasets.md`）

- GSM8K test（1319）：prompt 直接取训练用 parquet 的 `prompt` 字段（与训练逐字一致），greedy pass@1。
- MATH-500（500）：题干 + 与 GSM8K 相同的 `####` 指令，greedy pass@1。
- AIME 2024 / 2025（各 30）：同模板，temperature=0.6、top_p=0.95、n=16，报 avg@16 与逐题标准差。
- 判分器：`math-verify`（记录版本），`score(response, gt) -> dict`；保留 `####` 抽取作为诊断字段，但**以 Math-Verify 为准**。
- 与训练验证的差异（verl 内置 rule reward、温度 1.0、avg@1）在报告中并列解释，不混用。

### 3.3 Profiling 设计

每组参数 10 步 × 3 个取值 = 30 步/组（任务包"各 30 步"的解释：每组总步数 30，覆盖 3 个取值）。只用于测耗时/显存，不作精度结论。
- `EXP-W02-02-prof-gpu-mem`：0.25 / 0.30 / 0.35（0.25 预期起不来，保留失败记录）
- `EXP-W02-03-prof-rollout-n`：4 / 5 / 8（5 用作与 baseline 对齐的参照）
- `EXP-W02-04-prof-micro-bs`：1 / 2 / 4（OOM 即记录并降级）

## 4. 风险与缓解

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | 1024 response + 320 seq 在 16GB OOM | 训练中断 | micro-bs=1、grad ckpt、logprob micro=1；OOM 先看 dtype/权重副本（W1 教训），不先降有效 batch；仍不行则请示 |
| 2 | 4080 跑 116 步 > 48 h | 超时 | P1 20 步外推；>48h 停下请示（可等 3090/上云），不擅自减步数 |
| 3 | 分数差 > 2 分 | 验收不达标 | 按任务包顺序排查：reward/答案抽取 → prompt → 有效 batch/步数 → lr → 评测口径；差距原因入档 |
| 4 | 官方日志/脚本不可得（文件被删） | 无法对齐 | 已从 PR ref 取回日志并入库说明；脚本来自表格加入时的 commit；差异记录在案 |
| 5 | 评测 harness 与训练验证口径不一致 | 分数不可比 | 报告并列两套数字并解释；harness 完全独立于 verl；人工决策门审阅 |
| 6 | LoRA merge 前后输出不一致 | checkpoint 不可信 | merge 后先跑 20 题逐 token 对比；不一致则查 adapter/alpha/merge 实现 |
| 7 | AIME 仅 30 题、方差大 | 结论不稳 | 报 avg@16 ± 标准差；只作为相对基座的参考 |
| 8 | 数据污染（Qwen 见过 GSM8K） | 绝对分虚高 | 只做相对基座比较；重点看 AIME 2025；报告注明 |
| 9 | HF 镜像下载失败 | 评测阻塞 | `HF_ENDPOINT=hf-mirror.com` + `HF_HUB_DISABLE_XET=1`（W1 已记录）；GPU 任务不依赖下载 |
| 10 | wandb API 读不到 | profiling 解析失败 | `analysis/parse_profile.py` 支持本地日志 fallback；wandb 仅作补充 |
| 11 | 单卡排队（训练/评测/profiling 互斥） | 周内做不完 | 训练优先；评测可复用 checkpoint；profiling 用 10 步短跑；必要时顺延，不降标准 |
| 12 | 3090 本周未到位 | 任务 5 无法做 | 明确记录"未执行"，不编造两卡对比 |

## 5. 红线与请示点

- 本地 GPU，**不花钱**；不租云、不调付费 API。若 P1 外推 > 48 h，或需要云卡，先请示。
- 本周要**新建** `eval/` 评测代码（任务包明确要求）。W3 之前不改训练 reward/verifier；harness 建好后先交人审阅口径。
- 不修改 `weeks/` 任务包与验收标准；不降低步数/batch 换分数。
- 所有实验先登记 registry 再跑；git commit 后开跑，hash 入 registry。

## 6. 本周结束时要交的东西

`configs/EXP-W02-0{1,2,3,4}*.sh`、`eval/`（含测试）、`analysis/parse_profile.py`、`reports/profiling_baseline_4080.md`、`reports/week02_summary.md`、`reports/results_master.md` 两行、`experiments/registry.csv` 四条、`docs/troubleshooting.md` §4、`env/versions.md`（math-verify）。
