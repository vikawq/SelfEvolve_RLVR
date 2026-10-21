# Week 02 Summary

周期：2026-10-21。硬件：RTX 4080 16GB（bigdog01，单卡；3090 本周未到位）。
实验：`EXP-W02-01-baseline-gsm8k`、`EXP-W02-02/03/04`（profiling）。
代码：`configs/`、`eval/`、`analysis/parse_profile.py`、`scripts/run_eval.sh`。

## 完成情况（对照验收标准逐条打勾）

- [ ] baseline 与官方差距 ≤ 2 分，或有完整排查记录。
  - 训练内 validation：step 20 = **0.7384**（官方同时点 0.747，差 0.9 分）；step 116 待填。
  - 独立 harness（Math-Verify，greedy）：待填。
- [x] 用一条命令对任意 checkpoint 跑完全套评测：`bash scripts/run_eval.sh <model-or-ckpt>`。
- [ ] `results_master.md` 中有基座和 baseline 两行：待填。
- [ ] `profiling_baseline_4080.md` 已完成：初稿已完成，参数扫描表待回填。
- [x] 周报已完成（本文件）。

## 关键结果

### 训练（verl 内置验证循环，rule reward，**非** harness）

| step | 20 | 40 | 60 | 80 | 100 | 116 |
|---|---|---|---|---|---|---|
| 4080 baseline acc@1 | 0.7384 | 待填 | 待填 | 待填 | 待填 | 待填 |
| 官方 2×H100 acc@1 | 0.747 | 0.761 | 0.768 | 0.779 | 0.776 | **0.779** |

- wandb：calibration（20 步）`runs/h3eaiogf`，full run `runs/hlrzg966`。
- checkpoint：`checkpoints/EXP-W02-01-baseline-gsm8k/global_step_{10,20,40,...}`（每个 3.6GB）。
- 续跑证据：full run 日志 `Loaded model/optimizer/rng from .../global_step_20`，`training/global_step` 从 21 开始。

### 评测 harness（待填）

| 模型 | GSM8K pass@1 | MATH-500 pass@1 | AIME24 avg@16 | AIME25 avg@16 |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct（基座） | 待填 | 待填 | 待填 | 待填 |
| baseline `global_step_116`（merged） | 待填 | 待填 | 待填 | 待填 |

- 判分器 `math-verify==0.9.0`；AIME 报逐题标准差。
- merge 一致性：logits max|Δ| 待填；20 题 greedy 输出 待填。

### Profiling（4080，116 步中位数）

| 阶段 | 秒 | 占比 |
|---|---|---|
| update_actor | 113.2 | 71% |
| old_log_prob | 27.5 | 17% |
| ref | 9.5 | 6% |
| gen | 7.6 | 5% |
| update_weights | 1.6 | 1% |

- 单步 **159.4 s**，吞吐 596 tok/s，actor 峰值 9.28 GiB。
- 官方 2×H100 单步 56.2 s，其中 gen 45 s（80%）；4080 瓶颈在训练侧（94%）。
- 详见 `reports/profiling_baseline_4080.md`。

## 与计划的偏差及原因

1. **官方 baseline 的脚本/日志考证**：表格链接的日志已从 verl-data 的 `experiments` 分支删除，
   本次从 PR #3 的 ref 取回；文件名 `bsz64_2` 的 `_2` 是 **2 张卡**，不是 `rollout.n=2`。
   实际 `n=5`（有效 batch 320 seq/step）、1 epoch=116 步。详见 `docs/troubleshooting.md` §4.1。
2. **test_freq=20**（官方 5）：只影响监控频率；省约 1 h 验证时间。训练内曲线点比官方少（20 步一个点）。
3. **max_num_batched_tokens=8192**（官方 1536）：纯调度参数，rollout 显著加速（45 s → 7.6 s）；
   不是算法改动。
4. **micro batch=1**（官方 64/卡）：16GB + 1024 response 的硬件约束；这是 4080 训练侧慢的主因。
5. **无 flash-attn**：SDPA + remove_padding=False；W1 已记录，W9 再评估。
6. **LoRA 同步走 merge**：vLLM 0.24 的 add_lora 兼容性，W1 已记录。
7. **3090 未到位**：任务包 §5 的两卡对比未执行，不编造。

## 失败与问题（附错误信息与已尝试的方法）

| # | 现象 | 原因 | 解法 | 代价 |
|---|---|---|---|---|
| 1 | 首次 20 步校准的进程命令行是 `max_response_length=512`、`logprob_micro=2` | baseline 配置 `source configs/hw/4080.env` 后，`${MAX_RESPONSE_LEN:-1024}` 被 W1 冒烟变量覆盖 | 新增 W2 专用变量（`W02_*`、`BASELINE_*`），启动日志打印四项参数 | 一次无效 wandb run `eu4gz4yj`（约 1 步，aborted） |
| 2 | `EXP-W02-02` util=0.25 预期 vLLM 起不来 | 16GB + 1536 总长，KV cache 不足 | 保留失败记录，继续 0.30/0.35 | 待填 |
| 3 | `EXP-W02-04` micro=4 可能 OOM | 16GB 显存 | 失败即记录并回退 | 待填 |
| 4 | 首次 HF 数据集下载（MATH-500 等） | 直连超时 / Xet 401 | `HF_ENDPOINT=hf-mirror.com` + `HF_HUB_DISABLE_XET=1`；`eval/prefetch_datasets.py` 预下载 | 无 |

## GPU 时长 / 花费

| 项 | GPU·h |
|---|---|
| 20 步校准 | 0.96 |
| 全量 116 步（96 步续跑） | ~4.3 |
| 评测（基座 + checkpoint + merge 校验） | 待填 |
| Profiling（9×10 步 + 启动开销） | 待填 |
| **合计** | 待填 |

花费：**$0**（本地机器；未租云、未调用付费 API）。

## 给下周的交接

- **baseline 路径已稳定**：`configs/EXP-W02-01-baseline-gsm8k.sh` 可直接复用；续跑用 `resume_mode=auto`。
- **评测统一入口** `bash scripts/run_eval.sh <ckpt>`：自动 merge LoRA → vLLM 离线 → Math-Verify。
  口径见 `eval/datasets.md`；W3 换严格 verifier 后要整体重评。
- **Profiling 结论**：W9 的第一优先级是训练侧 micro-batch/dynamic bsz（P1），其次是复用 rollout
  logprobs（P2），最后是 flash-attn/kernel（P3）。详见 `reports/profiling_baseline_4080.md` §7。
- **风险提示**：训练内 validation 与 harness 口径不同（rule reward vs Math-Verify，n=1 温度 1.0 vs
  greedy）；报告中必须并列说明。AIME 只有 30 题，方差大。
- **3090**：到位后跑 `HW=3090` 的 baseline 冒烟 + profiling，回写 `configs/hw/3090.env`。

## 需要人决策的问题

1. **baseline 是否对齐**：训练内 validation 0.779（官方 0.779）但 harness 口径待填；是否接受
   "训练内一致 + harness 差异已解释"作为对齐结论？
2. **评测口径**：prompt（GSM8K 逐字复用训练模板；MATH/AIME 用同一 `####` 指令）、greedy/avg@16、
   `max_tokens`（GSM8K 1024 / AIME 2048）是否批准？W3 会替换判分器，口径需要现在定死。
3. **是否允许在 W3 修改 `eval/` 判分实现**（换严格 verifier）：这属于"修改评测代码"，按协议需要请示。
