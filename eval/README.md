# eval/

独立评测 harness（W2 起所有结论都用它出数）。与训练框架解耦：不 import verl，不读训练 config；
只依赖 transformers / tokenizers / vLLM / math-verify。

## 一条命令评任意 checkpoint

```bash
# 基座
python eval/run_eval.py \
  --model models/Qwen2.5-1.5B-Instruct \
  --datasets gsm8k,math500,aime2024,aime2025 \
  --output-dir outputs/eval/base-qwen2.5-1.5b

# LoRA checkpoint：先合并，再按普通 HF 模型评
python eval/merge_lora.py \
  --checkpoint checkpoints/EXP-W02-01-baseline-gsm8k/global_step_116 \
  --base-model models/Qwen2.5-1.5B-Instruct \
  --output models/merged/EXP-W02-01-step116

python eval/run_eval.py \
  --model models/merged/EXP-W02-01-step116 \
  --datasets gsm8k,math500,aime2024,aime2025 \
  --output-dir outputs/eval/EXP-W02-01-step116
```

冒烟：加 `--limit 20`。全量 1319 题约数分钟（取决于卡）。

## 长度预算（`--max-model-len`，默认 auto）

vLLM 在 `prompt_len + max_tokens > max_model_len` 时**不报错**，只会把生成预算静默裁到
`max_model_len - prompt_len`——分数掉了，看起来像模型不行，其实是被截断。所以：

- 默认 `--max-model-len 0` = **auto**：先用 tokenizer 量出每个数据集的最长 prompt，
  按 `最长 prompt + 该数据集的 max_tokens` 算预算（对齐到 64），取各数据集最大值初始化引擎。
- 显式传值时会**校验**预算，不够就直接报错并告诉你 auto 值是多少。
- `--max-model-len-cap`（默认 4096）防止 auto 值过大导致 16GB 上要不到 KV cache。
- 每次运行的预算写入 `config.json → length_budgets` 与每个数据集的
  `summary.json → length_budget`（含 `budget_ok`；为 false 时该数据集的截断率不可信）。

## 输出

- `<out>/<dataset>.jsonl`：逐样本，含 `id`、`messages`、`ground_truth`、每条生成的
  `text` / `finish_reason` / `extracted` / `correct` / 判分器版本。
- `<out>/summary.json`：数据集级指标（pass@1 / avg@k、误差、截断率、空答案率、长度预算、耗时）、
  判分器版本、生成参数。误差口径：`pass@1` 看 `binomial_stderr`，`avg@k` 看 `stderr`
  （= `std/sqrt(n)`），具体用哪个由 `reported_stderr_kind` 标明；`pass@1` 的 `std` 是 0/1
  样本标准差，**不要进报告**。
- `<out>/config.json`：命令行、模型路径、长度预算、依赖版本、git commit。

## 回填 `results_master.md`

```bash
python analysis/emit_results_row.py --summary outputs/eval/<run>/summary.json \
  --date 2026-10-21 --exp-id EXP-W02-01-baseline-gsm8k \
  --model 'Qwen2.5-1.5B-Instruct（基座，无 RL）' --gpu 'RTX 4080 16GB'
```

直接输出可粘贴的 Markdown 行（分数 / stderr / std / 截断率 / 空答案率 / 判分器版本），
不要手抄数字。`--summary` 也接受 `summary_rescored.json`。

## Merge 一致性

1. **权重级（权威）**：`python eval/check_merge.py ...` 把 PEFT adapter `merge_and_unload()` 后
   与 merged 模型逐参数比较，要求 max|Δweight| ≤ 1e-6；
2. **logits（参考）**：同一进程内 unloaded vs merged 的 20 条 prompt logits，应为 0；
3. **生成级（仅供参考）**：同 checkpoint 用 `--lora-adapter` 直接加载 adapter 跑 `--limit 20`，
   再对 merged 模型跑 `--limit 20`，`compare_outputs.py` 对比。注意 vLLM 的 LoRA runtime kernel
   与 merged kernel 不同，长链 greedy 会分叉，**逐字不一致不代表 merge 失败**；以权重级为准。
4. `merge_lora.py` 自己也会做 3 条固定文本的 logits 校验。

## 口径

见 [`datasets.md`](datasets.md)。测试集只用于评测；**W3 起判分器是
`src/selfevolve/verifier/math_verifier.py`（与训练奖励共用同一实现，协议 §4）**，本 harness
只是它的薄封装。W2 的 `math-verify` 口径（整段输出参与解析）已退役，但历史行保留在
`reports/results_master.md` 里做对照审计；两套口径的逐样本差异见 `reports/verifier_validation.md`。

用严格 verifier 重打分（不重新生成，零 GPU）：

```bash
python eval/rescore.py --run-dir outputs/eval/<run> --suffix strict \
  --out outputs/eval/<run>/summary_strict.json
```

`--suffix` 决定逐样本输出名（`<dataset>.strict.jsonl`），避免覆盖 W2 的 `*.rescored.jsonl`；
`--datasets` 可只重打分部分数据集。
