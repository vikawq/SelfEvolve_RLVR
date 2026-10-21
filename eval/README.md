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

## 输出

- `<out>/<dataset>.jsonl`：逐样本，含 `id`、`messages`、`ground_truth`、每条生成的
  `text` / `finish_reason` / `extracted` / `correct` / 判分器版本。
- `<out>/summary.json`：数据集级指标（pass@1 / avg@k ± std、截断率、空答案率、耗时）、
  判分器版本、生成参数。
- `<out>/config.json`：命令行、模型路径、依赖版本、git commit。

## Merge 一致性

1. **权重级（权威）**：`python eval/check_merge.py ...` 把 PEFT adapter `merge_and_unload()` 后
   与 merged 模型逐参数比较，要求 max|Δweight| ≤ 1e-6；
2. **logits（参考）**：同一进程内 unloaded vs merged 的 20 条 prompt logits，应为 0；
3. **生成级（仅供参考）**：同 checkpoint 用 `--lora-adapter` 直接加载 adapter 跑 `--limit 20`，
   再对 merged 模型跑 `--limit 20`，`compare_outputs.py` 对比。注意 vLLM 的 LoRA runtime kernel
   与 merged kernel 不同，长链 greedy 会分叉，**逐字不一致不代表 merge 失败**；以权重级为准。
4. `merge_lora.py` 自己也会做 3 条固定文本的 logits 校验。

## 口径

见 [`datasets.md`](datasets.md)。测试集只用于评测；W3 换严格 verifier 后，本 harness 的
判分实现整体替换并重评所有历史结果。
