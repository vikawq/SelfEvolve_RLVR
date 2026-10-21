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

1. `merge_lora.py` 在 CPU 上对比 merge 前后 logits（3 条固定输入，max|Δ| ≤ 0.02）。
2. 生成层面：同一 checkpoint 用 `--lora-adapter` 直接加载 adapter 跑 `--limit 20`，
   再对 merged 模型跑 `--limit 20`，最后
   `python eval/compare_outputs.py --left ... --right ... --datasets gsm8k --limit 20`
   要求 greedy 输出逐字一致。

## 口径

见 [`datasets.md`](datasets.md)。测试集只用于评测；W3 换严格 verifier 后，本 harness 的
判分实现整体替换并重评所有历史结果。
