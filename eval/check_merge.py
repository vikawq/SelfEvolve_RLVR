#!/usr/bin/env python
"""merge 前后一致性检查（CPU）：PEFT adapter vs 合并后的 HF 模型。

比 `merge_lora.py` 内置的 3 条固定文本更强：用 20 条真实 GSM8K prompt，逐位置比较 logits
与 argmax。vLLM 的 LoRA runtime 与 merged 权重走不同 kernel，生成长文本会因浮点差异级联
分叉；本检查在 HF 同一 kernel 下确认"权重确实等价"。

    python eval/check_merge.py \
        --base models/Qwen2.5-1.5B-Instruct \
        --adapter models/merged/<exp>_work/extracted/lora_adapter \
        --merged models/merged/<exp> \
        --num-prompts 20

输出 JSON（--out）与人类可读摘要；max|Δlogit| > 0.02 时退出码 1。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", default=None)
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval.harness.data import load_gsm8k

    print("[check] loading tokenizer and models on CPU")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    peft_model = PeftModel.from_pretrained(base, args.adapter)
    merged = AutoModelForCausalLM.from_pretrained(args.merged, torch_dtype=torch.bfloat16, device_map="cpu")
    peft_model.eval()
    merged.eval()

    examples = load_gsm8k(limit=args.num_prompts)
    rows = []
    max_diff = 0.0
    argmax_mismatch = 0
    total_positions = 0
    with torch.no_grad():
        for ex in examples:
            text = tokenizer.apply_chat_template(ex.messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            logits_a = peft_model(**enc).logits.float()
            logits_b = merged(**enc).logits.float()
            diff = (logits_a - logits_b).abs().max().item()
            mismatch = int((logits_a.argmax(-1) != logits_b.argmax(-1)).sum().item())
            total_positions += logits_a.shape[1]
            argmax_mismatch += mismatch
            max_diff = max(max_diff, diff)
            rows.append({"id": ex.id, "positions": logits_a.shape[1], "max_abs_diff": diff, "argmax_mismatch": mismatch})

    report = {
        "base": str(Path(args.base).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "merged": str(Path(args.merged).resolve()),
        "num_prompts": len(rows),
        "tolerance": args.tolerance,
        "max_abs_logit_diff": max_diff,
        "argmax_mismatch_positions": argmax_mismatch,
        "total_positions": total_positions,
        "per_prompt": rows,
    }
    print(f"[check] prompts={len(rows)} max|Δlogit|={max_diff:.3e} "
          f"argmax mismatch={argmax_mismatch}/{total_positions}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[check] report -> {args.out}")
    return 0 if max_diff <= args.tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
