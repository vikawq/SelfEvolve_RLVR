#!/usr/bin/env python
"""merge 前后一致性检查：**权重级**为准，生成/logits 只作参考。

背景（W2 实测）：vLLM 的 LoRA runtime kernel 与 merged 权重 kernel 不同，长链 greedy 生成会
因浮点差异级联分叉（20 题只有 1 题逐字一致）；HF CPU bf16 的跨进程 forward 也有非确定性。
因此"生成是否逐字一致"不能用来判断 merge 是否正确。

本脚本做两件事：
1. `peft.merge_and_unload()` 之后的 state_dict 与 merged 模型逐参数比较（权威判据，
   max|Δweight| > 1e-6 才算失败）；
2. 可选：在若干 GSM8K prompt 上比较 unloaded vs merged 的 logits（同进程、同 kernel，仅参考）。

    python eval/check_merge.py --base models/Qwen2.5-1.5B-Instruct \
        --adapter models/merged/<exp>_work/extracted/lora_adapter \
        --merged models/merged/<exp> --num-prompts 20
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
    parser.add_argument("--num-prompts", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", default=None)
    parser.add_argument("--weight-tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval.harness.data import load_gsm8k

    print("[check] loading base + adapter and merged model on CPU")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="cpu")
    peft_model = PeftModel.from_pretrained(base, args.adapter)
    merged = AutoModelForCausalLM.from_pretrained(args.merged, torch_dtype=torch.bfloat16, device_map="cpu")
    unloaded = peft_model.merge_and_unload()
    unloaded.eval()
    merged.eval()

    # --- 1. 权重级（权威）---
    sd_u = unloaded.state_dict()
    sd_m = merged.state_dict()
    max_weight_diff = 0.0
    missing = []
    for key, value in sd_u.items():
        if key not in sd_m:
            missing.append(key)
            continue
        max_weight_diff = max(max_weight_diff, (value.float() - sd_m[key].float()).abs().max().item())
    for key in sd_m:
        if key not in sd_u:
            missing.append(key)
    print(f"[check] weights: {len(sd_u)} tensors, max|Δ|={max_weight_diff:.3e}, missing={len(missing)}")

    # --- 2. logits（参考，不参与退出码）---
    logit_rows = []
    if args.num_prompts > 0:
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        for ex in load_gsm8k(limit=args.num_prompts):
            text = tokenizer.apply_chat_template(ex.messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            with torch.no_grad():
                a = unloaded(**enc).logits.float()
                b = merged(**enc).logits.float()
            logit_rows.append({
                "id": ex.id,
                "positions": int(a.shape[1]),
                "max_abs_diff": float((a - b).abs().max().item()),
                "argmax_mismatch": int((a.argmax(-1) != b.argmax(-1)).sum().item()),
            })
        worst = max(row["max_abs_diff"] for row in logit_rows) if logit_rows else 0.0
        print(f"[check] logits (same-process, informational): prompts={len(logit_rows)} max|Δ|={worst:.3e}")

    report = {
        "base": str(Path(args.base).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "merged": str(Path(args.merged).resolve()),
        "num_tensors": len(sd_u),
        "missing_tensors": missing[:10],
        "max_abs_weight_diff": max_weight_diff,
        "weight_tolerance": args.weight_tolerance,
        "weights_bit_identical": max_weight_diff <= args.weight_tolerance and not missing,
        "logit_probe": logit_rows,
        "note": "weights are authoritative; generation/logits differ by kernel/nondeterminism (see troubleshooting 4.7)",
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[check] report -> {args.out}")
    return 0 if report["weights_bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
