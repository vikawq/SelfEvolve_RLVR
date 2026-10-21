#!/usr/bin/env python
"""把 verl 的 LoRA checkpoint 合并成普通 HF 模型（评测用）。

流程（对应任务包 W2 §2）：
1. 调 verl 自带 model_merger，从 FSDP checkpoint 里导出 `lora_adapter/`；
2. 用 peft 的 `merge_and_unload()` 把 adapter 合进基座；
3. 在 CPU 上对比 merge 前后 logits（3 条固定输入，max|Δ| 超阈值直接失败）。

生成层面的 20 题一致性用 `run_eval.py --limit 20 --lora-adapter ...` 与合并后模型各跑一次，
再用 `eval/compare_outputs.py` 对比（任务包要求"合并后先跑 20 题"）。

示例：
    python eval/merge_lora.py \
        --checkpoint checkpoints/EXP-W02-01-baseline-gsm8k/global_step_116 \
        --base-model models/Qwen2.5-1.5B-Instruct \
        --output models/merged/EXP-W02-01-step116
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VERIFY_TEXTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "What is 2 + 2?",
    "Solve for x: 3x + 7 = 22.",
]


def _resolve_actor_dir(checkpoint: Path) -> Path:
    if (checkpoint / "fsdp_config.json").exists():
        return checkpoint
    actor = checkpoint / "actor"
    if (actor / "fsdp_config.json").exists():
        return actor
    raise FileNotFoundError(f"no fsdp_config.json under {checkpoint} or {checkpoint / 'actor'}")


def _run_merger(actor_dir: Path, work_dir: Path, python: str) -> Path:
    out = work_dir / "extracted"
    if (out / "lora_adapter" / "adapter_model.safetensors").exists():
        print(f"[merge] reuse extracted adapter: {out / 'lora_adapter'}")
        return out / "lora_adapter"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(out),
    ]
    print("[merge] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    adapter = out / "lora_adapter"
    if not (adapter / "adapter_model.safetensors").exists():
        raise RuntimeError(f"model_merger did not produce a LoRA adapter under {adapter}")
    return adapter


def _merge_peft(base_model: str, adapter_dir: Path, output: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] loading base on CPU: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="cpu")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    print(f"[merge] adapter: r={adapter_cfg.get('r')} alpha={adapter_cfg.get('lora_alpha')} "
          f"targets={len(adapter_cfg.get('target_modules', []))}")

    merged = peft_model.merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(output))
    print(f"[merge] saved merged model -> {output}")
    return merged, peft_model


def _verify_logits(merged, peft_model, base_model: str) -> dict:
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    max_diff = 0.0
    details = []
    merged.eval()
    peft_model.eval()
    with torch.no_grad():
        for text in VERIFY_TEXTS:
            enc = tokenizer(text, return_tensors="pt")
            logits_merged = merged(**enc).logits.float()
            logits_peft = peft_model(**enc).logits.float()
            diff = (logits_merged - logits_peft).abs().max().item()
            max_diff = max(max_diff, diff)
            details.append({"text": text[:60], "max_abs_diff": diff})
    return {"max_abs_diff": max_diff, "per_text": details, "tolerance": 0.02}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge verl LoRA checkpoint into a HF model")
    parser.add_argument("--checkpoint", required=True, help="global_step_N 目录（或其中的 actor/）")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", default=None, help="merger 中间产物目录，默认与 output 同级")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint).resolve()
    actor_dir = _resolve_actor_dir(checkpoint)
    output = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve() if args.work_dir else output.parent / (output.name + "_work")

    started = time.time()
    adapter_dir = _run_merger(actor_dir, work_dir, sys.executable)
    merged, peft_model = _merge_peft(args.base_model, adapter_dir, output)

    report = {
        "checkpoint": str(checkpoint),
        "actor_dir": str(actor_dir),
        "base_model": str(Path(args.base_model).resolve()),
        "output": str(output),
        "adapter_dir": str(adapter_dir),
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": None,
    }
    if not args.skip_verify:
        report["logit_equivalence"] = _verify_logits(merged, peft_model, args.base_model)
        if report["logit_equivalence"]["max_abs_diff"] > report["logit_equivalence"]["tolerance"]:
            raise RuntimeError(f"merge logits mismatch: {report['logit_equivalence']}")
        print(f"[merge] logit equivalence OK: max|Δ|={report['logit_equivalence']['max_abs_diff']:.2e}")
    report["elapsed_s"] = time.time() - started
    (output / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] report -> {output / 'merge_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
