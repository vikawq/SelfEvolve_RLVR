#!/usr/bin/env python
"""独立评测入口（W2）。

一条命令评任意基座 / merged LoRA checkpoint：

    python eval/run_eval.py --model models/Qwen2.5-1.5B-Instruct \
        --datasets gsm8k,math500,aime2024,aime2025 \
        --output-dir outputs/eval/base-qwen2.5-1.5b

输出：
    <out>/<dataset>.jsonl   逐样本：id、prompt、每条生成、抽取答案、是否正确
    <out>/summary.json      汇总：指标、标准差、截断率、判分器版本、生成配置
    <out>/config.json       本次运行的完整参数与依赖版本

评测红线：只读测试集；不改训练 reward；判分器版本写入每个结果文件。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.harness.data import DATASETS, DatasetSpec, Example, build_specs  # noqa: E402
from eval.harness.generate import VLLMGenerator  # noqa: E402
from eval.harness.scoring import SCORER_VERSION, score  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _process_dataset(
    spec: DatasetSpec,
    examples: list[Example],
    generator: VLLMGenerator,
    out_path: Path,
    *,
    seed: int,
    limit: int = 0,
) -> dict[str, Any]:
    prompts = [generator.render(ex.messages) for ex in examples]
    started = time.time()
    generations = generator.generate(
        prompts,
        n=spec.n,
        temperature=spec.temperature,
        top_p=spec.top_p,
        max_tokens=spec.max_tokens,
        seed=seed,
    )
    elapsed = time.time() - started

    per_question: list[float] = []
    n_correct = 0
    n_generations = 0
    truncated = 0
    empty = 0
    token_sums = 0
    errors: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as fh:
        for i, (ex, gens) in enumerate(zip(examples, generations)):
            scored = []
            for gen in gens:
                result = score(gen.text, ex.ground_truth, spec.answer_type)
                result.update(
                    finish_reason=gen.finish_reason,
                    num_tokens=gen.num_tokens,
                    text=gen.text,
                )
                scored.append(result)
                n_generations += 1
                token_sums += gen.num_tokens
                n_correct += int(result["correct"])
                truncated += int(gen.finish_reason == "length")
                empty += int(not result["extracted"])
                if result.get("error"):
                    errors[result["error"]] = errors.get(result["error"], 0) + 1
            accuracy = sum(int(r["correct"]) for r in scored) / len(scored)
            per_question.append(accuracy)
            fh.write(
                json.dumps(
                    {
                        "id": ex.id,
                        "dataset": spec.name,
                        "ground_truth": ex.ground_truth,
                        "messages": ex.messages,
                        "meta": ex.meta,
                        "accuracy": accuracy,
                        "outputs": scored,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 50 == 0:
                print(f"  [{spec.name}] {i + 1}/{len(examples)}", flush=True)

    mean_acc = sum(per_question) / len(per_question) if per_question else 0.0
    std = _sample_std(per_question)
    stderr = std / math.sqrt(len(per_question)) if per_question else 0.0
    return {
        "dataset": spec.name,
        "display_name": spec.display_name,
        "num_samples": len(examples),
        "metric": spec.metric,
        "score": mean_acc,
        "std": std,
        "stderr": stderr,
        "num_generations": n_generations,
        "n_per_sample": spec.n,
        "prompt_template": spec.prompt_template,
        "accuracy_over_generations": (n_correct / n_generations) if n_generations else 0.0,
        "truncation_rate": (truncated / n_generations) if n_generations else 0.0,
        "empty_answer_rate": (empty / n_generations) if n_generations else 0.0,
        "avg_output_tokens": (token_sums / n_generations) if n_generations else 0.0,
        "elapsed_s": elapsed,
        "scoring_errors": errors,
        "generation": {
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "n": spec.n,
            "max_tokens": spec.max_tokens,
            "seed": seed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent eval harness (vLLM + Math-Verify)")
    parser.add_argument("--model", required=True, help="基座目录或合并后的 HF 模型目录")
    parser.add_argument(
        "--datasets",
        default="gsm8k,math500,aime2024,aime2025",
        help="逗号分隔：gsm8k,math500,aime2024,aime2025,mmlu_pro",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "eval"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit", type=int, default=0, help="每个数据集只评前 N 条（0=全量）")
    parser.add_argument("--max-model-len", type=int, default=2560)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--lora-adapter", default=None, help="可选：直接加载 PEFT adapter（用于 merge 前后对比）")
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_name = args.run_name or Path(args.model.rstrip("/")).name
    if args.lora_adapter:
        run_name += "-lora"
    out_dir = Path(args.output_dir) / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        parser.error(f"output dir already exists: {out_dir} (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "argv": sys.argv if argv is None else [sys.argv[0], *argv],
        "model": str(Path(args.model).resolve()),
        "lora_adapter": str(Path(args.lora_adapter).resolve()) if args.lora_adapter else None,
        "datasets": datasets,
        "prompt_templates": {d: DATASETS[d].prompt_template for d in datasets},
        "limit": args.limit,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "seed": args.seed,
        "scorer": SCORER_VERSION,
        "git_commit": _git_commit(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "torch": _pkg_version("torch"),
            "vllm": _pkg_version("vllm"),
            "transformers": _pkg_version("transformers"),
            "math-verify": _pkg_version("math-verify"),
            "datasets": _pkg_version("datasets"),
        },
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)

    print(f"[eval] model={args.model} adapter={args.lora_adapter}")
    print(f"[eval] datasets={datasets} output={out_dir} scorer={SCORER_VERSION}")
    generator = VLLMGenerator(
        args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        lora_adapter=args.lora_adapter,
        max_lora_rank=args.max_lora_rank,
        seed=args.seed,
    )

    summary: dict[str, Any] = {"config": config, "datasets": {}}
    try:
        specs = build_specs(datasets, args.limit)
        for spec, examples in specs:
            print(f"[eval] {spec.name}: {len(examples)} samples, n={spec.n}, temp={spec.temperature}")
            result = _process_dataset(
                spec,
                examples,
                generator,
                out_dir / f"{spec.name}.jsonl",
                seed=args.seed,
            )
            summary["datasets"][spec.name] = result
            print(
                f"[eval] {spec.name}: {spec.metric}={result['score']:.4f} "
                f"±{result['std']:.4f} (n={result['num_samples']}, "
                f"trunc={result['truncation_rate']:.2%}, empty={result['empty_answer_rate']:.2%})"
            )
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        generator.close()

    print(f"[eval] summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
