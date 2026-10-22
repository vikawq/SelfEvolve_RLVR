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

`--max-model-len` 默认 0 = auto：先用 tokenizer 量出每个数据集最长 prompt，按
`最长 prompt + 该数据集的 max_tokens` 算出预算，取各数据集最大值初始化 vLLM。
显式传值时会校验预算，不够就直接报错——vLLM 自己不报错，只会静默压缩生成预算。

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

# 必须在 import transformers/vllm/datasets 之前设置：huggingface_hub 在 import 时就把 ENDPOINT
# 固化成常量，之后再 os.environ[...] 已经不生效（W2 实测：否则评测卡在 huggingface.co 超时）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.harness.data import DATASETS, DatasetSpec, Example, build_specs  # noqa: E402
from eval.harness.generate import (  # noqa: E402
    VLLMGenerator,
    count_prompt_tokens,
    load_tokenizer,
    render_chat,
    required_max_model_len,
)
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


def _binomial_stderr(p: float, n: int) -> float:
    """pass@1 的误差要报 binomial stderr；0/1 样本标准差恒在 0.4–0.5，没有信息量。"""
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _plan_budgets(
    specs: list[tuple[DatasetSpec, list[Example]]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """为每个数据集渲染 prompt、量 token 数、算出需要的 max_model_len。"""
    plans: list[dict[str, Any]] = []
    for spec, examples in specs:
        prompts = [render_chat(tokenizer, ex.messages) for ex in examples]
        lens = count_prompt_tokens(tokenizer, prompts)
        prompt_max = max(lens) if lens else 0
        plans.append(
            {
                "spec": spec,
                "examples": examples,
                "prompts": prompts,
                "prompt_tokens_max": prompt_max,
                "prompt_tokens_mean": (sum(lens) / len(lens)) if lens else 0.0,
                "required_max_model_len": required_max_model_len(prompt_max, spec.max_tokens),
            }
        )
    return plans


def _process_dataset(
    spec: DatasetSpec,
    examples: list[Example],
    prompts: list[str],
    generator: VLLMGenerator,
    out_path: Path,
    *,
    seed: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
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
                result = score(
                    gen.text,
                    ex.ground_truth,
                    spec.answer_type,
                    finished=gen.finish_reason != "length",
                )
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
        # pass@1 用 binomial stderr；avg@k 用逐题 accuracy 的 stderr（std/sqrt(n)）。
        "binomial_stderr": _binomial_stderr(mean_acc, len(examples)) if spec.n == 1 else None,
        "reported_stderr_kind": "binomial" if spec.n == 1 else "per_question_sample",
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
        "length_budget": {
            "prompt_tokens_max": budget["prompt_tokens_max"],
            "prompt_tokens_mean": budget["prompt_tokens_mean"],
            "required_max_model_len": budget["required_max_model_len"],
            "engine_max_model_len": generator.max_model_len,
            # False 表示生成预算被 vLLM 静默压缩过，该数据集的截断率不可信
            "budget_ok": budget["required_max_model_len"] <= generator.max_model_len,
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
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="0=auto（按各数据集最长 prompt + max_tokens 推算）；显式传值会校验预算是否够",
    )
    parser.add_argument(
        "--max-model-len-cap",
        type=int,
        default=4096,
        help="auto 推算值的上限，防止 KV cache 在 16GB 上要不到（超了直接报错）",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--lora-adapter", default=None, help="可选：直接加载 PEFT adapter（用于 merge 前后对比）")
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只量 prompt 长度并打印各数据集的 max_model_len 预算，不启动 vLLM、不写结果（零 GPU）",
    )
    args = parser.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_name = args.run_name or Path(args.model.rstrip("/")).name
    if args.lora_adapter and args.run_name is None:
        run_name += "-lora"
    out_dir = Path(args.output_dir) / run_name
    if not args.plan_only:
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            parser.error(f"output dir already exists: {out_dir} (use --overwrite)")
        out_dir.mkdir(parents=True, exist_ok=True)

    # 先加载数据、量 prompt 长度，再决定引擎的 max_model_len：vLLM 起来之后改不了。
    specs = build_specs(datasets, args.limit)
    tokenizer = load_tokenizer(args.model)
    plans = _plan_budgets(specs, tokenizer)
    auto_len = max(p["required_max_model_len"] for p in plans)
    for plan in plans:
        print(
            f"[eval] budget {plan['spec'].name}: prompt_max={plan['prompt_tokens_max']}"
            f" + max_tokens={plan['spec'].max_tokens}"
            f" -> need max_model_len>={plan['required_max_model_len']}"
        )

    if args.plan_only:
        # 用途之一：核查历史运行（固定 max_model_len=2560）有没有被静默压缩过生成预算。
        print(f"[eval] auto max_model_len = {auto_len}")
        for plan in plans:
            need = plan["required_max_model_len"]
            verdict = "ok" if need <= 2560 else "超过旧的固定值 2560 -> 历史结果需重跑"
            print(f"[eval] plan-only {plan['spec'].name}: need={need} ({verdict})")
        return 0

    if args.max_model_len > 0:
        engine_len = args.max_model_len
        short = {
            plan["spec"].name: plan["required_max_model_len"]
            for plan in plans
            if plan["required_max_model_len"] > engine_len
        }
        if short:
            parser.error(
                f"--max-model-len {engine_len} 不够：{short}。"
                " vLLM 不会报错，只会静默压缩生成预算（截断率失真）。"
                f" 去掉 --max-model-len 走 auto（{auto_len}），或调低对应数据集的 max_tokens。"
            )
        len_mode = "explicit"
    else:
        if auto_len > args.max_model_len_cap:
            parser.error(
                f"auto max_model_len={auto_len} 超过 --max-model-len-cap {args.max_model_len_cap}；"
                " 确认显存够再提高上限，或调低 max_tokens。"
            )
        engine_len = auto_len
        len_mode = "auto"
    print(f"[eval] max_model_len={engine_len} ({len_mode}; auto 推算值 {auto_len})")

    config = {
        "argv": sys.argv if argv is None else [sys.argv[0], *argv],
        "model": str(Path(args.model).resolve()),
        "lora_adapter": str(Path(args.lora_adapter).resolve()) if args.lora_adapter else None,
        "datasets": datasets,
        "prompt_templates": {d: DATASETS[d].prompt_template for d in datasets},
        "limit": args.limit,
        "max_model_len": engine_len,
        "max_model_len_mode": len_mode,
        "max_model_len_auto": auto_len,
        "length_budgets": {
            plan["spec"].name: {
                "prompt_tokens_max": plan["prompt_tokens_max"],
                "prompt_tokens_mean": plan["prompt_tokens_mean"],
                "max_tokens": plan["spec"].max_tokens,
                "required_max_model_len": plan["required_max_model_len"],
            }
            for plan in plans
        },
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
        max_model_len=engine_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        lora_adapter=args.lora_adapter,
        max_lora_rank=args.max_lora_rank,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    summary: dict[str, Any] = {"config": config, "datasets": {}}
    try:
        for plan in plans:
            spec = plan["spec"]
            print(
                f"[eval] {spec.name}: {len(plan['examples'])} samples,"
                f" n={spec.n}, temp={spec.temperature}"
            )
            result = _process_dataset(
                spec,
                plan["examples"],
                plan["prompts"],
                generator,
                out_dir / f"{spec.name}.jsonl",
                seed=args.seed,
                budget=plan,
            )
            summary["datasets"][spec.name] = result
            reported = (
                result["binomial_stderr"]
                if result["binomial_stderr"] is not None
                else result["stderr"]
            )
            print(
                f"[eval] {spec.name}: {spec.metric}={result['score']:.4f}"
                f" ±{reported:.4f} ({result['reported_stderr_kind']} stderr,"
                f" std={result['std']:.4f}, n={result['num_samples']},"
                f" trunc={result['truncation_rate']:.2%}, empty={result['empty_answer_rate']:.2%})"
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
