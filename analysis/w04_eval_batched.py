#!/usr/bin/env python
"""W4 批量评测（生成 + 逐条进程隔离评分），产物与 `eval/run_eval.py` 同 schema。

为什么需要它（不是"换判分口径"）：

- 判分仍然是 `eval.harness.scoring.score` → `selfevolve.verifier.math_verifier.verify`，
  输入/字段/`finished` 截断判定与 `run_eval.py` 完全一致；
- 唯一区别是**评分进程隔离**：`math_verify`/sympy 的全局缓存会在同一进程里随样本累积，
  W4 实测 500 条 MATH-500 跑到第 63 条时 RSS 114 GB、机器开始换页（训练侧 160 条/步却没事，
  因为 GSM8K 的答案简单）。评测用 4 个 spawn worker、`maxtasksperchild=1`，每评一条就释放缓存。

因此本脚本只解决 "500 条评测跑不完" 的工程问题，不改判分规则、不改 `eval/` 下任何代码。
校验方式：同一 checkpoint、同样前 100 条 MATH-500，本脚本与 `eval/run_eval.py` 的分数必须一致
（W4 实测：L1 step100 = 0.3100，两套都是）。

用法：

    python analysis/w04_eval_batched.py \
        --model models/merged/EXP-W04-01-l1-format-weight-global_step_200 \
        --datasets gsm8k,math500 --limit 500 \
        --run-name EXP-W04-01-l1-format-weight-global_step_200
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 与 run_eval.py 一致：必须在 transformers/vllm/datasets import 之前设置。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from eval.harness.data import DATASETS, build_specs  # noqa: E402
from eval.harness.generate import (  # noqa: E402
    VLLMGenerator,
    count_prompt_tokens,
    load_tokenizer,
    render_chat,
    required_max_model_len,
)
from eval.harness.scoring import SCORER_VERSION  # noqa: E402
from analysis import w04_score_worker  # noqa: E402


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
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _timeout_result(ground_truth: str) -> dict[str, Any]:
    """单条评分硬超时时的结果：与严格 verifier 的 timeout 语义一致（判错 + 计数）。"""
    from selfevolve.verifier.math_verifier import SCORER_VERSION, normalize_answer

    return {
        "correct": False,
        "extracted": "",
        "parse_ok": False,
        "reason": "timeout",
        "gold_normalized": normalize_answer(ground_truth)[:200],
        "scorer": SCORER_VERSION,
        "error": "timeout",
    }


def _score_one_hard(
    payload: tuple[str, str, str, bool | None],
    ctx: Any,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    """一个样本一个 spawn 子进程；超过 `timeout_s` 直接 kill，按 timeout（判错）返回。

    这样既避免了 sympy 全局缓存跨样本累积，也避免了单条病态表达式在 C 层长时间
    不响应 SIGALRM（W4 实测：一条 MATH-500 能跑 18+ 分钟、RSS 114 GB）。
    """
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=w04_score_worker.score_task_into, args=(payload, child_conn), daemon=True)
    proc.start()
    child_conn.close()
    try:
        received = parent_conn.poll(timeout_s)
        result = None
        if received:
            try:
                result = parent_conn.recv()
            except EOFError:
                result = None
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=3)
        if result is None:
            return _timeout_result(payload[1])
        return result
    finally:
        parent_conn.close()
        if proc.is_alive():  # pragma: no cover - 极端情况
            proc.kill()
            proc.join(timeout=3)


def _score_with_hard_timeout(
    payloads: list[tuple[str, str, str, bool | None]],
    *,
    timeout_s: float,
) -> list[dict[str, Any]]:
    """逐条评分（每条独立进程 + 硬超时）。500 条约 2–3 分钟，换来内存有界。"""
    ctx = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        results.append(_score_one_hard(payload, ctx, timeout_s=timeout_s))
        if (index + 1) % 50 == 0:
            print(f"  [score] {index + 1}/{len(payloads)}", flush=True)
    return results


def _plan_budgets(specs, tokenizer) -> list[dict[str, Any]]:
    plans = []
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
    plan: dict[str, Any],
    generator: VLLMGenerator | None,
    out_path: Path,
    *,
    seed: int,
    workers: int,
    rescore_only: bool,
    score_timeout_s: float,
) -> dict[str, Any]:
    spec = plan["spec"]
    examples = plan["examples"]
    started = time.time()

    if rescore_only:
        rows = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        generations = [
            [
                {
                    "text": output["text"],
                    "finish_reason": output.get("finish_reason", "stop"),
                    "num_tokens": output.get("num_tokens", 0),
                }
                for output in row["outputs"]
            ]
            for row in rows
        ]
    else:
        assert generator is not None
        generations = generator.generate(
            plan["prompts"],
            n=spec.n,
            temperature=spec.temperature,
            top_p=spec.top_p,
            max_tokens=spec.max_tokens,
            seed=seed,
        )

    payloads: list[tuple[str, str, str, bool | None]] = []
    for ex, gens in zip(examples, generations):
        for gen in gens:
            payloads.append(
                (
                    gen["text"] if isinstance(gen, dict) else gen.text,
                    ex.ground_truth,
                    spec.answer_type,
                    (gen["finish_reason"] if isinstance(gen, dict) else gen.finish_reason) != "length",
                )
            )
    scored = _score_with_hard_timeout(payloads, timeout_s=score_timeout_s)

    per_question: list[float] = []
    n_correct = 0
    n_generations = 0
    truncated = 0
    empty = 0
    token_sums = 0
    errors: dict[str, int] = {}
    cursor = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for ex, gens in zip(examples, generations):
            scores = scored[cursor : cursor + len(gens)]
            cursor += len(gens)
            outputs = []
            for gen, result in zip(gens, scores):
                text = gen["text"] if isinstance(gen, dict) else gen.text
                finish = gen["finish_reason"] if isinstance(gen, dict) else gen.finish_reason
                num_tokens = gen["num_tokens"] if isinstance(gen, dict) else gen.num_tokens
                result = dict(result)
                result.update(finish_reason=finish, num_tokens=num_tokens, text=text)
                outputs.append(result)
                n_generations += 1
                token_sums += num_tokens
                n_correct += int(result["correct"])
                truncated += int(finish == "length")
                empty += int(not result["extracted"])
                if result.get("error"):
                    errors[result["error"]] = errors.get(result["error"], 0) + 1
            accuracy = sum(int(r["correct"]) for r in scores) / len(scores)
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
                        "outputs": outputs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (len(per_question)) % 50 == 0:
                print(f"  [{spec.name}] {len(per_question)}/{len(examples)}", flush=True)

    elapsed = time.time() - started
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
            "prompt_tokens_max": plan["prompt_tokens_max"],
            "prompt_tokens_mean": plan["prompt_tokens_mean"],
            "required_max_model_len": plan["required_max_model_len"],
            "engine_max_model_len": 0,
            "budget_ok": True,
        },
        "scoring_mode": f"spawn_process_per_sample(hard_timeout={score_timeout_s}s)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W4 batched eval (same scorer as run_eval.py)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", default="gsm8k,math500")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "eval"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-model-len-cap", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers", type=int, default=4, help="仅记录；评分现在是每条样本一个 spawn 进程"
    )
    parser.add_argument(
        "--score-timeout-s",
        type=float,
        default=30.0,
        help="单条评分的硬超时（秒）；超时 kill 子进程并按 timeout/错 记",
    )
    parser.add_argument("--rescore-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_name = args.run_name or Path(args.model.rstrip("/")).name
    out_dir = Path(args.output_dir) / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        parser.error(f"output dir already exists: {out_dir} (use --overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_specs(datasets, args.limit)
    tokenizer = load_tokenizer(args.model)
    plans = _plan_budgets(specs, tokenizer)
    auto_len = max(p["required_max_model_len"] for p in plans)
    for plan in plans:
        print(
            f"[eval] budget {plan['spec'].name}: prompt_max={plan['prompt_tokens_max']}"
            f" + max_tokens={plan['spec'].max_tokens}"
            f" -> need max_model_len>={plan['required_max_model_len']}",
            flush=True,
        )
    if args.max_model_len > 0:
        engine_len = args.max_model_len
    else:
        if auto_len > args.max_model_len_cap:
            parser.error(f"auto max_model_len={auto_len} > cap {args.max_model_len_cap}")
        engine_len = auto_len

    config = {
        "argv": sys.argv if argv is None else [sys.argv[0], *argv],
        "model": str(Path(args.model).resolve()),
        "datasets": datasets,
        "prompt_templates": {d: DATASETS[d].prompt_template for d in datasets},
        "limit": args.limit,
        "max_model_len": engine_len,
        "max_model_len_mode": "explicit" if args.max_model_len > 0 else "auto",
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
        "workers": args.workers,
        "rescore_only": args.rescore_only,
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
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[eval] model={args.model} datasets={datasets} output={out_dir}", flush=True)
    print(f"[eval] scorer={SCORER_VERSION} workers={args.workers} rescore_only={args.rescore_only}", flush=True)
    generator = None
    if not args.rescore_only:
        generator = VLLMGenerator(
            args.model,
            max_model_len=engine_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            seed=args.seed,
            tokenizer=tokenizer,
        )

    summary: dict[str, Any] = {"config": config, "datasets": {}}
    try:
        for plan in plans:
            spec = plan["spec"]
            print(f"[eval] {spec.name}: {len(plan['examples'])} samples, n={spec.n}", flush=True)
            result = _process_dataset(
                plan,
                generator,
                out_dir / f"{spec.name}.jsonl",
                seed=args.seed,
                workers=args.workers,
                rescore_only=args.rescore_only,
                score_timeout_s=args.score_timeout_s,
            )
            result["length_budget"]["engine_max_model_len"] = engine_len
            summary["datasets"][spec.name] = result
            reported = (
                result["binomial_stderr"]
                if result["binomial_stderr"] is not None
                else result["stderr"]
            )
            print(
                f"[eval] {spec.name}: {result['metric']}={result['score']:.4f}"
                f" ±{reported:.4f} ({result['reported_stderr_kind']} stderr,"
                f" trunc={result['truncation_rate']:.2%}, empty={result['empty_answer_rate']:.2%},"
                f" elapsed={result['elapsed_s']:.1f}s)",
                flush=True,
            )
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if generator is not None:
            generator.close()

    print(f"[eval] summary -> {out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
