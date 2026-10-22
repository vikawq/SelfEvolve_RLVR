#!/usr/bin/env python
"""从评测 summary.json 生成 `reports/results_master.md` 的表格行。

results_master 是手写的 Markdown 表，W3 换严格 verifier 后所有行都要按新判分器重评一遍；
手抄 score / stderr / 截断率 容易出错，这里直接从 summary.json 生成可粘贴的行。

    python analysis/emit_results_row.py \
        --summary outputs/eval/<run>/summary.json \
        --date 2026-10-21 --exp-id EXP-W02-01-baseline-gsm8k \
        --model 'Qwen2.5-1.5B-Instruct（基座，无 RL）' --gpu 'RTX 4080 16GB'

`--summary` 也可以指向 `summary_rescored.json`（重打分后的产物）。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PLACEHOLDER = "—"


def binomial_stderr(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def reported_stderr(entry: dict[str, Any]) -> float:
    """老的 summary.json 没有 binomial_stderr 字段，这里按同样的规则补算。"""
    if entry.get("binomial_stderr") is not None:
        return float(entry["binomial_stderr"])
    if entry.get("n_per_sample", 1) == 1:
        return binomial_stderr(float(entry["score"]), int(entry["num_samples"]))
    return float(entry.get("stderr", 0.0))


def sampling_text(entry: dict[str, Any]) -> str:
    gen = entry.get("generation", {})
    temp = gen.get("temperature")
    top_p = gen.get("top_p")
    n = gen.get("n", entry.get("n_per_sample", 1))
    temp_s = "0" if temp == 0 else f"{temp}"
    return f"temperature={temp_s}, top_p={top_p}, n={n}"


def metric_text(entry: dict[str, Any]) -> str:
    return "greedy pass@1" if entry.get("n_per_sample", 1) == 1 else entry.get("metric", "avg@k")


def std_text(entry: dict[str, Any]) -> str:
    # pass@1 的 std 是 0/1 样本标准差，没有信息量，不进表。
    if entry.get("n_per_sample", 1) == 1:
        return "n/a"
    return f"{float(entry['std']):.4f}"


def emit_row(entry: dict[str, Any], config: dict[str, Any], meta: dict[str, str]) -> str:
    scorer = config.get("scorer", PLACEHOLDER)
    cells = [
        meta["date"],
        meta["exp_id"],
        meta["model"],
        entry.get("display_name", entry["dataset"]),
        str(entry["num_samples"]),
        sampling_text(entry),
        metric_text(entry),
        f"**{float(entry['score']):.4f}**",
        f"±{reported_stderr(entry):.4f}",
        std_text(entry),
        f"{float(entry['truncation_rate']):.2%}",
        f"{float(entry['empty_answer_rate']):.2%}",
        str(scorer),
        meta["gpu"],
        meta["wandb"],
    ]
    return "| " + " | ".join(cells) + " |"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="emit results_master.md rows from summary.json")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--model", required=True, help="results_master 的「模型 / checkpoint」列原文")
    parser.add_argument("--gpu", default="RTX 4080 16GB")
    parser.add_argument("--wandb", default=PLACEHOLDER)
    parser.add_argument("--datasets", default=None, help="逗号分隔；默认全部")
    args = parser.parse_args(argv)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    config = summary.get("config", {})
    names = [d.strip() for d in args.datasets.split(",")] if args.datasets else list(summary["datasets"])
    meta = {
        "date": args.date,
        "exp_id": args.exp_id,
        "model": args.model,
        "gpu": args.gpu,
        "wandb": args.wandb,
    }
    for name in names:
        entry = summary["datasets"][name]
        print(emit_row(entry, config, meta))
        budget = entry.get("length_budget")
        if budget is not None and not budget.get("budget_ok", True):
            print(
                f"  <!-- 警告：{name} 的生成预算不足"
                f"（need {budget['required_max_model_len']} >"
                f" engine {budget['engine_max_model_len']}），截断率不可信 -->"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
