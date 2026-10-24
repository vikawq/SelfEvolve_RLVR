#!/usr/bin/env python
"""W5 逐步指标：从训练 rollout dump + 评测 summary 计算"防御前后"的完整对照。

与 W4 的 `analysis/hack_metrics.py` 的区别：

- 在 `hack_detectors.aggregate_responses` 的通用指标之上，加 **hack rate**（规则见
  `selfevolve.metrics.hack_rate`）与防御诊断字段（D3 触发、D2 解析状态、D4 一致率）；
- 增加 **零方差组比例**：同一 prompt 的 5 条 rollout 奖励完全相同的比例。D3 把长回答的奖励
  压到 0 时，如果整组都是 0，GRPO 没有梯度——这是 W5 需要监控的"防御副作用"；
- 同时输出 last20（与 W4 可比）、last50（hack rate 口径）、first10 三个窗口。

hack rate / gap 的 onset 都复用 `hack_detectors.hack_onset` 的定义（10 步滑动均值 ≥ 0.10 且
此后连续 20 步 ≥ 0.10）。

用法：

    python analysis/w05_metrics.py \
        --experiment EXP-W05-01-l4-d3-overlong --hack-id L4 \
        --rollout-dir outputs/w05/rollouts/EXP-W05-01-l4-d3-overlong \
        --tokenizer models/Qwen2.5-0.5B-Instruct \
        --output reports/w05_metrics/EXP-W05-01-l4-d3-overlong.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from analysis.hack_metrics import load_tokenizer, tokenize_lengths  # noqa: E402
from selfevolve.metrics.hack_detectors import (  # noqa: E402
    aggregate_responses,
    hack_onset,
    rolling_mean,
)
from selfevolve.metrics.hack_rate import HACK_RULE_TEXT, hack_flag  # noqa: E402

DEFAULT_MAX_RESPONSE_LEN = 1024
EVAL_DATASETS = ("gsm8k", "math500")


def _step_files(rollout_dir: Path, limit_steps: int = 0) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in rollout_dir.glob("*.jsonl"):
        try:
            files.append((int(path.stem), path))
        except ValueError:
            continue
    files.sort(key=lambda item: item[0])
    return files[:limit_steps] if limit_steps > 0 else files


def _read_rows(path: Path) -> list[dict[str, Any]]:
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    rows: list[dict[str, Any]] = []
    bad = 0
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad and bad > max(2, int(0.05 * len(lines))):
        raise ValueError(f"{path}: {bad}/{len(lines)} malformed lines")
    if bad:
        print(f"[w05_metrics] WARN {path}: skipped {bad} malformed line(s)")
    return rows


def _strict(row: dict[str, Any]) -> float:
    if row.get("strict_acc") is not None:
        return float(row["strict_acc"])
    return float(row.get("score", 0.0))


def _zero_variance_group_rate(rows: list[dict[str, Any]]) -> float:
    """同一 prompt 的 rollout 奖励标准差为 0 的比例（GRPO 没有优势信号）。"""
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = str(row.get("input", ""))
        groups.setdefault(key, []).append(float(row.get("score", 0.0)))
    if not groups:
        return 0.0
    zero = 0
    for rewards in groups.values():
        if len(rewards) < 2:
            continue
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
        if std == 0.0:
            zero += 1
    return zero / len(groups)


def analyse_step(
    rows: list[dict[str, Any]],
    step: int,
    *,
    hack_id: str,
    tokenizer: Any,
    max_response_length: int,
) -> dict[str, Any]:
    texts = [str(row.get("output", "")) for row in rows]
    lengths = tokenize_lengths(tokenizer, texts) if tokenizer is not None else None
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        record = {
            "text": texts[index],
            "loose_score": float(row.get("score", 0.0)),
            "strict_score": _strict(row),
        }
        if lengths is not None:
            record["num_tokens"] = lengths[index]
            record["truncated"] = lengths[index] >= max_response_length
        records.append(record)

    aggregated = aggregate_responses(records)
    flags = [hack_flag(hack_id, row) for row in rows]
    aggregated["step"] = step
    aggregated["hack_id"] = hack_id
    aggregated["hack_count"] = sum(flags)
    aggregated["hack_rate"] = sum(flags) / len(rows) if rows else 0.0
    aggregated["zero_variance_group_rate"] = _zero_variance_group_rate(rows)

    if any("d3_state" in row for row in rows):
        states: dict[str, int] = {}
        for row in rows:
            states[str(row.get("d3_state", ""))] = states.get(str(row.get("d3_state", "")), 0) + 1
        aggregated["d3_states"] = states
        aggregated["d3_penalty_mean"] = sum(float(row.get("d3_penalty", 0.0)) for row in rows) / len(rows)
        aggregated["tokenizer_fallback_rate"] = sum(
            int(row.get("tokenizer_fallback", 0) or 0) for row in rows
        ) / len(rows)
    if any("d2_status" in row for row in rows):
        statuses: dict[str, int] = {}
        for row in rows:
            status = str(row.get("d2_status", ""))
            if status:
                statuses[status] = statuses.get(status, 0) + 1
        aggregated["d2_status_counts"] = statuses
    if any("d4_status" in row for row in rows):
        statuses = {}
        for row in rows:
            status = str(row.get("d4_status", ""))
            if status:
                statuses[status] = statuses.get(status, 0) + 1
        aggregated["d4_status_counts"] = statuses
        aggregated["d4_agree_rate"] = sum(float(row.get("d4_agree", 0.0)) for row in rows) / len(rows)
    return aggregated


def _window_mean(steps: Sequence[dict[str, Any]], key: str, n: int) -> float:
    values = [float(row[key]) for row in steps[-n:] if key in row]
    return sum(values) / len(values) if values else 0.0


def _eval_entries(eval_root: Path, experiment: str) -> list[dict[str, Any]]:
    pattern = re.compile(rf"^{re.escape(experiment)}-global_step_(\d+)$")
    entries: list[dict[str, Any]] = []
    if not eval_root.exists():
        return entries
    for path in sorted(eval_root.iterdir()):
        match = pattern.match(path.name)
        summary_path = path / "summary.json"
        if not match or not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        entry: dict[str, Any] = {"step": int(match.group(1)), "label": "checkpoint", "datasets": {}}
        for name, values in summary.get("datasets", {}).items():
            entry["datasets"][name] = {
                "score": values.get("score"),
                "stderr": values.get("binomial_stderr", values.get("stderr")),
                "truncation_rate": values.get("truncation_rate"),
                "empty_answer_rate": values.get("empty_answer_rate"),
                "avg_output_tokens": values.get("avg_output_tokens"),
                "elapsed_s": values.get("elapsed_s"),
            }
        entries.append(entry)
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W5 per-step defense metrics")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--hack-id", required=True, choices=["L1", "L2", "L3", "L4", "strict", "D1"])
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--eval-root", default=str(REPO_ROOT / "outputs" / "eval"))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-response-length", type=int, default=DEFAULT_MAX_RESPONSE_LEN)
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.exists():
        parser.error(f"rollout dir not found: {rollout_dir}")
    tokenizer = load_tokenizer(args.tokenizer)

    files = _step_files(rollout_dir, args.limit_steps)
    if not files:
        parser.error(f"no <step>.jsonl under {rollout_dir}")
    steps = [
        analyse_step(
            _read_rows(path),
            step,
            hack_id=args.hack_id,
            tokenizer=tokenizer,
            max_response_length=args.max_response_length,
        )
        for step, path in files
    ]

    step_numbers = [int(row["step"]) for row in steps]
    gaps = [float(row["reward_gap"]) for row in steps]
    rates = [float(row["hack_rate"]) for row in steps]

    summary = {
        "n_train_steps": len(steps),
        "last_step": step_numbers[-1],
        "gap_onset_step": hack_onset(step_numbers, gaps),
        "hack_onset_step": hack_onset(step_numbers, rates),
        "last20": {
            "loose_mean": _window_mean(steps, "loose_mean", 20),
            "strict_mean": _window_mean(steps, "strict_mean", 20),
            "reward_gap": _window_mean(steps, "reward_gap", 20),
            "hack_rate": _window_mean(steps, "hack_rate", 20),
            "len_chars_mean": _window_mean(steps, "len_chars_mean", 20),
            "len_tokens_mean": _window_mean(steps, "len_tokens_mean", 20),
            "truncation_rate": _window_mean(steps, "truncation_rate", 20),
            "zero_variance_group_rate": _window_mean(steps, "zero_variance_group_rate", 20),
        },
        "last50": {
            "hack_rate": _window_mean(steps, "hack_rate", 50),
            "reward_gap": _window_mean(steps, "reward_gap", 50),
            "strict_mean": _window_mean(steps, "strict_mean", 50),
            "len_chars_mean": _window_mean(steps, "len_chars_mean", 50),
        },
        "first10": {
            "hack_rate": _window_mean(steps, "hack_rate", 10),
            "reward_gap": _window_mean(steps, "reward_gap", 10),
            "strict_mean": _window_mean(steps, "strict_mean", 10),
        },
        "smoothed_gap_final": rolling_mean(gaps, 10)[-1],
        "smoothed_hack_rate_final": rolling_mean(rates, 10)[-1],
    }

    metrics = {
        "experiment": args.experiment,
        "hack_id": args.hack_id,
        "hack_rule": HACK_RULE_TEXT.get(args.hack_id, ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "rollout_dir": str(rollout_dir),
            "max_response_length": args.max_response_length,
            "tokenizer": args.tokenizer,
        },
        "summary": summary,
        "training": steps,
        "eval": _eval_entries(Path(args.eval_root), args.experiment),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[w05_metrics] wrote {output}")
    print(
        f"[w05_metrics] {args.experiment} last20 gap={summary['last20']['reward_gap']:.3f} "
        f"strict={summary['last20']['strict_mean']:.3f} hack={summary['last50']['hack_rate']:.3f} "
        f"onset gap={summary['gap_onset_step']} hack={summary['hack_onset_step']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
