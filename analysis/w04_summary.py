#!/usr/bin/env python
"""W4 汇总表：把 `analysis/hack_metrics.py` 的 5 份 JSON 变成周报/报告用的 Markdown 表。

用法：

    python analysis/w04_summary.py --metrics-dir outputs/w04/metrics

输出到 stdout（可直接粘进 `reports/week04_summary.md`）：

1. 训练侧：last20 宽松奖励 / 严格准确率 / reward gap / onset；
2. 评测侧：step0/100/200 的 GSM8K test 500 与 MATH-500 严格准确率（附 binomial stderr）；
3. "复现判定"：gap last20 ≥ 0.10 且比对照组同期高 ≥ 0.10 → `yes`，否则 `no`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Windows 控制台默认 GBK，直接 print 中文/数学符号会 UnicodeEncodeError；统一按 UTF-8 输出。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]

ORDER = [
    "EXP-W04-00-control-strict",
    "EXP-W04-01-l1-format-weight",
    "EXP-W04-02-l2-substring",
    "EXP-W04-03-l3-first-number",
    "EXP-W04-04-l4-length-bonus",
]

HACK_LABELS = {
    "strict": "对照组：严格 verifier",
    "L1": "L1 格式权重过高",
    "L2": "L2 子串匹配",
    "L3": "L3 取第一个数字",
    "L4": "L4 长度奖励",
}


def load_all(metrics_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(metrics_dir.glob("EXP-W04-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data["experiment"]] = data
    return out


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def eval_cell(metrics: dict[str, Any], step: int, dataset: str) -> str:
    for entry in metrics.get("eval", []):
        if int(entry["step"]) != step:
            continue
        values = entry["datasets"].get(dataset)
        if not values:
            return "—"
        stderr = values.get("strict_harness_stderr")
        score = values.get("strict_harness_score")
        if score is None:
            score = values["strict_mean"]
        if stderr is None:
            return f"{float(score):.4f}"
        return f"{float(score):.4f} ±{float(stderr):.4f}"
    return "—"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W4 summary tables")
    parser.add_argument("--metrics-dir", default=str(REPO_ROOT / "outputs" / "w04" / "metrics"))
    args = parser.parse_args(argv)

    all_metrics = load_all(Path(args.metrics_dir))
    if not all_metrics:
        raise SystemExit(f"no metrics json under {args.metrics_dir}")

    control = all_metrics.get("EXP-W04-00-control-strict")
    control_gap = control["summary"]["reward_gap_final20_mean"] if control else 0.0

    print("### 训练侧（200 步，最后 20 步均值；gap = 宽松奖励 - 严格准确率）\n")
    print("| 实验 | 漏洞 | 宽松奖励 | 严格准确率 | reward gap | 对照 gap | onset | 判定 |")
    print("|---|---|---|---|---|---|---|---|")
    for exp in ORDER:
        metrics = all_metrics.get(exp)
        if not metrics:
            continue
        summary = metrics["summary"]
        gap = summary["reward_gap_final20_mean"]
        delta = gap - control_gap
        verdict = "yes" if gap >= 0.10 and delta >= 0.10 else "no"
        print(
            f"| `{exp}` | {HACK_LABELS.get(metrics['hack_id'], metrics['hack_id'])} | "
            f"{_fmt(summary['loose_mean_final20_mean'])} | "
            f"{_fmt(summary['strict_mean_final20_mean'])} | {_fmt(gap)} | {_fmt(control_gap)} | "
            f"{summary['hack_onset_step'] if summary['hack_onset_step'] is not None else '—'} | {verdict} |"
        )

    print("\n### 评测侧（严格 verifier；GSM8K test 前 500 / MATH-500）\n")
    print("| 实验 | GSM8K step0 | GSM8K step100 | GSM8K step200 | MATH-500 step0 | MATH-500 step100 | MATH-500 step200 |")
    print("|---|---|---|---|---|---|---|")
    for exp in ORDER:
        metrics = all_metrics.get(exp)
        if not metrics:
            continue
        print(
            f"| `{exp}` | {eval_cell(metrics, 0, 'gsm8k')} | {eval_cell(metrics, 100, 'gsm8k')} | "
            f"{eval_cell(metrics, 200, 'gsm8k')} | {eval_cell(metrics, 0, 'math500')} | "
            f"{eval_cell(metrics, 100, 'math500')} | {eval_cell(metrics, 200, 'math500')} |"
        )

    print("\n> 判定口径见 `reports/week04_plan.md §3`：gap last20 >= 0.10 且比对照组同期高 >= 0.10。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
