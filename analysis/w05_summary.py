#!/usr/bin/env python
"""W5 汇总表：把 `analysis/w05_metrics.py` 的 JSON 变成审计报告/周报用的 Markdown 表。

输入：同一 schema 的 metrics JSON（W4 的 5 组已用 `w05_metrics.py` 重新算过 hack rate，
保证"防御 vs 未防御"用同一套规则）。输出到 stdout，可直接粘进报告。

三张表：

1. 每个 hack 的"未防御 vs 防御"：gap、hack rate、严格准确率、长度、step200 外部评测；
2. 防御代价：相对同 hack 未防御组的评测差；
3. D5 KL 扫描：0.001（W4 基线）/0.01/0.05 的 onset 与 hack rate。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]

CONTROL = "EXP-W04-00-control-strict"
BASE = {
    "L1": "EXP-W04-01-l1-format-weight",
    "L2": "EXP-W04-02-l2-substring",
    "L3": "EXP-W04-03-l3-first-number",
    "L4": "EXP-W04-04-l4-length-bonus",
}

ROWS = [
    (CONTROL, "strict", "D1 严格 verifier（根治）"),
    (BASE["L1"], "L1", "未防御"),
    ("EXP-W05-05-l1-d4-dual-parser", "L1", "D4 双解析器"),
    (BASE["L2"], "L2", "未防御"),
    ("EXP-W05-02-l2-d2-unique-answer", "L2", "D2 答案唯一性"),
    ("EXP-W05-03-l2-kl-0.01", "L2", "D5 KL=0.01"),
    ("EXP-W05-04-l2-kl-0.05", "L2", "D5 KL=0.05"),
    (BASE["L4"], "L4", "未防御"),
    ("EXP-W05-01-l4-d3-overlong", "L4", "D3 超长惩罚"),
]

HACK_LABELS = {
    "strict": "对照组",
    "L1": "L1 格式权重",
    "L2": "L2 子串匹配",
    "L3": "L3 首数字",
    "L4": "L4 长度奖励",
}


def load_metrics(metrics_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(metrics_dir.glob("EXP-W0*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data["experiment"]] = data
    return out


def _fmt(value: Any, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def eval_cell(metrics: dict[str, Any], step: int, dataset: str) -> str:
    for entry in metrics.get("eval", []):
        if int(entry.get("step", -1)) != step:
            continue
        values = entry.get("datasets", {}).get(dataset)
        if not values:
            return "—"
        score = values.get("score")
        stderr = values.get("stderr")
        if score is None:
            return "—"
        if stderr is None:
            return f"{float(score):.4f}"
        return f"{float(score):.4f}±{float(stderr):.4f}"
    return "—"


def eval_score(metrics: dict[str, Any], step: int, dataset: str) -> float | None:
    for entry in metrics.get("eval", []):
        if int(entry.get("step", -1)) != step:
            continue
        values = entry.get("datasets", {}).get(dataset)
        if values and values.get("score") is not None:
            return float(values["score"])
    return None


def last20(metrics: dict[str, Any], key: str) -> float | None:
    return metrics.get("summary", {}).get("last20", {}).get(key)


def last50(metrics: dict[str, Any], key: str) -> float | None:
    return metrics.get("summary", {}).get("last50", {}).get(key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W5 summary tables")
    parser.add_argument("--metrics-dir", default=str(REPO_ROOT / "reports" / "w05_metrics"))
    args = parser.parse_args(argv)

    all_metrics = load_metrics(Path(args.metrics_dir))
    if not all_metrics:
        raise SystemExit(f"no metrics JSON under {args.metrics_dir}")

    print("### 表 1：防御前后（训练侧 last20 / last50；评测 step200 为严格 verifier）\n")
    print("| 实验 | hack | 防御 | gap(last20) | hack rate(last50) | 严格 acc(last20) | 长度(chars) | GSM8K@200 | MATH-500@200 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for exp, hack, defense in ROWS:
        metrics = all_metrics.get(exp)
        if not metrics:
            print(f"| `{exp}` | {HACK_LABELS.get(hack, hack)} | {defense} | — | — | — | — | — | — |")
            continue
        print(
            f"| `{exp}` | {HACK_LABELS.get(hack, hack)} | {defense} | "
            f"{_fmt(last20(metrics, 'reward_gap'))} | {_fmt(last50(metrics, 'hack_rate'))} | "
            f"{_fmt(last20(metrics, 'strict_mean'))} | {_fmt(last20(metrics, 'len_chars_mean'), 0)} | "
            f"{eval_cell(metrics, 200, 'gsm8k')} | {eval_cell(metrics, 200, 'math500')} |"
        )

    print("\n### 表 2：防御代价（相对同 hack 未防御组的 step200 严格评测差；负 = 更差）\n")
    print("| 实验 | 防御 | Δ GSM8K@200 | Δ MATH-500@200 | 训练侧严格 acc 差 | 长度变化 |")
    print("|---|---|---|---|---|---|")
    for exp, hack, defense in ROWS:
        if exp == CONTROL or exp in BASE.values():
            continue
        metrics = all_metrics.get(exp)
        base = all_metrics.get(BASE.get(hack, ""))
        if not metrics or not base:
            continue
        d_gsm = (eval_score(metrics, 200, "gsm8k") or 0) - (eval_score(base, 200, "gsm8k") or 0)
        d_math = (eval_score(metrics, 200, "math500") or 0) - (eval_score(base, 200, "math500") or 0)
        d_strict = (last20(metrics, "strict_mean") or 0) - (last20(base, "strict_mean") or 0)
        d_len = (last20(metrics, "len_chars_mean") or 0) - (last20(base, "len_chars_mean") or 0)
        print(
            f"| `{exp}` | {defense} | {d_gsm:+.4f} | {d_math:+.4f} | {d_strict:+.3f} | {d_len:+.0f} |"
        )

    print("\n### 表 3：D5 KL 扫描（L2 子串，同奖励/同数据/同 seed）\n")
    print("| KL 系数 | 实验 | gap onset | hack onset | gap(last20) | hack rate(last50) | 严格 acc(last20) | GSM8K@200 |")
    print("|---|---|---|---|---|---|---|---|")
    for kl, exp in (("0.001", BASE["L2"]), ("0.01", "EXP-W05-03-l2-kl-0.01"), ("0.05", "EXP-W05-04-l2-kl-0.05")):
        metrics = all_metrics.get(exp)
        if not metrics:
            continue
        summary = metrics.get("summary", {})
        print(
            f"| {kl} | `{exp}` | {summary.get('gap_onset_step', '—')} | {summary.get('hack_onset_step', '—')} | "
            f"{_fmt(last20(metrics, 'reward_gap'))} | {_fmt(last50(metrics, 'hack_rate'))} | "
            f"{_fmt(last20(metrics, 'strict_mean'))} | {eval_cell(metrics, 200, 'gsm8k')} |"
        )

    print(
        "\n> hack rate 口径：最后 50 步全部 rollout 中满足 `metrics/hack_rate.py` 规则的样本比例；"
        "规则逐条见 `reports/P7_reward_hacking_audit.md` §4。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
