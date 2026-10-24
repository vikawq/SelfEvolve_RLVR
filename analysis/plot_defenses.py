#!/usr/bin/env python
"""W5 防御曲线图：每个 hack 一张三联图（reward gap / hack rate / 响应长度），防御组 vs 未防御组。

输入是 `analysis/w05_metrics.py` 的 JSON（同一套指标 schema）。

用法：

    python analysis/plot_defenses.py \
        --metrics 'reports/w05_metrics/*.json' \
        --out-dir reports/figures
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

SERIES: dict[str, list[tuple[str, str, str]]] = {
    "L1": [
        ("EXP-W04-01-l1-format-weight", "L1 undefended", "#1f77b4"),
        ("EXP-W05-05-l1-d4-dual-parser", "L1 + D4 dual parser", "#2ca02c"),
    ],
    "L2": [
        ("EXP-W04-02-l2-substring", "L2 undefended", "#1f77b4"),
        ("EXP-W05-02-l2-d2-unique-answer", "L2 + D2 unique answer", "#2ca02c"),
        ("EXP-W05-03-l2-kl-0.01", "L2 + KL 0.01", "#ff7f0e"),
        ("EXP-W05-04-l2-kl-0.05", "L2 + KL 0.05", "#d62728"),
    ],
    "L4": [
        ("EXP-W04-04-l4-length-bonus", "L4 undefended", "#1f77b4"),
        ("EXP-W05-01-l4-d3-overlong", "L4 + D3 overlong", "#2ca02c"),
    ],
}

TITLES = {
    "L1": "L1 format weight: D4 cannot fix a specification loophole",
    "L2": "L2 substring match: D2 vs stronger KL",
    "L4": "L4 length bonus: D3 overlong shaping",
}

CONTROL = "EXP-W04-00-control-strict"


def _rolling(values: list[float], window: int = 10) -> list[float]:
    out: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
            out.append(total / window)
        else:
            out.append(total / (index + 1))
    return out


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_hack(hack: str, metrics: dict[str, dict[str, Any]], control: dict[str, Any] | None, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    plotted = 0
    for exp, label, color in SERIES[hack]:
        data = metrics.get(exp)
        if not data:
            continue
        steps = [int(row["step"]) for row in data["training"]]
        gaps = _rolling([float(row["reward_gap"]) for row in data["training"]])
        rates = _rolling([float(row["hack_rate"]) for row in data["training"]])
        lengths = _rolling([float(row["len_chars_mean"]) for row in data["training"]])
        axes[0].plot(steps, gaps, color=color, linewidth=1.9, label=label)
        axes[1].plot(steps, rates, color=color, linewidth=1.9, label=label)
        axes[2].plot(steps, lengths, color=color, linewidth=1.9, label=label)
        plotted += 1

    if control:
        control_steps = [int(row["step"]) for row in control["training"]]
        control_gap = _rolling([float(row["reward_gap"]) for row in control["training"]])
        control_hack = _rolling([float(row["hack_rate"]) for row in control["training"]])
        control_len = _rolling([float(row["len_chars_mean"]) for row in control["training"]])
        axes[0].plot(control_steps, control_gap, color="#7f7f7f", linestyle="--", linewidth=1.1, label="D1 control")
        axes[1].plot(control_steps, control_hack, color="#7f7f7f", linestyle="--", linewidth=1.1, label="D1 control")
        axes[2].plot(control_steps, control_len, color="#7f7f7f", linestyle="--", linewidth=1.1, label="D1 control")

    axes[0].axhline(0.10, color="black", linestyle=":", linewidth=1.0)
    axes[0].set_title("reward gap (loose − strict), smoothed")
    axes[0].set_ylabel("gap")
    axes[1].axhline(0.10, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_title("hack rate (rule per hack), smoothed")
    axes[1].set_ylabel("fraction of rollouts")
    axes[2].set_title("response length (chars), smoothed")
    axes[2].set_ylabel("chars")
    for ax in axes:
        ax.set_xlabel("training step")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle(TITLES.get(hack, hack), fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"w05_{hack.lower()}_defense.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_defenses] wrote {out_path} ({plotted} series)")
    return out_path


def plot_summary(metrics: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows: list[tuple[str, float, str]] = []
    for hack, series in SERIES.items():
        for exp, label, color in series:
            data = metrics.get(exp)
            if not data:
                continue
            rate = float(data["summary"].get("last50", {}).get("hack_rate", 0.0))
            rows.append((label, rate, color))
    control = metrics.get(CONTROL)
    if control:
        rows.insert(0, ("D1 control (strict)", 0.0, "#7f7f7f"))

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=150)
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[2] for row in rows]
    positions = np.arange(len(rows))
    ax.bar(positions, values, color=colors, alpha=0.85)
    for pos, value in zip(positions, values):
        ax.text(pos, value + 0.008, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("hack rate (last 50 steps)")
    ax.set_title("W5 defense summary: hack rate on training rollouts")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "w05_hack_rate_summary.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot_defenses] wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="plot W5 defense curves")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "figures"))
    args = parser.parse_args(argv)

    metrics: dict[str, dict[str, Any]] = {}
    for path in args.metrics:
        data = _load(Path(path))
        metrics[data["experiment"]] = data
    control = metrics.get(CONTROL)
    out_dir = Path(args.out_dir)
    for hack in SERIES:
        plot_hack(hack, metrics, control, out_dir)
    plot_summary(metrics, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
