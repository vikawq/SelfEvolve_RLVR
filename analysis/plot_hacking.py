#!/usr/bin/env python
"""W4 绘图：每个实验一张"宽松奖励 vs 严格准确率"双曲线图，并标注 hack onset。

输入是 `analysis/hack_metrics.py` 产出的 JSON。图的读法：

- 蓝线 = 训练 rollout 上的**宽松奖励**（逐步均值，滑动 10 步平滑）；
- 红线 = **同一批 rollout** 上的严格 verifier 准确率（不是评测分数，但训练题不重复，
  200 步覆盖 6400 条 train 题，可当作留出曲线）；
- 红色菱形 / 橙色方块 = GSM8K test 前 500 / MATH-500 的外部严格评测（step0/100/200）；
- 灰色虚线 = 对照组（严格奖励训练）的严格准确率，作为"正常训练"参照；
- 竖线 = reward gap 持续扩大的起点（`hack_detectors.hack_onset`，判据见函数 docstring）。

用法：

    python analysis/plot_hacking.py \
        --metrics 'outputs/w04/metrics/*.json' \
        --control outputs/w04/metrics/EXP-W04-00-control-strict.json \
        --out-dir reports/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

HACK_LABELS = {
    "strict": "control: strict verifier",
    "L1": "L1: format weight",
    "L2": "L2: substring match",
    "L3": "L3: first number",
    "L4": "L4: length bonus",
}

FILE_STEMS = {
    "strict": "control",
    "L1": "l1_format",
    "L2": "l2_substring",
    "L3": "l3_first_number",
    "L4": "l4_length",
}


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


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _control_curve(control: dict[str, Any] | None) -> tuple[list[int], list[float]] | None:
    if not control:
        return None
    steps = [int(row["step"]) for row in control["training"]]
    strict = [float(row["strict_mean"]) for row in control["training"]]
    return steps, _rolling(strict)


def plot_one(
    metrics: dict[str, Any],
    control: dict[str, Any] | None,
    out_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np

    training = metrics["training"]
    steps = np.asarray([int(row["step"]) for row in training])
    loose = np.asarray([float(row["loose_mean"]) for row in training])
    strict = np.asarray([float(row["strict_mean"]) for row in training])
    loose_smooth = np.asarray(_rolling(loose.tolist()))
    strict_smooth = np.asarray(_rolling(strict.tolist()))

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    ax.plot(steps, loose, color="#1f77b4", alpha=0.18, linewidth=0.8)
    ax.plot(steps, strict, color="#d62728", alpha=0.18, linewidth=0.8)
    ax.plot(steps, loose_smooth, color="#1f77b4", linewidth=2.0, label="loose reward (train, smoothed)")
    ax.plot(steps, strict_smooth, color="#d62728", linewidth=2.0, label="strict accuracy (same rollouts)")

    control_curve = _control_curve(control)
    if control_curve and metrics["hack_id"] != "strict":
        ax.plot(
            control_curve[0],
            control_curve[1],
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="control strict accuracy",
        )

    for entry in metrics.get("eval", []):
        x = int(entry["step"])
        gsm = entry["datasets"].get("gsm8k")
        math = entry["datasets"].get("math500")
        if gsm:
            ax.scatter(
                [x],
                [gsm["strict_mean"]],
                marker="D",
                s=46,
                color="#d62728",
                edgecolor="white",
                zorder=5,
            )
        if math:
            ax.scatter(
                [x],
                [math["strict_mean"]],
                marker="s",
                s=42,
                color="#ff7f0e",
                edgecolor="white",
                zorder=5,
            )

    if metrics.get("eval"):
        ax.scatter([], [], marker="D", color="#d62728", label="GSM8K test 500 (strict)")
        ax.scatter([], [], marker="s", color="#ff7f0e", label="MATH-500 (strict)")

    onset = metrics.get("summary", {}).get("hack_onset_step")
    if onset is not None:
        ax.axvline(onset, color="#2ca02c", linestyle=":", linewidth=1.6)
        ax.annotate(
            f"hack onset step {onset}",
            xy=(onset, ax.get_ylim()[1] * 0.96),
            xytext=(6, -2),
            textcoords="offset points",
            color="#2ca02c",
            fontsize=9,
        )

    summary = metrics.get("summary", {})
    ax.set_title(
        f"{metrics['experiment']}  ({HACK_LABELS.get(metrics['hack_id'], metrics['hack_id'])})\n"
        f"final gap={summary.get('reward_gap_final20_mean', 0):.3f} "
        f"(last 20 steps mean), onset={onset}",
        fontsize=10,
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("reward / accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = FILE_STEMS.get(metrics["hack_id"], metrics["hack_id"].lower())
    out_path = out_dir / f"w04_{stem}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_summary(all_metrics: list[dict[str, Any]], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    colors = {"strict": "#7f7f7f", "L1": "#1f77b4", "L2": "#2ca02c", "L3": "#9467bd", "L4": "#d62728"}
    for metrics in all_metrics:
        steps = [int(row["step"]) for row in metrics["training"]]
        gaps = [float(row["reward_gap"]) for row in metrics["training"]]
        ax.plot(
            steps,
            _rolling(gaps),
            linewidth=2.0,
            color=colors.get(metrics["hack_id"], "black"),
            label=f"{metrics['experiment']} ({HACK_LABELS.get(metrics['hack_id'], metrics['hack_id'])})",
        )
    ax.axhline(0.10, color="black", linestyle=":", linewidth=1.0, label="onset threshold 0.10")
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("training step")
    ax.set_ylabel("reward gap (loose reward − strict accuracy), smoothed")
    ax.set_title("W4 reward gap by vulnerability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "w04_reward_gap_summary.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="plot W4 reward hacking curves")
    parser.add_argument("--metrics", nargs="+", required=True, help="一个或多个 metrics JSON")
    parser.add_argument("--control", default=None, help="对照组 metrics JSON（画灰虚线）")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "figures"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    control = load_metrics(Path(args.control)) if args.control else None
    all_metrics = [load_metrics(Path(path)) for path in args.metrics]

    outputs = []
    for metrics in all_metrics:
        path = plot_one(metrics, control, out_dir)
        outputs.append(path)
        print(f"[plot_hacking] wrote {path}")
    if len(all_metrics) > 1:
        path = plot_summary(all_metrics, out_dir)
        outputs.append(path)
        print(f"[plot_hacking] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
