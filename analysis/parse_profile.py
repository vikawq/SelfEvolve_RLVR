#!/usr/bin/env python
"""解析 verl 训练日志 / wandb，输出 profiling 表格与图。

W2 用法：
    python analysis/parse_profile.py \
        --logs logs/EXP-W02-0*.log \
        --out-dir reports/figures

    # 可选：从 wandb 拉取（自建实例）
    python analysis/parse_profile.py --wandb-run <entity>/<project>/<run_id>

输出：
    analysis/profile_summary.csv     每个 run / 每个阶段的统计
    analysis/profile_summary.json    同上，JSON（报告引用）
    reports/figures/profile_phase_pie.png
    reports/figures/profile_sensitivity.png

说明：
- 只解析 `timing_s/*`、`perf/*`、`response_length/*` 等数值键，不用 wandb 私有格式。
- 单步 warmup（step 1）默认剔除；含 testing / save_checkpoint 的步会单独标注，
  阶段占比用"纯训练步"计算，避免验证耗时污染结论。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STEP_RE = re.compile(r"step:(\d+) - (.+)$")
PAIR_RE = re.compile(r"([A-Za-z0-9_./-]+):(-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
TAG_RE = re.compile(r"^\[(EXP-W\d+[^\]]*)\]")

PHASES = {
    "timing_s/gen": "rollout 生成",
    "timing_s/old_log_prob": "old logprob",
    "timing_s/ref": "ref logprob",
    "timing_s/update_actor": "actor 更新",
    "timing_s/update_weights": "权重同步",
    "timing_s/reward": "reward",
    "timing_s/adv": "advantage",
}


@dataclass
class RunData:
    name: str
    source: str
    steps: list[dict[str, float]] = field(default_factory=list)

    def values(self, key: str, skip_first: bool = True) -> list[float]:
        rows = self.steps[1:] if skip_first and len(self.steps) > 1 else self.steps
        return [r[key] for r in rows if key in r and r[key] == r[key]]

    def median(self, key: str) -> float | None:
        vals = self.values(key)
        return statistics.median(vals) if vals else None


def parse_log(path: Path) -> list[RunData]:
    runs: dict[str, RunData] = {}
    current: RunData | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        mtag = TAG_RE.match(line.strip())
        if mtag:
            name = mtag.group(1)
            current = runs.setdefault(name, RunData(name=name, source=str(path)))
        m = STEP_RE.search(line)
        if not m:
            continue
        step = int(m.group(1))
        pairs = {k: float(v) for k, v in PAIR_RE.findall(m.group(2))}
        if not pairs:
            continue
        if current is None:
            current = runs.setdefault(path.stem, RunData(name=path.stem, source=str(path)))
        pairs["training/global_step"] = float(step)
        current.steps.append(pairs)
    return list(runs.values())


def fetch_wandb(run_path: str) -> RunData:
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    keys = ["training/global_step", "perf/throughput", "perf/max_memory_allocated_gb", *PHASES]
    history = run.history(keys=keys, pandas=False)
    steps = []
    for row in history:
        rec = {k: row[k] for k in keys if k in row and row[k] is not None}
        if "training/global_step" in rec:
            rec["training/global_step"] = float(rec["training/global_step"])
            steps.append(rec)
    return RunData(name=run.name, source=run_path, steps=steps)


def summarize(run: RunData) -> dict[str, object]:
    phases = {k: run.median(k) for k in PHASES}
    phase_sum = sum(v for v in phases.values() if v)
    step = run.median("timing_s/step")
    training_steps = []
    for row in run.steps[1:]:
        if row.get("timing_s/testing", 0) or row.get("timing_s/save_checkpoint", 0):
            continue
        if "timing_s/step" in row:
            training_steps.append(row)
    step_train = statistics.median([r["timing_s/step"] for r in training_steps]) if training_steps else step
    return {
        "name": run.name,
        "source": run.source,
        "num_steps": len(run.steps),
        "num_pure_training_steps": len(training_steps),
        "median_step_s": step,
        "median_step_train_s": step_train,
        "median_phases_s": phases,
        "phase_share": {k: (v / phase_sum if phase_sum and v else None) for k, v in phases.items()},
        "median_throughput_tok_s": run.median("perf/throughput"),
        "max_memory_allocated_gb": max(run.values("perf/max_memory_allocated_gb", skip_first=False) or [0.0]),
        "median_response_tokens": run.median("response_length/mean"),
    }


def print_table(summaries: list[dict[str, object]]) -> None:
    print(f"{'run':44s} {'steps':>5s} {'step_s':>8s} {'gen':>7s} {'oldlp':>7s} {'ref':>7s} "
          f"{'actor':>8s} {'sync':>6s} {'tok/s':>8s} {'memGB':>7s}")
    for s in summaries:
        ph = s["median_phases_s"]  # type: ignore[index]
        def fmt(key: str) -> str:
            v = ph.get(key)  # type: ignore[union-attr]
            return f"{v:.2f}" if v else "-"
        print(
            f"{s['name']:44s} {s['num_steps']:5d} "
            f"{(s['median_step_train_s'] or 0):8.2f} {fmt('timing_s/gen'):>7s} "
            f"{fmt('timing_s/old_log_prob'):>7s} {fmt('timing_s/ref'):>7s} "
            f"{fmt('timing_s/update_actor'):>8s} {fmt('timing_s/update_weights'):>6s} "
            f"{(s['median_throughput_tok_s'] or 0):8.1f} {s['max_memory_allocated_gb']:7.2f}"
        )


def plot_pie(summary: dict[str, object], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ph = {k: v for k, v in summary["median_phases_s"].items() if v}  # type: ignore[union-attr]
    labels = [PHASES[k] for k in ph]
    values = [ph[k] for k in ph]
    total = summary["median_step_train_s"] or sum(values)
    other = max(0.0, total - sum(values))
    if other > 0.01:
        labels.append("其他")
        values.append(other)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.pie(values, labels=[f"{l}\n{v:.1f}s" for l, v in zip(labels, values)], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"训练单步耗时构成（{summary['name']}，training steps）")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_sensitivity(summaries: list[dict[str, object]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {"EXP-W02-02": [], "EXP-W02-03": [], "EXP-W02-04": []}
    for s in summaries:
        for g in groups:
            if s["name"].startswith(g):
                groups[g].append(s)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    titles = {
        "EXP-W02-02": "vLLM gpu_memory_utilization",
        "EXP-W02-03": "rollout.n",
        "EXP-W02-04": "micro batch size",
    }
    for ax, (g, items) in zip(axes, groups.items()):
        if not items:
            ax.set_visible(False)
            continue
        labels = [s["name"].replace(g + "-prof-", "").replace("-", "\n") for s in items]
        step = [s["median_step_train_s"] or 0 for s in items]
        actor = [(s["median_phases_s"].get("timing_s/update_actor") or 0) for s in items]  # type: ignore[union-attr]
        gen = [(s["median_phases_s"].get("timing_s/gen") or 0) for s in items]  # type: ignore[union-attr]
        import numpy as np

        x = np.arange(len(items))
        ax.bar(x - 0.25, step, 0.25, label="step total")
        ax.bar(x, actor, 0.25, label="update_actor")
        ax.bar(x + 0.25, gen, 0.25, label="gen")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(titles[g], fontsize=10)
        ax.set_ylabel("seconds (median)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="*", default=[], help="训练日志 glob/路径")
    parser.add_argument("--wandb-run", action="append", default=[], help="entity/project/run_id，可重复")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "figures"))
    parser.add_argument("--csv", default=str(REPO_ROOT / "analysis" / "profile_summary.csv"))
    parser.add_argument("--json", default=str(REPO_ROOT / "analysis" / "profile_summary.json"))
    args = parser.parse_args(argv)

    runs: list[RunData] = []
    for pattern in args.logs:
        for path in sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]:
            if path.exists():
                runs.extend(parse_log(path))
            else:
                print(f"[parse] missing log: {path}", file=sys.stderr)
    for run_id in args.wandb_run:
        try:
            runs.append(fetch_wandb(run_id))
        except Exception as exc:
            print(f"[parse] wandb failed for {run_id}: {type(exc).__name__}: {exc}", file=sys.stderr)

    runs = [r for r in runs if len(r.steps) >= 2]
    if not runs:
        print("[parse] no runs with >=2 steps", file=sys.stderr)
        return 2

    summaries = [summarize(r) for r in runs]
    for s in summaries:
        print(f"[parse] {s['name']}: {len(s['median_phases_s'])} phases, {s['num_steps']} steps")  # type: ignore[arg-type]
    print()
    print_table(summaries)

    out_dir = Path(args.out_dir)
    baseline = next((s for s in summaries if str(s["name"]).startswith("EXP-W02-01")), summaries[0])
    plot_pie(baseline, out_dir / "profile_phase_pie.png")
    plot_sensitivity(summaries, out_dir / "profile_sensitivity.png")

    # CSV：每个 run 一行，阶段列展开
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    phase_keys = list(PHASES)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run", "num_steps", "median_step_train_s", *phase_keys, "median_throughput_tok_s",
                         "max_memory_allocated_gb", "median_response_tokens"])
        for s in summaries:
            ph = s["median_phases_s"]  # type: ignore[index]
            writer.writerow([
                s["name"], s["num_steps"], s["median_step_train_s"],
                *[ph.get(k) for k in phase_keys],
                s["median_throughput_tok_s"], s["max_memory_allocated_gb"], s["median_response_tokens"],
            ])
    Path(args.json).write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[parse] wrote {csv_path}, {args.json}, {out_dir}/profile_phase_pie.png, profile_sensitivity.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
