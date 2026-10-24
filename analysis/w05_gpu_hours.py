#!/usr/bin/env python
"""W5 GPU·h 统计：从训练日志的逐步 `timing_s/step` 与评测 summary 的 `elapsed_s` 汇总。

为什么用逐步耗时而不是墙钟：W4/W5 的实验是"训练 → merge → 评测"串行跑的，墙钟里混入了
模型加载、merge、评测和人工等待；registry 的 gpu_hours 只记实际占卡时间。

- 训练：对每个 `global_step` 取最后一次出现的 `timing_s/step`（断点续跑会有重复），求和；
- 评测：`outputs/eval/<EXP>-global_step_N/summary.json` 里每个数据集的 `elapsed_s` 求和；
- merge：`eval/merge_lora.py` 的耗时没有逐 run 日志，按 0 计并在输出里注明（<1 分钟，可忽略）。

用法：

    python analysis/w05_gpu_hours.py --experiments EXP-W05-01-l4-d3-overlong ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]

STEP_RE = re.compile(r"training/global_step:(\d+)")
TIME_RE = re.compile(r"timing_s/step:([0-9.]+)")


def training_seconds(exp: str, logs_dir: Path) -> tuple[float, int, Path | None]:
    logs = sorted(logs_dir.glob(f"{exp}_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not logs:
        return 0.0, 0, None
    log = logs[0]
    per_step: dict[int, float] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").split("\n"):
        step_match = STEP_RE.search(line)
        time_match = TIME_RE.search(line)
        if not step_match or not time_match:
            continue
        per_step[int(step_match.group(1))] = float(time_match.group(1))
    return sum(per_step.values()), len(per_step), log


def eval_seconds(exp: str, eval_root: Path) -> tuple[float, list[str]]:
    total = 0.0
    runs: list[str] = []
    pattern = re.compile(rf"^{re.escape(exp)}-global_step_(\d+)$")
    if not eval_root.exists():
        return 0.0, runs
    for path in sorted(eval_root.iterdir()):
        if not pattern.match(path.name):
            continue
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        elapsed = 0.0
        for values in summary.get("datasets", {}).values():
            elapsed += float(values.get("elapsed_s") or 0.0)
        total += elapsed
        runs.append(f"{path.name}:{elapsed:.0f}s")
    return total, runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W5 GPU hours")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--logs-dir", default=str(REPO_ROOT / "logs"))
    parser.add_argument("--eval-root", default=str(REPO_ROOT / "outputs" / "eval"))
    args = parser.parse_args(argv)

    print("| 实验 | 训练 GPU·h | 评测 GPU·h | 合计 GPU·h | 训练步数 | 评测 run |")
    print("|---|---|---|---|---|---|")
    for exp in args.experiments:
        train_s, n_steps, log = training_seconds(exp, Path(args.logs_dir))
        eval_s, runs = eval_seconds(exp, Path(args.eval_root))
        total_h = (train_s + eval_s) / 3600.0
        print(
            f"| `{exp}` | {train_s / 3600.0:.2f} | {eval_s / 3600.0:.2f} | **{total_h:.2f}** | "
            f"{n_steps} | {'; '.join(runs) if runs else '—'} |"
        )
    print("\n> merge 时间未计入（<1 分钟/checkpoint）；训练耗时是逐步 `timing_s/step` 之和，")
    print("> 不含模型加载/启动（约 1–2 分钟/次）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
