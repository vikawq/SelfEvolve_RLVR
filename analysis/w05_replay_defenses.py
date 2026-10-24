#!/usr/bin/env python
"""W5 离线反事实回放：把 W4 的 rollout 文本重新过一遍防御判分器，先估效果再烧 GPU。

用途：

1. 开跑前检查 D2/D4 是否会把奖励打到近乎全 0（那样 GRPO 组内无梯度，白跑 200 步）；
2. 给出"同一批生成文本上，防御前后"的 proxy 奖励、reward gap、hack rate；
3. 报告解析器状态分布（ok / multiple / none / truncated），这是补丁式防御的代价来源。

注意：这是**离线分析**，不是实验结果。报告里的结论只引用真实训练/评测数字；
本脚本用于选型与 sanity check。

用法：

    python analysis/w05_replay_defenses.py \
        --base-hack L2 --defenses D2 \
        --rollout-dir outputs/w04/rollouts/EXP-W04-02-l2-substring \
        --steps 20 --output reports/w05_metrics/replay_L2_D2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.metrics.hack_rate import HACK_RULE_TEXT, is_hack_row, summarize  # noqa: E402

BASE_MODULES = {
    "L1": "selfevolve.rewards.loose_L1_format",
    "L2": "selfevolve.rewards.loose_L2_substring",
    "L3": "selfevolve.rewards.loose_L3_first_number",
    "L4": "selfevolve.rewards.loose_L4_length",
}


def _step_files(rollout_dir: Path, last_n: int) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in rollout_dir.glob("*.jsonl"):
        try:
            files.append((int(path.stem), path))
        except ValueError:
            continue
    files.sort(key=lambda item: item[0])
    return files[-last_n:] if last_n > 0 else files


def _load_rows(files: list[tuple[int, Path]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, path in files:
        lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        for line in lines[:limit] if limit > 0 else lines:
            row = json.loads(line)
            row["_step"] = step
            rows.append(row)
    return rows


def _row_for_rules(base: dict[str, Any], text: str, gold: str, score: float) -> dict[str, Any]:
    return {
        "score": float(score),
        "strict_acc": float(base.get("strict_acc", 0.0)),
        "boxed_count": int(base.get("boxed_count", 0) or 0),
        "n_chars": len(text),
        "ground_truth": gold,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W5 offline counterfactual replay")
    parser.add_argument("--base-hack", required=True, choices=sorted(BASE_MODULES))
    parser.add_argument("--defenses", default="", help="如 D2 / D4 / D3 / D2,D4")
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="每个 step 只取前 N 条（0=全部）")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    os.environ["W05_BASE_HACK"] = args.base_hack
    os.environ["W05_DEFENSES"] = args.defenses

    import importlib

    from selfevolve.rewards import defended_reward

    base_compute = importlib.import_module(BASE_MODULES[args.base_hack]).compute_score
    files = _step_files(Path(args.rollout_dir), args.steps)
    if not files:
        raise SystemExit(f"no <step>.jsonl under {args.rollout_dir}")
    rows = _load_rows(files, args.limit)

    transitions = {"base_hack_defended_hack": 0, "base_hack_defended_ok": 0, "base_ok_defended_hack": 0, "base_ok_defended_ok": 0}
    status_counts: dict[str, int] = {}
    agree_count = 0
    rule_rows: list[dict[str, Any]] = []
    base_scores: list[float] = []
    defended_scores: list[float] = []

    for row in rows:
        text = str(row.get("output", ""))
        gold = str(row.get("gts", ""))
        data_source = "openai/gsm8k"
        base = base_compute(data_source=data_source, solution_str=text, ground_truth=gold)
        defended = defended_reward.compute_score(
            data_source=data_source, solution_str=text, ground_truth=gold
        )
        base_score = float(base.get("score", 0.0))
        def_score = float(defended.get("score", 0.0))
        base_scores.append(base_score)
        defended_scores.append(def_score)

        rule_row = _row_for_rules(base, text, gold, def_score)
        rule_row["_base_hack"] = is_hack_row(args.base_hack, _row_for_rules(base, text, gold, base_score))
        rule_row["_def_hack"] = is_hack_row(args.base_hack, rule_row)
        rule_rows.append(rule_row)

        if rule_row["_base_hack"] and rule_row["_def_hack"]:
            transitions["base_hack_defended_hack"] += 1
        elif rule_row["_base_hack"]:
            transitions["base_hack_defended_ok"] += 1
        elif rule_row["_def_hack"]:
            transitions["base_ok_defended_hack"] += 1
        else:
            transitions["base_ok_defended_ok"] += 1

        status = defended.get("d2_status") or defended.get("d4_status") or ""
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        if float(defended.get("d4_agree", 0.0)) > 0:
            agree_count += 1

    strict = [row["strict_acc"] for row in rule_rows]
    n = len(rule_rows)
    base_summary = summarize(args.base_hack, [{**r, "score": b} for r, b in zip(rule_rows, base_scores)])
    def_summary = summarize(args.base_hack, rule_rows)
    result = {
        "base_hack": args.base_hack,
        "defenses": args.defenses,
        "rollout_dir": args.rollout_dir,
        "steps": [step for step, _ in files],
        "n": n,
        "strict_acc": sum(strict) / n,
        "base": base_summary,
        "defended": def_summary,
        "transitions": transitions,
        "status_counts": status_counts,
        "d4_agree_rate": agree_count / n if n else 0.0,
        "rule": HACK_RULE_TEXT.get(args.base_hack, ""),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[replay] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
