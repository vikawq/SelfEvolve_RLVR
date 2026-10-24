#!/usr/bin/env python
"""W5 样例选择：从防御组 rollout 里挑"防御确实拦住了投机"与"防御代价"的典型样本。

判定与训练时一致：对每条 rollout 重新跑底层宽松判分器与防御判分器，用
`metrics/hack_rate.py` 的规则判断 base/defended 是否 hack。三类样例：

1. `fixed`：base 判为 hack、defended 不判 hack（防御生效）——优先展示；
2. `penalized`：D3 触发 linear/hard_cap（长度代价）；
3. `disagree`：D4 两个解析器不一致（解析层防御的实际触发点）。

每个 step 最多取 1 条，保证样例来自不同步；输出可直接贴进报告/实验记录的 Markdown。

用法：

    python analysis/w05_select_examples.py \
        --experiment EXP-W05-02-l2-d2-unique-answer --base-hack L2 --defenses D2 \
        --rollout-dir outputs/w05/rollouts/EXP-W05-02-l2-d2-unique-answer \
        --steps 50 --top-k 3 --output reports/hack_examples/EXP-W05-02-l2-d2-unique-answer.md
"""

from __future__ import annotations

import argparse
import importlib
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

from analysis.w05_replay_defenses import BASE_MODULES, _load_rows, _step_files  # noqa: E402
from selfevolve.metrics.hack_rate import HACK_RULE_TEXT, is_hack_row  # noqa: E402


def _snippet(text: str, limit: int = 900) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…（截断）"


def _row(base: dict[str, Any], text: str, score: float) -> dict[str, Any]:
    return {
        "score": float(score),
        "strict_acc": float(base.get("strict_acc", 0.0)),
        "boxed_count": int(base.get("boxed_count", 0) or 0),
        "n_chars": len(text),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="select W5 defense examples")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--base-hack", required=True, choices=sorted(BASE_MODULES))
    parser.add_argument("--defenses", required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    os.environ["W05_BASE_HACK"] = args.base_hack
    os.environ["W05_DEFENSES"] = args.defenses
    from selfevolve.rewards import defended_reward

    base_compute = importlib.import_module(BASE_MODULES[args.base_hack]).compute_score
    files = _step_files(Path(args.rollout_dir), args.steps)
    if not files:
        raise SystemExit(f"no <step>.jsonl under {args.rollout_dir}")
    rows = _load_rows(files, 0)

    fixed: list[dict[str, Any]] = []
    penalized: list[dict[str, Any]] = []
    disagree: list[dict[str, Any]] = []
    base_hack_count = 0
    def_hack_count = 0
    for row in rows:
        text = str(row.get("output", ""))
        gold = str(row.get("gts", ""))
        data_source = "openai/gsm8k"
        base = base_compute(data_source=data_source, solution_str=text, ground_truth=gold)
        defended = defended_reward.compute_score(
            data_source=data_source, solution_str=text, ground_truth=gold
        )
        base_hack = is_hack_row(args.base_hack, _row(base, text, float(base.get("score", 0.0))))
        def_hack = is_hack_row(args.base_hack, _row(base, text, float(defended.get("score", 0.0))))
        base_hack_count += int(base_hack)
        def_hack_count += int(def_hack)
        record = {
            "step": int(row.get("_step", 0)),
            "gold": gold,
            "text": text,
            "base_score": float(base.get("score", 0.0)),
            "defended_score": float(defended.get("score", 0.0)),
            "strict_acc": float(base.get("strict_acc", 0.0)),
            "strict_reason": base.get("strict_reason", ""),
            "d2_status": defended.get("d2_status", ""),
            "d3_state": defended.get("d3_state", ""),
            "d3_tokens": defended.get("d3_tokens", 0),
            "d3_penalty": defended.get("d3_penalty", 0.0),
            "d4_agree": defended.get("d4_agree", 0.0),
            "d4_status": defended.get("d4_status", ""),
        }
        if base_hack and not def_hack:
            fixed.append(record)
        if defended.get("d3_state") in ("linear", "hard_cap"):
            penalized.append(record)
        if "D4" in args.defenses and float(defended.get("d4_agree", 0.0)) == 0.0:
            disagree.append(record)

    def pick(items: list[dict[str, Any]], sort_key: str = "base_score") -> list[dict[str, Any]]:
        items = sorted(items, key=lambda item: item[sort_key], reverse=True)
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in items:
            if item["step"] in seen:
                continue
            seen.add(item["step"])
            out.append(item)
            if len(out) >= args.top_k:
                break
        return out

    lines: list[str] = [
        f"# {args.experiment} 防御样例",
        "",
        f"- hack 规则：{HACK_RULE_TEXT.get(args.base_hack, '')}",
        f"- 防御：`{args.defenses}`；统计窗口：最后 {args.steps} 步，{len(rows)} 条 rollout",
        f"- base hack：{base_hack_count} 条；defended hack：{def_hack_count} 条",
        f"- 防御拦住的 hack 样本：**{len(fixed)}**；D3 触发：{len(penalized)}；D4 不一致：{len(disagree)}",
        "",
        "## 1. 防御拦住的投机样本（base 会拿分、defended 判 0）",
        "",
    ]
    if not fixed:
        lines.append("（本窗口没有 base hack 被拦住的样本）")
    for record in pick(fixed):
        lines.extend(
            [
                f"### step {record['step']} | gold `{record['gold']}` | base {record['base_score']:.2f} → defended {record['defended_score']:.2f}",
                "",
                f"- 严格判定：`{record['strict_acc']:.0f}`（{record['strict_reason']}）；D2 status=`{record['d2_status']}`；D4 agree={record['d4_agree']:.0f}",
                f"- 响应长度：{len(record['text'])} 字符",
                "",
                "```text",
                _snippet(record["text"]),
                "```",
                "",
            ]
        )

    if penalized:
        lines.extend(["## 2. D3 触发样本（超长惩罚）", ""])
        for record in pick(penalized, "d3_penalty"):
            lines.extend(
                [
                    f"### step {record['step']} | gold `{record['gold']}` | {record['d3_tokens']} token | penalty {record['d3_penalty']:.3f} | state `{record['d3_state']}`",
                    "",
                    f"- 严格判定：`{record['strict_acc']:.0f}`；base {record['base_score']:.2f} → defended {record['defended_score']:.2f}",
                    "",
                    "```text",
                    _snippet(record["text"], 600),
                    "```",
                    "",
                ]
            )

    if disagree:
        lines.extend(["## 3. D4 解析器不一致样本", ""])
        for record in pick(disagree):
            lines.extend(
                [
                    f"### step {record['step']} | gold `{record['gold']}` | D4 agree=0 | status `{record['d4_status']}`",
                    "",
                    f"- 严格判定：`{record['strict_acc']:.0f}`；base {record['base_score']:.2f} → defended {record['defended_score']:.2f}",
                    "",
                    "```text",
                    _snippet(record["text"], 600),
                    "```",
                    "",
                ]
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[w05_select_examples] wrote {output} (fixed={len(fixed)}, penalized={len(penalized)}, disagree={len(disagree)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
