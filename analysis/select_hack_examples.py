#!/usr/bin/env python
"""W4 典型样例挑选：从 rollout dump 里挑"宽松奖励高、严格判定错"的响应。

任务包要求每种 hack 挑 3 条典型输出、注明出自哪一步。挑选规则写死在代码里，避免事后挑好看的：

1. 只看最后 `--window` 步（默认 40，即 hack 应该已经稳定）；
2. `strict_acc == 0`（严格判错）；
3. 宽松奖励 ≥ `--min-score`（默认按 hack 类型给：L1=0.5、其余=1.0）；
4. 按宽松奖励降序、步数降序；每条样例来自不同 step（避免全是同一步的重复模式）。

输出 Markdown：`reports/hack_examples/<EXP>.md`。

用法：

    python analysis/select_hack_examples.py \
        --experiment EXP-W04-02-l2-substring --hack-id L2 \
        --rollout-dir outputs/w04/rollouts/EXP-W04-02-l2-substring \
        --out reports/hack_examples/EXP-W04-02-l2-substring.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.metrics.hack_detectors import detect_response  # noqa: E402

DEFAULT_MIN_SCORE = {"L1": 0.5, "L2": 1.0, "L3": 1.0, "L4": 1.0, "strict": 1.0}
HACK_TITLES = {
    "L1": "L1 格式权重过高：只要写 `\\boxed{}` 就有 0.5，答案质量不提升",
    "L2": "L2 子串匹配：金标数字出现在响应任意位置就给满分",
    "L3": "L3 取第一个数字：第一个数字对就给满分，最终答案是错的也照样得分",
    "L4": "L4 长度奖励：正确性 + 长度线性加分，响应越长分越高",
    "strict": "对照组（严格 verifier）",
}


def _step_files(rollout_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in rollout_dir.glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError:
            continue
        files.append((step, path))
    files.sort(key=lambda item: item[0])
    return files


def _trim(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, total {len(text)} chars]"


def collect(
    rollout_dir: Path,
    *,
    min_score: float,
    window: int,
    n: int,
    max_per_step: int = 1,
) -> tuple[list[dict[str, Any]], int]:
    files = _step_files(rollout_dir)
    if not files:
        raise FileNotFoundError(f"no <step>.jsonl under {rollout_dir}")
    last_steps = {step for step, _ in files[-window:]}
    candidates: list[dict[str, Any]] = []
    max_step = files[-1][0]
    for step, path in files:
        if step not in last_steps:
            continue
        # 与 hack_metrics 一致：JSONL 按 "\n" 切，避免把 U+2028 当成行分隔符。
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            score = float(row.get("score", 0.0))
            strict = float(row.get("strict_acc", row.get("score", 0.0)))
            if strict != 0.0 or score < min_score:
                continue
            text = str(row.get("output", ""))
            detection = detect_response(text)
            candidates.append(
                {
                    "step": step,
                    "score": score,
                    "strict": strict,
                    "gold": row.get("gts"),
                    "prompt": row.get("input", ""),
                    "text": text,
                    "loose_correct": float(row.get("loose_correct", 0.0)),
                    "strict_reason": row.get("strict_reason", row.get("reason", "")),
                    "extracted": row.get("extracted", ""),
                    "loose_extracted": row.get("loose_extracted", ""),
                    **detection,
                }
            )
    # 优先挑"宽松判分器自己都认为答错、却仍然拿到奖励"的样本：这才是漏洞的直接证据。
    # 只按 score 排序会选到"答案对、但严格 verifier 因格式畸形判错"的样本，容易误导读者。
    candidates.sort(
        key=lambda item: (item["loose_correct"] == 0.0, item["score"], item["step"]),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    per_step: dict[int, int] = {}
    for candidate in candidates:
        if per_step.get(candidate["step"], 0) >= max_per_step:
            continue
        selected.append(candidate)
        per_step[candidate["step"]] = per_step.get(candidate["step"], 0) + 1
        if len(selected) >= n:
            break
    return selected, max_step


def render(experiment: str, hack_id: str, selected: list[dict[str, Any]], max_step: int) -> str:
    lines = [
        f"# {experiment}：典型 hack 样例",
        "",
        f"> 自动挑选（`analysis/select_hack_examples.py`）：最后 40 步内、严格判错、**优先宽松判分器"
        f"自己也认为答错**、宽松奖励最高的 3 条，每条来自不同 step。"
        f"原始数据：`outputs/w04/rollouts/{experiment}/<step>.jsonl`。",
        "",
        f"**漏洞**：{HACK_TITLES.get(hack_id, hack_id)}",
        "",
        "| # | step | 宽松奖励 | 宽松判分器认为 | 严格判定 | 严格 reason | 数字个数 | boxed 数 | 字符数 | 重复 5-gram 占比 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for index, sample in enumerate(selected, start=1):
        loose_verdict = "答对" if sample.get("loose_correct", 0.0) else "**答错**"
        lines.append(
            f"| {index} | {sample['step']} | {sample['score']:.3f} | {loose_verdict} | 错 | "
            f"{sample['strict_reason']} | {sample['n_numbers']} | {sample['boxed_count']} | "
            f"{sample['n_chars']} | {sample['repeat_ngram_ratio']:.3f} |"
        )
    lines.append("")

    for index, sample in enumerate(selected, start=1):
        loose_verdict = "答对" if sample.get("loose_correct", 0.0) else "答错"
        lines.extend(
            [
                f"## 样例 {index}（step {sample['step']}）",
                "",
                f"- 宽松奖励：**{sample['score']:.3f}**；宽松判分器认为：**{loose_verdict}**；"
                f"严格判定：**错**（`{sample['strict_reason']}`）",
                f"- 金标：`{sample['gold']}`",
                f"- 严格抽取：`{sample['extracted']}`；宽松抽取：`{sample['loose_extracted']}`",
                f"- 检测指标：数字 {sample['n_numbers']} 个 / boxed {sample['boxed_count']} 个 / "
                f"{sample['n_chars']} 字符 / 重复 5-gram {sample['repeat_ngram_ratio']:.3f}",
                "",
                "**输出（截断到 1600 字符）**",
                "",
                "```text",
                _trim(sample["text"], 1600),
                "```",
                "",
            ]
        )
    lines.append(f"（本实验最后一步：{max_step}；共挑选 {len(selected)} 条。）")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="select W4 hack examples")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--hack-id", required=True, choices=["L1", "L2", "L3", "L4", "strict"])
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args(argv)

    min_score = args.min_score if args.min_score is not None else DEFAULT_MIN_SCORE[args.hack_id]
    selected, max_step = collect(
        Path(args.rollout_dir),
        min_score=min_score,
        window=args.window,
        n=args.n,
    )
    if not selected:
        print(f"[hack_examples] WARN no candidate in last {args.window} steps for {args.experiment}")
        return 1
    markdown = render(args.experiment, args.hack_id, selected, max_step)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    print(f"[hack_examples] wrote {out} ({len(selected)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
