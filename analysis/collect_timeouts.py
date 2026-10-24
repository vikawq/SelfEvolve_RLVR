#!/usr/bin/env python
"""汇总 W4 MATH-500 评测里的硬超时样本（id + 全文 + sha256），落盘可审计。

数据来源：
1. `outputs/eval/<run>/math500.jsonl` —— L1 的 batched 全量评测（其余 run 的 harness 结果）；
2. `outputs/eval_batched/<run>/math500.jsonl` —— 其余 4 个实验 + 基座的 `--rescore-only`
   复核（同一批生成文本，见 `reports/w04_metrics/math500_path_compare.md`）。

同一条样本若在两个目录都出现，优先保留 batched 的那份（`outputs/eval_batched`），
因为报告里的 MATH-500 数字以 batched 路径为准。

用法（在 bigdog01 上）：
    python analysis/collect_timeouts.py
    python analysis/collect_timeouts.py --output reports/w04_metrics/math500_timeout_samples.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scan(path: Path, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    run = path.parent.name
    for index, line in enumerate(path.read_text(encoding="utf-8").split("\n")):
        if not line.strip():
            continue
        row = json.loads(line)
        for output_index, output in enumerate(row.get("outputs", [])):
            if output.get("reason") != "timeout" and output.get("error") != "timeout":
                continue
            text = str(output.get("text", ""))
            records.append(
                {
                    "run": run,
                    "dataset": "math500",
                    "sample_index": index,
                    "sample_id": row["id"],
                    "output_index": output_index,
                    "gold": row.get("ground_truth"),
                    "reason": output.get("reason"),
                    "error": output.get("error"),
                    "extracted": output.get("extracted"),
                    "num_tokens": output.get("num_tokens"),
                    "finish_reason": output.get("finish_reason"),
                    "text_chars": len(text),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "timeout_s": 30,
                    "messages": row.get("messages"),
                    "text": text,
                    "source": source,
                }
            )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="collect W4 MATH-500 timeout samples")
    parser.add_argument("--eval-root", default=str(REPO_ROOT / "outputs" / "eval"))
    parser.add_argument("--batched-root", default=str(REPO_ROOT / "outputs" / "eval_batched"))
    parser.add_argument(
        "--output", default=str(REPO_ROOT / "reports" / "w04_metrics" / "math500_timeout_samples.json")
    )
    args = parser.parse_args(argv)

    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    # 先扫 harness，再用 batched 覆盖（batched 是报告采用的口径）
    roots = [
        (Path(args.eval_root), "eval/run_eval.py"),
        (Path(args.batched_root), "w04_eval_batched.py --rescore-only"),
    ]
    for root, source in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/math500.jsonl")):
            for record in _scan(path, source):
                key = (record["run"], record["sample_index"], record["sample_id"])
                by_key[key] = record

    samples = sorted(by_key.values(), key=lambda r: (r["run"], r["sample_index"]))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": (
            "analysis/w04_eval_batched.py 每条样本 30 s 硬超时；超时 kill 子进程并按严格 verifier "
            "的 timeout 语义判错（correct=false, reason=timeout）。"
        ),
        "count": len(samples),
        "samples": samples,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect_timeouts] wrote {output} ({len(samples)} samples)")
    for sample in samples:
        print(
            f"  {sample['run']} index={sample['sample_index']} id={sample['sample_id']} "
            f"chars={sample['text_chars']} sha256={sample['text_sha256'][:12]}…"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
