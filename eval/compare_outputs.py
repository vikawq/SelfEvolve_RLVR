#!/usr/bin/env python
"""对比两次评测的逐样本输出（用于 LoRA merge 前后一致性检查）。

    python eval/compare_outputs.py --left outputs/eval/ckpt-lora --right outputs/eval/ckpt-merged \
        --datasets gsm8k --limit 20

判定：同一题、同一采样参数下，生成文本逐字一致才算一致（greedy 时应当 100%）。
AIME 的 avg@16 有随机性，默认不用于 merge 校验（用 greedy 数据集）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path, limit: int) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            rec = json.loads(line)
            rows[rec["id"]] = [o["text"] for o in rec["outputs"]]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--datasets", default="gsm8k,math500")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    failed = False
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        left_path = Path(args.left) / f"{ds}.jsonl"
        right_path = Path(args.right) / f"{ds}.jsonl"
        if not left_path.exists() or not right_path.exists():
            print(f"[compare] {ds}: SKIP (missing jsonl)")
            continue
        left = _load(left_path, args.limit)
        right = _load(right_path, args.limit)
        ids = list(left)[: args.limit]
        same = 0
        for qid in ids:
            if qid in right and left[qid] == right[qid]:
                same += 1
        total = len(ids)
        rate = same / total if total else 0.0
        print(f"[compare] {ds}: {same}/{total} identical ({rate:.1%})")
        if rate < 1.0:
            failed = True
            for qid in ids:
                if qid in right and left[qid] != right[qid]:
                    print(f"  first mismatch: {qid}")
                    print(f"    left : {left[qid][0][:120]!r}")
                    print(f"    right: {right[qid][0][:120]!r}")
                    break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
