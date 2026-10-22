#!/usr/bin/env python
"""用当前判分器重新给已保存的生成结果打分（不重新跑模型）。

用途：W2 发现 `#### 460 ####`（重复结尾标记）被旧抽取器误判为空答案，影响 37/1319 条。
生成文本已落盘，重打分是等价且可审计的修复方式；原 summary/JSONL 保留，另写 *_rescored。

    python eval/rescore.py --run-dir outputs/eval/EXP-W02-01-baseline-gsm8k-global_step_116
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.harness.data import DATASETS  # noqa: E402
from eval.harness.scoring import SCORER_VERSION, score  # noqa: E402


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _binomial_stderr(p: float, n: int) -> float:
    """pass@1 的误差要报 binomial stderr；0/1 样本标准差恒在 0.4–0.5，没有信息量。"""
    if n <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def rescore_dataset(run_dir: Path, name: str, suffix: str = "rescored") -> dict:
    spec = DATASETS[name]
    src = run_dir / f"{name}.jsonl"
    dst = run_dir / f"{name}.{suffix}.jsonl"
    per_question: list[float] = []
    n_correct = n_gen = truncated = empty = 0
    token_sums = 0
    before_correct = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            outputs = []
            for o in rec["outputs"]:
                before_correct += int(o.get("correct", False))
                result = score(
                    o["text"],
                    rec["ground_truth"],
                    spec.answer_type,
                    finished=o.get("finish_reason") != "length",
                )
                result.update(
                    finish_reason=o["finish_reason"],
                    num_tokens=o["num_tokens"],
                    text=o["text"],
                )
                outputs.append(result)
                n_gen += 1
                token_sums += o["num_tokens"]
                n_correct += int(result["correct"])
                truncated += int(o["finish_reason"] == "length")
                empty += int(not result["extracted"])
            accuracy = sum(int(o["correct"]) for o in outputs) / len(outputs)
            per_question.append(accuracy)
            rec["outputs"] = outputs
            rec["accuracy"] = accuracy
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    mean_acc = sum(per_question) / len(per_question)
    return {
        "dataset": name,
        "display_name": spec.display_name,
        "num_samples": len(per_question),
        "metric": spec.metric,
        "score": mean_acc,
        "std": _sample_std(per_question),
        "stderr": _sample_std(per_question) / math.sqrt(len(per_question)),
        # 与 run_eval.py 保持同一套误差口径，重打分后的行可以直接进 results_master。
        "binomial_stderr": _binomial_stderr(mean_acc, len(per_question)) if spec.n == 1 else None,
        "reported_stderr_kind": "binomial" if spec.n == 1 else "per_question_sample",
        "num_generations": n_gen,
        "n_per_sample": spec.n,
        "prompt_template": spec.prompt_template,
        "accuracy_over_generations": n_correct / n_gen,
        "truncation_rate": truncated / n_gen,
        "empty_answer_rate": empty / n_gen,
        "avg_output_tokens": token_sums / n_gen,
        "scorer": SCORER_VERSION,
        "rescored": True,
        "score_before": before_correct / n_gen,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--datasets", default=None, help="默认沿用 summary.json 的数据集列表")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--suffix",
        default="rescored",
        help="逐样本输出的后缀（W3 用 strict，避免覆盖 W2 的 *.rescored.jsonl）",
    )
    parser.add_argument(
        "--scorer-note",
        default="re-scored from saved generations; original summary.json preserved",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    names = [d.strip() for d in args.datasets.split(",")] if args.datasets else list(summary["datasets"])
    result = {
        "config": {**summary["config"], "rescored_at": datetime.now(timezone.utc).isoformat(),
                   "scorer": SCORER_VERSION},
        "datasets": {},
        "note": args.scorer_note,
    }
    for name in names:
        before = summary["datasets"][name]["score"]
        after = rescore_dataset(run_dir, name, args.suffix)
        result["datasets"][name] = after
        print(f"[rescore] {name}: {before:.4f} -> {after['score']:.4f} "
              f"(empty {after['empty_answer_rate']:.2%})")
    out = Path(args.out) if args.out else run_dir / f"summary_{args.suffix}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rescore] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
