"""Compute scorer accuracy on the reviewed (disagreement) set + spot-check the unanimous set.

人工核对口径（agent 初判，待项目负责人复核）：
`reference_correct` = 模型是否给出了正确的**最终答案**。
- 自动化部分：strict 抽取到的答案与 gold 归一化后字符串相等 -> 一定正确（96/110）。
- 手工部分：`MANUAL` 表里逐条写明 14 个边界样本的判定与理由。

    python analysis/verifier_accuracy.py --review reports/verifier_review_pack.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.verifier.math_verifier import normalize_answer  # noqa: E402

#: 无法用"抽取==gold 字符串相等"自动判定的边界样本：手工判定模型最终答案是否正确。
#: 每条都写了理由，人工复核时直接看这一列。
MANUAL: dict[str, tuple[bool, str]] = {
    "Qwen2.5-1.5B-Instruct/gsm8k/test/466#0": (True, "结尾 'Kelly paid $1.00 more'，数值 1.00 == gold 1"),
    "Qwen2.5-1.5B-Instruct/gsm8k/test/542#0": (True, "结尾 'Change = ... = $7.00'，gold 7"),
    "Qwen2.5-1.5B-Instruct/gsm8k/test/1083#0": (True, "结尾 'Jenny saved $2.00'，gold 2"),
    "Qwen2.5-1.5B-Instruct/gsm8k/test/23#0": (
        False,
        "模型答 L-8（用 L 表示），gold 是数值 8：模型没有给出题目要的数值",
    ),
    "EXP-W02-01-baseline-gsm8k-global_step_116/gsm8k/test/832#0": (True, "`#### 42.00`，2.00 形式的数值等价"),
    "EXP-W02-01-baseline-gsm8k-global_step_116/gsm8k/test/584#0": (True, "`#### 2.00`，数值等价"),
    "EXP-W02-01-baseline-gsm8k-global_step_116/gsm8k/test/164#0": (
        True,
        "`#### 15` 后紧跟 `-15`：模型先声明 15（=gold）再补 -15，判为**模型正确**（strict 的'最后一行优先'在"
        "这里误判，已记入验证报告 §5 的已知限制）",
    ),
    "EXP-W02-01-baseline-gsm8k-global_step_116/gsm8k/test/1083#0": (True, "`#### 2.00`，数值等价"),
    "Qwen2.5-1.5B-Instruct/test/prealgebra/1436.json#0": (
        True,
        "结尾 '56 Drinkets are equivalent to 6 Trinkets'，gold 6；句内另有一个题面数字 56，"
        "strict 因'结论句里有多个值'放弃抽取 -> no_answer（已知限制）",
    ),
    "Qwen2.5-1.5B-Instruct/test/intermediate_algebra/894.json#0": (
        False,
        "finish_reason=length，输出在推导中途被截断（未给出最终答案）；任务包要求截断判错",
    ),
    "EXP-W02-01-baseline-gsm8k-global_step_116/test/intermediate_algebra/1791.json#0": (
        False,
        "模型 `#### 8`，gold -3/8：模型错误（W2 口径因全文匹配 -3/8 而误判为对）",
    ),
    "EXP-W02-01-baseline-gsm8k-global_step_116/test/algebra/864.json#0": (
        False,
        "模型 `#### 6`，gold (-1,6)：模型只给了纵坐标",
    ),
    "EXP-W02-01-baseline-gsm8k-global_step_116/test/number_theory/22.json#0": (
        False,
        "模型 `#### 1`，gold 2：模型错误",
    ),
    "EXP-W02-01-baseline-gsm8k-global_step_116/test/algebra/1275.json#0": (
        False,
        "模型 `#### 3/3` = 1，gold √3/3 ≈ 0.577：模型错误",
    ),
}


def reference_correct(row: dict) -> tuple[bool, str]:
    sid = row["sample_id"]
    if sid in MANUAL:
        return MANUAL[sid]
    gold = normalize_answer(row["ground_truth"])
    extracted = normalize_answer(row["strict_extracted"])
    if extracted and extracted == gold:
        return True, "strict 抽取的最终答案与 gold 归一化后完全一致（确定性判定，不依赖 Math-Verify）"
    return False, "strict 未抽取到答案或抽取结果与 gold 字符串不一致（需人工复核）"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default=str(REPO_ROOT / "reports" / "verifier_review_pack.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "verifier_accuracy.json"))
    parser.add_argument("--total", type=int, default=420, help="抽样总条数（用于全量口径）")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.review).open(encoding="utf-8")]
    disputed = [r for r in rows if r["disagree"]]
    unanimous = [r for r in rows if not r["disagree"]]

    stats = {name: {"right": 0, "wrong": 0} for name in ("verl", "mathverify", "strict")}
    missing_manual = []
    for row in disputed:
        reference, reason = reference_correct(row)
        if row["sample_id"] not in MANUAL and not reason.startswith("strict 抽取"):
            missing_manual.append(row["sample_id"])
        row["reference_correct"] = reference
        row["reference_reason"] = reason
        for name in ("verl", "mathverify", "strict"):
            verdict = row["verdicts"][name]
            if verdict is None:
                continue
            if bool(verdict) == reference:
                stats[name]["right"] += 1
            else:
                stats[name]["wrong"] += 1

    print(f"review set: {len(disputed)} disputed / {len(rows)} sampled")
    result = {}
    for name, s in stats.items():
        total = s["right"] + s["wrong"]
        acc = s["right"] / total if total else float("nan")
        # 全量口径：一致样本按"全部判对"计入（另做了 20 条抽查，见验证报告）
        full = (len(unanimous) + s["right"]) / args.total
        result[name] = {
            "reviewed": total,
            "right": s["right"],
            "wrong": s["wrong"],
            "accuracy_on_reviewed": round(acc, 4),
            "estimated_accuracy_on_pool": round(full, 4),
        }
        print(
            f"  {name:11s} reviewed {s['right']:3d}/{total:3d} = {acc:.2%}"
            f"   full-pool estimate = {full:.2%} (n={args.total})"
        )
        if missing_manual:
            print("  !! unresolved samples:", missing_manual[:5])

    out = {
        "scorer_version": "selfevolve-strict-verifier-1.0+math-verify-0.9.0",
        "review_file": str(Path(args.review)),
        "sampled": len(rows),
        "disputed": len(disputed),
        "unanimous": len(unanimous),
        "adjudication": "agent 初判（逐条判定模型最终答案是否正确），待项目负责人复核",
        "manual_overrides": {k: {"reference_correct": v[0], "reason": v[1]} for k, v in MANUAL.items()},
        "results": result,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
