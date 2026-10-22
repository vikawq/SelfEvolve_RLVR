#!/usr/bin/env python
"""三种判分器对比 + 人工核对包生成（W3 任务 3）。

对 W2 保存的同一批生成结果，各打一次分：

1. `verl`：verl v0.9.1 内置 GSM8K 判分（`verl/utils/reward_score/gsm8k.py`，正则 `#### (-?[0-9.,]+)`；
   有 verl 安装时直接 import，否则用本文件里逐行复刻的版本并打警告）；
2. `mathverify`：W2 的口径（把整段输出交给 Math-Verify，任一候选命中即算对；
   逐行复刻自 W2 `eval/harness/scoring.py` @ `93b9682`）；
3. `strict`：W3 严格 verifier（`src/selfevolve/verifier/math_verifier.py`）。

产出：
- `<out>.csv`：逐样本三判分器判定 + 分歧标记 + agent 初判（待人工复核）
- `<out>.jsonl`：含完整响应，供人工核对时回查
- `<out>.md`：人工核对包（只含分歧样本）

    python analysis/verifier_compare.py --n 300 --seed 42 \
        --out reports/verifier_review_pack
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.verifier.math_verifier import (  # noqa: E402
    SCORER_VERSION as STRICT_SCORER_VERSION,
    extract_answer,
    verify,
)

# ---------------------------------------------------------------------------
# 1. verl 内置 GSM8K 判分（复刻自 verl v0.9.1，commit 1876b06）
# ---------------------------------------------------------------------------
_SOLUTION_CLIP_CHARS = 300


def _verl_extract_solution(solution_str: str, method: str = "strict") -> str | None:
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    if method == "strict":
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if not solutions:
            return None
        return solutions[-1].replace(",", "").replace("$", "")
    answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
    if not answer:
        return None
    for final_answer in reversed(answer):
        if final_answer not in ("", "."):
            return final_answer
    return None


def verl_gsm8k_score(solution_str: str, ground_truth: str) -> float:
    """优先用安装的 verl；没有就用上面的复刻实现（两者在单测里对拍）。"""
    try:  # pragma: no cover - 只在装了 verl 的机器上走这条
        from verl.utils.reward_score.gsm8k import compute_score as _verl_compute_score

        return float(_verl_compute_score(solution_str, ground_truth))
    except Exception:
        pass
    answer = _verl_extract_solution(solution_str, "strict")
    if answer is None:
        return 0.0
    return 1.0 if answer == ground_truth else 0.0


# ---------------------------------------------------------------------------
# 2. W2 口径：整段交给 Math-Verify（复刻自 W2 eval/harness/scoring.py）
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify

    _MV_AVAILABLE = True
except Exception:  # pragma: no cover
    _MV_AVAILABLE = False

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_W2_SCORER = "math-verify==0.9.0 (W2 口径，commit 93b9682)"


def _w2_normalize_gold(ground_truth: str) -> str:
    gt = str(ground_truth).strip()
    if not gt:
        return gt
    if "\\boxed" in gt or "$" in gt:
        return gt
    return "\\boxed{" + gt + "}"


def _w2_extract(response: str) -> str:
    text = response or ""
    if "####" in text:
        segments = [seg.strip() for seg in text.split("####")]
        for seg in reversed(segments):
            if seg:
                return seg[:200]
        return ""
    matches = list(_BOXED_RE.finditer(text))
    if matches:
        start = matches[-1].end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()[:200]
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else ""


def _w2_candidates(response: str) -> list[str]:
    text = response or ""
    cands: list[str] = []
    if "####" in text:
        for seg in [s.strip() for s in text.split("####") if s.strip()][::-1]:
            cands.append(seg)
            cands.append("\\boxed{" + seg + "}")
    cands.append(text)
    extracted = _w2_extract(text)
    if extracted:
        cands.append(extracted)
        cands.append("\\boxed{" + extracted + "}")
    out: list[str] = []
    for cand in cands:
        if cand and cand not in out:
            out.append(cand)
    return out


def w2_mathverify_score(response: str, ground_truth: str) -> dict[str, Any]:
    """W2 口径打分（整段输出参与解析，多候选任一命中即正确）。"""
    result = {"correct": False, "extracted": _w2_extract(response), "error": None}
    if not _MV_AVAILABLE:
        result["error"] = "math-verify not installed"
        return result
    try:
        gold = _mv_parse(_w2_normalize_gold(ground_truth), parsing_timeout=None)
        pred_sets = []
        for cand in _w2_candidates(response):
            try:
                parsed = _mv_parse(cand, parsing_timeout=None)
            except Exception:
                continue
            if parsed and parsed not in pred_sets:
                pred_sets.append(parsed)
        if not gold:
            result["error"] = "gold_parse_empty"
            return result
        if not pred_sets:
            result["error"] = "pred_parse_empty"
            return result
        result["correct"] = any(
            bool(_mv_verify(gold, pred, timeout_seconds=None)) for pred in pred_sets
        )
    except Exception as exc:  # pragma: no cover
        result["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return result


# ---------------------------------------------------------------------------
# 3. 抽样与对拍
# ---------------------------------------------------------------------------
def load_generations(
    run_dir: Path, dataset: str
) -> list[dict[str, Any]]:
    """读 `eval/rescore.py` 产出的逐样本 JSONL（含 finish_reason 与完整 text）。"""
    path = run_dir / f"{dataset}.rescored.jsonl"
    if not path.exists():
        path = run_dir / f"{dataset}.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for idx, out in enumerate(rec["outputs"]):
                rows.append(
                    {
                        "sample_id": f"{rec['id']}#{idx}",
                        "dataset": dataset,
                        "run": run_dir.name,
                        "ground_truth": rec["ground_truth"],
                        "question_tail": (rec["messages"][-1]["content"] or "")[-220:],
                        "response": out["text"],
                        "finish_reason": out.get("finish_reason"),
                        "num_tokens": out.get("num_tokens"),
                        "w2_correct": bool(out.get("correct")),
                        "w2_extracted": out.get("extracted") or "",
                    }
                )
    return rows


def score_sample(row: dict[str, Any]) -> dict[str, Any]:
    response = row["response"]
    gold = row["ground_truth"]
    finished = row["finish_reason"] != "length"

    verl = verl_gsm8k_score(response, gold) if row["dataset"] == "gsm8k" else None
    mv = w2_mathverify_score(response, gold)
    strict = verify(response, gold, finished=finished)

    verdicts = {
        "verl": None if verl is None else int(verl > 0.5),
        "mathverify": int(mv["correct"]),
        "strict": int(strict.correct),
    }
    values = [v for v in verdicts.values() if v is not None]
    row.update(
        verdicts=verdicts,
        disagree=len(set(values)) > 1,
        strict_reason=strict.reason,
        strict_extracted=strict.extracted or "",
        mathverify_extracted=mv["extracted"],
        mathverify_error=mv["error"],
    )
    return row


# ---------------------------------------------------------------------------
# 人工核对包
# ---------------------------------------------------------------------------
_REVIEW_MD_HEADER = """# 严格 Verifier 人工核对包（W3）

生成命令：

```bash
python analysis/verifier_compare.py --n 300 --seed 42 --out reports/verifier_review_pack
```

三个判分器：

| 代号 | 定义 |
|---|---|
| `verl` | verl v0.9.1 内置 GSM8K 判分：取最后 300 字符里最后一个 `#### (-?[0-9.,]+)`，与 gold 字符串相等 |
| `mathverify` | W2 口径：整段输出交给 Math-Verify，任一候选命中即算对（commit `93b9682`） |
| `strict` | W3 严格 verifier：只认最终答案，多候选/截断/无答案一律判错 |

核对方法：对每条样本看"金标 + 模型最终答案"，判断**每个判分器判得对不对**（不是模型答得对不对）。
把结论填进 `人工核对` 列（`verl=对/错`、`mv=对/错`、`strict=对/错`），或直接改 `agent 初判`。
"修正建议"列写你不同意的判定与理由。

统计口径（`reports/verifier_validation.md`）：准确率 = 判分器与人工核对一致的比例。

---
"""


def write_review_pack(rows: list[dict[str, Any]], out_md: Path, out_csv: Path, out_jsonl: Path) -> None:
    import csv

    fieldnames = [
        "sample_id",
        "dataset",
        "run",
        "ground_truth",
        "finish_reason",
        "w2_correct",
        "verl",
        "mathverify",
        "strict",
        "disagree",
        "strict_reason",
        "strict_extracted",
        "mathverify_extracted",
        "agent_note",
        "human_verdict",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    disputed = [r for r in rows if r["disagree"]]
    with out_md.open("w", encoding="utf-8") as fh:
        fh.write(_REVIEW_MD_HEADER)
        fh.write(f"\n共 {len(rows)} 条抽样，其中三判分器不一致 **{len(disputed)}** 条。\n\n")
        for i, row in enumerate(disputed, 1):
            fh.write(f"\n## {i}. `{row['sample_id']}`（{row['dataset']} / {row['run']}）\n\n")
            fh.write(f"- 金标：`{row['ground_truth']}`\n")
            fh.write(f"- 生成结束原因：`{row['finish_reason']}`\n")
            fh.write(
                f"- 判定：`verl={row['verdicts']['verl']}` / `mathverify={row['verdicts']['mathverify']}`"
                f" / `strict={row['verdicts']['strict']}`（strict reason=`{row['strict_reason']}`）\n"
            )
            fh.write(f"- W2 抽取：`{row['w2_extracted'][:120]}`｜strict 抽取：`{row['strict_extracted'][:120]}`\n")
            fh.write(f"- 题面尾部：{row['question_tail'].strip()[:200]}\n")
            fh.write("\n```text\n" + row["response"][-900:] + "\n```\n")
            fh.write(f"\n- agent 初判：{row.get('agent_note', '待填')}\n")
            fh.write("- 人工核对：\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="three-scorer comparison for the strict verifier")
    parser.add_argument(
        "--run-dirs",
        default=None,
        help="逗号分隔的 eval 输出目录；默认读 outputs/eval 下两个 W2 run",
    )
    parser.add_argument("--n", type=int, default=300, help="GSM8K 抽样条数（两个模型各一半）")
    parser.add_argument("--math-n", type=int, default=120, help="MATH-500 补充抽样条数（只用两个数学判分器）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="reports/verifier_review_pack")
    args = parser.parse_args(argv)

    if args.run_dirs:
        run_dirs = [Path(p) for p in args.run_dirs.split(",")]
    else:
        base = REPO_ROOT / "outputs" / "eval"
        run_dirs = [
            base / "Qwen2.5-1.5B-Instruct",
            base / "EXP-W02-01-baseline-gsm8k-global_step_116",
        ]

    rng = random.Random(args.seed)
    gsm_rows: list[dict[str, Any]] = []
    math_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        g = load_generations(run_dir, "gsm8k")
        per_run = max(1, args.n // len(run_dirs))
        gsm_rows.extend(rng.sample(g, min(per_run, len(g))))
        m = load_generations(run_dir, "math500")
        per_run_m = max(1, args.math_n // len(run_dirs))
        math_rows.extend(rng.sample(m, min(per_run_m, len(m))))

    rows = [score_sample(r) for r in gsm_rows + math_rows]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_review_pack(rows, out.with_suffix(".md"), out.with_suffix(".csv"), out.with_suffix(".jsonl"))

    gsm = [r for r in rows if r["dataset"] == "gsm8k"]
    print(f"[compare] samples: {len(rows)} (gsm8k={len(gsm)}, math500={len(rows) - len(gsm)})")
    for name in ("verl", "mathverify", "strict"):
        vals = [r["verdicts"][name] for r in gsm if r["verdicts"][name] is not None]
        if vals:
            print(f"[compare] {name:11s} gsm8k accuracy on sample = {sum(vals) / len(vals):.4f} (n={len(vals)})")
    for name in ("mathverify", "strict"):
        vals = [r["verdicts"][name] for r in rows]
        print(f"[compare] {name:11s} all-sample accuracy = {sum(vals) / len(vals):.4f} (n={len(vals)})")
    agree = Counter(
        (
            r["verdicts"]["mathverify"],
            r["verdicts"]["strict"],
        )
        for r in rows
    )
    print(f"[compare] mathverify vs strict joint counts: {dict(agree)}")
    print(f"[compare] disagreements: {sum(1 for r in rows if r['disagree'])}/{len(rows)}")
    print(f"[compare] -> {out}.md / .csv / .jsonl")
    print(f"[compare] strict scorer version: {STRICT_SCORER_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
