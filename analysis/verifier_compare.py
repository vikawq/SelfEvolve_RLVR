#!/usr/bin/env python
"""三种判分器对比 + 人工核对包生成（W3 任务 3）。

对 W2 保存的同一批生成结果，各打一次分：

1. `verl`：verl v0.9.1 内置 GSM8K 判分（`verl/utils/reward_score/gsm8k.py`，正则 `#### (-?[0-9.,]+)`；
   有 verl 安装时直接 import，否则用本文件里逐行复刻的版本并打警告）；
2. `mathverify`：W2 的口径（把整段输出交给 Math-Verify，任一候选命中即算对；
   逐行复刻自 W2 `eval/harness/scoring.py` @ `93b9682`）；
3. `strict`：W3 严格 verifier（`src/selfevolve/verifier/math_verifier.py`）。

产出：
- `<out>.csv`：逐样本三判分器判定（`verl`/`mathverify`/`strict` 三列）+ 分歧标记
  + agent 初判（`agent_verdict`/`agent_note`）+ 留给人填的 `human_verdict`/`human_note`
- `<out>.jsonl`：含完整响应，供人工核对时回查
- `<out>.md`：人工核对包（只含分歧样本，每条都带 agent 初判与理由）

判定口径与 agent 初判的实现都在 `analysis/review_adjudication.py`（与 `verifier_accuracy.py` 共用）。

**重新生成不会抹掉已完成的人工核对**：输出路径上已有 CSV 时，先把里面填好的
`human_verdict` / `human_note` 读出来，再原样写回新 CSV。

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

from analysis.review_adjudication import (  # noqa: E402
    HumanVerdict,
    agent_verdict,
    is_manually_adjudicated,
    read_human_verdicts,
    write_review_csv,
)
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
                        # 带上 run 名：base 与 ckpt 的 sample id 会重复，人工核对时必须能区分
                        "sample_id": f"{run_dir.name}/{rec['id']}#{idx}",
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
    # agent 初判要同时落到 CSV / MD / JSONL，统一在这里算一次
    # （必须在上面写入 strict_extracted 之后：自动规则读的就是它）
    agent, agent_note = agent_verdict(row)
    row.update(
        agent_verdict=int(agent),
        agent_note=agent_note,
        agent_source="manual" if is_manually_adjudicated(row) else "auto",
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

核对方法（与 `analysis/verifier_accuracy.py` 一致）：对每条样本只判断一个事实——
**"模型最终给出的答案对不对"**（不是"这个判分器说得对不对"）：

- 没有 marker、但结尾是一句明确结论（`Therefore, Janet makes $18 ...`）→ 算模型**对**；
- 正确数字只出现在推理里、最终答案写错（`…= -50 … #### 50`）→ 算模型**错**；
- 输出被 `max_tokens` 截断、没给出最终答案 → 算模型**错**（任务包要求截断判错）。

判分器的准确性 = 它的判定与上面这个事实一致的比例。

## 怎么填

每条下面都写了 **agent 初判**（自动跑出来的草稿）和它的理由。核对时只需要：

1. 在 `reports/verifier_review_pack.csv` 的 **`human_verdict`** 列填 `1`（模型答对）或 `0`（模型答错）
   —— 也接受 `对`/`错`、`true`/`false`、`yes`/`no`；
2. 与 agent 初判不一致、或想留个理由时，写在 **`human_note`** 列；
3. 跑 `python analysis/verifier_accuracy.py` 刷新准确率。

**agent 初判不能替代人工核对**：它有 96/110 条是用 strict 自己的抽取结果当参考答案
（`strict_extracted == gold` -> 判模型正确），对 strict 而言是循环论证。
所以 `verifier_accuracy.py` 把两种口径分开报（`results_agent` / `results_human`），
只有 `human_verdict` 填满 110 条之后，`human_review.status` 才会变成 `complete`，
才能对外说"人工核对通过"。

---
"""


def write_review_pack(
    rows: list[dict[str, Any]],
    out_md: Path,
    out_csv: Path,
    out_jsonl: Path,
    human: dict[str, HumanVerdict] | None = None,
) -> None:
    """写 CSV / JSONL / MD 三份核对包。

    `human` 是已完成的人工核对（通常是上一版 CSV 里读出来的），原样写回，不会被覆盖。
    """
    human = human or {}
    # CSV 的列定义与逐行序列化都在 review_adjudication 里：
    # 这里曾经把嵌套的 row["verdicts"] 直接丢给 DictWriter，三家判定被静默写成空列。
    write_review_csv(rows, out_csv, human)

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows:
            record = dict(row)
            got = human.get(row["sample_id"])
            if got is not None:
                record["human_verdict"] = int(got.verdict)
                record["human_note"] = got.note
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    disputed = [r for r in rows if r["disagree"]]
    reviewed = sum(1 for r in disputed if r["sample_id"] in human)
    with out_md.open("w", encoding="utf-8") as fh:
        fh.write(_REVIEW_MD_HEADER)
        fh.write(f"\n共 {len(rows)} 条抽样，其中三判分器不一致 **{len(disputed)}** 条。\n")
        fh.write(f"人工核对进度：**{reviewed}/{len(disputed)}**。\n\n")
        for i, row in enumerate(disputed, 1):
            fh.write(f"\n## {i}. `{row['sample_id']}`（{row['dataset']} / {row['run']}）\n\n")
            fh.write(f"- 金标：`{row['ground_truth']}`\n")
            fh.write(f"- 生成结束原因：`{row['finish_reason']}`\n")
            fh.write(
                f"- 判定：`verl={row['verdicts']['verl']}`"
                f" / `mathverify={row['verdicts']['mathverify']}`"
                f" / `strict={row['verdicts']['strict']}`（strict reason=`{row['strict_reason']}`）\n"
            )
            fh.write(
                f"- W2 抽取：`{row['w2_extracted'][:120]}`"
                f"｜strict 抽取：`{row['strict_extracted'][:120]}`\n"
            )
            fh.write(f"- 题面尾部：{row['question_tail'].strip()[:200]}\n")
            fh.write("\n```text\n" + row["response"][-900:] + "\n```\n")
            source = "手工判定" if row["agent_source"] == "manual" else "自动规则"
            label = "模型答对" if row["agent_verdict"] else "模型答错"
            fh.write(
                f"\n- agent 初判：**{label}**（`{row['agent_verdict']}`，{source}）"
                f"——{row['agent_note']}\n"
            )
            got = human.get(row["sample_id"])
            if got is None:
                fh.write("- 人工核对：_待填_（在 CSV 的 `human_verdict` 列填 1 或 0）\n")
            else:
                mark = "模型答对" if got.verdict else "模型答错"
                agrees = (
                    "与初判一致" if int(got.verdict) == row["agent_verdict"] else "**与初判不一致**"
                )
                note = f"——{got.note}" if got.note else ""
                fh.write(f"- 人工核对：**{mark}**（`{int(got.verdict)}`，{agrees}）{note}\n")


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
    parser.add_argument(
        "--from-jsonl",
        default=None,
        help="跳过抽样与打分，直接读已有的 <out>.jsonl 重建 CSV/MD（不需要 outputs/eval，"
        "也不需要装 math-verify；改了核对包格式或 agent 初判时用这个）",
    )
    parser.add_argument("--out", default="reports/verifier_review_pack")
    parser.add_argument(
        "--drop-human",
        action="store_true",
        help="不保留已有 CSV 里的 human_verdict（默认保留，避免重新生成抹掉人工核对结果）",
    )
    args = parser.parse_args(argv)

    if args.from_jsonl:
        rows = [json.loads(line) for line in Path(args.from_jsonl).open(encoding="utf-8")]
        # 旧版 jsonl 里没有 agent 初判字段，在这里补齐（打分结果原样不动）
        for row in rows:
            agent, agent_note = agent_verdict(row)
            row.update(
                agent_verdict=int(agent),
                agent_note=agent_note,
                agent_source="manual" if is_manually_adjudicated(row) else "auto",
            )
        print(f"[compare] 从 {args.from_jsonl} 重建核对包（{len(rows)} 条，不重新打分）")
        return _finish(args, rows)

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
    return _finish(args, rows)


def _finish(args: argparse.Namespace, rows: list[dict[str, Any]]) -> int:
    """写三份核对包并打印汇总（正常抽样路径与 --from-jsonl 共用）。"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 重新生成前先把已填好的人工核对捞出来，否则一次重跑就把人工成果覆盖掉了。
    # 旧版 CSV 没有 human_verdict 列（require_column=False -> 当成"还没核对"）。
    human: dict[str, HumanVerdict] = {}
    if not args.drop_human:
        human = read_human_verdicts(out.with_suffix(".csv"), require_column=False)
        if human:
            print(f"[compare] 保留已有人工核对 {len(human)} 条（--drop-human 可丢弃）")

    write_review_pack(
        rows, out.with_suffix(".md"), out.with_suffix(".csv"), out.with_suffix(".jsonl"), human
    )
    stale = sorted(set(human) - {r["sample_id"] for r in rows})
    if stale:
        print(f"[compare] !! {len(stale)} 条人工核对在本次抽样里不存在，已丢弃：{stale[:3]}")

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
    disputed = [r for r in rows if r["disagree"]]
    print(f"[compare] disagreements: {len(disputed)}/{len(rows)}")
    manual = sum(1 for r in disputed if r["agent_source"] == "manual")
    print(f"[compare] agent 初判: {manual} 条手工判定 + {len(disputed) - manual} 条自动规则")
    print(f"[compare] 人工核对: {sum(1 for r in disputed if r['sample_id'] in human)}/{len(disputed)} 已填")
    print(f"[compare] -> {out}.md / .csv / .jsonl")
    print(f"[compare] strict scorer version: {STRICT_SCORER_VERSION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
