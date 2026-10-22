"""三判分器在人工核对集上的准确率（W3 任务 3 / 验收标准 2）。

两种口径**分开统计、分开报告**，不可互相替代：

- **`results_agent`（agent 初判）**：自动跑出来的草稿。`MANUAL` 表里 14 条逐条手工判定，
  其余走"strict 抽取 == gold 归一化后相等"的自动规则。这一档对 strict 是**循环论证**
  （用被测对象自己的抽取结果当参考答案），所以**不能**用来声称"严格 verifier 准确率 ≥98%"。
- **`results_human`（人工核对）**：读核对包 CSV 的 `human_verdict` 列。只有这一档能满足
  验收标准 2 的"交给人核对"。覆盖不满时 `human_review.status` 是 `partial`，
  此时**不**给全量口径估计（样本已被人为挑选过，估不出来）。

判定口径与 CSV 读写都在 `analysis/review_adjudication.py`（与 `verifier_compare.py` 共用）。

    # 刷新准确率（默认连 reports/verifier_review_pack.csv 一起读）
    python analysis/verifier_accuracy.py
    # 只看 agent 初判
    python analysis/verifier_accuracy.py --no-human
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from analysis.review_adjudication import (  # noqa: E402
    AUTO_FALSE_REASON,
    MANUAL,
    SCORERS,
    HumanVerdict,
    agent_verdict,
    read_human_verdicts,
)
from selfevolve.verifier.math_verifier import SCORER_VERSION  # noqa: E402


def _rel(path: Path | None) -> str | None:
    """落盘的 JSON 要能在别的机器上看懂：仓库内的路径写成相对路径。"""
    if path is None:
        return None
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _tally(
    graded: list[tuple[dict[str, Any], bool]],
    unanimous_right: int | None,
    total: int,
) -> dict[str, dict[str, Any]]:
    """按判分器统计"判定与参考答案一致"的条数。

    `unanimous_right`：三家一致的样本里算作"判对"的条数；`None` 表示不给全量口径估计。
    """
    out: dict[str, dict[str, Any]] = {}
    for name in SCORERS:
        right = wrong = 0
        for row, reference in graded:
            verdict = row["verdicts"][name]
            if verdict is None:  # verl 对非 GSM8K 不适用
                continue
            if bool(verdict) == reference:
                right += 1
            else:
                wrong += 1
        reviewed = right + wrong
        entry: dict[str, Any] = {
            "reviewed": reviewed,
            "right": right,
            "wrong": wrong,
            "accuracy_on_reviewed": round(right / reviewed, 4) if reviewed else None,
        }
        if unanimous_right is None:
            entry["estimated_accuracy_on_pool"] = None
            entry["estimated_accuracy_note"] = "人工核对未覆盖全部分歧样本，不给全量口径估计"
        else:
            entry["estimated_accuracy_on_pool"] = round((unanimous_right + right) / total, 4)
        out[name] = entry
    return out


def _print_block(title: str, results: dict[str, dict[str, Any]]) -> None:
    print(f"\n[{title}]")
    for name, s in results.items():
        if not s["reviewed"]:
            print(f"  {name:11s} 无可比样本")
            continue
        acc = f"{s['accuracy_on_reviewed']:.2%}"
        pool = s["estimated_accuracy_on_pool"]
        pool_txt = "全量口径 n/a" if pool is None else f"全量口径估计 {pool:.2%}"
        print(f"  {name:11s} {s['right']:3d}/{s['reviewed']:3d} = {acc:>7s}   {pool_txt}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", default=str(REPO_ROOT / "reports" / "verifier_review_pack.jsonl"))
    parser.add_argument(
        "--human",
        default=str(REPO_ROOT / "reports" / "verifier_review_pack.csv"),
        help="人工核对 CSV（读 human_verdict / human_note 两列）",
    )
    parser.add_argument("--no-human", action="store_true", help="忽略人工核对，只报 agent 初判")
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "verifier_accuracy.json"))
    parser.add_argument("--total", type=int, default=420, help="抽样总条数（用于全量口径）")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.review).open(encoding="utf-8")]
    disputed = [r for r in rows if r["disagree"]]
    unanimous = [r for r in rows if not r["disagree"]]

    # --- agent 初判 -------------------------------------------------------
    graded_agent: list[tuple[dict[str, Any], bool]] = []
    unsettled: list[str] = []
    for row in disputed:
        reference, reason = agent_verdict(row)
        row["agent_reference"] = reference
        row["agent_reason"] = reason
        if reason == AUTO_FALSE_REASON:
            # 自动规则在这里只能默认判"模型错"，是猜的 —— 必须人来定
            unsettled.append(row["sample_id"])
        graded_agent.append((row, reference))
    results_agent = _tally(graded_agent, unanimous_right=len(unanimous), total=args.total)

    # --- 人工核对 ---------------------------------------------------------
    human: dict[str, HumanVerdict] = {}
    human_path = None if args.no_human else Path(args.human)
    if human_path is not None:
        human = read_human_verdicts(human_path)

    disputed_ids = {r["sample_id"] for r in disputed}
    graded_human = [(r, human[r["sample_id"]].verdict) for r in disputed if r["sample_id"] in human]

    conflicts = [
        {
            "sample_id": row["sample_id"],
            "agent": int(row["agent_reference"]),
            "human": int(human[row["sample_id"]].verdict),
            "agent_reason": row["agent_reason"],
            "human_note": human[row["sample_id"]].note,
        }
        for row, _ in graded_human
        if human[row["sample_id"]].verdict != row["agent_reference"]
    ]

    # 三家一致的样本上的人工抽查：人说"模型错"而三家都说"对"（或反过来）= 三家一起错了
    spot_confirmed = 0
    spot_contradicted: list[str] = []
    for row in unanimous:
        got = human.get(row["sample_id"])
        if got is None:
            continue
        agreed = {bool(v) for v in row["verdicts"].values() if v is not None}
        if agreed and bool(got.verdict) in agreed:
            spot_confirmed += 1
        else:
            spot_contradicted.append(row["sample_id"])

    complete = bool(disputed) and disputed_ids <= set(human)
    status = "complete" if complete else ("partial" if graded_human else "agent-only")
    results_human = None
    if graded_human:
        results_human = _tally(
            graded_human,
            # 全量口径只有核对完整时才给；三家一致那档扣掉抽查发现的反例
            unanimous_right=(len(unanimous) - len(spot_contradicted)) if complete else None,
            total=args.total,
        )

    orphans = sorted(set(human) - disputed_ids - {r["sample_id"] for r in unanimous})
    n_manual = len(set(MANUAL) & disputed_ids)

    # --- 输出 -------------------------------------------------------------
    print(f"review set: {len(disputed)} disputed / {len(rows)} sampled")
    print(f"human_verdict: {len(graded_human)}/{len(disputed)} 分歧样本已核对  ->  status={status}")
    if spot_confirmed or spot_contradicted:
        print(f"  三家一致样本抽查: {spot_confirmed} 条确认 / {len(spot_contradicted)} 条反例")
    if unsettled:
        print(f"  !! 自动规则未能判定（默认判错，需人工定）: {len(unsettled)} 条 {unsettled[:5]}")
    if conflicts:
        print(f"  !! 人工核对与 agent 初判不一致: {len(conflicts)} 条")
        for c in conflicts[:5]:
            print(f"     {c['sample_id']}: agent={c['agent']} human={c['human']} {c['human_note'][:60]}")
    if orphans:
        print(f"  !! {len(orphans)} 条人工核对不在本次抽样里，已忽略: {orphans[:3]}")

    _print_block("results_agent  agent 初判（对 strict 是循环论证，不可作为验收依据）", results_agent)
    if results_human is None:
        print("\n[results_human  人工核对] 还没有任何 human_verdict —— 验收标准 2 未达成。")
        print(f"  填 {args.human} 的 human_verdict 列（1=模型答对 / 0=模型答错）后重跑本脚本。")
    else:
        _print_block(f"results_human  人工核对（status={status}）", results_human)
        if status != "complete":
            print(f"  覆盖 {len(graded_human)}/{len(disputed)}，未填满前不能声称'人工核对通过'。")

    out = {
        "scorer_version": SCORER_VERSION,
        "review_file": _rel(Path(args.review)),
        "human_review_file": _rel(human_path),
        "sampled": len(rows),
        "disputed": len(disputed),
        "unanimous": len(unanimous),
        "human_review": {
            "status": status,
            "disputed_reviewed": len(graded_human),
            "disputed_total": len(disputed),
            "coverage": round(len(graded_human) / len(disputed), 4) if disputed else None,
            "agreement_with_agent": {
                "same": len(graded_human) - len(conflicts),
                "differ": len(conflicts),
                "conflicts": conflicts,
            },
            "unanimous_spotchecks": {
                "n": spot_confirmed + len(spot_contradicted),
                "confirmed": spot_confirmed,
                "contradicted": spot_contradicted,
            },
            "ignored_sample_ids": orphans,
        },
        "agent_adjudication": {
            "manual": n_manual,
            "auto_rule": len(disputed) - n_manual,
            "unsettled_by_auto_rule": unsettled,
            "caveat": "自动规则用 strict 自己的抽取结果当参考答案，对 strict 是循环论证；"
            "只有 results_human 能作为验收标准 2 的依据",
        },
        "results_basis": "human" if status == "complete" else "agent",
        "results_agent": results_agent,
        "results_human": results_human,
        "manual_overrides": {k: {"reference_correct": v[0], "reason": v[1]} for k, v in MANUAL.items()},
    }
    # 向后兼容：旧字段 `results` 指向当前权威口径，`results_basis` 说明是哪一档
    out["results"] = results_human if status == "complete" else results_agent
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {args.out}  (results_basis={out['results_basis']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
