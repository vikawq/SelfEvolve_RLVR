"""人工核对流水线：CSV 列、human_verdict 解析、两种口径的切分（W3 验收标准 2）。

这组用例盯死三个曾经让"人工核对"白做的 bug：

1. 核对包 CSV 里 `verl`/`mathverify`/`strict`/`agent_*` 四列被静默写成空（嵌套 dict 交给
   `DictWriter` + `extrasaction="ignore"`）—— 人打开 Excel 看不到任何判定；
2. MD 核对包里 agent 初判全是字面的"待填"；
3. `verifier_accuracy.py` 根本不读 `human_verdict`，填完重跑还是 agent 初判的数字 ——
   看起来像"人工核对通过了"。
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from analysis.review_adjudication import (  # noqa: E402
    CSV_FIELDNAMES,
    NOT_APPLICABLE,
    HumanVerdict,
    HumanVerdictError,
    agent_verdict,
    csv_record,
    parse_human_verdict,
    read_human_verdicts,
    write_review_csv,
)

SHIPPED_CSV = REPO_ROOT / "reports" / "verifier_review_pack.csv"
SHIPPED_MD = REPO_ROOT / "reports" / "verifier_review_pack.md"


def _row(
    sample_id: str,
    gold: str,
    extracted: str,
    verdicts: dict[str, int | None],
    disagree: bool,
) -> dict:
    return {
        "sample_id": sample_id,
        "dataset": "gsm8k",
        "run": "unit-test",
        "ground_truth": gold,
        "finish_reason": "stop",
        "w2_correct": True,
        "question_tail": "...?",
        "response": "blah\n#### " + extracted,
        "verdicts": verdicts,
        "disagree": disagree,
        "strict_reason": "correct" if verdicts["strict"] else "wrong",
        "strict_extracted": extracted,
        "mathverify_extracted": extracted,
    }


# ---------------------------------------------------------------------------
# human_verdict 解析
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("0", False),
        ("对", True),
        ("错", False),
        ("true", True),
        ("FALSE", False),
        ("yes", True),
        ("n", False),
        (" 1 ", True),
        ("✓", True),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_parse_human_verdict(raw, expected) -> None:
    assert parse_human_verdict(raw) is expected


@pytest.mark.parametrize("raw", ["maybe", "2", "对吧", "?", "-"])
def test_parse_human_verdict_rejects_garbage(raw) -> None:
    """不认识的值必须抛错：静默跳过就是'填了但没统计'。"""
    with pytest.raises(HumanVerdictError):
        parse_human_verdict(raw)


# ---------------------------------------------------------------------------
# CSV：三家判定必须真的落进去
# ---------------------------------------------------------------------------
def test_csv_record_flattens_nested_verdicts() -> None:
    row = _row("r/1", "42", "42", {"verl": 0, "mathverify": 1, "strict": 1}, True)
    rec = csv_record(row)
    assert rec["verl"] == "0"
    assert rec["mathverify"] == "1"
    assert rec["strict"] == "1"
    assert set(rec) == set(CSV_FIELDNAMES)


def test_csv_record_marks_verl_not_applicable() -> None:
    """verl 对非 GSM8K 不适用（None）；写成 n/a，别和'没填'混在一起。"""
    row = _row("r/2", "6", "6", {"verl": None, "mathverify": 1, "strict": 1}, False)
    assert csv_record(row)["verl"] == NOT_APPLICABLE


def test_csv_record_carries_agent_verdict() -> None:
    row = _row("r/3", "42", "42", {"verl": 1, "mathverify": 1, "strict": 1}, False)
    rec = csv_record(row)
    assert rec["agent_verdict"] == "1"
    assert rec["agent_note"].strip()


def test_csv_record_human_columns_blank_by_default() -> None:
    row = _row("r/4", "42", "42", {"verl": 1, "mathverify": 1, "strict": 1}, False)
    rec = csv_record(row)
    assert rec["human_verdict"] == ""
    assert rec["human_note"] == ""


def test_write_then_read_human_roundtrip(tmp_path: Path) -> None:
    rows = [_row("r/5", "42", "42", {"verl": 0, "mathverify": 1, "strict": 1}, True)]
    out = tmp_path / "pack.csv"
    write_review_csv(rows, out, {"r/5": HumanVerdict(False, "最终答案写错了")})
    got = read_human_verdicts(out)
    assert got == {"r/5": HumanVerdict(False, "最终答案写错了")}


def test_regenerating_pack_keeps_human_verdicts(tmp_path: Path) -> None:
    """重新生成核对包不能抹掉已完成的人工核对。"""
    rows = [_row("r/6", "42", "42", {"verl": 0, "mathverify": 1, "strict": 1}, True)]
    out = tmp_path / "pack.csv"
    write_review_csv(rows, out)
    assert read_human_verdicts(out) == {}

    with out.open(encoding="utf-8-sig", newline="") as fh:
        data = list(csv.DictReader(fh))
    data[0]["human_verdict"] = "0"
    data[0]["human_note"] = "人工：模型答错"
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(data)

    carried = read_human_verdicts(out)
    write_review_csv(rows, out, carried)
    assert read_human_verdicts(out) == {"r/6": HumanVerdict(False, "人工：模型答错")}


def test_read_human_verdicts_missing_file(tmp_path: Path) -> None:
    assert read_human_verdicts(tmp_path / "nope.csv") == {}


def test_read_human_verdicts_requires_column(tmp_path: Path) -> None:
    out = tmp_path / "old.csv"
    out.write_text("sample_id,strict\nr/7,1\n", encoding="utf-8-sig")
    with pytest.raises(HumanVerdictError, match="human_verdict"):
        read_human_verdicts(out)
    assert read_human_verdicts(out, require_column=False) == {}


def test_read_human_verdicts_reports_all_bad_cells(tmp_path: Path) -> None:
    out = tmp_path / "bad.csv"
    out.write_text(
        "sample_id,human_verdict,human_note\nr/a,maybe,\nr/b,1,\nr/c,huh,\n",
        encoding="utf-8-sig",
    )
    with pytest.raises(HumanVerdictError) as exc:
        read_human_verdicts(out)
    message = str(exc.value)
    assert "r/a" in message and "r/c" in message
    assert "2 处" in message


# ---------------------------------------------------------------------------
# agent 初判
# ---------------------------------------------------------------------------
def test_agent_verdict_auto_rule_matches_gold() -> None:
    row = _row("r/8", "1,234", "1234", {"verl": 1, "mathverify": 1, "strict": 1}, False)
    verdict, reason = agent_verdict(row)
    assert verdict is True
    assert "归一化" in reason


def test_agent_verdict_defaults_to_wrong_without_extraction() -> None:
    row = _row("r/9", "42", "", {"verl": 0, "mathverify": 0, "strict": 0}, False)
    verdict, _ = agent_verdict(row)
    assert verdict is False


def test_agent_verdict_manual_table_wins() -> None:
    """MANUAL 表里的手工判定必须盖掉自动规则（这条自动规则会判对，人判错）。"""
    sid = "Qwen2.5-1.5B-Instruct/gsm8k/test/23#0"
    row = _row(sid, "8", "8", {"verl": 0, "mathverify": 1, "strict": 0}, True)
    verdict, reason = agent_verdict(row)
    assert verdict is False
    assert "L-8" in reason


# ---------------------------------------------------------------------------
# verifier_accuracy.py：两种口径必须分开
# ---------------------------------------------------------------------------
def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    rows = [
        # 分歧：verl 漏判，另两家抽到正确答案 -> agent 初判"模型答对"
        _row("s/A", "42", "42", {"verl": 0, "mathverify": 1, "strict": 1}, True),
        # 分歧：strict 抽不到 -> agent 初判"模型答错"（自动规则的默认值，需人来定）
        _row("s/B", "6", "", {"verl": None, "mathverify": 1, "strict": 0}, True),
        # 三家一致
        _row("s/C", "7", "7", {"verl": 1, "mathverify": 1, "strict": 1}, False),
    ]
    jsonl = tmp_path / "pack.jsonl"
    jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    csv_path = tmp_path / "pack.csv"
    write_review_csv(rows, csv_path)
    return jsonl, csv_path, rows


def _run_accuracy(tmp_path: Path, jsonl: Path, csv_path: Path | None) -> dict:
    from analysis.verifier_accuracy import main as accuracy_main

    out = tmp_path / "accuracy.json"
    argv = ["--review", str(jsonl), "--out", str(out), "--total", "3"]
    argv += ["--no-human"] if csv_path is None else ["--human", str(csv_path)]
    assert accuracy_main(argv) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_accuracy_agent_only_when_nothing_reviewed(tmp_path: Path) -> None:
    jsonl, csv_path, _ = _fixture(tmp_path)
    got = _run_accuracy(tmp_path, jsonl, csv_path)
    assert got["human_review"]["status"] == "agent-only"
    assert got["results_human"] is None
    assert got["results_basis"] == "agent"
    # agent 初判：A 参考为"对"，B 参考为"错"
    assert got["results_agent"]["strict"] == pytest.approx(
        {"reviewed": 2, "right": 2, "wrong": 0, "accuracy_on_reviewed": 1.0,
         "estimated_accuracy_on_pool": 1.0}
    )
    assert got["agent_adjudication"]["unsettled_by_auto_rule"] == ["s/B"]


def _fill(csv_path: Path, verdicts: dict[str, str], notes: dict[str, str] | None = None) -> None:
    notes = notes or {}
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        data = list(csv.DictReader(fh))
    for row in data:
        if row["sample_id"] in verdicts:
            row["human_verdict"] = verdicts[row["sample_id"]]
            row["human_note"] = notes.get(row["sample_id"], "")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(data)


def test_accuracy_partial_coverage_gives_no_pool_estimate(tmp_path: Path) -> None:
    jsonl, csv_path, _ = _fixture(tmp_path)
    _fill(csv_path, {"s/A": "1"})
    got = _run_accuracy(tmp_path, jsonl, csv_path)
    assert got["human_review"]["status"] == "partial"
    assert got["human_review"]["disputed_reviewed"] == 1
    assert got["human_review"]["coverage"] == 0.5
    # 覆盖不满时不给全量口径估计：抽样已被人为挑选过，估不出来
    assert got["results_human"]["strict"]["estimated_accuracy_on_pool"] is None
    assert got["results_basis"] == "agent"


def test_accuracy_human_basis_overrides_agent(tmp_path: Path) -> None:
    """人说 B 的模型答案是对的，agent 初判说错 —— 以人为准，strict 因此被记为判错。"""
    jsonl, csv_path, _ = _fixture(tmp_path)
    _fill(csv_path, {"s/A": "1", "s/B": "1"}, {"s/B": "结论句里给了 6，模型是对的"})
    got = _run_accuracy(tmp_path, jsonl, csv_path)

    assert got["human_review"]["status"] == "complete"
    assert got["results_basis"] == "human"
    human = got["results_human"]
    assert human["strict"]["right"] == 1 and human["strict"]["wrong"] == 1
    assert human["mathverify"]["right"] == 2 and human["mathverify"]["wrong"] == 0
    # verl 只在 A 上适用，且判了 0
    assert human["verl"] == pytest.approx(
        {"reviewed": 1, "right": 0, "wrong": 1, "accuracy_on_reviewed": 0.0,
         "estimated_accuracy_on_pool": round(1 / 3, 4)}
    )
    # agent 初判那一档必须还在，而且和人工口径不同
    assert got["results_agent"]["strict"]["accuracy_on_reviewed"] == 1.0
    assert got["results_agent"] != got["results_human"]


def test_accuracy_reports_agent_human_conflicts(tmp_path: Path) -> None:
    jsonl, csv_path, _ = _fixture(tmp_path)
    _fill(csv_path, {"s/A": "1", "s/B": "1"}, {"s/B": "人工：模型其实答对了"})
    got = _run_accuracy(tmp_path, jsonl, csv_path)
    agreement = got["human_review"]["agreement_with_agent"]
    assert agreement["same"] == 1
    assert agreement["differ"] == 1
    conflict = agreement["conflicts"][0]
    assert conflict["sample_id"] == "s/B"
    assert (conflict["agent"], conflict["human"]) == (0, 1)
    assert conflict["human_note"] == "人工：模型其实答对了"


def test_accuracy_unanimous_spotcheck_contradiction(tmp_path: Path) -> None:
    """人在三家一致的样本上发现反例 -> 全量口径要把这条扣掉。"""
    jsonl, csv_path, _ = _fixture(tmp_path)
    _fill(csv_path, {"s/A": "1", "s/B": "0", "s/C": "0"})
    got = _run_accuracy(tmp_path, jsonl, csv_path)
    spot = got["human_review"]["unanimous_spotchecks"]
    assert spot["n"] == 1
    assert spot["contradicted"] == ["s/C"]
    # 全部 3 条都核对过 -> 全量口径是实测值，不再用"三家一致都算判对"的估计
    strict = got["results_human"]["strict"]
    assert got["human_review"]["full_sample_coverage"] is True
    assert strict["estimated_accuracy_on_pool"] is None
    assert strict["accuracy_on_pool"] == pytest.approx(2 / 3, abs=1e-4)
    assert (strict["pool_right"], strict["pool_n"]) == (2, 3)


def test_accuracy_exact_pool_uses_per_scorer_denominator(tmp_path: Path) -> None:
    """verl 在非 GSM8K 上不适用，不能算进它的分母 —— 旧的估计公式就是这样把它算高的。"""
    jsonl, csv_path, _ = _fixture(tmp_path)
    _fill(csv_path, {"s/A": "1", "s/B": "1", "s/C": "1"})
    got = _run_accuracy(tmp_path, jsonl, csv_path)
    assert got["human_review"]["full_sample_coverage"] is True
    # s/B 的 verl 判定是 None（MATH 行），只剩 A/C 两条可比
    assert got["results_human"]["verl"]["pool_n"] == 2
    assert got["results_human"]["mathverify"]["pool_n"] == 3
    # 估计口径会把 B 当成 verl "判对"，实测不会
    assert got["results_human"]["verl"]["accuracy_on_pool"] == pytest.approx(0.5, abs=1e-4)


def test_accuracy_rejects_unparseable_human_column(tmp_path: Path) -> None:
    jsonl, csv_path, _ = _fixture(tmp_path)
    _fill(csv_path, {"s/A": "大概对吧"})
    with pytest.raises(HumanVerdictError):
        _run_accuracy(tmp_path, jsonl, csv_path)


# ---------------------------------------------------------------------------
# 入库的核对包本身
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SHIPPED_CSV.exists(), reason="review pack not generated")
def test_shipped_csv_has_every_scorer_verdict() -> None:
    """回归：这四列曾经整列是空的，人工核对包等于没有判定可看。"""
    with SHIPPED_CSV.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for column in ("verl", "mathverify", "strict", "agent_verdict", "agent_note"):
        assert all(row[column].strip() for row in rows), f"{column} 有空格子"
    for row in rows:
        assert row["mathverify"] in {"0", "1"}
        assert row["verl"] in {"0", "1", NOT_APPLICABLE}


@pytest.mark.skipif(not SHIPPED_MD.exists(), reason="review pack not generated")
def test_shipped_md_has_real_agent_verdicts() -> None:
    """回归：MD 里的 agent 初判曾经 110 条全是字面的'待填'。"""
    text = SHIPPED_MD.read_text(encoding="utf-8")
    assert "agent 初判：待填" not in text
    samples = len(re.findall(r"^## \d+\. ", text, re.MULTILINE))
    assert samples > 0
    assert text.count("- agent 初判：") == samples
    assert text.count("- 人工核对：") == samples
