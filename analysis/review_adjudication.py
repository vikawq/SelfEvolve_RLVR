"""人工核对口径的单一实现（W3 任务 3）。

`verifier_compare.py`（生成核对包）与 `verifier_accuracy.py`（统计准确率）都从这里取判定逻辑，
避免两边各写一份然后漂移。这里定义三件事：

1. **agent 初判**（`agent_verdict`）：`MANUAL` 表里逐条写明的 14 个边界样本，
   其余走"strict 抽取 == gold 归一化后字符串相等"的自动规则。
2. **CSV 列定义**（`CSV_FIELDNAMES`）+ 逐行序列化（`csv_record`）：核对包 CSV 的唯一写法。
3. **human_verdict 的读取**（`read_human_verdicts`）：把人工填的那一列解析成布尔，
   无法识别的值**报错**而不是静默丢掉。

口径（两个脚本、MD/CSV 核对包、验证报告必须一致）：对每条样本只判断一个事实——
**"模型最终给出的答案对不对"**，不是"判分器说得对不对"。判分器的准确率 = 它的判定与这个事实一致的比例。

> **自动规则的循环性（必须知道）**：自动那一档用 strict 自己的抽取结果当参考答案
> （`strict_extracted == gold` -> 模型正确），而 strict 这时判的就是 1，必然计为"判对"。
> 也就是说 **agent 初判无法独立证明 strict 的准确率**，它只能把需要人判断的样本收敛到
> `MANUAL` 这一档。打破循环的唯一办法是人工填 `human_verdict` 列——这正是验收标准 2
> 要求"交给人核对"的原因。两种口径在 `verifier_accuracy.py` 的输出里分别叫
> `results_agent` / `results_human`，**不可混用，也不可互相替代**。
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from selfevolve.verifier.math_verifier import normalize_answer  # noqa: E402

SCORERS = ("verl", "mathverify", "strict")

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
        "这里误判，已记入验证报告 §6 的已知限制）",
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

AUTO_TRUE_REASON = "strict 抽取的最终答案与 gold 归一化后完全一致（确定性判定，不依赖 Math-Verify）"
AUTO_FALSE_REASON = "strict 未抽取到答案或抽取结果与 gold 字符串不一致（需人工复核）"


def agent_verdict(row: dict[str, Any]) -> tuple[bool, str]:
    """agent 初判：模型最终答案是否正确，附理由。

    `MANUAL` 优先；其余走"strict 抽取 == gold"的自动规则（见模块 docstring 里的循环性说明）。
    """
    sid = row["sample_id"]
    if sid in MANUAL:
        return MANUAL[sid]
    gold = normalize_answer(row["ground_truth"])
    extracted = normalize_answer(row.get("strict_extracted") or "")
    if extracted and extracted == gold:
        return True, AUTO_TRUE_REASON
    return False, AUTO_FALSE_REASON


def is_manually_adjudicated(row: dict[str, Any]) -> bool:
    """这一条的 agent 初判是不是人写的（而不是自动规则推的）。"""
    return row["sample_id"] in MANUAL


# ---------------------------------------------------------------------------
# 核对包 CSV
# ---------------------------------------------------------------------------
#: 核对包 CSV 的列定义（唯一出处）。`human_verdict` / `human_note` 留给人填。
CSV_FIELDNAMES = [
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
    "agent_verdict",
    "agent_note",
    "human_verdict",
    "human_note",
]

#: verl 对非 GSM8K 数据集不适用（判定是 None），CSV 里写成这个，区别于"没填"。
NOT_APPLICABLE = "n/a"

TRUE_TOKENS = frozenset({"1", "1.0", "true", "t", "yes", "y", "correct", "right", "ok", "对", "正确", "√", "✓"})
FALSE_TOKENS = frozenset({"0", "0.0", "false", "f", "no", "n", "wrong", "incorrect", "错", "错误", "×", "✗"})


class HumanVerdictError(ValueError):
    """`human_verdict` 列里有无法识别的值，或 CSV 缺这一列。

    这里**必须**抛错：静默跳过会让"填了但没被统计"看起来像"核对通过"。
    """


def parse_human_verdict(raw: Any) -> bool | None:
    """把一个 `human_verdict` 单元格解析成 True/False；空白返回 None（= 没核对）。

    无法识别的值抛 `HumanVerdictError`。
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in TRUE_TOKENS:
        return True
    if lowered in FALSE_TOKENS:
        return False
    raise HumanVerdictError(
        f"无法识别的 human_verdict 值 {text!r}；"
        f"请填 1/0（也接受 对/错、true/false、yes/no）"
    )


@dataclass(frozen=True)
class HumanVerdict:
    """人工核对的一条结论。"""

    verdict: bool
    note: str = ""


def _verdict_cell(value: Any) -> str:
    if value is None:
        return NOT_APPLICABLE
    return str(int(bool(value)))


def csv_record(row: dict[str, Any], human: HumanVerdict | None = None) -> dict[str, str]:
    """把一条打分结果拍平成 CSV 行。

    `row["verdicts"]` 是嵌套 dict，直接交给 `DictWriter` 会被静默丢成空列——
    所有写 CSV 的地方都必须走这个函数。
    """
    verdicts = row.get("verdicts") or {}
    agent, agent_note = agent_verdict(row)
    return {
        "sample_id": row["sample_id"],
        "dataset": row.get("dataset", ""),
        "run": row.get("run", ""),
        "ground_truth": str(row.get("ground_truth", "")),
        "finish_reason": str(row.get("finish_reason") or ""),
        "w2_correct": _verdict_cell(row.get("w2_correct")),
        "verl": _verdict_cell(verdicts.get("verl")),
        "mathverify": _verdict_cell(verdicts.get("mathverify")),
        "strict": _verdict_cell(verdicts.get("strict")),
        "disagree": str(bool(row.get("disagree"))),
        "strict_reason": str(row.get("strict_reason") or ""),
        "strict_extracted": str(row.get("strict_extracted") or "")[:200],
        "mathverify_extracted": str(row.get("mathverify_extracted") or "")[:200],
        "agent_verdict": str(int(agent)),
        "agent_note": agent_note,
        "human_verdict": "" if human is None else str(int(human.verdict)),
        "human_note": "" if human is None else human.note,
    }


def write_review_csv(
    rows: Iterable[dict[str, Any]],
    out_csv: Path,
    human: dict[str, HumanVerdict] | None = None,
) -> None:
    """写核对包 CSV（`utf-8-sig`，Excel 直接打开不乱码）。

    `human` 里已有的判定会原样写回，重新生成核对包不会抹掉已完成的人工核对。
    """
    human = human or {}
    with Path(out_csv).open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_record(row, human.get(row["sample_id"])))


def read_human_verdicts(csv_path: Path, require_column: bool = True) -> dict[str, HumanVerdict]:
    """读 CSV 的 `human_verdict` / `human_note` 两列。

    只返回**填了**的行。文件不存在返回空 dict；列不存在时按 `require_column` 决定抛错还是返回空。
    单元格里有无法识别的值时，**一次报出全部坏值**再抛错——不静默跳过。
    """
    path = Path(csv_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if "human_verdict" not in fieldnames:
            if require_column:
                raise HumanVerdictError(
                    f"{path} 里没有 human_verdict 列（现有列：{fieldnames}）；"
                    f"请用 analysis/verifier_compare.py 重新生成核对包"
                )
            return {}
        out: dict[str, HumanVerdict] = {}
        problems: list[str] = []
        for lineno, row in enumerate(reader, start=2):
            sid = (row.get("sample_id") or "").strip()
            if not sid:
                continue
            try:
                verdict = parse_human_verdict(row.get("human_verdict"))
            except HumanVerdictError as exc:
                problems.append(f"  第 {lineno} 行 ({sid}): {exc}")
                continue
            if verdict is None:
                continue
            out[sid] = HumanVerdict(verdict=verdict, note=(row.get("human_note") or "").strip())
    if problems:
        raise HumanVerdictError(f"{path} 的 human_verdict 列有 {len(problems)} 处无法识别：\n" + "\n".join(problems))
    return out


__all__ = [
    "AUTO_FALSE_REASON",
    "AUTO_TRUE_REASON",
    "CSV_FIELDNAMES",
    "FALSE_TOKENS",
    "HumanVerdict",
    "HumanVerdictError",
    "MANUAL",
    "NOT_APPLICABLE",
    "SCORERS",
    "TRUE_TOKENS",
    "agent_verdict",
    "csv_record",
    "is_manually_adjudicated",
    "parse_human_verdict",
    "read_human_verdicts",
    "write_review_csv",
]
