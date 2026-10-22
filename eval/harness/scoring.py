"""判分器封装：W3 起统一走严格 verifier。

红线（协议 §4）：训练奖励与评测必须共用同一实现。训练侧是
`src/selfevolve/rewards/strict_reward.py`，评测侧是本文件，两者都只调用
`src/selfevolve/verifier/math_verifier.py`。

W2 的历史口径（原样 Math-Verify 解析全文）已退役：它会把"正确数字只出现在推理过程里"的样本判对，
W3 的验证报告 `reports/verifier_validation.md` 量化了这部分差异。旧分数保留在 `results_master.md`
里用于对比审计，不覆盖。

接口约定（评测 harness 内部）：

    score(response, ground_truth, answer_type="math", finished=None) -> dict

返回 dict 含 `correct`（bool）、`extracted`、`reason`、`parse_ok`、`scorer`、`error`。
`finished=False` 表示生成被 `max_tokens` 截断（`finish_reason == "length"`），按任务包要求判错。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from selfevolve.verifier.math_verifier import (  # noqa: E402
    REASON_CORRECT,
    REASON_MULTIPLE,
    REASON_NO_ANSWER,
    REASON_TIMEOUT,
    REASON_TRUNCATED,
    REASON_WRONG,
    SCORER_NAME,
    SCORER_VERSION,
    VERIFIER_VERSION,
    extract_answer,
    normalize_answer,
    verify,
)

#: 这些 reason 是"判定结果"，不是判分器故障。
_EXPECTED_REASONS = frozenset(
    {REASON_CORRECT, REASON_WRONG, REASON_NO_ANSWER, REASON_MULTIPLE, REASON_TRUNCATED}
)


def normalize_gold(ground_truth: str) -> str:
    """把 gold 包成 `\\boxed{...}`（W2 的约定，保留给老脚本；verifier 内部已自己做）。

    Math-Verify 对裸答案解析不出东西（`parse("460") == []`），必须包裹。
    """
    gt = str(ground_truth).strip()
    if not gt:
        return gt
    if "\\boxed" in gt or "$" in gt:
        return gt
    return "\\boxed{" + gt + "}"


def extract_predicted_answer(response: str) -> str:
    """只用于日志展示；判分以 `verify()` 的 `extracted` 为准（没有答案时返回空串）。"""
    answer, _ = extract_answer(response)
    return (answer or "")[:200]


def score_math(response: str, ground_truth: str, *, finished: bool | None = None) -> dict[str, Any]:
    """数学判分：严格 verifier 的薄封装。"""
    result = verify(response, ground_truth, finished=finished)
    return {
        "correct": result.correct,
        "extracted": result.extracted or "",
        "parse_ok": result.parse_ok,
        "reason": result.reason,
        "gold_normalized": normalize_answer(ground_truth)[:200],
        "scorer": SCORER_VERSION,
        # 超时是判分器故障（值得计数），其余 reason 都是正常判定结果
        "error": result.reason if result.reason == REASON_TIMEOUT else None,
    }


def score_mc(response: str, ground_truth: str, *, finished: bool | None = None) -> dict[str, Any]:
    """选择题判分：最后一行 `Answer: X` 优先，否则取最后一个独立选项字母。"""
    import re

    text = response or ""
    pred = ""
    matches = list(re.finditer(r"answer\s*[:：]\s*\(?\s*([A-Ja-j])\s*\)?", text, re.IGNORECASE))
    if matches:
        pred = matches[-1].group(1).upper()
    else:
        letters = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", text)
        if letters:
            pred = letters[-1].upper()
    gold = str(ground_truth).strip().upper()
    correct = bool(pred) and pred == gold
    reason = REASON_CORRECT if correct else REASON_WRONG
    if not pred:
        reason = REASON_NO_ANSWER
    return {
        "correct": correct,
        "extracted": pred,
        "parse_ok": bool(pred),
        "reason": reason,
        "gold_normalized": gold,
        "scorer": SCORER_VERSION,
        "error": None,
    }


def score(
    response: str,
    ground_truth: str,
    answer_type: str = "math",
    *,
    finished: bool | None = None,
) -> dict[str, Any]:
    """统一入口，供 harness 与 rescore 使用。"""
    if answer_type == "mc":
        return score_mc(response, ground_truth, finished=finished)
    return score_math(response, ground_truth, finished=finished)


__all__ = [
    "SCORER_NAME",
    "SCORER_VERSION",
    "VERIFIER_VERSION",
    "extract_predicted_answer",
    "normalize_gold",
    "score",
    "score_mc",
    "score_math",
]
