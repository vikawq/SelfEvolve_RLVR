"""判分器封装（W2 暂用 Math-Verify；W3 换成严格 verifier）。

接口约定（任务包要求）：

    score(response, ground_truth) -> dict

返回 dict 至少含 `correct`（bool），并保留抽取结果，便于逐样本审计。
注意：W3 会把训练奖励与评测统一到同一严格 verifier；届时本文件的实现要整体替换，
所有历史结果需按新 verifier 重评（协议 §4）。
"""

from __future__ import annotations

import logging
import re
from importlib.metadata import PackageNotFoundError, version
from typing import Any

logger = logging.getLogger(__name__)

try:  # 允许在没有安装 math-verify 的机器上 import（测试用 importorskip）
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify

    _MV_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖缺失只在本地开发环境发生
    _mv_parse = None
    _mv_verify = None
    _MV_AVAILABLE = False

try:
    _MV_VERSION = version("math-verify")
except PackageNotFoundError:  # pragma: no cover
    _MV_VERSION = "not-installed"

SCORER_NAME = "math-verify"
SCORER_VERSION = f"{SCORER_NAME}=={_MV_VERSION}"

_HASH_RE = re.compile(r"####")
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_MC_ANSWER_RE = re.compile(r"answer\s*[:：]\s*\(?\s*([A-Ja-j])\s*\)?", re.IGNORECASE)


def _boxed(expr: str) -> str:
    return "\\boxed{" + expr + "}"


def normalize_gold(ground_truth: str) -> str:
    """把 gold 统一包装成 Math-Verify 能稳定抽取的形式。

    纯 LaTeX（如 `\\left(3,\\frac{\\pi}{2}\\right)`）直接 parse 会丢结构甚至只取第一个数；
    包成 `\\boxed{...}` 后与标准 MATH 答案格式一致。已经带 `\\boxed` 或 `$` 的不再重复包。
    """
    gt = str(ground_truth).strip()
    if not gt:
        return gt
    if "\\boxed" in gt or "$" in gt:
        return gt
    return _boxed(gt)


def extract_predicted_answer(response: str) -> str:
    """只用于日志展示；判分仍以 Math-Verify 的 parse/verify 为准。"""
    text = response or ""
    if "####" in text:
        return text.rsplit("####", 1)[1].strip()[:200]
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


def _candidate_texts(response: str) -> list[str]:
    """列出可能包含最终答案的文本片段（去重，保持优先级）。

    纯 LaTeX 直接 parse 会丢结构（如 `\\left(3,\\frac{\\pi}{2}\\right)` 只取到 3），
    因此对 `####` 尾部额外生成一个 `\\boxed{...}` 版本；两种解析结果都参与 verify。
    """
    text = response or ""
    candidates: list[str] = []
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip()
        if tail:
            candidates.append(tail)
            candidates.append(_boxed(tail))
    candidates.append(text)
    extracted = extract_predicted_answer(text)
    if extracted:
        candidates.append(extracted)
        candidates.append(_boxed(extracted))
    out: list[str] = []
    for c in candidates:
        if c and c not in out:
            out.append(c)
    return out


def _parse_pred_candidates(response: str) -> list[list[Any]]:
    """返回多个候选解析结果；verify 时任一命中即算对。"""
    assert _mv_parse is not None
    parsed_sets: list[list[Any]] = []
    for cand in _candidate_texts(response):
        try:
            parsed = _mv_parse(cand)
        except Exception as exc:  # Math-Verify 对畸形公式可能抛异常
            logger.debug("math_verify.parse failed on %r: %s", cand[:80], exc)
            continue
        if parsed and parsed not in parsed_sets:
            parsed_sets.append(parsed)
    return parsed_sets


def score_math(response: str, ground_truth: str) -> dict[str, Any]:
    """数学判分。返回 correct / extracted / gold / pred，异常时记 parse_error。"""
    result: dict[str, Any] = {
        "correct": False,
        "extracted": extract_predicted_answer(response),
        "scorer": SCORER_VERSION,
        "error": None,
    }
    if not _MV_AVAILABLE:
        result["error"] = "math-verify not installed"
        return result
    try:
        gold = _mv_parse(normalize_gold(ground_truth))
        pred_sets = _parse_pred_candidates(response or "")
        result["gold_parsed"] = repr(gold)[:300]
        result["pred_parsed"] = repr(pred_sets[0])[:300] if pred_sets else "[]"
        if not gold:
            result["error"] = "gold_parse_empty"
            return result
        if not pred_sets:
            result["error"] = "pred_parse_empty"
            return result
        result["correct"] = any(bool(_mv_verify(gold, pred)) for pred in pred_sets)
    except Exception as exc:  # pragma: no cover - 防御性
        result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return result


def score_mc(response: str, ground_truth: str) -> dict[str, Any]:
    """选择题判分：最后一行 `Answer: X` 优先，否则取最后一个独立选项字母。"""
    text = response or ""
    pred = ""
    matches = list(_MC_ANSWER_RE.finditer(text))
    if matches:
        pred = matches[-1].group(1).upper()
    else:
        letters = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", text)
        if letters:
            pred = letters[-1].upper()
    gold = str(ground_truth).strip().upper()
    return {
        "correct": bool(pred) and pred == gold,
        "extracted": pred,
        "gold": gold,
        "scorer": SCORER_VERSION,
        "error": None if pred else "no_option_letter",
    }


def score(response: str, ground_truth: str, answer_type: str = "math") -> dict[str, Any]:
    """统一入口，供 harness 与后续 verifier 对齐使用。"""
    if answer_type == "mc":
        return score_mc(response, ground_truth)
    return score_math(response, ground_truth)
