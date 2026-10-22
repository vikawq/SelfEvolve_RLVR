"""verl 自定义奖励函数：严格 verifier 的薄封装（W3）。

接入方式（verl v0.9.1 实际字段，不是任务包写的顶层键）：

    reward.custom_reward_function.path=src/selfevolve/rewards/strict_reward.py
    reward.custom_reward_function.name=compute_score

调用约定来自 `verl/experimental/reward_loop/reward_manager/naive.py`：verl 以**关键字**调用

    compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)

并把返回值当 dict 处理（`score = result["score"]`，其余键进 `reward_extra_info` 落 wandb/parquet）。
`verl/trainer/main_eval.py` 则是**位置**调用，因此这里的签名两种都兼容。

奖励定义（任务包 §1）：正确 1.0，其余 0.0；格式奖励默认 0，保留成可配置项但不启用。
训练侧拿不到 `finish_reason`，截断只能靠文本推断（评测侧会显式传 `finished`）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# verl 用 importlib 从文件路径加载本模块（不在包内 import），所以这里自己把 `src` 放进 sys.path。
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from selfevolve.verifier.math_verifier import REASON_CORRECT, SCORER_VERSION, verify  # noqa: E402

logger = logging.getLogger(__name__)

#: 诊断字段的长度上限：这些值会进 non_tensor_batch 并最终写 parquet。
_MAX_EXTRACT_CHARS = 120

#: 内部异常计数（每 100 次打一条 warning，避免刷屏但也不静默）。
_error_count = 0

#: data_source 里带这些关键词时按选择题处理（与评测 harness 的口径一致）。
_MC_DATA_SOURCES = ("mmlu", "gpqa", "arc_", "hellaswag", "ceval")


def _score_mc(solution_str: str, ground_truth: str) -> tuple[float, str]:
    """选择题：最后一个 `Answer: X` 优先，否则取最后一个独立选项字母。"""
    import re

    text = solution_str or ""
    matches = list(re.finditer(r"answer\s*[:：]\s*\(?\s*([A-Ja-j])\s*\)?", text, re.IGNORECASE))
    pred = matches[-1].group(1).upper() if matches else ""
    if not pred:
        letters = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", text)
        pred = letters[-1].upper() if letters else ""
    gold = str(ground_truth or "").strip().upper()
    correct = bool(pred) and pred == gold
    return (1.0 if correct else 0.0), (pred or "no_answer")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """verl 自定义奖励入口。返回 dict：`score` 是唯一被用于训练的值。"""
    global _error_count

    format_score = float(kwargs.get("format_score", 0.0))
    source = str(data_source or "")

    if any(key in source for key in _MC_DATA_SOURCES):
        score, extracted = _score_mc(solution_str, ground_truth)
        return {
            "score": score,
            "acc": score,
            "reason": REASON_CORRECT if score > 0 else "wrong",
            "extracted": extracted[:_MAX_EXTRACT_CHARS],
            "scorer": SCORER_VERSION,
        }

    try:
        result = verify(solution_str, ground_truth)
        score = result.score
        if format_score and result.parse_ok:
            score += format_score
        return {
            "score": float(score),
            "acc": float(result.score),
            "reason": result.reason,
            "extracted": (result.extracted or "")[:_MAX_EXTRACT_CHARS],
            "scorer": SCORER_VERSION,
        }
    except Exception as exc:  # 判分器内部异常不能拖垮训练，但必须留痕
        _error_count += 1
        if _error_count == 1 or _error_count % 100 == 0:
            logger.warning(
                "strict_reward compute_score failed (%d occurrences so far): %s: %s",
                _error_count,
                type(exc).__name__,
                str(exc)[:300],
            )
        return {
            "score": 0.0,
            "acc": 0.0,
            "reason": "error",
            "extracted": "",
            "scorer": SCORER_VERSION,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        }


__all__ = ["compute_score"]
