"""W5 防御构件（D2 答案唯一性 / D3 overlong shaping / D4 双解析器一致）。

设计约束：

1. **不碰评测**：本模块只服务训练奖励；评测仍然只用 `eval/run_eval.py` 的严格 verifier。
2. **不改变严格判定**：`strict_acc` 仍由 W3 的 `verify()` 给出，防御只改"训练用的宽松分数"，
   因此 rollout 里的 `strict_acc` 与未防御组逐字可比。
3. **两条独立解析路径**：D4 的 B 解析器**不复用** `verifier.extract_answer` 的任何内部实现，
   只复用 `normalize_answer` 做最后的字符串归一化；否则"双解析器一致"会退化成"同一个解析器
   自己和自己一致"。
4. **可诊断**：所有防御字段在每条样本上都必须存在（verl 组 batch 要求键集合一致），
   分析脚本靠这些键算 hack rate 和防御触发率。

D3 的语义：任务文档写"超过硬上限直接判 0"，DAPO 原文是 `R_correct - 1`。本周按文档实现
（奖励恒非负、硬上限 0），差异记在 `reports/P7_reward_hacking_audit.md` 与
`docs/troubleshooting.md`。
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from selfevolve.verifier.math_verifier import (  # noqa: E402
    extract_answer,
    is_equivalent,
    normalize_answer,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# D3 参数与 token 计数
# ---------------------------------------------------------------------------

D3_SOFT_TOKENS = int(os.environ.get("W05_D3_SOFT_TOKENS", "768"))
D3_HARD_TOKENS = int(os.environ.get("W05_D3_HARD_TOKENS", "1024"))
D3_WEIGHT = float(os.environ.get("W05_D3_WEIGHT", "1.0"))

#: tokenizer 不可用时的字符/ token 经验比（W4 五个实验逐 step 的中位数在 3.26–3.93，
#: 取 3.5；只在回退路径使用，并在样本上打 `tokenizer_fallback=1`）。
FALLBACK_CHARS_PER_TOKEN = 3.5

_TOKENIZER: Any = None
_TOKENIZER_FAILED = False
_TOKENIZER_WARNED = False


def _tokenizer_path() -> str:
    return (
        os.environ.get("W05_TOKENIZER_PATH")
        or os.environ.get("W05_MODEL_PATH")
        or os.environ.get("MODEL_PATH")
        or ""
    )


def _load_tokenizer() -> Any:
    """懒加载 tokenizer；失败只记一次 warning，之后走字符回退。"""
    global _TOKENIZER, _TOKENIZER_FAILED, _TOKENIZER_WARNED
    if _TOKENIZER is not None or _TOKENIZER_FAILED:
        return _TOKENIZER
    path = _tokenizer_path()
    if not path:
        _TOKENIZER_FAILED = True
        if not _TOKENIZER_WARNED:
            logger.warning("W05_TOKENIZER_PATH not set; D3 falls back to char-length estimate")
            _TOKENIZER_WARNED = True
        return None
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        logger.info("D3 tokenizer loaded from %s", path)
    except Exception as exc:  # pragma: no cover - 取决于训练机环境
        _TOKENIZER_FAILED = True
        if not _TOKENIZER_WARNED:
            logger.warning(
                "D3 tokenizer load failed (%s: %s); falling back to char-length estimate",
                type(exc).__name__,
                str(exc)[:200],
            )
            _TOKENIZER_WARNED = True
    return _TOKENIZER


def count_tokens(text: str) -> tuple[int, int]:
    """返回 `(token 数, 是否回退)`。回退标志 1 = 没用到真 tokenizer。"""
    raw = "" if text is None else str(text)
    tokenizer = _load_tokenizer()
    if tokenizer is not None:
        try:
            ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
            return len(ids), 0
        except Exception as exc:  # pragma: no cover
            logger.warning("D3 tokenize failed (%s); falling back", type(exc).__name__)
    return max(1, int(math.ceil(len(raw) / FALLBACK_CHARS_PER_TOKEN))), 1


def overlong_penalty(
    n_tokens: int,
    *,
    soft: int = D3_SOFT_TOKENS,
    hard: int = D3_HARD_TOKENS,
    weight: float = D3_WEIGHT,
) -> tuple[float, str]:
    """DAPO 式超长惩罚：`(penalty, state)`，state ∈ {none, linear, hard_cap}。

    - `n_tokens <= soft`：0；
    - `soft < n_tokens < hard`：`weight * (n - soft) / (hard - soft)`；
    - `n_tokens >= hard`：返回 0 + `hard_cap`，由调用方直接把整条奖励判 0（按任务文档）。
    """
    if hard <= soft:
        raise ValueError(f"hard must be > soft, got soft={soft}, hard={hard}")
    if weight < 0:
        raise ValueError(f"weight must be >= 0, got {weight}")
    if n_tokens >= hard:
        return 0.0, "hard_cap"
    if n_tokens <= soft:
        return 0.0, "none"
    frac = (float(n_tokens) - soft) / float(hard - soft)
    return weight * frac, "linear"


# ---------------------------------------------------------------------------
# 独立解析器（D2 / D4-B）
# ---------------------------------------------------------------------------

_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_HASH_MARKER = "####"
_PHRASE_RE = re.compile(
    r"(?:the\s+(?:final\s+)?answer(?:\s+\S+){0,4}?\s+(?:is|would\s+be)"
    r"|final\s+answer\b|answer\s*[:：]"
    r"|答案\s*(?:是|为)?\s*[:：]?|答\s*[:：])",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer|final\s+answer|answer|ans|result|答案|最终答案|答)"
    r"\s*(?:is|are|为|是)?\s*[:：]?\s*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w.])-?\$?\d+(?:,\d{3})*(?:\.\d+)?\$?(?![\w]|\.\d)")
_TRAILING_PUNCT = ".,;:。，；：、"


def _boxed_spans(text: str) -> list[tuple[int, int, str, bool]]:
    """找所有 `\\boxed{...}`（含嵌套大括号）；返回 (start, end, content, closed)。"""
    spans: list[tuple[int, int, str, bool]] = []
    for match in _BOXED_OPEN_RE.finditer(text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text):
            ch = text[i]
            escaped = i > 0 and text[i - 1] == "\\"
            if ch == "{" and not escaped:
                depth += 1
            elif ch == "}" and not escaped:
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), i + 1, text[start:i], True))
                    break
            i += 1
        else:
            spans.append((match.start(), len(text), text[start:], False))
    return spans


def _clean_value(value: str) -> str:
    return normalize_answer(value.strip().strip(_TRAILING_PUNCT))


def _values_equal(a: str, b: str) -> bool:
    """同一候选的判定：字符串相同，或数值相等（`7` vs `7.0`）。

    刻意不在这里调 Math-Verify：`\\frac{1}{2}` vs `0.5` 这类跨表示等价留给严格 verifier，
    独立解析器的职责只是"多候选检测"，宁可把不确定的当不同候选（更严）。
    """
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _distinct_values(values: list[str]) -> list[str]:
    """归一化去重；`2` 与 `2 dollars` 视为同一个值（normalize 已去单位）。"""
    out: list[str] = []
    for value in values:
        norm = _clean_value(value)
        if norm and not any(_values_equal(norm, prev) for prev in out):
            out.append(norm)
    return out


def _first_number(text: str) -> str | None:
    match = _NUMBER_RE.search(text)
    return match.group(0) if match else None


def _hash_candidates(text: str) -> list[str]:
    positions = [m.start() for m in re.finditer(re.escape(_HASH_MARKER), text)]
    values: list[str] = []
    for index, pos in enumerate(positions):
        end = positions[index + 1] if index + 1 < len(positions) else len(text)
        segment = text[pos + len(_HASH_MARKER) : end]
        lines = [line.strip() for line in segment.splitlines() if line.strip()]
        for line in reversed(lines[:6]):
            spans = [content for _, _, content, closed in _boxed_spans(line) if closed]
            if spans:
                values.append(spans[-1])
                break
            stripped = _LABEL_RE.sub("", line).strip().strip(_TRAILING_PUNCT)
            if not stripped:
                continue
            if len(stripped) <= 80 and (re.search(r"\d|\\[A-Za-z]+", stripped) or len(stripped) <= 24):
                values.append(stripped)
                break
        else:
            number = _first_number(segment)
            if number:
                values.append(number)
    return values


def _phrase_candidates(text: str) -> list[str]:
    tail_start = max(0, len(text) - 400)
    tail = text[tail_start:]
    values: list[str] = []
    for match in _PHRASE_RE.finditer(tail):
        rest = tail[match.end() :][:200]
        if not rest.strip():
            continue
        first_line = rest.splitlines()[0].strip() if rest.splitlines() else ""
        candidate = _LABEL_RE.sub("", first_line).strip().strip(_TRAILING_PUNCT)
        if candidate and len(candidate) <= 80:
            values.append(candidate)
            continue
        number = _first_number(rest)
        if number:
            values.append(number)
    return values


def independent_final_answer(text: str) -> tuple[str | None, str]:
    """第二套独立解析器：返回 `(answer, state)`，state ∈ {ok, multiple, truncated, none}。

    优先级与严格 verifier 相同（boxed → #### → 收尾句），但实现完全独立：
    不调用 `extract_answer`、不调用 Math-Verify，只做字符串级抽取。
    它的职责是"提供一个可对照的第二意见"，不是替代严格判分器。
    """
    raw = "" if text is None else str(text)
    if not raw.strip():
        return None, "none"
    without_code = _FENCE_RE.sub(lambda m: " " * len(m.group(0)), raw)

    spans = _boxed_spans(without_code)
    closed = [content for _, _, content, ok in spans if ok and content.strip()]
    if closed:
        values = _distinct_values(closed)
        if len(values) > 1:
            return None, "multiple"
        if values:
            return values[0], "ok"
    if any(not ok for _, _, _, ok in spans):
        # 有 `\\boxed{` 没闭合：畸形/截断，按"不可给分"处理
        return None, "truncated"

    hashes = _hash_candidates(without_code)
    values = _distinct_values(hashes)
    if len(values) > 1:
        return None, "multiple"
    if values:
        return values[0], "ok"

    phrases = _phrase_candidates(without_code)
    values = _distinct_values(phrases)
    if len(values) > 1:
        return None, "multiple"
    if values:
        return values[0], "ok"
    return None, "none"


# ---------------------------------------------------------------------------
# D2 / D4
# ---------------------------------------------------------------------------


def unique_answer_status(text: str) -> tuple[str | None, str]:
    """D2：只有存在**唯一**最终答案时才返回它；多候选/无答案都算不可给分。"""
    return independent_final_answer(text)


def dual_parser_agreement(text: str, parser_a_answer: str | None) -> dict[str, Any]:
    """D4：严格 verifier 抽出的 A 与独立解析器抽出的 B 是否一致。

    返回诊断 dict：`agree` / `status` / `parser_a` / `parser_b`。
    A 为空（严格 verifier 判 no_answer/multiple/truncated）时一律不一致。
    """
    parser_b, status = independent_final_answer(text)
    a_norm = normalize_answer(parser_a_answer) if parser_a_answer else ""
    b_norm = normalize_answer(parser_b) if parser_b else ""
    agree = False
    if a_norm and b_norm:
        agree = a_norm == b_norm
        if not agree:
            try:
                agree = bool(is_equivalent(a_norm, b_norm))
            except Exception:  # pragma: no cover - is_equivalent 自己已兜底
                agree = False
    return {
        "agree": agree,
        "status": status,
        "parser_a": (parser_a_answer or "")[:120],
        "parser_b": (parser_b or "")[:120],
    }


__all__ = [
    "D3_HARD_TOKENS",
    "D3_SOFT_TOKENS",
    "D3_WEIGHT",
    "FALLBACK_CHARS_PER_TOKEN",
    "count_tokens",
    "dual_parser_agreement",
    "independent_final_answer",
    "overlong_penalty",
    "unique_answer_status",
]
