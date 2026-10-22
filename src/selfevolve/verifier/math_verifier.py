"""严格数学答案判分器（W3）。

设计目标（`weeks/week03_strict_verifier.md`）：
- **只奖励正确性**：默认 1.0（正确）/ 0.0（错误或无法解析）；不启用格式奖励。
- **宁严勿松**：情况模糊一律判错（多候选、截断、解析失败）。
- **确定性 + 超时保护**：相同输入永远得到相同结果；单次判定有超时，不会卡死训练。

对外契约：

    verify(response, ground_truth) -> VerifyResult
    verify_many(pairs) -> list[VerifyResult]

`eval/harness/scoring.py`（评测）与 `src/selfevolve/rewards/strict_reward.py`（训练奖励）都只调用
本模块，保证训练和评测共用同一实现（协议 §4）。

与上游 Math-Verify 的关系：答案**抽取**、候选冲突判定、截断判定、归一化都在本模块里做；Math-Verify
只负责最后一步"两个已抽取表达式是否等价"。这样做是因为 Math-Verify 的 `parse` 会"任意匹配"——
`3 or 5` 只取到一个数、`\boxed{3} and \boxed{5}` 变成一个集合、`(3,5)` 被当成开区间——这类行为不能
直接当作最终答案判定用（见 `docs/troubleshooting.md §5`）。

已实测的上游坑（2026-10-22，math-verify 0.9.0）：
1. 裸 gold（`"460"`）`parse` 返回空列表，必须包成 `\\boxed{460}`（W2 的 `normalize_gold` 已处理）；
2. `parse`/`verify` 内置超时在 POSIX 用 `signal.alarm`，**非主线程会直接抛 ValueError**；
   Windows 用 multiprocessing，每次调用起一个进程。本模块因此**关闭**上游超时
   （`parsing_timeout=None` / `timeout_seconds=None`），自己实现超时（见 `_run_with_timeout`）；
3. `\\dfrac` 直接 `parse` 失败（包进 `\\boxed{}` 才成功）、`3.5\\times 10^{3}` 在未包裹时只解析出 10、
   `1e3` 被当成 `1*E*3`、`\\%` 会被当成 `*(1/100)` 也当成单位（两种解释都接受）。归一化层负责在
   送进 Math-Verify 之前把这些形式统一掉。
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing
import os
import re
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 版本与常量
# ---------------------------------------------------------------------------

VERIFIER_VERSION = "1.0"

REASON_CORRECT = "correct"
REASON_WRONG = "wrong"
REASON_NO_ANSWER = "no_answer"
REASON_MULTIPLE = "multiple_answers"
REASON_TRUNCATED = "truncated"
REASON_TIMEOUT = "timeout"

REASONS: frozenset[str] = frozenset(
    {REASON_CORRECT, REASON_WRONG, REASON_NO_ANSWER, REASON_MULTIPLE, REASON_TRUNCATED, REASON_TIMEOUT}
)

#: 单次判定（解析 + 等价比较）的默认超时，秒。论文/任务包建议 5 秒。
DEFAULT_TIMEOUT_S: float = float(os.environ.get("STRICT_VERIFIER_TIMEOUT_S", "5"))

#: 送进 Math-Verify 的字符串长度上限。真正被解析的是"最终答案"，正常远短于这个值；
#: 超长输入是攻击面（解析复杂度不可控），直接判错。
MAX_PARSE_CHARS: int = int(os.environ.get("STRICT_VERIFIER_MAX_CHARS", "2000"))

#: 子进程池大小（仅在无法用 signal 的环境下启用）。
POOL_WORKERS: int = int(os.environ.get("STRICT_VERIFIER_POOL_WORKERS", "4"))

#: 没有 marker 时，"尾部长句 + 没有句末标点" 视为截断的文本长度下限。
TRUNCATION_MIN_CHARS = 40

#: 只在尾部这么多字符里检查未闭合的数学环境（截断一定发生在这里；中段的 `\[` 可能后文才闭合）。
TRUNCATION_TAIL_CHARS = 300

#: 只在尾部这么多字符里找 "The answer is ..." 这类收尾句，避免把推理中的句子当答案。
PHRASE_TAIL_CHARS = 400


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    """一次判定的结构化结果（任务包要求的四个字段）。"""

    score: float  # 1.0 或 0.0
    extracted: str | None  # 抽取到的最终答案（归一化前的原文）
    parse_ok: bool  # 抽取到了答案且等价比较正常执行
    reason: str  # correct / wrong / no_answer / multiple_answers / truncated / timeout

    def __post_init__(self) -> None:
        if self.reason not in REASONS:
            raise ValueError(f"unknown reason: {self.reason!r}")
        if self.score not in (0.0, 1.0):
            raise ValueError(f"score must be 0.0 or 1.0, got {self.score!r}")

    @property
    def correct(self) -> bool:
        return self.score == 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "extracted": self.extracted,
            "parse_ok": self.parse_ok,
            "reason": self.reason,
        }


def _ok(extracted: str | None = None) -> VerifyResult:
    return VerifyResult(score=1.0, extracted=extracted, parse_ok=True, reason=REASON_CORRECT)


def _fail(reason: str, extracted: str | None = None, parse_ok: bool = False) -> VerifyResult:
    return VerifyResult(score=0.0, extracted=extracted, parse_ok=parse_ok, reason=reason)


# ---------------------------------------------------------------------------
# Math-Verify 导入（延迟 + 显式失败）
# ---------------------------------------------------------------------------
_MV_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - 依赖在 CI/训练机上总是存在
    from math_verify import ExprExtractionConfig, LatexExtractionConfig
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify

    _MV_AVAILABLE = True
    # 我们主动关掉了上游超时（见模块 docstring），它会在每次调用时往 stdout 打一行警告，
    # 污染训练/评测日志。这里把两个模块的"已提示"标志置位，保持输出干净。
    try:
        import math_verify.grader as _mv_grader
        import math_verify.parser as _mv_parser

        _mv_parser.TIMEOUT_WARNING_SHOWN = True
        _mv_grader.TIMEOUT_WARNING_SHOWN = True
    except Exception:  # pragma: no cover - 上游改名时不影响功能
        pass
except Exception as _exc:  # pragma: no cover
    _MV_AVAILABLE = False
    _MV_IMPORT_ERROR = _exc
    _mv_parse = None  # type: ignore[assignment]
    _mv_verify = None  # type: ignore[assignment]
    ExprExtractionConfig = None  # type: ignore[assignment]
    LatexExtractionConfig = None  # type: ignore[assignment]


def _math_verify_version() -> str:
    try:
        from importlib.metadata import version

        return version("math-verify")
    except Exception:  # pragma: no cover
        return "not-installed"


SCORER_NAME = "selfevolve-strict-verifier"
SCORER_VERSION = f"{SCORER_NAME}-{VERIFIER_VERSION}+math-verify-{_math_verify_version()}"


def _require_math_verify() -> None:
    """Math-Verify 缺失时显式报错，避免静默把所有答案判错（协议 §4：训练奖励不能悄悄退化）。"""
    if not _MV_AVAILABLE:
        raise RuntimeError(
            "math-verify is required by the strict verifier but could not be imported: "
            f"{_MV_IMPORT_ERROR!r}. Install it with `pip install math-verify==0.9.0`."
        )


# ---------------------------------------------------------------------------
# 1. 答案抽取
# ---------------------------------------------------------------------------
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")
_HASH_MARKER = "####"
_FENCE_RE = re.compile(r"```.*?```", re.S)

#: 抽取到的最终答案若是这些收尾词，直接判定为标签残留而不是答案。
_LABEL_PREFIX_RE = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer|final\s+answer|answer|ans|result|答案|最终答案|答)"
    r"\s*(?:is|are|为|是)?\s*[:：]?\s*",
    re.IGNORECASE,
)

#: "The answer is ..." / "答案是 ..." 收尾句。允许中间插几个词
#: （真实输出里常见 "the answer Ringo should have found is ..."）。
_PHRASE_RE = re.compile(
    r"(?:the\s+(?:final\s+)?answer(?:\s+\S+){0,6}?\s+(?:is|would\s+be)"
    r"|final\s+answer\b|answer\s*[:：]"
    r"|答案\s*(?:是|为)?\s*[:：]?|答\s*[:：])",
    re.IGNORECASE,
)

#: 明确的多答案表达（“3 or 5”“3或5”“3 and 5”）。
_DISJUNCTION_RE = re.compile(r"(?:\bor\b|\band\b|或|或者|\balso\b|以及)", re.IGNORECASE)
#: 裸数字串（注意不含 `/`：`14/3` 是单个分数而不是两个答案）。
_NUMBER_LIST_RE = re.compile(r"^[\(\[\{]?[\s\d.,\-+]+[\)\]\}]?$")

#: 结尾追加的"注解"（单位/说明）不能当成数学符号的那批词。
_MATH_TOKEN_BLACKLIST = frozenset(
    {
        "pi",
        "e",
        "i",
        "sqrt",
        "log",
        "ln",
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "arcsin",
        "arccos",
        "arctan",
        "exp",
        "abs",
        "mod",
        "x",
        "y",
        "z",
        "n",
        "m",
        "k",
        "a",
        "b",
        "c",
        "p",
        "q",
        "r",
        "t",
    }
)


def _find_boxed_spans(text: str) -> list[tuple[int, int, str, bool]]:
    """找出所有 `\\boxed{...}`。返回 (start, end, content, closed)。

    `closed=False` 表示大括号没闭合——要么是模型输出被截断，要么是畸形 LaTeX。
    """
    spans: list[tuple[int, int, str, bool]] = []
    for match in _BOXED_OPEN_RE.finditer(text):
        content_start = match.end()
        depth = 1
        i = content_start
        closed = False
        while i < len(text):
            ch = text[i]
            escaped = i > 0 and text[i - 1] == "\\"
            if ch == "{" and not escaped:
                depth += 1
            elif ch == "}" and not escaped:
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), i + 1, text[content_start:i], True))
                    closed = True
                    break
            i += 1
        if not closed:
            spans.append((match.start(), len(text), text[content_start:], False))
    return spans


def _strip_hash_segment(segment: str) -> str | None:
    """从 `####` 之后的那段文本里取出"看起来是最终答案"的一行。

    真实输出里这段可能是 `460`、`\\boxed{2}`、`Final Answer: 24`，也可能是 `#### Step 2: ...`
    这种分节标题；还有一种常见形态是 `#### 3\\n$(3, \\frac{\\pi}{2})$`（先给个残缺值再补全）。
    因此**从后往前**找第一个合格行，而不是取第一行。
    """
    lines = [line.strip() for line in segment.splitlines() if line.strip()]
    for line in reversed(lines[:6]):
        # 段里自带 boxed 时以 boxed 内容为准
        spans = _find_boxed_spans(line)
        closed = [c for _, _, c, ok in spans if ok]
        if closed:
            return closed[-1].strip()
        stripped = _LABEL_PREFIX_RE.sub("", line).strip()
        if not stripped:
            continue
        if _looks_like_answer_line(stripped):
            return _strip_trailing_punctuation(stripped)
    return None


#: 形似答案但其实是备注/单位的单词（`#### 42 units` 不能被当成文本答案）。
_NON_ANSWER_WORDS = frozenset(
    {"units", "unit", "approximately", "approx", "about", "and", "or", "so", "thus", "therefore"}
)


def _looks_like_answer_line(text: str) -> bool:
    """区分"最终答案"与分节标题 / 散文。

    - 必须含数字或 LaTeX 命令（`\\frac`、`\\pi`、`\\sqrt`…），或本身就是短文本答案；
    - 不能是一串小写英文单词（"Prize but Not Grand Prize" 这类标题）；
    - 长度受限。
    """
    if not text or len(text) > 80:
        return False
    has_math = bool(re.search(r"\d|\\[a-zA-Z]+", text))
    if not has_math:
        # 文本答案："Evelyn" / "Yes" / "east"（MATH-500 里有 36.8% 的 gold 不是纯数字）
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z\-]{1,24}", text)) and text.lower() not in _NON_ANSWER_WORDS
    # 以数字/LaTeX 开头的短行就是答案本身（`36 batches of oatmeal cookies`），
    # 分节标题（`Step 2: ...`、`Prize but Not Grand Prize`）不会有这个开头。
    if re.match(r"^[\-\+\(\[\{]?\d", text) or text.startswith("\\"):
        return True
    words = re.findall(r"[A-Za-z]{2,}", re.sub(r"\\[a-zA-Z]+", " ", text))
    # 出现 3 个以上普通单词 -> 更像句子/标题；"Final Answer" 两个词仍可接受
    if len(words) >= 3:
        return False
    return True


def _strip_trailing_punctuation(text: str) -> str:
    """去掉首尾标点，以及被 `\\( ... \\)` / `\\[ ... \\]` 包裹时残留的环境符。"""
    out = text.strip().strip(".,;:。，；：、").strip()
    for closer in (r"\)", r"\]"):
        if out.endswith(closer):
            out = out[: -len(closer)].strip()
    for opener in (r"\(", r"\["):
        if out.startswith(opener):
            out = out[len(opener) :].strip()
    return out


def _hash_candidates(text: str) -> list[tuple[int, str]]:
    """`####` 候选：每个 `####` 到下一个 `####` 之间的一段，取 (位置, 答案原文)。"""
    out: list[tuple[int, str]] = []
    positions = [m.start() for m in re.finditer(re.escape(_HASH_MARKER), text)]
    for idx, pos in enumerate(positions):
        end = positions[idx + 1] if idx + 1 < len(positions) else len(text)
        cleaned = _strip_hash_segment(text[pos + len(_HASH_MARKER) : end])
        if cleaned:
            out.append((pos, cleaned))
    return out


def _phrase_candidates(text: str) -> list[tuple[int, str]]:
    """尾部收尾句候选（只在没有任何显式 marker 时才启用）。

    真实输出里收尾句后面常常不是裸答案，而是 "Final answer: The rink has 23 yellow cars." 这种
    整句复述，所以这里复用 `_value_from_sentence` 从句子内部取值，而不是要求整行都是答案。
    """
    tail_start = max(0, len(text) - PHRASE_TAIL_CHARS)
    tail = text[tail_start:]
    out: list[tuple[int, str]] = []
    for match in _PHRASE_RE.finditer(tail):
        rest = tail[match.end() :][:200].strip()
        if not rest:
            continue
        first_line = rest.splitlines()[0].strip() if rest.splitlines() else ""
        candidate = _strip_trailing_punctuation(first_line)
        if candidate and len(candidate) <= 80 and _looks_like_answer_line(candidate):
            out.append((tail_start + match.start(), candidate))
            continue
        sentence_value = _value_from_sentence(rest)
        if sentence_value:
            out.append((tail_start + match.start(), sentence_value))
    return out


def _value_from_sentence(sentence: str) -> str | None:
    """从一句自然语言里取"唯一可判定的值"：强调 > 系动词/等号之后 > 全句唯一值。"""
    value = _emphasized_value(sentence) or _value_after_copula(sentence)
    if value is None:
        unique = _unique_values(sentence)
        if len(unique) != 1:
            return None
        value = next(iter(unique))
    if not value or _is_disjunctive(value):
        return None
    return value


#: 结尾"结论句"的引导词（第四优先级抽取器用）。
_CONCLUSION_CUE_RE = re.compile(
    r"^[\s\*_\-\"'“”]*"
    r"(?:therefore|thus|hence|so\b|in\s+total|in\s+conclusion|altogether|overall"
    r"|最终|因此|所以|综上|总之)",
    re.IGNORECASE,
)
#: 结论句里能当答案的数/表达式片段。
_VALUE_TOKEN_RE = re.compile(
    r"\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}"  # \frac{a}{b}
    r"|\\sqrt\s*\{?[^{}\s]+\}?"  # \sqrt{5}
    # 数字（可带千分位/美元符）。句末的 `.` 允许（`$1520.`），但小数点后必须跟数字。
    r"|(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?\$?(?![\w]|\.\d)"
)


def _closing_sentence_candidate(text: str) -> str | None:
    """第四优先级：结尾"结论句"里的最终答案。

    动机（实测）：Qwen2.5-1.5B-Instruct 在 GSM8K 上有 45% 的样本**根本不写 `####`**，
    结尾是 "Therefore, the total is **273 yards**." 这种一句结论。如果一律判 `no_answer`，
    会把"格式不合规"误报成"能力不行"（W2 数据上 base GSM8K 会从 0.72 掉到 0.38）。

    取值顺序（越具体越优先）：
    1. 结论句里被加粗/强调的值（`**273**`）——模型自己标出来的最终答案；
    2. 最后一个 "is / was / 是 / =" 之后的第一个值——英文结论句的主语补足语位置；
    3. 整句恰好只有一个值。

    仍然只对**结论句**生效、且不允许出现多个互斥候选：正确数字埋在推理里、结论句没有数字时依旧
    判 `no_answer`；结论句里出现 "3 or 5" 这类多候选则判 `multiple_answers`。
    """
    tail = text[-PHRASE_TAIL_CHARS:]
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?。！？])\s+|\n+", tail) if chunk.strip()]
    if not chunks:
        return None
    sentence = chunks[-1]
    if not _CONCLUSION_CUE_RE.match(sentence) and len(sentence.split()) > 25:
        # 没有结论引导词时只接受短句：长段落的最后一句更可能是补充说明
        return None
    return _value_from_sentence(sentence)


def _unique_values(text: str) -> set[str]:
    values = _VALUE_TOKEN_RE.findall(text)
    cleaned = {_strip_trailing_punctuation(v.replace("$", "")).strip() for v in values}
    return {v for v in cleaned if v and v not in {"-", "+"}}


def _emphasized_value(sentence: str) -> str | None:
    for match in re.finditer(r"\*\*(.+?)\*\*|__(.+?)__", sentence):
        content = match.group(1) or match.group(2) or ""
        unique = _unique_values(content)
        if len(unique) == 1:
            return next(iter(unique))
    return None


def _value_after_copula(sentence: str) -> str | None:
    last = None
    for match in re.finditer(r"\bis\b|\bwas\b|\bequals\b|是|=", sentence):
        last = match.end()
    if last is None:
        return None
    for value in _VALUE_TOKEN_RE.findall(sentence[last:]):
        cleaned = _strip_trailing_punctuation(value.replace("$", "")).strip()
        if cleaned and cleaned not in {"-", "+"}:
            return cleaned
    return None


def _is_disjunctive(text: str) -> bool:
    """最终答案里出现多个互斥候选：`3 or 5`、`3或5`、`3 and 5`。"""
    if not text:
        return False
    if _DISJUNCTION_RE.search(text):
        return True
    stripped = _THOUSANDS_RE.sub("", text).strip()
    # 被括号/花括号整体包住的数字串是集合/坐标/区间，属于**一个**答案，不是多答案
    if re.fullmatch(r"[\(\[\{].*[\)\]\}]", stripped):
        return False
    # "3, 5" / "3 5" 这种裸数字串（没有括号包裹成集合/坐标）。
    if _NUMBER_LIST_RE.match(stripped) and len(re.findall(r"-?\d+(?:\.\d+)?", stripped)) >= 2:
        return True
    return False


def extract_answer(response: str) -> tuple[str | None, str]:
    """抽取最终答案。

    返回 `(answer, state)`，`state ∈ {"ok", "no_answer", "multiple_answers", "truncated"}`。
    抽取优先级（任务包 §1）：最后一个 `\\boxed{...}` → `####` → 结尾收尾句；
    同类候选之间按**出现位置**取最后一个（"最终答案"是位置概念，且真实输出里 boxed 与 `####`
    同时出现的样本占 0.4%–3%，见 `reports/verifier_validation.md`）。

    防投机：
    - 多个 `\\boxed{}` 且互不一致 → `multiple_answers`；
    - `####` 后出现多个互不一致的合格答案 → `multiple_answers`；
    - 答案本身是 "3 or 5" 这类多候选 → `multiple_answers`；
    - `\\boxed{` 未闭合 / 数学环境未闭合 → `truncated`。
    """
    text = "" if response is None else str(response)
    if not text.strip():
        return None, REASON_NO_ANSWER

    # 代码块里出现的 marker 不算（模型常在 ```python 里打印中间值）
    without_code = _FENCE_RE.sub(lambda m: " " * len(m.group(0)), text)

    # 尾部数学环境没闭合 = 输出被截断：先把后续所有抽取作废（宁严勿松）
    if _has_unbalanced_math(without_code[-TRUNCATION_TAIL_CHARS:]):
        return None, REASON_TRUNCATED

    spans = _find_boxed_spans(without_code)
    open_spans = [s for s in spans if not s[3]]
    closed_spans = [s for s in spans if s[3]]

    # --- boxed 冲突检测 ---
    boxed_values: list[tuple[int, str]] = [(start, content.strip()) for start, _, content, _ in closed_spans]
    boxed_values = [(pos, val) for pos, val in boxed_values if val]
    conflict = _has_conflicting_values([val for _, val in boxed_values])
    if conflict:
        return None, REASON_MULTIPLE

    # --- `####` 候选 ---
    hashes = _hash_candidates(without_code)

    # 注：`####` 之间不互不一致就不判错——GSM8K 的约定是"最后一个 `####` 之后是最终答案"，
    # 模型先写草稿值再补全（`#### 3#### 3` + 下一行完整答案）是常态，verl 内置判分也取最后一个。
    # 实测这类样本占 ckpt MATH-500 的 5.4%，若按"多答案"判错会系统性压低分数。

    # --- 位置选择：显式 marker（boxed / ####）优先，其次收尾句，最后结论句 ---
    candidates: list[tuple[int, str]] = []
    candidates.extend(boxed_values)
    candidates.extend(hashes)
    if not candidates:
        candidates.extend(_phrase_candidates(without_code))

    if open_spans:
        # 有 `\boxed{` 没闭合：输出是畸形的（被截断或 LaTeX 写坏），按"宁严勿松"整条判错
        return None, REASON_TRUNCATED

    if not candidates:
        closing = _closing_sentence_candidate(without_code)
        if closing:
            return closing, "ok"
        return None, _no_answer_state(without_code)

    pos, answer = max(candidates, key=lambda item: item[0])
    answer = _strip_trailing_punctuation(answer)
    if not answer:
        return None, REASON_NO_ANSWER
    if _is_disjunctive(answer):
        return None, REASON_MULTIPLE
    if _has_unbalanced_math(answer):
        return None, REASON_TRUNCATED
    return answer, "ok"


def _no_answer_state(text: str) -> str:
    """没有抽到答案时，区分 `truncated` 与 `no_answer`。"""
    tail = text.rstrip()
    if not tail:
        return REASON_NO_ANSWER
    if _has_unbalanced_math(tail):
        return REASON_TRUNCATED
    if len(tail) >= TRUNCATION_MIN_CHARS and tail[-1] not in ".!?。！？:：)]}>\"'”’":
        return REASON_TRUNCATED
    return REASON_NO_ANSWER


def _has_unbalanced_math(text: str) -> bool:
    """检测未闭合的 `\\(` / `\\[` 数学环境。

    刻意**不看** `$`：真实输出里货币符号 `$18` 到处都是，奇偶性完全没有诊断价值
    （W2 数据里曾把 10.9% 的正常结尾误判成截断）。
    """
    for opener, closer in ((r"\(", r"\)"), (r"\[", r"\]")):
        if text.count(opener) > text.count(closer):
            return True
    return False


def _has_conflicting_values(values: Sequence[str]) -> bool:
    """判断一组候选答案是否互不一致（先用字符串归一化，再做数学等价）。

    `2` / `2 trips` / `2 dollars` 这种"同一个值 + 注解"不算冲突。
    """
    if len(values) <= 1:
        return False
    normalized: list[str] = []
    for value in values:
        norm = normalize_answer(value)
        if not norm:
            continue
        if not normalized:
            normalized.append(norm)
            continue
        if any(_same_value(norm, prev) for prev in normalized):
            continue
        normalized.append(norm)
    return len(normalized) > 1


def _same_value(a: str, b: str) -> bool:
    if a == b:
        return True
    # `2` vs `2 trips`：后者是前者的注解
    if a.startswith(b + " ") or b.startswith(a + " "):
        return True
    return _equivalent_safe(a, b)


# ---------------------------------------------------------------------------
# 2. 归一化
# ---------------------------------------------------------------------------
_UNIT_SUFFIXES = (
    "dollars",
    "dollar",
    "usd",
    "cents",
    "cent",
    "euros",
    "euro",
    "pounds",
    "pound",
    "yuan",
    "rmb",
    "kg",
    "g",
    "mg",
    "cm",
    "mm",
    "km",
    "m",
    "ft",
    "feet",
    "inches",
    "inch",
    "miles",
    "mile",
    "yards",
    "yard",
    "seconds",
    "second",
    "secs",
    "sec",
    "minutes",
    "minute",
    "mins",
    "min",
    "hours",
    "hour",
    "hrs",
    "hr",
    "days",
    "day",
    "weeks",
    "week",
    "months",
    "month",
    "years",
    "year",
    "degrees",
    "degree",
    "radians",
    "units",
    "unit",
    "people",
    "students",
    "apples",
    "eggs",
    "items",
    "times",
)

_TEXT_WRAPPER_RE = re.compile(r"\\(?:text|textrm|textbf|textit|mathrm|mbox)\s*\{([^{}]*)\}")
_THIN_SPACE_RE = re.compile(r"\\[!,;:]|\\ |~")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMERIC_FRACTION_RE = re.compile(r"\\frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}")
_LEADING_DOT_RE = re.compile(r"(^|[\s\(\[\{,])\.(\d)")

#: 模型常用 Unicode 数学符号，Math-Verify 只认 LaTeX（例：`(-∞, 2) ∪ (3, ∞)`）。
_UNICODE_MATH_MAP = {
    "∞": r"\infty ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "≤": r"\le ",
    "≥": r"\ge ",
    "≠": r"\neq ",
    "×": r"\times ",
    "÷": "/",
    "π": r"\pi ",
    "±": r"\pm ",
    "∈": r"\in ",
    "√": r"\sqrt ",
    "−": "-",
    "–": "-",
}
_SCI_LATEX_RE = re.compile(
    r"^(?P<mant>[+-]?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|\\ast|\*|×)\s*10\s*\^?\s*\{?\s*(?P<exp>[+-]?\d+)\s*\}?$"
)
_SCI_E_RE = re.compile(r"^(?P<mant>[+-]?\d+(?:\.\d+)?)[eE](?P<exp>[+-]?\d+)$")


def normalize_answer(answer: str | None) -> str:
    """把答案归一化成可比较的形式（对 pred 和 gold 用同一套规则）。

    做过的变换（都在 `reports/verifier_validation.md` 里有对应用例）：
    去美元符号/千分位逗号/百分号单位/单位后缀、`\\left`/`\\right`、`\\dfrac→\\frac`、
    `\\text{...}` 取内部文本、科学计数法求值、多余空白、首尾标点。
    """
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""

    # 外层包裹：引号/反引号/加粗/单个 $...$ / 残留 \boxed{...}
    text = text.strip("`'\"“”‘’").strip()
    text = _strip_markdown_emphasis(text)
    if text.startswith("$") and text.endswith("$") and len(text) > 1 and not text.startswith("\\$"):
        text = text[1:-1].strip()
    spans = _find_boxed_spans(text)
    if len(spans) == 1 and spans[0][3] and spans[0][0] == 0 and spans[0][1] == len(text):
        text = spans[0][2].strip()

    # \text{...} 取内部
    prev = None
    while prev != text:
        prev = text
        text = _TEXT_WRAPPER_RE.sub(r"\1", text)

    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # `11sqrt2` / `2sqrt{3}`：模型常漏掉反斜杠，补成合法 LaTeX
    text = re.sub(r"(\d)\s*sqrt\s*\{?\s*(\d+)\s*\}?", r"\1\\sqrt{\2}", text)
    # Unicode 数学符号 -> LaTeX
    for symbol, latex in _UNICODE_MATH_MAP.items():
        text = text.replace(symbol, latex)
    # 纯数字分数 `\frac{16}{49}` -> `16/49`：Math-Verify 在矩阵/多行环境里会解析失败，
    # 先做无歧义的规范化（只处理分子分母都是纯数字的情况）。
    text = _NUMERIC_FRACTION_RE.sub(r"\1/\2", text)
    text = _THIN_SPACE_RE.sub("", text)
    text = text.replace("\\$", "").replace("$", "")
    # 百分号按单位处理：数据里 gold 全是裸数字（见验证报告 §3），统一去掉"%"再比较
    text = text.replace("\\%", "").replace("%", "")
    # 千分位逗号（可能多段）
    prev = None
    while prev != text:
        prev = text
        text = _THOUSANDS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    # `.5` -> `0.5`（Math-Verify 解析裸 `.5` 会失败）
    text = _LEADING_DOT_RE.sub(r"\g<1>0.\g<2>", text)

    # 科学计数法：只有整个答案就是一个科学计数法时才求值
    text = _maybe_scientific(text)

    text = _strip_trailing_punctuation(text)
    text = _strip_unit_suffix(text)
    text = _strip_trailing_annotation(text)
    text = text.strip()
    # 去掉外层 $...$ 之后可能又出现首尾标点
    text = _strip_trailing_punctuation(text)
    return text


def _maybe_scientific(text: str) -> str:
    match = _SCI_LATEX_RE.match(text) or _SCI_E_RE.match(text)
    if not match:
        return text
    try:
        value = Decimal(match.group("mant")) * (Decimal(10) ** int(match.group("exp")))
    except (InvalidOperation, ValueError):
        return text
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _strip_markdown_emphasis(text: str) -> str:
    """去掉把答案整个包住的 markdown 强调：`**5**` / `__5__` / `*5*` / `_5_`。"""
    for open_mark, close_mark in (("**", "**"), ("__", "__"), ("*", "*"), ("_", "_")):
        if len(text) > 2 * len(open_mark) and text.startswith(open_mark) and text.endswith(close_mark):
            inner = text[len(open_mark) : -len(close_mark)].strip()
            if inner and (open_mark not in inner and close_mark not in inner):
                return inner
    return text


def _strip_unit_suffix(text: str) -> str:
    """去掉 `100 dollars` / `3 cm` 这类单位后缀（保守：只认白名单里的词）。"""
    match = re.match(r"^(?P<value>.+?)\s+(?P<unit>[a-zA-Z]+)$", text)
    if not match:
        return text
    unit = match.group("unit").lower()
    if unit not in _UNIT_SUFFIXES:
        return text
    value = match.group("value").strip()
    # 前一段必须真的像数值/表达式，否则 "No solution" 这类文本答案会被误砍
    if not re.search(r"[\d\}\)\]]$", value):
        return text
    return value


def _strip_trailing_annotation(text: str) -> str:
    """去掉答案后面跟的人话注解：`2 trips` / `36 batches of oatmeal cookies`。

    规则保守：被保留的前缀必须以数字/右括号结尾，被去掉的部分必须全是字母（≤6 个词），
    且第一个词不能是数学符号（`2 pi` 不能变成 `2`）。
    """
    match = re.match(r"^(?P<value>.*[\d\}\)])\s+(?P<rest>[A-Za-z][A-Za-z\s\-]{0,60})$", text)
    if not match:
        return text
    rest = match.group("rest")
    rest_words = rest.split()
    if not rest_words or len(rest_words) > 12:
        return text
    if any(ch.isdigit() for ch in rest):  # "5 times 3" 不能被砍成 5
        return text
    if rest_words[0].lower() in _MATH_TOKEN_BLACKLIST:
        return text
    return match.group("value").strip()


# ---------------------------------------------------------------------------
# 3. 等价判断（Math-Verify 只做这一步）
# ---------------------------------------------------------------------------
def _box_wrap(text: str) -> str:
    """Math-Verify 对裸答案解析不出东西，必须包成 `\\boxed{...}`（W2 的结论）。"""
    return "\\boxed{" + text + "}"


def _parse_answer(text: str) -> list[Any]:
    _require_math_verify()
    assert _mv_parse is not None
    parsed = _mv_parse(
        _box_wrap(text),
        [LatexExtractionConfig()],
        parsing_timeout=None,  # 上游超时在本进程里不可用/代价高，外层自己管
        raise_on_error=False,
    )
    return list(parsed or [])


def _compare_sync(pred: str, gold: str) -> bool:
    """纯函数：两个**已归一化**字符串是否等价。会被送进子进程执行，必须可 pickle。"""
    if pred == gold:
        return True
    if len(pred) > MAX_PARSE_CHARS or len(gold) > MAX_PARSE_CHARS:
        return False
    try:
        gold_parsed = _parse_answer(gold)
        pred_parsed = _parse_answer(pred)
    except Exception as exc:  # Math-Verify 对畸形输入可能直接抛异常
        logger.debug("parse failed pred=%r gold=%r: %s", pred[:80], gold[:80], exc)
        return False
    if not gold_parsed or not pred_parsed:
        return False
    try:
        return bool(_mv_verify(gold_parsed, pred_parsed, timeout_seconds=None, raise_on_error=False))
    except Exception as exc:  # pragma: no cover - 防御性
        logger.debug("verify failed pred=%r gold=%r: %s", pred[:80], gold[:80], exc)
        return False


class _Timeout(Exception):
    """内部超时信号。"""


#: 超时模式：auto（默认）/ signal / pool / none。
#: - auto：POSIX 主线程用 SIGALRM，其它情况用进程池；
#: - signal：强制 SIGALRM（不在主线程时无意义）；
#: - pool：强制进程池；
#: - none：关掉超时直接同步执行——只在本地调试/离线分析时用，别用于训练。
#:   注意：进程池在 Windows 上要求调用方脚本有 `if __name__ == "__main__"` 保护。
TIMEOUT_MODE: str = os.environ.get("STRICT_VERIFIER_TIMEOUT_MODE", "auto").lower()


def _can_use_sigalrm() -> bool:
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


_POOL: ProcessPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _get_pool() -> ProcessPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ProcessPoolExecutor(
                max_workers=max(1, POOL_WORKERS),
                mp_context=multiprocessing.get_context("spawn"),
            )
        return _POOL


def _reset_pool() -> None:
    """超时后池里的 worker 可能卡死，整池重建。"""
    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # pragma: no cover
            pass


@atexit.register
def _shutdown_pool() -> None:  # pragma: no cover - 进程退出路径
    _reset_pool()


def _run_with_timeout(func: Callable[..., Any], *args: Any, timeout_s: float) -> Any:
    """带超时执行 `func(*args)`。

    - POSIX 主线程：`signal.setitimer`，无进程开销；
    - 其它情况（Windows / 子线程 / Ray actor 的 executor 线程）：进程池 + `future.result(timeout)`。
      Math-Verify 自带的超时在子线程里会抛 ValueError，所以必须走这条。

    进程池本身出问题（例如 Windows 上调用方脚本没有 `if __name__ == "__main__"` 保护）时
    退化为同步执行：**判定结果不变**，只是失去超时保护。
    """
    if timeout_s <= 0 or TIMEOUT_MODE == "none":
        return func(*args)
    if TIMEOUT_MODE == "signal" or (TIMEOUT_MODE == "auto" and _can_use_sigalrm()):
        if not _can_use_sigalrm():  # 显式要求 signal 但当前不在主线程
            return func(*args)

        def _handler(signum: int, frame: Any) -> None:
            raise _Timeout()

        previous = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            return func(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    try:
        pool = _get_pool()
        future = pool.submit(func, *args)
    except Exception as exc:  # pragma: no cover - 池创建失败（Windows 无 main guard 等）
        logger.warning("strict verifier process pool unavailable (%s); running inline", exc)
        return func(*args)
    try:
        return future.result(timeout=timeout_s)
    except FuturesTimeoutError as exc:
        future.cancel()
        _reset_pool()
        raise _Timeout() from exc
    except Exception as exc:  # pragma: no cover - 池被杀死（BrokenProcessPool）
        logger.warning("strict verifier process pool failed (%s); running inline", exc)
        _reset_pool()
        return func(*args)


def is_equivalent(pred: str, gold: str, *, timeout_s: float | None = None) -> bool:
    """比较两个答案是否等价；超时/异常一律返回 False（宁严勿松）。"""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if not pred_norm or not gold_norm:
        return False
    timeout = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    try:
        return bool(_run_with_timeout(_compare_sync, pred_norm, gold_norm, timeout_s=timeout))
    except _Timeout:
        logger.warning("strict verifier timed out: pred=%r gold=%r", pred_norm[:80], gold_norm[:80])
        return False
    except Exception as exc:  # pragma: no cover - 进程池不可用等
        logger.warning("strict verifier equivalence failed (%s); falling back to string equality", exc)
        return pred_norm == gold_norm


def _equivalent_safe(a: str, b: str) -> bool:
    """供内部冲突检测用：不抛异常，超时按"不相等"处理。"""
    if a == b:
        return True
    try:
        return is_equivalent(a, b)
    except Exception:  # pragma: no cover
        return a == b


# ---------------------------------------------------------------------------
# 4. 对外接口
# ---------------------------------------------------------------------------
def verify(
    response: str,
    ground_truth: str,
    *,
    finished: bool | None = None,
    timeout_s: float | None = None,
) -> VerifyResult:
    """判定一条响应。

    Args:
        response: 模型输出全文。
        ground_truth: 数据集金标（任意格式，内部会归一化）。
        finished: 是否正常结束。评测侧传 `finish_reason != "length"`；训练侧拿不到，传 None 走文本推断。
        timeout_s: 单次判定超时（秒）。None 用 `DEFAULT_TIMEOUT_S`。

    Returns:
        VerifyResult：score 只有 0.0/1.0，reason 见 `REASONS`。
    """
    extracted, state = extract_answer(response)
    if state == REASON_MULTIPLE:
        return _fail(REASON_MULTIPLE)
    if state == REASON_TRUNCATED:
        return _fail(REASON_TRUNCATED, extracted)
    if state == REASON_NO_ANSWER or extracted is None:
        return _fail(REASON_NO_ANSWER)
    if finished is False:
        # 明确被截断：任务包要求判错，且实测代价极小（W2 数据里截断样本被判对的不到 5%）
        return _fail(REASON_TRUNCATED, extracted)

    timeout = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    try:
        matched = _run_with_timeout(
            _compare_sync, normalize_answer(extracted), normalize_answer(ground_truth), timeout_s=timeout
        )
    except _Timeout:
        return _fail(REASON_TIMEOUT, extracted)
    if matched:
        return _ok(extracted)
    return _fail(REASON_WRONG, extracted, parse_ok=_predicates_parsed(extracted))


def _predicates_parsed(answer: str) -> bool:
    """诊断用：pred 能不能被 Math-Verify 解析（不影响 score）。"""
    try:
        return bool(_parse_answer(normalize_answer(answer)))
    except Exception:  # pragma: no cover
        return False


def verify_many(
    pairs: Iterable[tuple[str, str]],
    *,
    finished: Sequence[bool | None] | None = None,
    timeout_s: float | None = None,
) -> list[VerifyResult]:
    """批量判定（重打分用）。与逐条 `verify` 结果一致，只是复用进程池/超时设施。"""
    pair_list = list(pairs)
    if finished is None:
        finished_list: list[bool | None] = [None] * len(pair_list)
    else:
        finished_list = list(finished)
        if len(finished_list) != len(pair_list):
            raise ValueError("finished must have the same length as pairs")
    timeout = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
    results: list[VerifyResult] = []
    for (response, gold), fin in zip(pair_list, finished_list):
        results.append(verify(response, gold, finished=fin, timeout_s=timeout))
    return results


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "REASONS",
    "REASON_CORRECT",
    "REASON_MULTIPLE",
    "REASON_NO_ANSWER",
    "REASON_TIMEOUT",
    "REASON_TRUNCATED",
    "REASON_WRONG",
    "SCORER_NAME",
    "SCORER_VERSION",
    "VERIFIER_VERSION",
    "VerifyResult",
    "extract_answer",
    "is_equivalent",
    "normalize_answer",
    "verify",
    "verify_many",
]
