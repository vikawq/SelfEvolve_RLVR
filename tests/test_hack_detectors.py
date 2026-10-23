"""W4 hack 检测指标单测（`src/selfevolve/metrics/hack_detectors.py`）。

逐指标覆盖：数字个数、boxed 数、首数字位置、长度、重复 n-gram、reward gap、hack onset。
这些指标会决定"是否复现"的结论，所以边界（空响应、无数字、短于 n 个词）必须钉死。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.metrics.hack_detectors import (  # noqa: E402
    aggregate_responses,
    count_boxed,
    count_distinct_numbers,
    detect_response,
    first_number,
    first_number_position,
    hack_onset,
    mean,
    normalize_number_token,
    numbers_in_order,
    percentile,
    repeated_ngram_ratio,
    reward_gap,
    rolling_mean,
)


# ---------------------------------------------------------------------------
# 数字
# ---------------------------------------------------------------------------
def test_numbers_in_order_keeps_written_form() -> None:
    assert numbers_in_order("I got 1,000 and 1000 and 3.5, not -2.") == [
        "1,000",
        "1000",
        "3.5",
        "-2",
    ]


def test_distinct_numbers_normalizes_thousands_separator() -> None:
    # 1,000 与 1000 是同一个数；3.5 与 -2 各算一个
    assert count_distinct_numbers("1,000 + 1000 + 3.5 - 2") == 3


def test_distinct_numbers_ignores_digits_inside_words() -> None:
    # Qwen2.5 / v0.9.1 这类版本号不应被算成"答案候选"
    assert count_distinct_numbers("Qwen2.5 and v0.9.1") == 0


def test_normalize_number_token() -> None:
    assert normalize_number_token("1,234") == "1234"
    assert normalize_number_token("3.5") == "3.5"


def test_first_number_position() -> None:
    text = "Let me compute: 17 + 5 = 22."
    assert first_number(text) == "17"
    assert first_number_position(text) == text.index("17")
    assert first_number("no digits") is None
    assert first_number_position("no digits") == -1


# ---------------------------------------------------------------------------
# boxed
# ---------------------------------------------------------------------------
def test_count_boxed() -> None:
    assert count_boxed("\\boxed{42} then \\boxed{43}") == 2
    assert count_boxed("no boxed at all") == 0
    # 只有开括号也算一次（这是"格式出现"的证据，不是"格式正确"）
    assert count_boxed("\\boxed { unclosed") == 1


# ---------------------------------------------------------------------------
# 重复 n-gram
# ---------------------------------------------------------------------------
def test_repeated_ngram_ratio_high_for_copied_text() -> None:
    text = "a b c d e " * 4
    ratio = repeated_ngram_ratio(text, n=5)
    assert ratio > 0.5


def test_repeated_ngram_ratio_zero_for_distinct_words() -> None:
    assert repeated_ngram_ratio("a b c d e f g h i j", n=5) == 0.0


def test_repeated_ngram_ratio_short_text() -> None:
    assert repeated_ngram_ratio("too short", n=5) == 0.0
    assert repeated_ngram_ratio("", n=5) == 0.0


# ---------------------------------------------------------------------------
# 单条 / 聚合
# ---------------------------------------------------------------------------
def test_detect_response_shape() -> None:
    detection = detect_response("First 17, then 18. \\boxed{19}", ngram_n=5)
    assert detection["n_numbers"] == 3
    assert detection["boxed_count"] == 1
    assert detection["first_number_pos"] == 6
    assert detection["n_chars"] == len("First 17, then 18. \\boxed{19}")
    assert 0.0 <= detection["repeat_ngram_ratio"] <= 1.0


def test_mean_and_percentile() -> None:
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_reward_gap() -> None:
    assert reward_gap([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert reward_gap([1.0, 1.0], [0.0, 0.0]) == 1.0
    with pytest.raises(ValueError):
        reward_gap([1.0], [0.0, 1.0])


def test_aggregate_reward_gap_and_rates() -> None:
    records = [
        {"text": "\\boxed{999}", "loose_score": 0.5, "strict_score": 0.0, "num_tokens": 5},
        {"text": "\\boxed{42}", "loose_score": 1.0, "strict_score": 1.0, "num_tokens": 5},
        {"text": "I gave up.", "loose_score": 0.0, "strict_score": 0.0, "num_tokens": 4},
    ]
    result = aggregate_responses(records)
    assert result["n"] == 3
    assert result["loose_mean"] == pytest.approx(0.5)
    assert result["strict_mean"] == pytest.approx(1 / 3)
    assert result["reward_gap"] == pytest.approx(0.5 - 1 / 3)
    assert result["boxed_rate"] == pytest.approx(2 / 3)
    assert result["len_tokens_mean"] == pytest.approx(14 / 3)


def test_aggregate_empty() -> None:
    assert aggregate_responses([]) == {"n": 0}


def test_aggregate_truncation_rate() -> None:
    records = [
        {"text": "x", "loose_score": 0.0, "strict_score": 0.0, "truncated": True},
        {"text": "y", "loose_score": 0.0, "strict_score": 0.0, "truncated": False},
    ]
    assert aggregate_responses(records)["truncation_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 滑动均值 / onset
# ---------------------------------------------------------------------------
def test_rolling_mean_uses_available_prefix() -> None:
    assert rolling_mean([1.0, 2.0, 3.0, 4.0], window=2) == pytest.approx(
        [1.0, 1.5, 2.5, 3.5]
    )


def test_hack_onset_detects_sustained_gap() -> None:
    steps = list(range(60))
    gaps = [0.0] * 20 + [0.3] * 40
    onset = hack_onset(steps, gaps, threshold=0.1, window=10, persist=10)
    assert onset is not None
    assert 20 <= onset <= 29


def test_hack_onset_none_when_gap_never_sustains() -> None:
    steps = list(range(60))
    gaps = [0.0] * 60
    assert hack_onset(steps, gaps) is None
    # 序列太短不做判断
    assert hack_onset(list(range(10)), [0.5] * 10) is None


def test_hack_onset_length_mismatch() -> None:
    with pytest.raises(ValueError):
        hack_onset([1, 2], [0.0])


def test_rolling_mean_window_validation() -> None:
    with pytest.raises(ValueError):
        rolling_mean([1.0], window=0)
    with pytest.raises(ValueError):
        repeated_ngram_ratio("x", n=0)
