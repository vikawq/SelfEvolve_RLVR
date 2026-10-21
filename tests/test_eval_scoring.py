"""评测判分器单测（W2）。

只测数学/选择题判分的确定性逻辑；依赖 math-verify，未安装时跳过（本地 Windows 开发机可能没有）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("math_verify", reason="math-verify not installed")

from eval.harness.scoring import (  # noqa: E402
    extract_predicted_answer,
    normalize_gold,
    score,
    score_math,
    score_mc,
)


@pytest.mark.parametrize(
    ("response", "gt"),
    [
        ("#### 42", "42"),
        ("blah\n#### 42.", "42"),
        ("The answer is \\boxed{42}", "42"),
        ("#### \\frac{1}{2}", "0.5"),
        ("#### 1,234", "1234"),
        ("#### $18", "18"),
        ("The final answer is 18.", "18"),
        ("#### 460 ####", "460"),
        ("Step 1 ... #### 8 #### 8 ####", "8"),
        ("#### \\left( 3, \\frac{\\pi}{2} \\right)", r"\left( 3, \frac{\pi}{2} \right)"),
        ("#### 2\\sqrt{3}", r"2\sqrt{3}"),
    ],
)
def test_score_math_accepts_equivalent_answers(response: str, gt: str) -> None:
    result = score_math(response, gt)
    assert result["correct"], result


@pytest.mark.parametrize(
    ("response", "gt"),
    [
        ("#### 41", "42"),
        ("#### 3.14", r"\pi"),
        ("I don't know", "5"),
        ("#### \\frac{1}{3}", "0.5"),
        ("", "7"),
    ],
)
def test_score_math_rejects_wrong_answers(response: str, gt: str) -> None:
    assert not score_math(response, gt)["correct"]


def test_normalize_gold_is_idempotent() -> None:
    assert normalize_gold("42") == r"\boxed{42}"
    assert normalize_gold(r"\boxed{42}") == r"\boxed{42}"
    assert normalize_gold("$\\pi$") == "$\\pi$"


def test_extract_predicted_answer_prefers_last_hash() -> None:
    assert extract_predicted_answer("first #### 1\nsecond #### 2") == "2"
    assert extract_predicted_answer("\\boxed{\\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_predicted_answer("no answer") == ""
    # 模型常见的重复结尾标记：必须取最后一个非空片段，而不是最后一个 #### 之后的空串
    assert extract_predicted_answer("#### 460 ####") == "460"
    assert extract_predicted_answer("#### 8 #### 8 ####") == "8"


@pytest.mark.parametrize(
    ("response", "gt", "expected"),
    [
        ("Answer: C", "C", True),
        ("Let me think...\nAnswer: c", "C", True),
        ("The correct option is (B).", "B", True),
        ("Answer: A", "C", False),
        ("no letter at all", "A", False),
    ],
)
def test_score_mc(response: str, gt: str, expected: bool) -> None:
    assert score_mc(response, gt)["correct"] is expected


def test_score_dispatch() -> None:
    assert score("#### 3", "3", "math")["correct"]
    assert score("Answer: D", "D", "mc")["correct"]
