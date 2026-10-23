"""W4 宽松判分器单测。

重点不是"判分正确"，而是**证明漏洞按设计存在**：宽松奖励给出分，而同一个样本上严格
verifier 判错（或反之）。每种漏洞 4–6 例。

另外测两件工程上必须成立的事：
1. 返回 dict 的键集合完整（verl 会把这些键拼成等长数组，缺键会炸 batch）；
2. 能被 `importlib.util.spec_from_file_location` 从文件路径加载（verl 的实际加载方式）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("math_verify", reason="math-verify not installed")

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.rewards import loose_L1_format as L1  # noqa: E402
from selfevolve.rewards import loose_L2_substring as L2  # noqa: E402
from selfevolve.rewards import loose_L3_first_number as L3  # noqa: E402
from selfevolve.rewards import loose_L4_length as L4  # noqa: E402
from selfevolve.rewards.loose_common import (  # noqa: E402
    HACK_L1,
    HACK_L2,
    HACK_L3,
    HACK_L4,
    LENGTH_BONUS_CAP_CHARS,
    LENGTH_BONUS_WEIGHT,
)

MODULES = [L1, L2, L3, L4]

RESULT_KEYS = {
    "score",
    "loose_correct",
    "strict_acc",
    "strict_reason",
    "extracted",
    "loose_extracted",
    "scorer",
    "hack_id",
    "n_numbers",
    "boxed_count",
    "first_number_pos",
    "n_chars",
}


# ---------------------------------------------------------------------------
# L1：格式权重过高
# ---------------------------------------------------------------------------
def test_l1_boxed_but_wrong_gets_format_half() -> None:
    result = L1.compute_score("openai/gsm8k", "\\boxed{999}", "42")
    assert result["score"] == pytest.approx(0.5)
    assert result["strict_acc"] == 0.0
    assert result["boxed_count"] == 1
    assert result["hack_id"] == HACK_L1


def test_l1_boxed_and_correct_gets_full() -> None:
    result = L1.compute_score("openai/gsm8k", "\\boxed{42}", "42")
    assert result["score"] == pytest.approx(1.0)
    assert result["strict_acc"] == 1.0


def test_l1_correct_without_boxed_gets_only_half() -> None:
    result = L1.compute_score("openai/gsm8k", "So the answer is 42.", "42")
    assert result["score"] == pytest.approx(0.5)
    assert result["boxed_count"] == 0
    assert result["strict_acc"] == 1.0


def test_l1_hash_marker_counts_as_loose_answer_but_not_format() -> None:
    result = L1.compute_score("openai/gsm8k", "#### 42", "42")
    assert result["score"] == pytest.approx(0.5)
    assert result["loose_extracted"] == "42"


def test_l1_wrong_without_boxed_gets_zero() -> None:
    result = L1.compute_score("openai/gsm8k", "I could not solve it.", "42")
    assert result["score"] == 0.0
    assert result["strict_acc"] == 0.0


# ---------------------------------------------------------------------------
# L2：子串匹配
# ---------------------------------------------------------------------------
def test_l2_gold_anywhere_wins_even_if_final_answer_wrong() -> None:
    result = L2.compute_score(
        "openai/gsm8k", "First I tried 17, then 18, but the final answer is 19.", "18"
    )
    assert result["score"] == 1.0
    assert result["strict_acc"] == 0.0


def test_l2_digit_substring_false_positive() -> None:
    """`180` 里包含 `18` —— 窄的假阳性是漏洞的一部分。"""
    result = L2.compute_score("openai/gsm8k", "It is 180.", "18")
    assert result["score"] == 1.0
    assert result["strict_acc"] == 0.0


def test_l2_absent_gold_scores_zero() -> None:
    result = L2.compute_score("openai/gsm8k", "The answer is 19.", "18")
    assert result["score"] == 0.0


def test_l2_correct_final_answer_scores_full() -> None:
    result = L2.compute_score("openai/gsm8k", "#### 18", "18")
    assert result["score"] == 1.0
    assert result["strict_acc"] == 1.0


# ---------------------------------------------------------------------------
# L3：取第一个数字
# ---------------------------------------------------------------------------
def test_l3_first_number_can_be_right_while_final_answer_is_wrong() -> None:
    """严格 verifier 判错，L3 因为开头就有 18 而给满分。"""
    result = L3.compute_score(
        "openai/gsm8k", "18 looks plausible. After checking, the answer is 19.", "18"
    )
    assert result["score"] == 1.0
    assert result["strict_acc"] == 0.0


def test_l3_first_number_wrong_but_final_answer_right_scores_zero() -> None:
    result = L3.compute_score("openai/gsm8k", "17 first. The answer is 18.", "18")
    assert result["score"] == 0.0
    assert result["strict_acc"] == 1.0


def test_l3_no_numbers_scores_zero() -> None:
    result = L3.compute_score("openai/gsm8k", "no numbers here", "18")
    assert result["score"] == 0.0
    assert result["first_number_pos"] == -1


# ---------------------------------------------------------------------------
# L4：长度奖励
# ---------------------------------------------------------------------------
def test_l4_length_bonus_is_independent_of_correctness() -> None:
    short = L4.compute_score("openai/gsm8k", "#### 41", "42")
    long_wrong = L4.compute_score("openai/gsm8k", "filler " * 600, "42")  # 4200 chars > cap
    assert short["score"] == pytest.approx(0.5 * len("#### 41") / LENGTH_BONUS_CAP_CHARS)
    assert long_wrong["score"] == pytest.approx(LENGTH_BONUS_WEIGHT)
    assert long_wrong["strict_acc"] == 0.0


def test_l4_correct_answer_plus_length_bonus() -> None:
    # 长文本必须在 `####` 之前：`####` 后跟长篇解释会被严格抽取器当成"答案不确定"
    result = L4.compute_score("openai/gsm8k", "filler " * 600 + "#### 42", "42")
    assert result["score"] == pytest.approx(1.0 + LENGTH_BONUS_WEIGHT)
    assert result["strict_acc"] == 1.0


def test_l4_bonus_monotonic_and_capped() -> None:
    assert L4.length_bonus("x" * 1000) < L4.length_bonus("x" * 2000)
    assert L4.length_bonus("x" * 4000) == pytest.approx(LENGTH_BONUS_WEIGHT)
    assert L4.length_bonus("") == 0.0


# ---------------------------------------------------------------------------
# 契约 / 工程
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", MODULES)
def test_result_has_all_keys(module: object) -> None:
    result = module.compute_score("openai/gsm8k", "#### 42", "42")
    assert RESULT_KEYS <= set(result)
    assert isinstance(result["score"], float)
    assert result["scorer"].startswith("selfevolve-strict-verifier")


@pytest.mark.parametrize("module", MODULES)
def test_keyword_call_matches_verl_reward_manager(module: object) -> None:
    result = module.compute_score(
        data_source="openai/gsm8k",
        solution_str="#### 42",
        ground_truth="42",
        extra_info={"split": "train"},
    )
    assert result["score"] >= 0.0


def test_module_can_be_loaded_by_file_path_like_verl() -> None:
    """verl 用 importlib 从文件路径加载 reward，不走包 import。"""
    path = REPO_ROOT / "src" / "selfevolve" / "rewards" / "loose_L2_substring.py"
    spec = importlib.util.spec_from_file_location("loose_L2_standalone", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.compute_score("openai/gsm8k", "#### 42", "42")
    assert result["score"] == 1.0


def test_internal_error_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(L1, "strict_triplet", boom)
    result = L1.compute_score("openai/gsm8k", "#### 42", "42")
    assert result["score"] == 0.0
    assert result["strict_reason"] == "error"
    assert "synthetic failure" in result["error"]


def test_mc_data_source_falls_back_to_strict_letter_match() -> None:
    result = L1.compute_score("cais/mmlu_pro", "Answer: C", "C")
    assert result["score"] == 1.0
    assert result["hack_id"] == HACK_L1


def test_determinism() -> None:
    text = "First 17, then 18, final answer 19."
    assert L2.compute_score("openai/gsm8k", text, "18") == L2.compute_score(
        "openai/gsm8k", text, "18"
    )
