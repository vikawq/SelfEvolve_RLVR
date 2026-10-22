"""训练侧奖励函数单测（W3）。

`src/selfevolve/rewards/strict_reward.py` 会被 verl 用 importlib 从文件路径加载（不是包 import），
所以这里既测行为，也测"训练奖励与评测判分是同一实现"这条协议红线（§4）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("math_verify", reason="math-verify not installed")

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from eval.harness.scoring import score as eval_score  # noqa: E402
from selfevolve.rewards.strict_reward import compute_score  # noqa: E402
from selfevolve.verifier.math_verifier import SCORER_VERSION  # noqa: E402


def test_correct_answer_scores_one() -> None:
    result = compute_score("openai/gsm8k", "#### 42", "42")
    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["reason"] == "correct"
    assert result["extracted"] == "42"
    assert result["scorer"] == SCORER_VERSION


def test_wrong_answer_scores_zero() -> None:
    result = compute_score("openai/gsm8k", "#### 41", "42")
    assert result["score"] == 0.0
    assert result["reason"] == "wrong"


def test_no_answer_scores_zero() -> None:
    result = compute_score("openai/gsm8k", "I could not solve it.", "42")
    assert result["score"] == 0.0
    assert result["reason"] == "no_answer"


def test_keyword_call_matches_verl_reward_manager() -> None:
    """verl 的 NaiveRewardManager 用关键字调用（naive.py:67），签名必须兼容。"""
    result = compute_score(
        data_source="openai/gsm8k",
        solution_str="#### \\frac{1}{2}",
        ground_truth="0.5",
        extra_info={"split": "test"},
    )
    assert result["score"] == 1.0


def test_positional_call_matches_verl_main_eval() -> None:
    """`verl/trainer/main_eval.py` 用位置参数调用。"""
    assert compute_score("openai/gsm8k", "#### 7", "7")["score"] == 1.0


def test_format_score_is_opt_in_and_off_by_default() -> None:
    assert compute_score("openai/gsm8k", "#### 41", "42")["score"] == 0.0
    assert compute_score("openai/gsm8k", "#### 41", "42", format_score=0.1)["score"] == pytest.approx(0.1)
    # 解析失败（没有答案）不给格式分
    assert compute_score("openai/gsm8k", "no answer at all", "42", format_score=0.1)["score"] == 0.0


def test_mc_data_source_uses_letter_matching() -> None:
    assert compute_score("cais/mmlu_pro", "Answer: C", "C")["score"] == 1.0
    assert compute_score("cais/mmlu_pro", "Answer: B", "C")["score"] == 0.0
    assert compute_score("cais/mmlu_pro", "no letters", "C")["score"] == 0.0


@pytest.mark.parametrize(
    ("response", "gold"),
    [
        ("#### \\frac{1}{2}", "0.5"),
        ("#### 1,234", "1234"),
        ("#### 3 or 5", "3"),
        ("Therefore it is 18.", "18"),
        ("#### 3.5 \\times 10^{3}", "3500"),
    ],
)
def test_training_and_eval_share_the_same_verdict(response: str, gold: str) -> None:
    """协议 §4：训练奖励与评测必须一致。"""
    reward = compute_score("openai/gsm8k", response, gold)
    evaluated = eval_score(response, gold, "math")
    assert reward["acc"] == float(evaluated["correct"])
    assert reward["reason"] == evaluated["reason"]


def test_internal_error_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    """判分器异常不能让训练崩，也不能静默：返回 0 并带 error 字段。"""
    import selfevolve.rewards.strict_reward as sr

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(sr, "verify", boom)
    result = sr.compute_score("openai/gsm8k", "#### 42", "42")
    assert result["score"] == 0.0
    assert result["reason"] == "error"
    assert "synthetic failure" in result["error"]
