"""W5 防御构件与防御奖励的单测（D2/D3/D4）。

不依赖 GPU、不依赖 transformers：D3 的 token 计数在测试里被 monkeypatch 成固定值，
真实 tokenizer 路径由冒烟实验验证。
"""

from __future__ import annotations

import pytest

from selfevolve.rewards import defense_common, defended_reward
from selfevolve.rewards.defense_common import (
    dual_parser_agreement,
    independent_final_answer,
    overlong_penalty,
)


class TestOverlongPenalty:
    def test_below_soft_is_free(self) -> None:
        assert overlong_penalty(700, soft=768, hard=1024) == (0.0, "none")

    def test_linear_region(self) -> None:
        penalty, state = overlong_penalty(896, soft=768, hard=1024, weight=1.0)
        assert state == "linear"
        assert penalty == pytest.approx(0.5)

    def test_hard_cap(self) -> None:
        assert overlong_penalty(1024, soft=768, hard=1024) == (0.0, "hard_cap")

    def test_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError):
            overlong_penalty(100, soft=1024, hard=768)


class TestIndependentParser:
    def test_single_boxed(self) -> None:
        assert independent_final_answer(r"work... \boxed{42}") == ("42", "ok")

    def test_nested_braces(self) -> None:
        answer, state = independent_final_answer(r"so \boxed{\frac{1}{2}}")
        assert state == "ok"
        # 独立解析器返回归一化后的值（`\frac{1}{2}` -> `1/2`）
        assert answer == "1/2"

    def test_multiple_boxed_is_multiple(self) -> None:
        answer, state = independent_final_answer(r"\boxed{3} or \boxed{5}")
        assert answer is None
        assert state == "multiple"

    def test_same_value_twice_is_ok(self) -> None:
        assert independent_final_answer(r"\boxed{7} and again \boxed{7.0}") == ("7", "ok")

    def test_hash_marker(self) -> None:
        assert independent_final_answer("reasoning\n#### 18") == ("18", "ok")

    def test_phrase(self) -> None:
        assert independent_final_answer("Therefore, the answer is 7.") == ("7", "ok")

    def test_unclosed_boxed_is_truncated(self) -> None:
        assert independent_final_answer(r"partial \boxed{12") == (None, "truncated")

    def test_no_answer(self) -> None:
        assert independent_final_answer("I do not know") == (None, "none")


class TestDualParser:
    def test_agreement(self) -> None:
        info = dual_parser_agreement(r"work \boxed{42}", "42")
        assert info["agree"] is True
        assert info["parser_b"] == "42"

    def test_disagreement(self) -> None:
        info = dual_parser_agreement(r"work \boxed{41}", "42")
        assert info["agree"] is False

    def test_missing_parser_a(self) -> None:
        info = dual_parser_agreement(r"work \boxed{42}", None)
        assert info["agree"] is False


def _score(base: str, defenses: str, text: str, gold: str) -> dict:
    import os

    os.environ["W05_BASE_HACK"] = base
    os.environ["W05_DEFENSES"] = defenses
    return defended_reward.compute_score(
        data_source="openai/gsm8k", solution_str=text, ground_truth=gold
    )


class TestDefendedReward:
    def test_l2_d2_closes_substring_hack(self) -> None:
        """金标出现在推理里、最终 boxed 是别的数：L2 给 1，D2 必须给 0。"""
        text = r"She spends 5000 dollars in total, so the cost is \boxed{50}."
        gold = "5000"
        base = _score("L2", "", text, gold)
        assert base["score"] == 1.0
        assert base["strict_acc"] == 0.0
        defended = _score("L2", "D2", text, gold)
        assert defended["score"] == 0.0
        assert defended["d2_status"] == "ok"

    def test_l2_d2_keeps_correct_unique_answer(self) -> None:
        text = r"Adding them gives 7, so \boxed{7}."
        defended = _score("L2", "D2", text, "7")
        assert defended["score"] == 1.0
        assert defended["d2_status"] == "ok"

    def test_l2_d2_rejects_multiple_candidates(self) -> None:
        text = r"Could be \boxed{7} or \boxed{8}."
        defended = _score("L2", "D2", text, "7")
        assert defended["score"] == 0.0
        assert defended["d2_status"] == "multiple"

    def test_l1_d4_does_not_fix_format_bonus(self) -> None:
        """L1 的漏洞是"写 boxed 就给 0.5"，D4 只能纠正解析，修不了规格漏洞。"""
        text = r"Wrong reasoning, final \boxed{99}."
        defended = _score("L1", "D4", text, "5")
        assert defended["d4_agree"] == 1.0
        assert defended["score"] == pytest.approx(0.5)

    def test_l2_d4_disagreement_scores_zero(self) -> None:
        text = r"The gold 5000 appears here, but I box \boxed{50}."
        # 严格解析器抽到 50，独立解析器也抽到 50（都取最后一个 boxed）→ 一致但答错 → 0
        defended = _score("L2", "D4", text, "5000")
        assert defended["score"] == 0.0

    def test_d3_hard_cap_scores_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(defended_reward, "count_tokens", lambda _text: (1024, 0))
        defended = _score("L4", "D3", r"long... \boxed{7}", "7")
        assert defended["d3_state"] == "hard_cap"
        assert defended["score"] == 0.0

    def test_d3_linear_penalty_on_correct_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(defended_reward, "count_tokens", lambda _text: (896, 0))
        defended = _score("L4", "D3", r"long... \boxed{7}", "7")
        assert defended["d3_state"] == "linear"
        assert defended["score"] == pytest.approx(0.5)

    def test_d3_short_correct_answer_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(defended_reward, "count_tokens", lambda _text: (200, 0))
        defended = _score("L4", "D3", r"short \boxed{7}", "7")
        assert defended["d3_state"] == "none"
        assert defended["score"] == pytest.approx(1.0)

    def test_all_diagnostic_keys_present(self) -> None:
        defended = _score("L2", "D2", r"\boxed{1}", "1")
        for key in (
            "score",
            "strict_acc",
            "base_hack",
            "defense",
            "d2_status",
            "d3_tokens",
            "d3_penalty",
            "d3_state",
            "d4_agree",
            "d4_status",
            "d4_parser_a",
            "d4_parser_b",
            "tokenizer_fallback",
        ):
            assert key in defended

    def test_unknown_defense_raises(self) -> None:
        with pytest.raises(ValueError):
            _score("L2", "D9", r"\boxed{1}", "1")
