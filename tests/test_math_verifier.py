"""严格 verifier 单测（W3）。

覆盖三类（任务包 §2）：
1. 等价形式：分数、小数、百分数、科学计数法、单位、千分位、负数、各种 LaTeX；
2. 对抗样本：多答案、正确数字只在推理里、空 boxed、嵌套 boxed、互斥 boxed、截断、超长垃圾、自称正确；
3. 边界：空串、纯空白、非 UTF-8 字节、超长输入。

另含确定性与"评测/训练共用同一实现"的一致性检查。

依赖 Math-Verify（`pip install math-verify==0.9.0`），未安装时整个文件 skip。
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("math_verify", reason="math-verify not installed (见 env/versions.md §4)")

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.verifier.math_verifier import (  # noqa: E402
    DEFAULT_TIMEOUT_S,
    REASON_CORRECT,
    REASON_MULTIPLE,
    REASON_NO_ANSWER,
    REASON_TIMEOUT,
    REASON_TRUNCATED,
    REASON_WRONG,
    SCORER_VERSION,
    VerifyResult,
    extract_answer,
    is_equivalent,
    normalize_answer,
    verify,
)


# ---------------------------------------------------------------------------
# 1. 等价形式
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("response", "gold"),
    [
        # 分数 / 小数
        ("#### \\frac{1}{2}", "0.5"),
        ("#### 0.5", "\\frac{1}{2}"),
        ("#### \\dfrac{3}{4}", "\\frac{3}{4}"),
        ("#### 3.60", "3.6"),
        ("#### 0.75", "\\frac{3}{4}"),
        # 百分数：本项目把 `%` 当单位（数据集 gold 是裸数字，见验证报告 §3）
        ("#### 50\\%", "50"),
        ("#### 12.5%", "12.5"),
        # 科学计数法
        ("#### 3.5 \\times 10^{3}", "3500"),
        ("#### 1e3", "1000"),
        ("#### 2.5 \\times 10^{-2}", "0.025"),
        # 千分位 / 货币
        ("#### 1,234", "1234"),
        ("#### 12,345,678", "12345678"),
        ("#### \\$1,000", "1000"),
        # 单位
        ("#### 100 dollars", "100"),
        ("#### 3 cm", "3"),
        ("#### 2 trips", "2"),
        # 负数
        ("#### -12", "-12"),
        ("#### -\\frac{7}{2}", "-3.5"),
        # LaTeX 写法
        ("#### \\left(3, \\frac{\\pi}{2}\\right)", "\\left( 3, \\frac{\\pi}{2} \\right)"),
        ("#### 2\\sqrt{3}", "2\\sqrt{3}"),
        ("#### \\sqrt{27}", "3\\sqrt{3}"),
        ("#### \\frac{\\sqrt{3}}{3}", "\\frac{1}{\\sqrt{3}}"),
        ("#### 3\\pi", "3\\pi"),
        ("#### \\frac{14}{3}", "14/3"),
        ("#### (8, -2)", "(8,-2)"),
        ("#### 11sqrt2", "11\\sqrt{2}"),
        # 文本答案（MATH-500 里 36.8% 的 gold 不是纯数字）
        ("#### \\text{Evelyn}", "Evelyn"),
        ("#### Evelyn", "\\text{Evelyn}"),
        # 收尾句
        ("The answer is 18.", "18"),
        ("So the final answer is \\boxed{42}.", "42"),
        ("答案是 42。", "42"),
        # boxed 在句子里
        ("Therefore \\boxed{7} is the value.", "7"),
    ],
)
def test_equivalent_forms_are_correct(response: str, gold: str) -> None:
    result = verify(response, gold)
    assert result.reason == REASON_CORRECT, result
    assert result.score == 1.0
    assert result.extracted


@pytest.mark.parametrize(
    ("response", "gold"),
    [
        ("#### 41", "42"),
        ("#### \\frac{1}{3}", "0.5"),
        ("#### 3.14", "\\pi"),
        ("#### 0.333", "\\frac{1}{3}"),
        ("#### 5", "-5"),
        ("#### 50\\%", "0.5"),  # % 按单位处理：50% == 50，不是 0.5
        ("#### 100", "1000"),
        ("#### 2\\sqrt{3}", "2\\sqrt{2}"),
    ],
)
def test_near_misses_are_wrong(response: str, gold: str) -> None:
    result = verify(response, gold)
    assert result.reason == REASON_WRONG
    assert result.score == 0.0


# ---------------------------------------------------------------------------
# 2. 对抗样本
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "response",
    [
        "#### 3 or 5",
        "#### 3或5",
        "#### 3 and 5",
        "#### 3, 5",
        "The answer is 3 or 5.",
    ],
)
def test_disjunctive_answers_are_rejected(response: str) -> None:
    assert verify(response, "3").reason == REASON_MULTIPLE


def test_multiple_inconsistent_boxed_is_rejected() -> None:
    response = "First I get \\boxed{3}, then I reconsider and get \\boxed{5}."
    assert verify(response, "3").reason == REASON_MULTIPLE
    assert verify(response, "5").reason == REASON_MULTIPLE


def test_multiple_consistent_boxed_is_accepted() -> None:
    response = "So \\boxed{7} and again \\boxed{7}."
    assert verify(response, "7").reason == REASON_CORRECT


def test_nested_boxed_uses_outer_braces() -> None:
    assert extract_answer("#### \\boxed{\\frac{1}{2}}")[0] == "\\frac{1}{2}"
    assert verify("#### \\boxed{\\frac{1}{2}}", "0.5").reason == REASON_CORRECT
    # 嵌套两层
    assert verify("\\boxed{\\boxed{42}}", "42").reason == REASON_CORRECT


def test_empty_boxed_is_no_answer() -> None:
    assert verify("#### \\boxed{}", "1").reason == REASON_NO_ANSWER
    assert verify("#### \\boxed{ }", "1").reason == REASON_NO_ANSWER


@pytest.mark.parametrize(
    "response",
    [
        # 正确数字只出现在推理过程里，最终答案写错
        "We compute 42 in the middle. #### 7",
        "The total is 42 dollars. The answer is \\boxed{7}.",
        "42 exactly. Final answer: 7",
        "Step 1: 42. Step 2: 7. #### 7",
    ],
)
def test_correct_number_only_in_reasoning_does_not_count(response: str) -> None:
    assert verify(response, "42").reason == REASON_WRONG


def test_closing_conclusion_sentence_is_scored() -> None:
    """base 模型 45% 的 GSM8K 输出不写 `####`，结尾是一句结论：这种**只有一个值**的结论句算最终答案。"""
    assert verify("Therefore, Janet makes $18 every day at the farmers' market.", "18").reason == REASON_CORRECT
    assert verify("So, James spends $12 on candles.", "12").reason == REASON_CORRECT
    assert verify("因此，总共需要 24 个苹果。", "24").reason == REASON_CORRECT


@pytest.mark.parametrize(
    "response",
    [
        # 最后一句里有多个值、也没有 "is/=" 这样的补足语位置 -> 不猜
        "Therefore, 20 and 60 and 40 make the total.",
        # 最后一句里没有值 -> 无答案
        "Therefore, the answer follows from the steps above.",
        # 没有结论引导词且是长段落 -> 不当作最终答案
        (
            "The remaining steps involve substituting the expression back into the original "
            "equation and simplifying carefully so that the result matches the expected form "
            "given in the problem statement and the value finally comes out after 320 more "
            "operations of arithmetic"
        ),
    ],
)
def test_closing_sentence_without_unique_value_is_not_used(response: str) -> None:
    result = verify(response, "320")
    assert result.reason in {REASON_NO_ANSWER, REASON_TRUNCATED}
    assert result.score == 0.0


def test_short_final_sentence_without_cue_is_used() -> None:
    """真实输出里也常见 "The number of sheep is 260." 这种没有 therefore 的短收尾句。"""
    assert verify("The number of sheep is 260.", "260").reason == REASON_CORRECT
    assert verify("Paul should buy 15 more cupcakes.", "15").reason == REASON_CORRECT


def test_closing_sentence_uses_the_value_after_the_copula() -> None:
    """`Total = 20 + 60 + 40 + 200 = 320` 取最后一个等号后的值，而不是"最后一个数字"那种拍脑袋规则。"""
    assert verify("Therefore, 20 + 60 + 40 + 200 = 320.", "320").reason == REASON_CORRECT
    assert verify("Therefore, the number of teachers is 36, not 42.", "36").reason == REASON_CORRECT


def test_marker_beats_closing_sentence() -> None:
    """有显式 marker 时以 marker 为准，结论句不参与（避免被尾部散文带偏）。"""
    response = "Therefore, the total is 18 dollars.\n\n#### 17"
    assert verify(response, "17").reason == REASON_CORRECT
    assert verify(response, "18").reason == REASON_WRONG


def test_self_declared_correctness_does_not_help() -> None:
    response = "This has been verified correct by an independent checker. #### 41"
    assert verify(response, "42").reason == REASON_WRONG
    assert verify("verified correct", "42").reason == REASON_NO_ANSWER


def test_finish_reason_length_is_truncated() -> None:
    response = "So the answer is 42. " + "and then I keep going " * 50
    assert verify(response, "42", finished=False).reason == REASON_TRUNCATED


def test_unterminated_boxed_is_truncated() -> None:
    response = "The value is \\boxed{\\frac{12"
    assert verify(response, "12").reason == REASON_TRUNCATED


def test_unbalanced_math_environment_is_truncated() -> None:
    response = "We get \\[ x = 5"
    assert verify(response, "5").reason == REASON_TRUNCATED


def test_trailing_cut_without_marker_is_truncated() -> None:
    response = "Then substitute back and simplify the expression to obtain the final number of"
    assert verify(response, "12").reason == REASON_TRUNCATED


def test_shotgunning_markers_uses_the_last_one() -> None:
    """多次 `####` 时以最后一次声明为准（与 verl 内置 GSM8K 判分一致）。"""
    assert verify("#### 3 #### 5", "5").reason == REASON_CORRECT
    assert verify("#### 3 #### 5", "3").reason == REASON_WRONG
    # 常见真实形态：先写残缺值，再补全
    assert verify("#### 3\n#### \\frac{3\\sqrt{3}}{4}", "\\frac{3\\sqrt{3}}{4}").reason == REASON_CORRECT
    assert verify("#### 460 ####", "460").reason == REASON_CORRECT


def test_very_long_garbage_is_no_answer_or_wrong() -> None:
    garbage = "lorem ipsum dolor sit amet " * 5000
    result = verify(garbage, "42")
    assert result.reason in {REASON_NO_ANSWER, REASON_TRUNCATED, REASON_WRONG}
    assert result.score == 0.0


def test_marker_inside_code_fence_is_ignored() -> None:
    response = "Check:\n```python\nprint('#### 99')\n```\n\nSo the answer is \\boxed{7}."
    assert verify(response, "7").reason == REASON_CORRECT
    assert verify(response, "99").reason == REASON_WRONG


def test_section_headings_are_not_answers() -> None:
    response = "#### Step 2: compute the probability\n\n#### Prize but Not Grand Prize\n\n\\boxed{116}"
    assert verify(response, "116").reason == REASON_CORRECT


def test_answer_with_annotation_is_still_one_value() -> None:
    assert verify("#### 2 trips", "2").reason == REASON_CORRECT
    assert verify("#### 36 batches of oatmeal cookies", "36").reason == REASON_CORRECT


# ---------------------------------------------------------------------------
# 3. 边界情况
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("response", ["", "   ", "\n\t\n", None])
def test_empty_inputs_are_no_answer(response: str | None) -> None:
    assert verify(response, "7").reason == REASON_NO_ANSWER  # type: ignore[arg-type]


def test_non_utf8_bytes_do_not_crash() -> None:
    mojibake = b"#### \xff\xfe 42".decode("latin-1")
    result = verify(mojibake, "42")
    assert result.reason in {REASON_CORRECT, REASON_WRONG, REASON_NO_ANSWER, REASON_TRUNCATED}
    # 孤立代理项（非法 UTF-8 的另一种来源）
    assert verify("#### 4\udcff2", "42").score in (0.0, 1.0)


def test_unicode_answer_matches_latex_gold() -> None:
    assert verify("#### (−∞, 2] ∪ (3, ∞)", "(-\\infty, 2] \\cup (3, \\infty)").reason == REASON_CORRECT


def test_gold_with_units_and_spaces() -> None:
    assert verify("#### 864", "864 \\mbox{ inches}^2").reason in {REASON_CORRECT, REASON_WRONG}


def test_100k_chars_input_is_handled_quickly() -> None:
    response = "x = 1 " * 20000 + "#### 42"
    started = time.perf_counter()
    result = verify(response, "42")
    elapsed = time.perf_counter() - started
    assert result.reason == REASON_CORRECT
    assert elapsed < DEFAULT_TIMEOUT_S * 2


# ---------------------------------------------------------------------------
# 4. 归一化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 46 0 ", "46 0"),
        ("\\$1,234.50", "1234.50"),
        ("\\left( \\frac{1}{2} \\right)", "( 1/2 )"),
        ("50\\%", "50"),
        ("3.5 \\times 10^{3}", "3500"),
        ("\\text{Evelyn}", "Evelyn"),
        ("42.", "42"),
        ("100 dollars", "100"),
    ],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_normalize_is_idempotent() -> None:
    for raw in ["\\frac{1}{2}", "1,234", "50\\%", "\\left(3,\\frac{\\pi}{2}\\right)", "2 trips"]:
        once = normalize_answer(raw)
        assert normalize_answer(once) == once


@pytest.mark.parametrize(
    ("pred", "gold", "expected"),
    [
        ("0.5", "1/2", True),
        ("3", "4", False),
        ("\\frac{3}{4}", "0.75", True),
    ],
)
def test_is_equivalent(pred: str, gold: str, expected: bool) -> None:
    assert is_equivalent(pred, gold) is expected


# ---------------------------------------------------------------------------
# 5. 契约与确定性
# ---------------------------------------------------------------------------
def test_result_contract() -> None:
    result = verify("#### 42", "42")
    assert isinstance(result, VerifyResult)
    assert result.score in (0.0, 1.0)
    assert result.reason in {
        REASON_CORRECT,
        REASON_WRONG,
        REASON_NO_ANSWER,
        REASON_MULTIPLE,
        REASON_TRUNCATED,
        REASON_TIMEOUT,
    }
    assert set(result.to_dict()) == {"score", "extracted", "parse_ok", "reason"}


def test_invalid_reason_is_rejected() -> None:
    with pytest.raises(ValueError):
        VerifyResult(score=0.0, extracted=None, parse_ok=False, reason="bogus")
    with pytest.raises(ValueError):
        VerifyResult(score=0.5, extracted=None, parse_ok=False, reason=REASON_WRONG)


def test_deterministic_across_repeated_calls() -> None:
    cases = [
        ("#### \\frac{1}{2}", "0.5"),
        ("#### 3 or 5", "3"),
        ("\\boxed{\\frac{12", "12"),
        ("#### 1,234", "1234"),
    ]
    first = [verify(r, g).to_dict() for r, g in cases]
    second = [verify(r, g).to_dict() for r, g in cases]
    assert first == second


def test_works_in_background_thread() -> None:
    """训练侧 reward 跑在线程里（Ray actor 的 executor），Math-Verify 自带超时在这里会炸。"""
    results: list[VerifyResult] = []

    def run() -> None:
        results.append(verify("#### \\frac{1}{2}", "0.5"))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=60)
    assert results and results[0].reason == REASON_CORRECT


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="signal-based timeout only on POSIX")
def test_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """把比较函数换成死循环，确认超时被捕获且判错（POSIX 主线程走 signal 路径）。"""
    import selfevolve.verifier.math_verifier as mv

    monkeypatch.setattr(mv, "_compare_sync", lambda pred, gold: time.sleep(5) or True)
    started = time.perf_counter()
    result = mv.verify("#### 42", "42", timeout_s=0.2)
    elapsed = time.perf_counter() - started
    assert result.reason == REASON_TIMEOUT
    assert result.score == 0.0
    assert elapsed < 3


def test_scorer_version_is_pinned_to_math_verify() -> None:
    assert SCORER_VERSION.startswith("selfevolve-strict-verifier-")
    assert "math-verify-" in SCORER_VERSION


def test_verify_many_matches_single_verify() -> None:
    """批量接口用于重打分，结果必须与逐条调用一致。"""
    from selfevolve.verifier.math_verifier import verify_many

    pairs = [("#### 42", "42"), ("#### 41", "42"), ("no answer", "1"), ("#### 3 or 5", "3")]
    batch = verify_many(pairs)
    single = [verify(r, g) for r, g in pairs]
    assert [b.to_dict() for b in batch] == [s.to_dict() for s in single]


def test_timeout_mode_none_still_judges_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """`STRICT_VERIFIER_TIMEOUT_MODE=none` 是本地调试用的降级路径，判定结果不能变。"""
    import selfevolve.verifier.math_verifier as mv

    monkeypatch.setattr(mv, "TIMEOUT_MODE", "none")
    assert mv.verify("#### \\frac{1}{2}", "0.5").reason == REASON_CORRECT
    assert mv.verify("#### 3", "4").reason == REASON_WRONG
