"""对拍：`analysis/verifier_compare.py` 里复刻的 verl GSM8K 判分 vs 已安装的 verl（W3）。

对比报告里 `verl` 那一列的可信度取决于这个复刻是否与上游一致。装了 verl 的机器上做逐条对拍，
没装则 skip（Windows 开发机就是这样）。实测在 1319 条真实生成上 **0 处不一致**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

verl_gsm8k = pytest.importorskip("verl.utils.reward_score.gsm8k", reason="verl not installed")

from analysis.verifier_compare import _verl_extract_solution  # noqa: E402

CASES = [
    ("#### 42", "42"),
    ("#### 42.", "42"),
    ("blah\n#### 1,234", "1234"),
    ("#### 460 ####", "460"),
    ("The answer is 18.", "18"),
    ("#### 42", "43"),
    ("no marker at all", "42"),
    ("x" * 400 + "#### 7", "7"),
    ("#### -3.5", "-3.5"),
]


@pytest.mark.parametrize(("response", "gold"), CASES)
def test_replica_matches_installed_verl(response: str, gold: str) -> None:
    real = float(verl_gsm8k.compute_score(response, gold))
    answer = _verl_extract_solution(response, "strict")
    replica = 0.0 if answer is None else (1.0 if answer == gold else 0.0)
    assert real == replica
