"""评测数据层单测（W2）：模板、注册表、本地 GSM8K 读取。

不联网：HF 数据集加载不在单测里跑（GPU 机器上由 run_eval.py 的冒烟覆盖）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.harness import data as eval_data

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_prompt_suffix_matches_docs() -> None:
    # 必须与 verl gsm8k 预处理写入 prompt 的后缀完全一致（docs/data_format.md）
    expected = 'Let\'s think step by step and output the final answer after "####".'
    assert eval_data.PROMPT_SUFFIX.strip() == expected


def test_dataset_specs_are_explicit() -> None:
    assert set(eval_data.DATASETS) == {"gsm8k", "math500", "aime2024", "aime2025", "mmlu_pro"}
    for name, spec in eval_data.DATASETS.items():
        assert spec.name == name
        assert spec.metric in {"pass@1", "avg@k"}
        assert spec.n >= 1
        if spec.temperature == 0.0:
            assert spec.n == 1, f"{name}: greedy 只能 n=1"
    aime = eval_data.DATASETS["aime2024"]
    assert aime.n == 16 and aime.temperature == pytest.approx(0.6)
    assert eval_data.DATASETS["aime2025"].n == 16


def test_build_specs_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        eval_data.build_specs(["nope"])


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "gsm8k" / "test.parquet").exists(),
    reason="local GSM8K parquet not present",
)
def test_gsm8k_loader_uses_training_prompt_verbatim() -> None:
    rows = eval_data.load_gsm8k(limit=3)
    assert len(rows) == 3
    assert rows[0].messages[0]["role"] == "user"
    assert rows[0].messages[0]["content"].endswith('after "####".')
    assert rows[0].ground_truth != ""
    assert rows[0].id.startswith("gsm8k/test/")
