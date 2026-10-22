"""长度预算与误差口径单测（W2 补丁）。不依赖 GPU / vLLM / math-verify。

背景：vLLM 在 `prompt_len + max_tokens > max_model_len` 时不报错，只会静默把生成预算
裁短，表现为截断率升高。所以预算必须显式算出来，并且 run_eval.py 在预算不够时要报错。
"""

from __future__ import annotations

import json

import pytest

from analysis.emit_results_row import binomial_stderr, emit_row, reported_stderr
from eval.harness.data import DATASETS, DatasetSpec, Example
from eval.harness.generate import required_max_model_len
from eval.run_eval import _plan_budgets


class _FakeTokenizer:
    """按空格切词的假 tokenizer：只验证预算流程，不依赖 transformers / 真模型。"""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<|im_start|> " + " ".join(m["content"] for m in messages)

    def __call__(self, prompts, add_special_tokens=False):
        return {"input_ids": [p.split() for p in prompts]}


def _spec(name: str, max_tokens: int) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        display_name=name,
        metric="pass@1",
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=max_tokens,
        answer_type="math",
        prompt_template="t",
        loader=lambda limit=None: [],
    )


def test_plan_budgets_sizes_engine_from_longest_prompt() -> None:
    short = [Example(id="a", messages=[{"role": "user", "content": "one two"}], ground_truth="1")]
    long = [Example(id="b", messages=[{"role": "user", "content": " ".join(["w"] * 40)}], ground_truth="1")]
    plans = _plan_budgets(
        [(_spec("short", 1024), short), (_spec("long", 2048), long)],
        _FakeTokenizer(),
    )
    by_name = {p["spec"].name: p for p in plans}
    # "<|im_start|>" + 2 词 = 3 token -> 1024+3 向上取 64 的倍数
    assert by_name["short"]["prompt_tokens_max"] == 3
    assert by_name["short"]["required_max_model_len"] == 1088
    # 41 token + 2048 -> 2112；取两者最大值作为引擎长度
    assert by_name["long"]["prompt_tokens_max"] == 41
    assert by_name["long"]["required_max_model_len"] == 2112
    assert max(p["required_max_model_len"] for p in plans) == 2112
    assert len(by_name["long"]["prompts"]) == 1


def test_required_len_covers_prompt_plus_budget() -> None:
    # 512 + 2048 = 2560，刚好对齐到 64 的倍数
    assert required_max_model_len(512, 2048) == 2560
    # 不足 64 的部分向上取整，保证预算不被削掉
    assert required_max_model_len(513, 2048) == 2624
    assert required_max_model_len(0, 1) == 64


def test_required_len_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        required_max_model_len(-1, 1024)
    with pytest.raises(ValueError):
        required_max_model_len(512, 0)


def test_aime_budget_exceeds_old_fixed_2560() -> None:
    """旧的固定 max_model_len=2560 对 AIME（max_tokens=2048）只留 512 token 给 prompt。

    任何长于 512 token 的题干都会悄悄吃掉生成预算，这是本次补丁要消灭的失真来源。
    """
    aime = DATASETS["aime2024"]
    assert aime.max_tokens == 2048
    assert required_max_model_len(700, aime.max_tokens) > 2560


def test_binomial_stderr_beats_zero_one_std() -> None:
    # GSM8K 1319 题、76.65% 时的 binomial stderr 约 1.16 个百分点
    se = binomial_stderr(0.7665, 1319)
    assert 0.0110 < se < 0.0120
    assert binomial_stderr(0.5, 0) == 0.0


def test_reported_stderr_backfills_legacy_summary() -> None:
    """老 summary.json 没有 binomial_stderr 字段，也要能算出同样的值。"""
    legacy = {"score": 0.7665, "num_samples": 1319, "n_per_sample": 1, "stderr": 0.0116}
    assert reported_stderr(legacy) == pytest.approx(binomial_stderr(0.7665, 1319))
    # avg@k 用逐题 accuracy 的 stderr
    avg = {"score": 0.0271, "num_samples": 30, "n_per_sample": 16, "stderr": 0.0122, "std": 0.067}
    assert reported_stderr(avg) == pytest.approx(0.0122)


def test_emit_row_shape_matches_results_master() -> None:
    entry = {
        "dataset": "gsm8k",
        "display_name": "GSM8K test",
        "num_samples": 1319,
        "metric": "pass@1",
        "score": 0.7665,
        "std": 0.4231,
        "stderr": 0.0116,
        "n_per_sample": 1,
        "truncation_rate": 0.0053,
        "empty_answer_rate": 0.0023,
        "generation": {"temperature": 0.0, "top_p": 1.0, "n": 1},
    }
    row = emit_row(entry, {"scorer": "math-verify==0.9.0"}, {
        "date": "2026-10-21",
        "exp_id": "EXP-W02-01-baseline-gsm8k",
        "model": "base",
        "gpu": "RTX 4080 16GB",
        "wandb": "—",
    })
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert len(cells) == 15, row
    assert cells[6] == "greedy pass@1"
    assert cells[8] == "±0.0116"
    assert cells[9] == "n/a"  # pass@1 不报 0/1 样本 std
    assert cells[10] == "0.53%"
    assert cells[11] == "0.23%"


def test_emit_row_keeps_std_for_avg_at_k(tmp_path) -> None:
    summary = {
        "config": {"scorer": "math-verify==0.9.0"},
        "datasets": {
            "aime2024": {
                "dataset": "aime2024",
                "display_name": "AIME 2024",
                "num_samples": 30,
                "metric": "avg@16",
                "score": 0.0271,
                "std": 0.0670,
                "stderr": 0.0122,
                "n_per_sample": 16,
                "truncation_rate": 0.1,
                "empty_answer_rate": 0.02,
                "generation": {"temperature": 0.6, "top_p": 0.95, "n": 16},
            }
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    entry = summary["datasets"]["aime2024"]
    cells = [c.strip() for c in emit_row(entry, summary["config"], {
        "date": "2026-10-21", "exp_id": "E", "model": "m", "gpu": "g", "wandb": "—"
    }).strip().strip("|").split("|")]
    assert cells[6] == "avg@16"
    assert cells[9] == "0.0670"
