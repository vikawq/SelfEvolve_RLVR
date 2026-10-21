"""Profiling 日志解析单测（W2）。不依赖 GPU / wandb。"""

from __future__ import annotations

import math

from analysis.parse_profile import PHASES, parse_log, summarize


def _write_log(tmp_path, name: str, steps: list[int], *, tag: str = "EXP-W02-09-test") -> "Path":
    lines = []
    for step in steps:
        lines.append(f"[{tag}] HW=4080 running")
        lines.append(
            f"step:{step} - timing_s/gen:{10 + step}.0 - timing_s/old_log_prob:20.0 - "
            f"timing_s/ref:5.0 - timing_s/update_actor:60.0 - timing_s/update_weights:2.0 - "
            f"timing_s/reward:1.0 - timing_s/adv:0.01 - timing_s/step:{98 + step}.0 - "
            f"perf/throughput:700.0 - perf/max_memory_allocated_gb:np.float64(9.5)"
        )
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parse_merges_split_runs_and_reads_np_float(tmp_path) -> None:
    p1 = _write_log(tmp_path, "a.log", [1, 2, 3])
    p2 = _write_log(tmp_path, "b.log", [4, 5])
    runs = parse_log(p1) + parse_log(p2)

    merged = {}
    for run in runs:
        merged.setdefault(run.name, run)
        if merged[run.name] is not run:
            merged[run.name].steps.extend(run.steps)
    run = merged["EXP-W02-09-test"]
    assert len(run.steps) == 5
    assert run.max_of(("perf/max_memory_allocated_gb",)) == 9.5
    # 默认剔除 step 1，剩余 [2,3,4,5] 的中位数
    assert run.median("training/global_step") == 3.5


def test_summarize_shares_and_medians(tmp_path) -> None:
    run = parse_log(_write_log(tmp_path, "a.log", [1, 2, 3]))[0]
    summary = summarize(run)
    # step 1 被剔除，median step = 100（step2）与 101（step3）的中位数 100.5
    assert math.isclose(summary["median_step_train_s"], 100.5)
    phase_sum = sum(v for v in summary["median_phases_s"].values() if v)
    # 中位数不可加：phase sum 与 step median 允许有小偏差，但不能超过 step 太多
    assert phase_sum <= summary["median_step_train_s"] + 1.0
    assert math.isclose(sum(summary["phase_share"].values()), 1.0, rel_tol=1e-6)
    for key in PHASES:
        assert key in summary["median_phases_s"]
    assert summary["max_memory_allocated_gb"] == 9.5
