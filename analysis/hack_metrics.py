#!/usr/bin/env python
"""W4 离线分析：从 training rollout dump 与评测 JSONL 计算 hack 检测指标。

数据来源（都不改评测代码）：

1. `outputs/w04/rollouts/<EXP>/<step>.jsonl`
   —— verl v0.9.1 `trainer.rollout_data_dir` 的原生落盘，每行含 `input` / `output` /
   `gts` / `score`（宽松奖励）以及 reward dict 的全部诊断键（`strict_acc`、`n_numbers`…）。
2. `outputs/eval/<EXP>-global_step_<N>/{gsm8k,math500}.jsonl`
   —— `eval/run_eval.py` 的逐样本输出，`correct` 已经是严格 verifier 的判定；
   本脚本再把同一个宽松判分器作用到这些文本上，得到评测侧的 reward gap。
3. `outputs/eval/Qwen2.5-0.5B-Instruct/...` —— step0 基座参照。

输出：`outputs/w04/metrics/<EXP>.json`（训练逐步曲线 + 各 checkpoint 评测指标 + hack onset）。
可选 `--wandb-run-id` 把同一批指标回填到原来的 wandb run（训练循环本身不记录 reward dict 的
额外键，只有 validation 会；这是 v0.9.1 的实际行为，见 `docs/troubleshooting.md`）。

用法：

    python analysis/hack_metrics.py \
        --experiment EXP-W04-02-l2-substring --hack-id L2 \
        --rollout-dir outputs/w04/rollouts/EXP-W04-02-l2-substring \
        --eval-root outputs/eval --base-eval-dir outputs/eval/Qwen2.5-0.5B-Instruct \
        --tokenizer models/Qwen2.5-0.5B-Instruct \
        --output outputs/w04/metrics/EXP-W04-02-l2-substring.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.metrics.hack_detectors import (  # noqa: E402
    aggregate_responses,
    hack_onset,
    rolling_mean,
)

HACK_MODULES = {
    "L1": ("selfevolve.rewards.loose_L1_format", "L1-format-weight"),
    "L2": ("selfevolve.rewards.loose_L2_substring", "L2-substring"),
    "L3": ("selfevolve.rewards.loose_L3_first_number", "L3-first-number"),
    "L4": ("selfevolve.rewards.loose_L4_length", "L4-length-bonus"),
    "strict": ("selfevolve.rewards.strict_reward", "strict"),
}

#: 与 configs/EXP-W04-00-control-strict.sh 的 `actor_rollout_ref.rollout.n` 一致。
DEFAULT_ROLLOUT_N = 5
DEFAULT_MAX_RESPONSE_LEN = 1024


def load_compute_score(hack_id: str) -> tuple[Callable[..., dict[str, Any]], str]:
    module_name, label = HACK_MODULES[hack_id]
    module = importlib.import_module(module_name)
    return module.compute_score, label


def load_tokenizer(model_path: str | None):
    if not model_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return tokenizer


def tokenize_lengths(tokenizer, texts: list[str], batch_size: int = 64) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        lengths.extend(len(ids) for ids in encoded)
    return lengths


def _step_files(rollout_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in rollout_dir.glob("*.jsonl"):
        try:
            step = int(path.stem)
        except ValueError:
            continue
        files.append((step, path))
    files.sort(key=lambda item: item[0])
    return files


def _check_unique_steps(files: list[tuple[int, Path]]) -> None:
    steps = [step for step, _ in files]
    duplicates = sorted({step for step in steps if steps.count(step) > 1})
    if duplicates:
        raise RuntimeError(f"duplicate step files in rollout dir: {duplicates}")


def analyse_rollouts(
    rollout_dir: Path,
    tokenizer,
    *,
    max_response_length: int,
    limit_steps: int = 0,
) -> list[dict[str, Any]]:
    files = _step_files(rollout_dir)
    if not files:
        raise FileNotFoundError(f"no <step>.jsonl under {rollout_dir}")
    _check_unique_steps(files)
    if limit_steps > 0:
        files = files[:limit_steps]

    steps: list[dict[str, Any]] = []
    for step, path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = [str(row.get("output", "")) for row in rows]
        lengths = tokenize_lengths(tokenizer, texts) if tokenizer is not None else None
        records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            record: dict[str, Any] = {
                "text": texts[index],
                "loose_score": float(row.get("score", 0.0)),
                # 对照组（strict_reward）没有 strict_acc 键：它的 score 就是严格准确率。
                "strict_score": float(row.get("strict_acc", row.get("score", 0.0))),
            }
            if lengths is not None:
                record["num_tokens"] = lengths[index]
                record["truncated"] = lengths[index] >= max_response_length
            records.append(record)
        aggregated = aggregate_responses(records)
        aggregated["step"] = step
        steps.append(aggregated)
    return steps


def _eval_dirs(eval_root: Path, experiment: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(experiment)}-global_step_(\d+)$")
    found: list[tuple[int, Path]] = []
    for path in eval_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match and (path / "summary.json").exists():
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    return found


def analyse_eval_dir(
    eval_dir: Path,
    compute_score: Callable[..., dict[str, Any]],
    *,
    step: int,
    label: str,
    hack_id: str,
) -> dict[str, Any]:
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {"step": step, "label": label, "eval_dir": str(eval_dir), "datasets": {}}
    for dataset in ("gsm8k", "math500"):
        jsonl = eval_dir / f"{dataset}.jsonl"
        if not jsonl.exists():
            continue
        records: list[dict[str, Any]] = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gold = str(row.get("ground_truth", ""))
            data_source = "openai/gsm8k" if dataset == "gsm8k" else "math500"
            for output in row.get("outputs", []):
                text = str(output.get("text", ""))
                strict_correct = float(bool(output.get("correct", False)))
                if hack_id == "strict":
                    # 对照组的"宽松奖励"就是严格判定本身；用 harness 的 correct（含截断判错），
                    # 而不是离线 verify，否则截断样本会制造一个假的正 gap。
                    loose = strict_correct
                else:
                    loose = float(compute_score(data_source, text, gold)["score"])
                records.append(
                    {
                        "text": text,
                        "loose_score": loose,
                        # 评测侧的严格判定以 eval harness 的 correct 为准（它带 finished 截断判定）
                        "strict_score": strict_correct,
                        "num_tokens": output.get("num_tokens"),
                        "truncated": output.get("finish_reason") == "length",
                    }
                )
        aggregated = aggregate_responses(records)
        entry = summary.get("datasets", {}).get(dataset, {})
        aggregated["strict_harness_score"] = entry.get("score")
        aggregated["strict_harness_stderr"] = (
            entry.get("binomial_stderr")
            if entry.get("binomial_stderr") is not None
            else entry.get("stderr")
        )
        aggregated["truncation_rate_harness"] = entry.get("truncation_rate")
        result["datasets"][dataset] = aggregated
    return result


def find_wandb_run_id(experiment: str, logs_dir: Path) -> str | None:
    logs = sorted(logs_dir.glob(f"{experiment}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"/runs/([a-z0-9]{8})", text)
        if match:
            return match.group(1)
    return None


def backfill_wandb(metrics: dict[str, Any], *, run_id: str, project: str, entity: str | None) -> None:
    import wandb

    kwargs: dict[str, Any] = {"project": project, "id": run_id, "resume": "allow", "reinit": True}
    if entity:
        kwargs["entity"] = entity
    run = wandb.init(**kwargs)
    if run is None:  # pragma: no cover - wandb 返回 None 只会在异常模式下出现
        raise RuntimeError("wandb.init returned None")
    for step_metrics in metrics["training"]:
        x = int(step_metrics["step"])
        run.log(
            {
                "hack/loose_reward": step_metrics["loose_mean"],
                "hack/strict_acc": step_metrics["strict_mean"],
                "hack/reward_gap": step_metrics["reward_gap"],
                "hack/n_numbers": step_metrics["n_numbers_mean"],
                "hack/boxed_rate": step_metrics["boxed_rate"],
                "hack/len_chars_mean": step_metrics["len_chars_mean"],
                "hack/repeat_ngram_ratio": step_metrics["repeat_ngram_ratio_mean"],
                "hack/n_samples": step_metrics["n"],
            },
            step=x,
        )
    for eval_entry in metrics["eval"]:
        x = int(eval_entry["step"])
        payload: dict[str, float] = {}
        for dataset, values in eval_entry["datasets"].items():
            payload[f"eval/{dataset}_loose_reward"] = values["loose_mean"]
            payload[f"eval/{dataset}_strict_acc"] = values["strict_mean"]
            payload[f"eval/{dataset}_reward_gap"] = values["reward_gap"]
            payload[f"eval/{dataset}_truncation_rate"] = values.get("truncation_rate", 0.0)
        run.log(payload, step=x)
    run.finish()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W4 reward hacking metrics")
    parser.add_argument("--experiment", required=True, help="如 EXP-W04-02-l2-substring")
    parser.add_argument("--hack-id", required=True, choices=sorted(HACK_MODULES))
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--eval-root", default=str(REPO_ROOT / "outputs" / "eval"))
    parser.add_argument("--base-eval-dir", default=str(REPO_ROOT / "outputs" / "eval" / "Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--tokenizer", default=None, help="模型目录；给了才统计 token 长度/截断率")
    parser.add_argument("--max-response-length", type=int, default=DEFAULT_MAX_RESPONSE_LEN)
    parser.add_argument("--limit-steps", type=int, default=0, help="调试用：只分析前 N 步")
    parser.add_argument("--output", required=True)
    parser.add_argument("--logs-dir", default=str(REPO_ROOT / "logs"))
    parser.add_argument("--wandb-run-id", default=None, help="不传则从 logs/<EXP>_*.log 里取")
    parser.add_argument("--wandb-project", default="selfevolve-rlvr")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)

    rollout_dir = Path(args.rollout_dir)
    if not rollout_dir.exists():
        parser.error(f"rollout dir not found: {rollout_dir}")

    compute_score, hack_label = load_compute_score(args.hack_id)
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"[hack_metrics] {args.experiment} hack={hack_label} rollouts={rollout_dir}")
    training = analyse_rollouts(
        rollout_dir,
        tokenizer,
        max_response_length=args.max_response_length,
        limit_steps=args.limit_steps,
    )
    print(f"[hack_metrics] training steps: {len(training)}")

    eval_entries: list[dict[str, Any]] = []
    base_dir = Path(args.base_eval_dir)
    if base_dir.exists() and (base_dir / "summary.json").exists():
        eval_entries.append(
            analyse_eval_dir(base_dir, compute_score, step=0, label="base", hack_id=args.hack_id)
        )
    for step, eval_dir in _eval_dirs(Path(args.eval_root), args.experiment):
        eval_entries.append(
            analyse_eval_dir(
                eval_dir, compute_score, step=step, label="checkpoint", hack_id=args.hack_id
            )
        )
    print(f"[hack_metrics] eval entries: {len(eval_entries)}")

    steps = [int(row["step"]) for row in training]
    gaps = [float(row["reward_gap"]) for row in training]
    onset = hack_onset(steps, gaps)
    final = training[-1]

    metrics = {
        "experiment": args.experiment,
        "hack_id": args.hack_id,
        "hack_label": hack_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "rollout_dir": str(rollout_dir),
            "max_response_length": args.max_response_length,
            "tokenizer": args.tokenizer,
            "rollout_n": DEFAULT_ROLLOUT_N,
            "eval_limit": 500,
        },
        "summary": {
            "n_train_steps": len(training),
            "reward_gap_final": final["reward_gap"],
            "loose_mean_final": final["loose_mean"],
            "strict_mean_final": final["strict_mean"],
            "reward_gap_final20_mean": sum(gaps[-20:]) / len(gaps[-20:]),
            "strict_mean_final20_mean": sum(float(r["strict_mean"]) for r in training[-20:])
            / len(training[-20:]),
            "hack_onset_step": onset,
            "smoothed_gap_final": rolling_mean(gaps, 10)[-1],
        },
        "training": training,
        "eval": eval_entries,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hack_metrics] wrote {output}")
    print(
        f"[hack_metrics] gap final={metrics['summary']['reward_gap_final']:.3f} "
        f"(last20={metrics['summary']['reward_gap_final20_mean']:.3f}), "
        f"onset={metrics['summary']['hack_onset_step']}"
    )

    if not args.no_wandb:
        run_id = args.wandb_run_id or find_wandb_run_id(args.experiment, Path(args.logs_dir))
        if run_id is None:
            print("[hack_metrics] WARN no wandb run id found; skip backfill")
        else:
            try:
                backfill_wandb(
                    metrics,
                    run_id=run_id,
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                )
                print(f"[hack_metrics] wandb backfilled to run {run_id}")
            except Exception as exc:  # pragma: no cover - 网络/权限问题不应毁掉分析
                print(f"[hack_metrics] WARN wandb backfill failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
