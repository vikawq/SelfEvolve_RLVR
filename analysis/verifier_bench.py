"""Per-sample latency of the strict verifier (task: 单样本平均耗时 / p99)。

两种路径都测：
- main：POSIX 主线程 -> signal.setitimer（评测脚本走这条）；
- thread：线程 -> 进程池 + future.result(timeout)（verl reward manager 走这条）。

    python analysis/verifier_bench.py --n 500
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from selfevolve.verifier.math_verifier import SCORER_VERSION, verify  # noqa: E402


def load_samples(n: int) -> list[tuple[str, str]]:
    path = REPO_ROOT / "outputs" / "eval" / "Qwen2.5-1.5B-Instruct" / "gsm8k.jsonl"
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            for out in rec["outputs"]:
                pairs.append((out["text"], rec["ground_truth"]))
                if len(pairs) >= n:
                    return pairs
    return pairs


def run(pairs: list[tuple[str, str]]) -> list[float]:
    durations: list[float] = []
    for text, gold in pairs:
        started = time.perf_counter()
        verify(text, gold, finished=True)
        durations.append(time.perf_counter() - started)
    return durations


def summarize(name: str, durations: list[float]) -> dict:
    ordered = sorted(durations)
    stats = {
        "path": name,
        "n": len(ordered),
        "mean_ms": 1000 * statistics.fmean(ordered),
        "p50_ms": 1000 * ordered[len(ordered) // 2],
        "p95_ms": 1000 * ordered[int(len(ordered) * 0.95)],
        "p99_ms": 1000 * ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "max_ms": 1000 * ordered[-1],
    }
    print(
        f"[bench] {name:7s} n={stats['n']} mean={stats['mean_ms']:.2f}ms p50={stats['p50_ms']:.2f}ms "
        f"p95={stats['p95_ms']:.2f}ms p99={stats['p99_ms']:.2f}ms max={stats['max_ms']:.1f}ms"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "verifier_bench.json"))
    args = parser.parse_args(argv)

    pairs = load_samples(args.n)
    main_stats = summarize("main", run(pairs))

    thread_result: dict = {}

    def worker() -> None:
        thread_result["durations"] = run(pairs)

    thread = threading.Thread(target=worker)
    started = time.perf_counter()
    thread.start()
    thread.join()
    wall = time.perf_counter() - started
    thread_stats = summarize("thread", thread_result["durations"])
    # 320 seq/step 的训练步：reward 以 8 个 worker 并行
    per_step = 320 * main_stats["mean_ms"] / 1000
    print(f"[bench] wall in thread: {wall:.2f}s; 320 样本串行 {per_step:.2f}s；按 reward.num_workers=8 并行约 {per_step / 8:.2f}s/step")
    print(f"[bench] scorer={SCORER_VERSION}")

    out = {
        "scorer": SCORER_VERSION,
        "samples": len(pairs),
        "path_main": main_stats,
        "path_thread": thread_stats,
        "projection": {
            "samples_per_step": 320,
            "serial_s_per_step": round(per_step, 3),
            "parallel_s_per_step_num_workers_8": round(per_step / 8, 3),
            "w2_step_seconds": 162.4,
            "share_of_step_pct": round(100 * (per_step / 8) / 162.4, 3),
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
