"""W4 批量评测的评分 worker（每个进程只评一条，评完即退出）。

为什么单独一个模块：`multiprocessing` 用 spawn，子进程会 import 本模块；这里**只**依赖
`eval.harness.scoring`（→ 严格 verifier），不 import torch/vLLM，因此子进程不会初始化 CUDA，
也不会把父进程的 vLLM 状态复制过去。

背景：`math_verify` 在一条进程里连续评 500 条 MATH-500 时，sympy 的全局缓存会随样本累积
（W4 实测 500 条跑到 63 条时 RSS 114 GB、机器开始换页）；把评分放进 `maxtasksperchild=1`
的进程池，每条评完就释放缓存，评分函数本身**不变**。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from eval.harness.scoring import score  # noqa: E402


def score_task(payload: tuple[str, str, str, bool | None]) -> dict[str, Any]:
    """评一条：`(text, ground_truth, answer_type, finished)` → `score()` 的 dict。"""
    text, ground_truth, answer_type, finished = payload
    return score(text, ground_truth, answer_type, finished=finished)


def score_task_into(payload: tuple[str, str, str, bool | None], conn: Any) -> None:
    """评一条并把结果写回管道；父进程超时会直接 kill 本进程，所以这里不做超时。"""
    try:
        conn.send(score_task(payload))
    except BaseException as exc:  # pragma: no cover - 子进程崩溃路径
        conn.send({"correct": False, "extracted": "", "parse_ok": False, "reason": "error",
                   "gold_normalized": "", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
    finally:
        conn.close()


__all__ = ["score_task", "score_task_into"]
