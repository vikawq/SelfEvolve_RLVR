#!/usr/bin/env python
"""预下载/校验评测数据集（W2）。

在 GPU 任务开始前先跑一次，避免评测时被 HF 镜像抖动卡住：

    python eval/prefetch_datasets.py

只做下载与 schema 校验，不生成、不判分。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from eval.harness.data import DATASETS

    failed = []
    for name, spec in DATASETS.items():
        try:
            rows = spec.loader(1)  # 只取 1 条验证 schema
            assert rows and rows[0].ground_truth != "", "empty ground truth"
            print(f"[prefetch] {name}: OK, one sample id={rows[0].id}, gt={rows[0].ground_truth!r}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"[prefetch] {name}: FAIL {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
