"""仓库骨架的冒烟单测（W1）。

只做不依赖 GPU、不依赖 verl 的结构检查，保证 `pytest -q` 在任何机器上都能全绿
（协议 02_AGENT_PROTOCOL.md §8：判分/奖励/过滤逻辑的真实单测从 W3 起加在本目录）。
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# 协议 §1 要求的目录
REQUIRED_DIRS = [
    "analysis",
    "configs",
    "configs/hw",
    "data",
    "docs",
    "env",
    "eval",
    "experiments",
    "reports",
    "scripts",
    "src/selfevolve",
    "tests",
]

SUBPACKAGES = ["challenger", "metrics", "rewards", "solver", "verifier"]

# 协议 §3 要求 registry 登记与补全的字段
REGISTRY_COLUMNS = [
    "id",
    "date",
    "purpose",
    "config_path",
    "model",
    "gpu",
    "seed",
    "git_commit",
    "status",
    "wandb_url",
    "key_metrics",
    "gpu_hours",
    "cost_usd",
    "conclusion",
]


@pytest.mark.parametrize("rel", REQUIRED_DIRS)
def test_required_dir_exists(rel: str) -> None:
    assert (REPO_ROOT / rel).is_dir(), f"协议 §1 要求的目录缺失: {rel}"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name: str) -> None:
    importlib.import_module(f"selfevolve.{name}")


def _exported_keys(hw: str) -> set[str]:
    """读取硬件档位文件里 export 的变量名。"""
    text = (REPO_ROOT / "configs" / "hw" / f"{hw}.env").read_text(encoding="utf-8")
    return {
        line.removeprefix("export ").split("=", 1)[0].strip()
        for line in text.splitlines()
        if line.startswith("export ")
    }


def test_hw_profiles_define_same_keys() -> None:
    """换卡只改 HW 一个变量 —— 两个档位必须定义同一组变量，否则 source 后会漏参数。"""
    assert _exported_keys("4080") == _exported_keys("3090")


def test_registry_header_matches_protocol() -> None:
    with (REPO_ROOT / "experiments" / "registry.csv").open(encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == REGISTRY_COLUMNS
