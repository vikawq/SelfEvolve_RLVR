"""评测数据集定义：加载、prompt 模板、采样参数。

红线（协议 §4）：测试集只用于评测。本模块不向训练侧暴露任何写接口。

prompt 约定（与训练一致）：
- GSM8K 直接使用训练 parquet 的 `prompt` 字段（逐字一致，已含 `####` 指令与 chat role）。
- MATH-500 / AIME 使用同一句 `####` 指令，仅把题干替换掉；这样基座与 RL 模型的格式要求相同。
- MMLU-Pro（可选）使用选择题模板，要求最后一行 `Answer: X`。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

# 与 verl examples/data_preprocess/gsm8k.py 写入 prompt 的后缀完全一致
PROMPT_SUFFIX = '\n\nLet\'s think step by step and output the final answer after "####".'

MC_INSTRUCTION = (
    "Answer the following multiple choice question. "
    'The last line of your response must be exactly "Answer: $LETTER" (without quotes), '
    "where LETTER is one of the option letters.\n\n"
)


@dataclass
class Example:
    """一条评测样本。messages 为 chat 格式，生成前由 tokenizer 套 chat template。"""

    id: str
    messages: list[dict[str, str]]
    ground_truth: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    metric: str  # "pass@1" | "avg@k"
    temperature: float
    top_p: float
    n: int
    max_tokens: int
    answer_type: str  # "math" | "mc"
    prompt_template: str  # 显式记录模板来源，写入 summary
    loader: Callable[[int | None], list[Example]]


def _user(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def _limit(rows: list[Example], limit: int | None) -> list[Example]:
    return rows if limit is None or limit <= 0 else rows[:limit]


# ---------------------------------------------------------------------------
# GSM8K：读本地训练同源 parquet（只读 test split）
# ---------------------------------------------------------------------------
def load_gsm8k(limit: int | None = None) -> list[Example]:
    import pandas as pd

    path = Path(os.environ.get("GSM8K_TEST_PARQUET", REPO_ROOT / "data" / "gsm8k" / "test.parquet"))
    if not path.exists():
        raise FileNotFoundError(
            f"GSM8K test parquet not found: {path}. 先跑 scripts/prepare_data.sh（见 docs/data_format.md）。"
        )
    df = pd.read_parquet(path)
    rows: list[Example] = []
    for i, rec in enumerate(df.to_dict("records")):
        messages = [dict(m) for m in rec["prompt"]]
        gt = str(rec["reward_model"]["ground_truth"])
        rows.append(
            Example(
                id=f"gsm8k/test/{i}",
                messages=messages,
                ground_truth=gt,
                meta={"question": rec.get("extra_info", {}).get("question", "")},
            )
        )
    return _limit(rows, limit)


# ---------------------------------------------------------------------------
# HF 数据集
# ---------------------------------------------------------------------------
def _load_hf(name: str, split: str, limit: int | None) -> Any:
    from datasets import load_dataset

    # 与 W1 相同：本机直连 HF 超时，走镜像；Xet CAS 会被镜像重定向后 401。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ds = load_dataset(name, split=split)
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_math500(limit: int | None = None) -> list[Example]:
    ds = _load_hf("HuggingFaceH4/MATH-500", "test", limit)
    rows = []
    for rec in ds:
        rows.append(
            Example(
                id=str(rec["unique_id"]),
                messages=_user(str(rec["problem"]).strip() + PROMPT_SUFFIX),
                ground_truth=str(rec["answer"]).strip(),
                meta={"subject": rec.get("subject"), "level": rec.get("level")},
            )
        )
    return rows


def load_aime2024(limit: int | None = None) -> list[Example]:
    ds = _load_hf("Maxwell-Jia/AIME_2024", "train", limit)
    rows = []
    for rec in ds:
        rows.append(
            Example(
                id=f"aime2024/{rec['ID']}",
                messages=_user(str(rec["Problem"]).strip() + PROMPT_SUFFIX),
                ground_truth=str(rec["Answer"]).strip(),
            )
        )
    return rows


def load_aime2025(limit: int | None = None) -> list[Example]:
    ds = _load_hf("yentinglin/aime_2025", "train", limit)
    rows = []
    for rec in ds:
        rows.append(
            Example(
                id=f"aime2025/{rec['id']}",
                messages=_user(str(rec["problem"]).strip() + PROMPT_SUFFIX),
                ground_truth=str(rec["answer"]).strip(),
                meta={"url": rec.get("url")},
            )
        )
    return rows


def load_mmlu_pro(limit: int | None = 1000) -> list[Example]:
    """可选：从 MMLU-Pro test 随机抽 1000 题（seed=42，固定集合）。"""
    ds = _load_hf("TIGER-Lab/MMLU-Pro", "test", None)
    if limit is not None and limit > 0:
        ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    rows = []
    letters = "ABCDEFGHIJ"
    for rec in ds:
        options = list(rec["options"])
        body = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        content = f"{MC_INSTRUCTION}{rec['question'].strip()}\n\n{body}"
        rows.append(
            Example(
                id=f"mmlu_pro/{rec['question_id']}",
                messages=_user(content),
                ground_truth=str(rec["answer"]).strip().upper(),
                meta={"category": rec.get("category")},
            )
        )
    return rows


DATASETS: dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        name="gsm8k",
        display_name="GSM8K test",
        metric="pass@1",
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=1024,
        answer_type="math",
        prompt_template="gsm8k_training_verbatim",
        loader=load_gsm8k,
    ),
    "math500": DatasetSpec(
        name="math500",
        display_name="MATH-500",
        metric="pass@1",
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=1024,
        answer_type="math",
        prompt_template="####_suffix",
        loader=load_math500,
    ),
    "aime2024": DatasetSpec(
        name="aime2024",
        display_name="AIME 2024",
        metric="avg@16",
        temperature=0.6,
        top_p=0.95,
        n=16,
        max_tokens=2048,
        answer_type="math",
        prompt_template="####_suffix",
        loader=load_aime2024,
    ),
    "aime2025": DatasetSpec(
        name="aime2025",
        display_name="AIME 2025",
        metric="avg@16",
        temperature=0.6,
        top_p=0.95,
        n=16,
        max_tokens=2048,
        answer_type="math",
        prompt_template="####_suffix",
        loader=load_aime2025,
    ),
    "mmlu_pro": DatasetSpec(
        name="mmlu_pro",
        display_name="MMLU-Pro (1000 sampled)",
        metric="pass@1",
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=512,
        answer_type="mc",
        prompt_template="mc_answer_letter",
        loader=load_mmlu_pro,
    ),
}


def build_specs(names: list[str], limit: int | None = None) -> list[tuple[DatasetSpec, list[Example]]]:
    """按名称加载数据集；未知名直接报错，避免静默少评。"""
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise KeyError(f"unknown datasets: {unknown}; available: {sorted(DATASETS)}")
    out = []
    for n in names:
        spec = DATASETS[n]
        out.append((spec, spec.loader(limit)))
    return out
