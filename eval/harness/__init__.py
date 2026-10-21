"""独立评测 harness（W2）。

与训练框架解耦：不 import verl，不读训练 config；只依赖 transformers/tokenizers/vLLM/math-verify。
"""

from eval.harness.data import DATASETS, DatasetSpec, Example, build_specs
from eval.harness.scoring import SCORER_VERSION, score_math, score_mc

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "Example",
    "build_specs",
    "SCORER_VERSION",
    "score_math",
    "score_mc",
]
