"""vLLM 离线生成后端（评测专用）。

评测时没有训练进程占卡，16GB 可以用到 0.85。所有采样参数显式传入，不用默认值。
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Generation:
    text: str
    finish_reason: str
    num_tokens: int


class VLLMGenerator:
    def __init__(
        self,
        model_path: str,
        *,
        max_model_len: int = 2560,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "bfloat16",
        lora_adapter: str | None = None,
        max_lora_rank: int = 32,
        tensor_parallel_size: int = 1,
        seed: int = 42,
    ) -> None:
        from vllm import LLM

        self.model_path = model_path
        self.lora_adapter = lora_adapter
        kwargs: dict[str, Any] = {
            "model": model_path,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": False,
            "seed": seed,
            "disable_log_stats": True,
        }
        if lora_adapter:
            kwargs.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
        logger.info("initializing vLLM: %s", {k: v for k, v in kwargs.items() if k != "model"})
        self.llm = LLM(**kwargs)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

    def render(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(
        self,
        prompts: list[str],
        *,
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        seed: int = 42,
    ) -> list[list[Generation]]:
        from vllm import SamplingParams

        if temperature <= 0.0 and n != 1:
            raise ValueError("greedy decoding (temperature=0) requires n=1")
        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        lora_request = None
        if self.lora_adapter:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("eval_adapter", 1, self.lora_adapter)
        outputs = self.llm.generate(prompts, params, lora_request=lora_request)
        results: list[list[Generation]] = []
        for out in outputs:
            results.append(
                [
                    Generation(
                        text=o.text,
                        finish_reason=str(o.finish_reason),
                        num_tokens=len(o.token_ids),
                    )
                    for o in out.outputs
                ]
            )
        return results

    def close(self) -> None:
        try:
            del self.llm
        finally:
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
