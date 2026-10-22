"""vLLM 离线生成后端（评测专用）。

评测时没有训练进程占卡，16GB 可以用到 0.85。所有采样参数显式传入，不用默认值。

`max_model_len` 不能拍一个全局值：vLLM 在 `prompt_len + max_tokens > max_model_len` 时
**不报错**，而是静默把生成预算裁到 `max_model_len - prompt_len`，表现为截断率升高，
看起来像"模型不行"。所以预算按数据集算出来（见 `required_max_model_len`），
由 `run_eval.py` 取各数据集的最大值初始化引擎。
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


def load_tokenizer(model_path: str) -> Any:
    """在 vLLM 初始化之前拿到 tokenizer，用于按数据集规划 max_model_len。"""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """套 chat template（不 tokenize），与训练时的 rollout 输入保持一致。"""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def count_prompt_tokens(tokenizer: Any, prompts: list[str]) -> list[int]:
    """逐条 prompt 的 token 数。chat template 已把特殊 token 写成文本，故不再加特殊 token。"""
    if not prompts:
        return []
    encoded = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    return [len(ids) for ids in encoded]


def required_max_model_len(prompt_tokens_max: int, max_tokens: int, *, align: int = 64) -> int:
    """某个数据集需要的 max_model_len = 最长 prompt + 完整生成预算，向上取 align 的倍数。

    低于这个值时 vLLM 会静默压缩生成预算（见模块 docstring），必须显式避免。
    """
    if prompt_tokens_max < 0:
        raise ValueError(f"prompt_tokens_max 必须 >= 0，收到 {prompt_tokens_max}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens 必须 >= 1，收到 {max_tokens}")
    need = prompt_tokens_max + max_tokens
    return ((need + align - 1) // align) * align


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
        tokenizer: Any | None = None,
    ) -> None:
        from vllm import LLM

        self.model_path = model_path
        self.lora_adapter = lora_adapter
        self.max_model_len = max_model_len
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

        # tokenizer 由调用方在引擎之前建好并传进来（规划 max_model_len 时就需要），避免重复加载。
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(model_path)

    def render(self, messages: list[dict[str, str]]) -> str:
        return render_chat(self.tokenizer, messages)

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
