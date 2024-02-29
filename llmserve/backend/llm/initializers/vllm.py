from typing import Any, Dict

import torch

from llmserve.backend.logger import get_logger

from ._base import LLMInitializer
from .models import AsyncEngineArgsFilter
import ray

logger = get_logger(__name__)

class VllmInitializer(LLMInitializer):
    """Initialize vllm model and tokenizer.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        **model_init_kwargs: Keyword arguments to pass to the llama_cpp model init.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        **model_init_kwargs,
    ):
        super().__init__(
            device=device,
            world_size=world_size,
        )
        self.model_init_kwargs = model_init_kwargs

    def _get_model_init_kwargs(self) -> Dict[str, Any]:
        logger.info(f"Vllm initialize model parameters {self.model_init_kwargs}")
        return AsyncEngineArgsFilter.from_model_init_kwargs(self.model_init_kwargs)

    def load_model(self, model_id: str) -> "AsyncLLMEngine":
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        
        current_placement_group = ray.util.get_current_placement_group()
        logger.info(f"ray current placement group: {current_placement_group}")
        if current_placement_group:
            bundles = current_placement_group.bundle_specs
            logger.info(f"ray current placement group bundle: {bundles}")
            for bundle in bundles:
                bundle_gpus = bundle.get("GPU", 0)
                logger.info(f"ray current placement group bundle gpus: {bundle_gpus}")
        
        logger.info(f"VllmInitializer loading {model_id}")
        engine_args = AsyncEngineArgs(
            model=model_id,
            **self._get_model_init_kwargs(),
        )

        return AsyncLLMEngine.from_engine_args(engine_args)

    def load_tokenizer(self, tokenizer_name: str) -> None:
        return None
