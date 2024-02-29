from pydantic import BaseModel
from typing import Optional
import dataclasses
from dataclasses import dataclass


@dataclass
class AsyncEngineArgsFilter(BaseModel):
    """Arguments for vLLM engine."""
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    dtype: str = 'auto'
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    max_paddings: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192

    @classmethod
    def from_model_init_kwargs(
        cls,
        kwargs: dict,
    ) -> dict:
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = {k: v for k, v in kwargs.items() if k in attrs}

        return engine_args
