import torch
from typing import List, Optional, Any
from ray.air import ScalingConfig
from ray.util.placement_group import PlacementGroup
from llmserve.backend.server.models import Prompt

from llmserve.backend.logger import get_logger

import asyncio
import gc
import os
import traceback
from typing import List, Optional

import ray
import ray.util
import torch
import torch.backends.cuda
from ray.air import ScalingConfig
from ray.air.util.torch_dist import TorchDistributedWorker

from llmserve.backend.llm.initializers import get_initializer_cls_by_name
from llmserve.backend.llm.pipelines import get_pipeline_cls_by_name
from llmserve.backend.llm.pipelines._base import BasePipeline
from llmserve.backend.llm.utils import (
    init_torch_dist_process_group_async,
    timeit,
    get_max_token_size,
)
from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import Args, LLMConfig, Prompt, Response
from llmserve.backend.server.utils import render_gradio_params
from ._base import LLMEngine

from typing import AsyncGenerator, Generator, Union
from queue import Empty
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logger = get_logger(__name__)

@timeit
def init_model(
    llm_config: LLMConfig,
    world_size: int,
    local_rank: int,
    max_batch_size: Optional[int] = None,
):
    """Initialize the model.

    Args:
        llm_config (LLM): LLM configuration.
        world_size (int): Number of GPUs.
        local_rank (int): Local rank of the current GPU.
        max_batch_size (Optional[int], optional): Maximum batch size. Defaults to None.
    """
    logger.info(
        f"Initializing model {llm_config.model_id} with local_rank: {local_rank}")

    # Lazy import so that the new cache location is used
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        # device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    initializer_name = llm_config.initialization.initializer
    if not isinstance(initializer_name, str):
        initializer_name = initializer_name.type

    logger.info(f"Initializer name is {initializer_name} on device {device}")
    initializer = get_initializer_cls_by_name(initializer_name)(
        device=device,
        world_size=world_size,
        **llm_config.initialization.initializer.get_initializer_kwargs(),
    )

    pipeline_name = llm_config.initialization.pipeline
    additional_kwargs = dict(
        task=llm_config.model_task,
    )
    logger.info(f"Pipeline name is {pipeline_name} on device {device}")
    pipeline = get_pipeline_cls_by_name(pipeline_name).from_initializer(
        initializer,
        llm_config.actual_hf_model_id,
        prompt_format=(
            llm_config.generation.prompt_format if llm_config.generation else None),
        device=device,
        **additional_kwargs
    )

    # Warmup
    # For DS w/ kernel inject, first batch the model gets MUST be of maximum batch size,
    # otherwise subsequent batches with more entries than the first batch
    # will raise CUDA errors if use_kernel=True.
    batch_size = max_batch_size or 1

    logger.info(f"Model {llm_config.model_id} batch_size is {batch_size}")
    model_task_info = render_gradio_params(llm_config.model_task)
    warmup_inputs = model_task_info["warmup"] if "warmup" in model_task_info else None

    if llm_config.warmup and warmup_inputs:
        prowarmup_inputs_max = Prompt(prompt=warmup_inputs * (
            int(get_max_token_size(llm_config) / (len(warmup_inputs.split()) + 1))
        ), use_prompt_format=False)

        logger.info(
            f"Model {llm_config.model_id} is warming up, input is {warmup_inputs}...")
        if llm_config.generation:
            generate_kwargs = llm_config.generation.all_generate_kwargs.copy()
            if "max_new_tokens" in generate_kwargs:
                generate_kwargs["min_new_tokens"] = generate_kwargs["max_new_tokens"]
        else:
            generate_kwargs = {}

    warmup_success = False
    while not warmup_success and llm_config.warmup and warmup_inputs:
        try:
            logger.info("start to test with single prompt")
            logger.info(f"warmpup prompt is: {warmup_inputs}")
            resp = generate(
                [Prompt(prompt=warmup_inputs, use_prompt_format=True)],
                pipeline,
                **generate_kwargs,
            )
            logger.info(f"warmpup response is {str(resp)}")
            assert len(resp) > 0
            assert all(x.generated_text for x in resp)

            logger.info("start to test with max batch prompts, try to find a suitable batch size")
            assert batch_size > 0
            resp_batch = generate(
                [prowarmup_inputs_max] * batch_size,
                pipeline,
                **generate_kwargs,
            )
            logger.info(str(resp_batch))
            assert len(resp_batch) == batch_size
            assert all(x.generated_text for x in resp_batch)

            warmup_success = True
        except torch.cuda.OutOfMemoryError:
            batch_size -= 2
            logger.warning(
                f"Warmup failed due to CUDA OOM, reducing batch size to {batch_size}")

    logger.info(
        f"Model {llm_config.model_id} succesfully initialized, final batch size {batch_size}!")

    gc.collect()

    return pipeline


@timeit
def generate(
    prompts: List[Prompt], pipeline: BasePipeline, **generate_kwargs
) -> List[Response]:
    """Generate predictions using a Pipeline.

    Args:
        prompts (List[Prompt]): List of prompts.
        pipeline (BasePipeline): Pipeline to use.
        **generate_kwargs: Keyword arguments to pass to the pipeline's `generate` method.
    """
    logger.info(f"Call pipeline with generate_kwargs: {generate_kwargs}")
    outputs = pipeline(
        prompts,
        **generate_kwargs,
    )
    return outputs

import logging
@ray.remote
class PredictionWorker(TorchDistributedWorker):
    """A PredictionWorker is a Ray remote actor that runs a single shard of a DeepSpeed job.

    Multiple PredictionWorkers of the same WorkerGroup will form a PyTorch DDP process
    group and work together under the orchestration of DeepSpeed.

    Args:
        llm_config (LLM): LLM configuration.
        world_size (int): Number of GPUs.
    """

    def __init__(self, llm_config: LLMConfig, world_size: int):
        self.llm_config = llm_config
        self.world_size = world_size

    def init_model(
        self,
        local_rank: int,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: int = 0,
    ):
        """Initialize model for inference.

        Args:
            local_rank (int): Local rank of the current GPU.
            num_cpus_per_worker (int, optional): Number of CPUs to use per worker. Defaults to 1.
        """
        
        # Recreate the logger to make sure it takes precedence over
        # other logger configurations.
        # get_logger(__name__, force=True)

        # for DDP, comment now, still consider if ddp is a case
        # rank = torch.distributed.get_rank()
        # logger.info(rank)

        logger.info(
            f"num_gpus_per_worker: {num_gpus_per_worker}, num_cpus_per_worker: {num_cpus_per_worker}")
        os.environ["OMP_NUM_THREADS"] = str(int(num_cpus_per_worker))

        logger.info("Prediction Worker calling init model")
        self.generator = init_model(
            self.llm_config,
            self.world_size,
            local_rank,
            max_batch_size=(
                self.llm_config.generation.max_batch_size if self.llm_config.generation else None),
        )

    def generate(
        self,
        data: List[Prompt],
        *,
        timeout_s: Optional[float] = None,
        start_timestamp: Optional[float] = None,
        oom_retry: bool = True,
        **kwargs,
    ) -> List[str]:
        """Generate text from prompts.

        Args:
            data (List[Prompt]): Batch of prompts.
            timeout_s (Optional[float], optional): Timeout for the generation.
                Ignored if start_timestamp is None.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation. Ignored if timeout_s is None.
            oom_retry (bool, optional): Whether to retry if CUDA OOM occurs.
        """
        try:
            logger.info(
                f"Prediction Worker generate text from prompts with kwargs: {kwargs}")
            return generate(
                data,
                self.generator,
                timeout_s=timeout_s,
                start_timestamp=start_timestamp,
                **kwargs,
            )
        except torch.cuda.OutOfMemoryError as e:
            if not oom_retry:
                raise e
            else:
                logger.error(
                    "[FIXME] Prediction failed due to CUDA OOM, trying again...\n"
                    f"{traceback.format_exc()}"
                )
                data_1, data_2 = data[: len(data) // 2], data[len(data) // 2:]
                responses_1 = generate(
                    data_1,
                    self.generator,
                    timeout_s=timeout_s,
                    start_timestamp=start_timestamp,
                    **kwargs,
                )
                responses_2 = generate(
                    data_2,
                    self.generator,
                    timeout_s=timeout_s,
                    start_timestamp=start_timestamp,
                    **kwargs,
                )
                return responses_1 + responses_2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.llm_config.model_id}"

    def ping(self) -> bool:
        """Ping the worker."""
        return True
    
    async def worker_stream_generate_texts(self, prompt: Union[Prompt, List[Prompt]], **kwargs) -> Generator[str, None, None]: # type: ignore
        logger.info(f"Call PredictionWorker.worker_stream_generate_texts with kwargs: {kwargs}")
        for s in self.generator.streamGenerate(prompt, **kwargs):
            # logger.info(f"PredictionWorker.worker_stream_generate_texts -> yield ->{s}")
            yield s
    
class GenericEngine(LLMEngine):
    base_worker_group = None

    async def launch_engine(
            self, 
            scaling_config: ScalingConfig,
            placement_group: PlacementGroup,
            scaling_options: dict,
        ) -> Any:
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """

        config: Args = self.args  # pylint:disable=no-member
        llm_config = config.model_config
        runtime_env = llm_config.initialization.runtime_env or {}
        prediction_worker_cls = PredictionWorker.options(  # pylint:disable=no-member
            **scaling_options, runtime_env=runtime_env
        )
        
        # Create the prediction workers.
        logger.info("Creating prediction workers...")
        worker_group = [
            prediction_worker_cls.remote(
                llm_config, scaling_config.num_workers)
            for i in range(scaling_config.num_workers)
        ]

        logger.info("Initializing torch_dist process group on workers...")
        # Initialize torch distributed process group for the workers.
        local_ranks = await init_torch_dist_process_group_async(
            worker_group,
            backend="nccl" if scaling_config.use_gpu else "gloo",
        )

        # Initialize model on each worker.
        logger.info(
            f"Initializing model on workers with local_ranks: {local_ranks}")
        await asyncio.gather(
            *[
                worker.init_model.remote(
                    local_rank = local_rank,
                    num_cpus_per_worker=scaling_config.num_cpus_per_worker,
                    num_gpus_per_worker=scaling_config.num_gpus_per_worker
                )
                for worker, local_rank in zip(worker_group, local_ranks)
                # for worker in worker_group
            ]
        )

        self.base_worker_group = worker_group
        return worker_group

    async def predict(
            self,
            prompts: List[Prompt],
            *,
            timeout_s: float = 60,
            start_timestamp: Optional[float] = None,
            lock: asyncio.Lock,
        ) -> List[str]:
        """Generate text for a list of prompts.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.
            timeout_s (float, optional): Timeout for the generation. Defaults
                to 60. Ignored if start_timestamp is None.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation.

        Returns:
            A list of generated texts.
        """
        def slice_prompts(worker_num: int, worker_index: int, prompts: list[str]):
            prompts = ray.get(prompts)

            slice_size = len(prompts)//worker_num if len(prompts)//worker_num != 0 else 1
            if worker_index == worker_num - 1:
                return prompts[slice_size * worker_index:]
            else:
                return prompts[slice_size * worker_index: slice_size * worker_index + slice_size]

        logger.info('LLM GenericEngine do async predict')

        async with lock:
            # prediction = (
            #     await asyncio.gather(
            #         *[
            #             worker.generate.remote(
            #                 slice_prompts(len(self.base_worker_group), index, prompts),
            #                 # prompts,
            #                 timeout_s=timeout_s,
            #                 start_timestamp=start_timestamp,
            #                 **self.args.model_config.generation.all_generate_kwargs if self.args.model_config.generation else {},  # pylint:disable=no-member
            #             ) if len(slice_prompts(len(self.base_worker_group), index, prompts)) > 0 else ray.put([])

            #             for index, worker in enumerate(self.base_worker_group)
            #             # for worker in self.base_worker_group
            #         ]
            #     )
            # )
        # return [response for responses in prediction for response in responses]
            prediction = (
                await asyncio.gather(
                    *[
                        worker.generate.remote(
                            # slice_prompts(len(self.base_worker_group), index, prompts),
                            prompts,
                            timeout_s=timeout_s,
                            start_timestamp=start_timestamp,
                            **self.args.model_config.generation.all_generate_kwargs if self.args.model_config.generation else {},  # pylint:disable=no-member
                        )

                        # for index, worker in enumerate(self.base_worker_group)
                        for worker in self.base_worker_group
                    ]
                )
            )[0]

            return prediction
        
    async def check_health(self):
        if self._new_worker_group_lock.locked():
            logger.info("Rollover in progress, skipping health check")
            return
        if self.pg and self.base_worker_group:
            dead_actors = []
            for actor in self.base_worker_group:
                actor_state = ray.state.actors(actor._ray_actor_id.hex())
                if actor_state["State"] == "DEAD":
                    dead_actors.append(actor)
            if dead_actors:
                raise RuntimeError(
                    f"At least one prediction worker is dead. Dead workers: {dead_actors}. "
                    "Reinitializing worker group."
                )
    
    def stream_generate_texts(self, prompt: Union[Prompt, List[Prompt]]) -> Generator[str, None, None]: # type: ignore
        logger.info(f"GenericEngine.stream_generate_texts -> worker.length: {len(self.base_worker_group)}")
        worker0 = self.base_worker_group[0]
        for strHandle in worker0.worker_stream_generate_texts.remote(
            prompt,
            **self.args.model_config.generation.all_generate_kwargs if self.args.model_config.generation else {}
        ):
            val = ray.get(strHandle)
            logger.info(f"GenericEngine.stream_generate_texts -> yield -> {val}")
            yield val
