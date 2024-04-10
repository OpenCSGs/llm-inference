import asyncio
import gc
from typing import List, Optional

import ray
import ray.util
from ray.air import ScalingConfig
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from llmserve.backend.llm.engines import get_engine_cls_by_name
from llmserve.backend.llm.utils import (
    initialize_node,
)
from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import Args, Prompt

from typing import AsyncGenerator, Generator, Union

initialize_node_remote = ray.remote(initialize_node)
logger = get_logger(__name__)


class LLMPredictor:
    """Predictor for LLM models."""

    def __init__(self) -> None:
        logger.info('LLM Predictor Initialize')
        self.base_worker_group = None
        self.new_worker_group = None
        self._base_worker_group_lock = asyncio.Lock()
        self._new_worker_group_lock = asyncio.Lock()


    async def rollover(self, scaling_config: ScalingConfig, pg_timeout_s: float = 600):
        """Roll over to a new worker group.

        The new worker group is created asynchronously and the old worker group
        is replaced with the new worker group once it is ready.

        Args:
            scaling_config (ScalingConfig): Scaling configuration for the new worker group.
            pg_timeout_s (float, optional): Timeout for the new worker group to be ready. Defaults to 600.
        """
        if self._new_worker_group_lock.locked():
            logger.info("Rollover already in progress")
            return
        async with self._new_worker_group_lock:
            logger.info(f"Initializing new worker group {scaling_config}")

            # init engine here
            config: Args = self.args  # pylint:disable=no-member
            llm_config = config.model_config
            initializer_name = llm_config.initialization.initializer
            if not isinstance(initializer_name, str):
                initializer_name = initializer_name.type
            
            engine_name = ("vllm" if initializer_name == "Vllm" else "generic")
                
            logger.info(f"Engine name is {engine_name}")
            self.engine = get_engine_cls_by_name(engine_name)(
                args = self.args
            )

            self.new_worker_group = await self._create_worker_group(
                scaling_config, pg_timeout_s=pg_timeout_s
            )
            async with self._base_worker_group_lock:
                logger.info(
                    f"Rolling over to new worker group {self.new_worker_group}")
                self.base_worker_group = self.new_worker_group
                self.new_worker_group = None
            gc.collect()

    async def _create_worker_group(
        self, scaling_config: ScalingConfig, pg_timeout_s: float = 600
    ) -> List[ray.ObjectRef]:
        """Create a new worker group.

        Args:
            scaling_config (ScalingConfig): Scaling configuration for the new worker group.
            pg_timeout_s (float, optional): Timeout for the new worker group to be ready. Defaults to 600.
        """
        logger.info("LLM Predictor creating a new worker group")
        gc.collect()

        config: Args = self.args  # pylint:disable=no-member
        llm_config = config.model_config

        # Start a placement group for the workers.
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        scaling_options = dict(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            resources=scaling_config.additional_resources_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
        )
        runtime_env = llm_config.initialization.runtime_env or {}
        logger.info("Build Prediction Worker with runtime_env:")
        logger.info(llm_config.initialization.runtime_env)
        # prediction_worker_cls = PredictionWorker.options(  # pylint:disable=no-member
        #     **scaling_options, runtime_env=runtime_env
        # )
        initialize_node_remote_pg = initialize_node_remote.options(
            **scaling_options, runtime_env=runtime_env
        )

        logger.info("Waiting for placement group to be ready...")
        # This will raise a timeout error.
        await asyncio.wait_for(self.pg.ready(), timeout=pg_timeout_s)

        logger.info("Starting initialize_node tasks...")
        await asyncio.gather(
            *[
                initialize_node_remote_pg.remote(
                    llm_config.model_id,
                    llm_config.initialization.s3_mirror_config,
                )
                for i in range(scaling_config.num_workers)
            ]
        )

        # Create the prediction workers.
        # logger.info("Creating prediction workers...")
        # worker_group = [
        #     prediction_worker_cls.remote(
        #         llm_config, scaling_config.num_workers)
        #     for i in range(scaling_config.num_workers)
        # ]

        # logger.info("Initializing torch_dist process group on workers...")
        # # Initialize torch distributed process group for the workers.
        # local_ranks = await init_torch_dist_process_group_async(
        #     worker_group,
        #     backend="nccl" if scaling_config.use_gpu else "gloo",
        # )

        # # Initialize model on each worker.
        # logger.info(
        #     f"Initializing model on workers with local_ranks: {local_ranks}")
        # await asyncio.gather(
        #     *[
        #         worker.init_model.remote(
        #             local_rank = local_rank,
        #             num_cpus_per_worker=scaling_config.num_cpus_per_worker,
        #             num_gpus_per_worker=scaling_config.num_gpus_per_worker
        #         )
        #         for worker, local_rank in zip(worker_group, local_ranks)
        #         # for worker in worker_group
        #     ]
        # )

        engine = await self.engine.launch_engine(scaling_config, self.pg, scaling_options)
        return engine

    async def _predict_async(
        self,
        prompts: List[Prompt],
        *,
        timeout_s: float = 60,
        start_timestamp: Optional[float] = None,
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
        prediction = await self.engine.predict(prompts, timeout_s=timeout_s, start_timestamp=start_timestamp, lock=self._base_worker_group_lock)
        return prediction

    async def _stream_async(
        self,
        prompts: List[Prompt],
        *,
        timeout_s: float = 60,
        start_timestamp: Optional[float] = None,
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
        async for s in self.engine.stream(prompts, timeout_s=timeout_s, start_timestamp=start_timestamp, lock=self._base_worker_group_lock):
            yield s
    
    # Called by Serve to check the replica's health.
    async def check_health(self):
        self.engine.check_health()