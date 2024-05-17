
from .._base import LLMEngine
import asyncio
import torch
import gc
from typing import List, Optional, Any, Dict, List, Optional, AsyncIterator, Iterator
from ray.air import ScalingConfig
from ray.util.placement_group import PlacementGroup
from llmserve.backend.server.models import Args, Prompt, Response
from llmserve.backend.logger import get_logger
from .vllm_compatibility import AsyncLLMEngineRay
from llmserve.backend.llm.pipelines.utils import construct_prompts
from .models import VLLMSamplingParams
from .generation import FinishReason
import time
import uuid
import ray
from llmserve.backend.server.utils import render_gradio_params
import json

from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = get_logger(__name__)
@ray.remote(num_gpus=0, num_cpus=0)
class EngineWorker:
    _engine_cls = AsyncLLMEngineRay

    def __init__(
        self,
        args: Args,

    ):
        if not (args.scaling_config.num_gpus_per_worker > 0):
            raise ValueError("The VLLM Engine Requires > 0 GPUs to run.")
        self.running = False
        self.args = args

    async def launch_engine(
        self, 
        scaling_config: ScalingConfig,
        placement_group: PlacementGroup,
        scaling_options: dict,
    ) -> Any:
        if self.running:
            # The engine is already running!
            logger.info("Skipping engine restart because the engine is already running")
            return

        config: Args = self.args  # pylint:disable=no-member
        llm_config = config.model_conf
        runtime_env = llm_config.initialization.runtime_env or {}

        self.engine = self._engine_cls.from_llm_app(
                self.args,
                scaling_options,
                placement_group,
                runtime_env,
            )
        
        # warmp up
        model_task_info = render_gradio_params(llm_config.model_task)
        warmup_inputs = model_task_info["warmup"] if "warmup" in model_task_info else None
        warmup_success = False
        while not warmup_success and llm_config.warmup and warmup_inputs:
            prompt = [Prompt(prompt=warmup_inputs, use_prompt_format=False)]
            try:
                logger.info("start to test with single prompt")
                logger.info(f"warmpup prompt is: {warmup_inputs}")
                resp = await self.predict(
                    prompt,
                    [{}],
                    timeout_s=120,
                    start_timestamp=None,
                    lock=None  
                )
                logger.info(f"warmpup response is {str(resp)}")
                assert len(resp) > 0
                assert all(x.generated_text for x in resp)

                warmup_success = True
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"Warmup failed due to CUDA OOM")

        logger.info(
            f"Model {llm_config.model_id} succesfully initialized!")

        gc.collect()
        self.running = True

    def _parse_sampling_params(
        self, generate_kwargs: Dict[str, Any]
    ) -> "SamplingParams":
        sampling_params = generate_kwargs.copy()

        try:
            if sampling_params.n != 1:
                raise ValueError("n>1 is not supported yet in llm-inference")
            return SamplingParams(
                n=1,
                best_of=sampling_params.best_of,
                presence_penalty=sampling_params.presence_penalty
                if sampling_params.presence_penalty is not None
                else 0.0,
                frequency_penalty=sampling_params.frequency_penalty
                if sampling_params.frequency_penalty is not None
                else 0.0,
                temperature=sampling_params.temperature
                if sampling_params.temperature is not None
                else 1.0,
                top_p=sampling_params.top_p
                if sampling_params.top_p is not None
                else 1.0,
                top_k=sampling_params.top_k
                if sampling_params.top_k is not None
                else -1,
                use_beam_search=False,
                stop=sampling_params.stop,
                ignore_eos=False,
                # vLLM will cancel internally if input+output>max_tokens
                max_tokens=sampling_params.max_tokens,
                # or self.engine_config.max_total_tokens,
                logprobs=sampling_params.logprobs,
            )
        except Exception as e:
            # Wrap the error in ValidationError so the status code
            # returned to the user is correct.
            raise SystemError(str(e)) from e
        
    async def predict(
            self,
            prompts: List[Prompt],
            generate: dict[str, str] = {},
            *,
            timeout_s: float = 60,
            start_timestamp: Optional[float] = None,
            lock: asyncio.Lock,
        ) -> List[str]:
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        streams = [list() for _ in range(len(prompts))]
        async for batch_response in self.stream(prompts, generate, timeout_s=timeout_s, start_timestamp=start_timestamp, lock=lock):
            for i, response in enumerate(batch_response):
                streams[i].append(response)

        return [Response.merge_stream(*stream) for stream in streams]

    async def stream(
        self,
        prompts: List[Prompt],
        generate: dict[str, str] = {},
        *,
        timeout_s: float = 60,
        start_timestamp: Optional[float] = None,
        lock: asyncio.Lock,
    ) -> Iterator[List[Response]]:
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        if isinstance(prompts, ray.ObjectRef):
            prompts = ray.get(prompts)

        if isinstance(generate, ray.ObjectRef):
            generate = ray.get(generate)

        generate = generate[0]
        logger.info(f"get on fly params for generation: {generate}")

        prompt_format=(self.args.model_conf.generation.prompt_format if self.args.model_conf.generation else None)
        
        logger.info(f"Get prompt: {prompts}")
        logger.info(f"Get prompt format: {prompt_format}")
        inputs = construct_prompts(
            prompts, prompt_format=prompt_format)

        logger.info(f"Get input: {inputs}")
        if len(inputs) > 1:
            logger.warn("vllm cannot handle more than 1 prompt with one line engine, try 'LLMEngine' if you want try static batch")

        kwargs = self.args.model_conf.generation.all_generate_kwargs if self.args.model_conf.generation else {}
        logger.info(f"predefined generate params: {kwargs}")

        sampling_params = VLLMSamplingParams.merge_generation_params(
            kwargs, generate
        )

        logger.info(f"final generate params: {sampling_params}")
        request_id = str(uuid.uuid4())
        tokenizer = self.engine.engine.tokenizer
        prompt_text = inputs[0]

        try:
            prompt_text_bak = prompt_text
            prompt_text = json.loads(prompt_text, strict=False)
            prompt_text = tokenizer.apply_chat_template(prompt_text, tokenize=False, add_generation_prompt=True)
        except Exception as ex:
            logger.warn(f"Exception apply_chat_template: {ex}")
            logger.info("Seems no chat template from user or the model donot has a 'chat template'")
            prompt_text = prompt_text_bak

        logger.info(f"final prompt is: {prompt_text}")
        # Construct a results generator from VLLM
        st = time.monotonic()
        results_generator: AsyncIterator[RequestOutput] = self.engine.generate(
            prompt_text,
            self._parse_sampling_params(sampling_params),
            request_id,
        )

        index = 0
        try:
            async for request_output in results_generator:
                # TODO(pengli): handle more than one output
                # if request_output.finished:
                assert (
                    len(request_output.outputs) == 1
                ), "Received more than 1 output from vllm, aborting"
                gen_time = time.monotonic() - st
                output = request_output.outputs[0]
                text_output = output.text
                text_output = text_output[index:]
                index = len(output.text)
                num_text_returned = len(text_output)
                num_input_tokens = len(request_output.prompt_token_ids)
                finish_reason = FinishReason.from_vllm_finish_reason(
                    output.finish_reason
                )

                yield [
                    Response(
                        generated_text=text_output,
                        num_generated_tokens=1,
                        num_generated_tokens_batch=1,
                        num_input_tokens=num_input_tokens,
                        num_input_tokens_batch=num_input_tokens,
                        preprocessing_time=None,
                        postprocessing_time=None,
                        generation_time=gen_time,
                    )
                ]
                st = time.monotonic()
            logger.info(
                f"Request {request_id} finished ({finish_reason}). "
                f"Total time: {(gen_time)}s, "
            )
        finally:
            # Ensure that we cancel on the engine once we have exited the streaming
            # phase
            self.engine._abort(request_id)


class VllmEngine(LLMEngine):
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
        llm_config = config.model_conf
        runtime_env = llm_config.initialization.runtime_env or {}
        scaling_options.pop("num_cpus")
        scaling_options.pop("num_gpus")
        scaling_options.pop("resources")
        logger.info(f"Schedule engine launcher with: {scaling_options}")
        engine_worker_cls = EngineWorker.options(  # pylint:disable=no-member
            **scaling_options, runtime_env=runtime_env
        )
        
        # Create the prediction workers.
        logger.info("Creating vllm engine launcher...")
        worker = engine_worker_cls.remote(config)
        await asyncio.gather(
            *[
                worker.launch_engine.remote(
                    scaling_config=scaling_config,
                    placement_group=placement_group,
                    scaling_options=scaling_options
                )
            ]
        )

        self.base_worker_group = [worker]
        return [worker]

    async def check_health(self):
        logger.info("not implements yet...")

    async def predict(
            self,
            prompts: List[Prompt],
            generate: dict[str, str] = {},
            *,
            timeout_s: float = 60,
            start_timestamp: Optional[float] = None,
            lock: asyncio.Lock,
        ) -> List[str]:
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        prediction = (
            await asyncio.gather(
                *[
                    worker.predict.remote(
                        prompts,
                        generate,
                        timeout_s=timeout_s,
                        start_timestamp=start_timestamp,
                        lock=lock
                    )

                    for worker in self.base_worker_group
                ]
            )
        )[0]

        return prediction

    async def stream(
        self,
        prompts: List[Prompt],
        generate: dict[str, str] = {},
        *,
        timeout_s: float = 60,
        start_timestamp: Optional[float] = None,
        lock: asyncio.Lock,
    ) -> Iterator[List[Response]]:
        async with lock:

            tasks = [
                worker.stream.remote(
                    prompts,
                    generate,
                    timeout_s=timeout_s,
                    start_timestamp=start_timestamp,
                    lock=lock,
                )
                for worker in self.base_worker_group
            ]
            async for result in tasks[0]:
                yield await result
