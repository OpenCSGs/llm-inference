import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, AsyncIterator

import torch
import uuid

from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import Response

from ...initializers.vllm import VllmInitializer
from ..utils import decode_stopping_sequences_where_needed, construct_prompts
from .generation import FinishReason
from .models import (
    VLLMSamplingParams,
)

import asyncio
from .._base import BasePipeline


logger = get_logger(__name__)


class VllmPipeline(BasePipeline):
    """Text generation pipeline using vllm.

    May not support all features."""

    def __init__(
        self,
        model: "AsyncLLMEngine",
        tokenizer: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> None:
        self.engine = model
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_format = prompt_format

        self.loop = asyncio.get_event_loop()


    def _parse_sampling_params(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> "SamplingParams":
        try:
            from vllm.sampling_params import SamplingParams
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            
        sampling_params = generate_kwargs.copy()
        try:
            if sampling_params.n != 1:
                raise ValueError("n>1 is not supported yet in aviary")
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
    
    def __call__(self, inputs: List[str], **kwargs) -> List[Response]:
        return self.loop.run_until_complete(self._generate(inputs, **kwargs))
    

    async def _generate(self, inputs: List[str], **kwargs) -> List[Response]:
        try:
            from vllm.outputs import RequestOutput
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        logger.info(inputs)
        inputs = construct_prompts(
            inputs, prompt_format=self.prompt_format)
        
        logger.info(inputs)

        sampling_params = VLLMSamplingParams.merge_generation_params(
            dict(), kwargs
        )

        st = time.monotonic()
        request_id = str(uuid.uuid4())
        # Construct a results generator from VLLM
        results_generator: AsyncIterator[RequestOutput] = self.engine.generate(
            inputs[0],
            self._parse_sampling_params(sampling_params),
            request_id,
        )

        responses = []
        try:
            async for request_output in results_generator:
                # TODO(pengli): handle more than one output
                if request_output.finished:
                    assert (
                        len(request_output.outputs) == 1
                    ), "Received more than 1 output from vllm, aborting"
                    gen_time = time.monotonic() - st
                    output = request_output.outputs[0]
                    text_output = output.text
                    num_text_returned = len(text_output)
                    num_input_tokens = len(request_output.prompt_token_ids)
                    finish_reason = FinishReason.from_vllm_finish_reason(
                        output.finish_reason
                    )

                    responses.append(
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
                    )
            logger.info(
                f"Request {request_id} finished ({finish_reason}). "
                f"Total time: {(gen_time)}s, "
            )
        finally:
            # Ensure that we cancel on the engine once we have exited the streaming
            # phase
            self.engine._abort(request_id)
            
        return responses

    def preprocess(self, prompts: List[str], **generate_kwargs):
        pass

    def forward(self, model_inputs, **generate_kwargs):
        pass
    
    @classmethod
    def from_initializer(
        cls,
        initializer: "VllmInitializer",
        model_id: str,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> "VllmPipeline":
        assert isinstance(initializer, VllmInitializer)
        logger.info(f"vllm initializer loading model: {model_id}")
        model, tokenizer = initializer.load(model_id)
        logger.info(f"vllm loaded model: {model}")
        return cls(
            model,
            tokenizer,
            device=device,
            **kwargs,
        )