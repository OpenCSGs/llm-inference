import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import torch

from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import Prompt, Response

from ...initializers.llamacpp import LlamaCppInitializer, LlamaCppTokenizer
from .._base import StreamingPipeline
from ..utils import decode_stopping_sequences_where_needed, construct_prompts
import json

if TYPE_CHECKING:
    from llama_cpp import Llama, LogitsProcessorList, StoppingCriteriaList

logger = get_logger(__name__)


class LlamaCppPipeline(StreamingPipeline):
    """Text generation pipeline using llama.cpp.

    May not support all features."""

    def __init__(
        self,
        model: "Llama",
        tokenizer: LlamaCppTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> None:
        from llama_cpp import Llama

        if not isinstance(model, Llama):
            raise TypeError("Model must be an instance of llama_cpp.Llama.")
        self.model = model
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_format = prompt_format

    def _get_logits_processors(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> "LogitsProcessorList":
        from llama_cpp import LogitsProcessorList

        from llmserve.backend.llm.pipelines.llamacpp.processors import (
            LlamaCppMinNewTokensLengthLogitsProcessor,
        )
        lst = []
        if "min_new_tokens" in generate_kwargs:
            lst.append(
                LlamaCppMinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=len(
                        model_inputs["tokenized_inputs"]),
                    min_new_tokens=generate_kwargs.pop("min_new_tokens", 4),
                    eos_token_id=self.model.token_eos(),
                )
            )

        return LogitsProcessorList(lst)

    def _get_stopping_criteria(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> "StoppingCriteriaList":
        from llama_cpp import StoppingCriteriaList

        from llmserve.backend.llm.pipelines.llamacpp.processors import (
            LlamaMaxTimeCriteria,
        )

        lst = []

        timeout_s = generate_kwargs.pop("timeout_s", None)
        start_timestamp = generate_kwargs.pop("start_timestamp", None)
        if timeout_s is not None and start_timestamp is not None:
            lst.append(LlamaMaxTimeCriteria(timeout_s, start_timestamp))

        return StoppingCriteriaList(lst)

    def _add_default_generate_kwargs(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> Dict[str, Any]:
        generate_kwargs = generate_kwargs.copy()
        generate_kwargs.setdefault("echo", False)
        stopping_sequences = generate_kwargs.pop("stopping_sequences")
        stopping_sequences = decode_stopping_sequences_where_needed(
            self.tokenizer, stopping_sequences
        )
        generate_kwargs.setdefault("stop", stopping_sequences)
        generate_kwargs["logits_processor"] = self._get_logits_processors(
            generate_kwargs, model_inputs=model_inputs
        )
        generate_kwargs["stopping_criteria"] = self._get_stopping_criteria(
            generate_kwargs, model_inputs=model_inputs
        )
        return generate_kwargs

    def __call__(self, inputs: List[str], **kwargs) -> List[Response]:
        streams = [list() for _ in range(len(inputs))]
        for batch_response in self.stream(inputs, **kwargs):
            for i, response in enumerate(batch_response):
                streams[i].append(response)

        return [Response.merge_stream(*stream) for stream in streams]

    def preprocess(self, prompts: List[str], **generate_kwargs):
        pass

    def forward(self, model_inputs, **generate_kwargs):
        pass

    @classmethod
    def from_initializer(
        cls,
        initializer: "LlamaCppInitializer",
        model_id: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> "LlamaCppPipeline":
        assert isinstance(initializer, LlamaCppInitializer)
        logger.info(f"LlamaCppPipeline initializer loading model: {model_id}")
        model, tokenizer = initializer.load(model_id)
        logger.info(f"LlamaCppPipeline loaded model: {model}")
        return cls(
            model,
            tokenizer,
            prompt_format,
            device=device,
            **kwargs,
        )

    def stream(
        self,
        inputs: List[Union[str, Prompt]],
        **kwargs,
    ) -> Iterator[List[Response]]:
        logger.info(f"stream prompt: {inputs}")
        inputs = construct_prompts(inputs, prompt_format=self.prompt_format)
        logger.info(f"stream inputs: {inputs}")

        tokenized_inputs = self.tokenizer.encode(inputs)
        kwargs = self._add_default_generate_kwargs(
            kwargs,
            model_inputs={"inputs": inputs,
                          "tokenized_inputs": tokenized_inputs},
        )

        chat_completion = False
        try:
            inputs_bak = inputs
            inputs = [json.loads(prompt, strict=False) for prompt in inputs]
            chat_completion = True
        except Exception as ex:
            logger.error(f"Exception apply_chat_template: {ex}")
            logger.info("Seems no chat template from user")
            inputs = inputs_bak
            
        logger.info(f"stream generate_kwargs: {kwargs}")
        logger.info(f"model inputs: {inputs}")

        kwargs.pop('start_timestamp', None)
        kwargs.pop('timeout_s', None)
        batch_size = len(inputs)
        for idx, input in enumerate(inputs):
            tokenized_inputs = self.tokenizer.encode(input)
            if chat_completion:
                # kwargs.pop('stopping_sequences', None)
                kwargs.pop('stopping_criteria', None)
                kwargs.pop('echo', None)
                logger.info(f"chat generate_kwargs: {kwargs}")
                st = time.monotonic()
                output = self.model.create_chat_completion(messages=input, stream=True, **kwargs)
                for chunk in output:
                    gen_time = time.monotonic() - st
                    delta = chunk['choices'][0]['delta']
                    
                    val = ''
                    if 'role' in delta:
                        val = ''
                    elif 'content' in delta:
                        val = delta['content']
                    # logger.info(f'LlamaCppPipeline -> create_chat_completion -> Yield -> "{val}"')
                    if val:
                        yield [
                            Response(
                                generated_text=val,
                                num_generated_tokens=1,
                                num_input_tokens=len(tokenized_inputs),
                                num_generated_tokens_batch=1,
                                num_input_tokens_batch=len(tokenized_inputs),
                                preprocessing_time=None,
                                postprocessing_time=None,
                                generation_time=gen_time,
                            )
                            if i == idx else 
                            Response(
                                generated_text="",
                                num_generated_tokens=0,
                                num_input_tokens=len(tokenized_inputs),
                                num_generated_tokens_batch=0,
                                num_input_tokens_batch=len(tokenized_inputs),
                                preprocessing_time=None,
                                postprocessing_time=None,
                                generation_time=gen_time,
                            )
                            for i in range(batch_size)   
                        ]
                        st = time.monotonic()
            else:
                logger.info(f"generate_kwargs: {kwargs}")
                st = time.monotonic()
                output = self.model(input, stream=True, **kwargs)
                for token in output:
                    gen_time = time.monotonic() - st
                    chunk = token["choices"][0]["text"].replace("\u200b", "")
                    # logger.info(f'LlamaCppPipeline -> generate -> Yield -> "{chunk}"')
                    # if chunk:
                    yield [
                        Response(
                            generated_text=chunk,
                            num_generated_tokens=1,
                            num_input_tokens=len(tokenized_inputs),
                            num_generated_tokens_batch=1,
                            num_input_tokens_batch=len(tokenized_inputs),
                            preprocessing_time=None,
                            postprocessing_time=None,
                            generation_time=gen_time,
                        )
                        if i == idx else 
                        Response(
                            generated_text="",
                            num_generated_tokens=0,
                            num_input_tokens=len(tokenized_inputs),
                            num_generated_tokens_batch=0,
                            num_input_tokens_batch=len(tokenized_inputs),
                            preprocessing_time=None,
                            postprocessing_time=None,
                            generation_time=gen_time,
                        )
                        for i in range(batch_size)   
                    ]
                    st = time.monotonic()


    def _sanitize_gen_parameters(
        self,
        generate_params: dict[str, str]
        ):
        pass