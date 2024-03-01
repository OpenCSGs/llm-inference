from typing import TYPE_CHECKING, List, Optional, Union

import torch
import time
from transformers import Pipeline as TransformersPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline

from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import Prompt, Response

from ._base import BasePipeline
from .utils import construct_prompts_experimental, truncate_to_first_stop_token
from llmserve.backend.server.utils import render_gradio_params
from .default_pipeline import DefaultPipeline

try:
    import transformers
    from transformers import pipelines
except ImportError as ie:
    raise ImportError(
        "transformers not installed. Please try `pip install transformers`"
    ) from ie

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer

logger = get_logger(__name__)


class DefaultTransformersPipeline(BasePipeline):
    """Text generation pipeline using Transformers Pipeline.

    May not support all features.

    Args:
        model (PreTrainedModel): Hugging Face model.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        prompt_format (Optional[str], optional): Prompt format. Defaults to None.
        device (Optional[Union[str, int, torch.device]], optional): Device to place model on. Defaults to model's
            device.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        # device: Optional[torch.device] = None,
        task: str = None,
    ) -> None:
        if not hasattr(model, "generate"):
            raise ValueError("Model must have a generate method.")
        
        logger.info(f"DefaultTransformersPipeline.init.device: {device}")
        super().__init__(model, tokenizer, prompt_format, device)

        self.pipeline = None
        self.preprocess = None
        self.postprocess = None

    def _get_transformers_pipeline(self, **kwargs) -> TransformersPipeline:
        logger.info(f"DefaultTransformersPipeline.device: {self.device}")
        default_kwargs = dict(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        transformers_pipe = pipeline(**{**default_kwargs, **kwargs})
        transformers_pipe.device = self.device
        return transformers_pipe

    @torch.inference_mode()
    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[Response]:
        if not self.pipeline:
            self.pipeline = self._get_transformers_pipeline()

        logger.info(f"input from pipeline: ****** {inputs}")
        inputs = construct_prompts_experimental(inputs, prompt_format=self.prompt_format)
        
        logger.info(f"input from pipeline: ****** {inputs}")

        preprocess_st = time.monotonic()
        if self.preprocess:
            data = self.preprocess(inputs)
        preprocess_time = time.monotonic() - preprocess_st

        kwargs.pop("stopping_sequences", None)
        kwargs.pop("timeout_s", None)
        kwargs.pop("start_timestamp", None)
        logger.info(f"input data: {data}")
        # special cases that needs to be handled differently

        generation_st = time.monotonic()
        logger.info(f"self.pipeline.device: {self.pipeline.device}")
        if isinstance(
            self.pipeline,
            (
                pipelines.text_classification.TextClassificationPipeline,
                pipelines.text2text_generation.Text2TextGenerationPipeline,
                pipelines.text2text_generation.TranslationPipeline,
            ),
        ):
            logger.info(f"TextClassificationPipeline|Text2TextGenerationPipeline|TranslationPipeline : {data}")
            logger.info(f"TextClassificationPipeline|Text2TextGenerationPipeline|TranslationPipeline : {kwargs}")
            data = self.pipeline(*data, **kwargs)
        # elif isinstance(
        #     self.pipeline,
        #     (transformers.pipelines.text_generation.TextGenerationPipeline)
        # ):
        #     textInputs = data['text_inputs']
        #     logger.info(f"Call TextGenerationPipeline : {textInputs}")
        #     logger.info(f"Call TextGenerationPipeline : {kwargs}")
        #     data = self.pipeline(text_inputs=textInputs, **kwargs)
        else:
            logger.info(f"Call {self.pipeline} : {data}")
            logger.info(f"Call {self.pipeline} : {kwargs}")
            data = self.pipeline(**data, **kwargs)
        generation_time = time.monotonic() - generation_st

        logger.info(f"output data from pipeline: ****** {data}")
        if self.postprocess:
            output = self.postprocess(data)
        output = self.format_output(data[0], inputs, preprocess_time, generation_time)

        return output

    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_id: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        # device: torch.device = None,
        stopping_sequences: List[Union[int, str]] = None,
        **kwargs,
    ) -> "DefaultTransformersPipeline":
        logger.info(f"DefaultTransformersPipeline.from_initializer on device {device} for {model_id}")
        model_from_pretrained_kwargs = initializer.get_model_from_pretrained_kwargs()
        default_kwargs = dict(
            model=model_id,
            # device=device,
            **kwargs,
            **model_from_pretrained_kwargs
        )
        model_kwargs = initializer.get_model_init_kwargs()
        logger.info(f"DefaultTransformersPipeline default_kwargs {default_kwargs}")
        logger.info(f"DefaultTransformersPipeline model_kwargs {model_kwargs}")
        transformers_pipe = pipeline(
            **default_kwargs,
            model_kwargs=model_kwargs,
        )
        # transformers_pipe.model = initializer.postprocess_model(transformers_pipe.model)
        pipe = cls(
            model=transformers_pipe.model,
            tokenizer=transformers_pipe.tokenizer,
            prompt_format=prompt_format,
            device=device,
            # stopping_sequences=stopping_sequences,
            **kwargs,
        )
        pipe.pipeline = transformers_pipe
        transformers_pipe.device = pipe.device
        logger.info(f"pipe.device: {pipe.device}, transformers_pipe.device: {transformers_pipe.device}")
        if "task" in kwargs:
            pipeline_info = render_gradio_params(kwargs["task"])
            pipe.preprocess = pipeline_info["preprocess"]
            pipe.postprocess = pipeline_info["postprocess"]

        return pipe

    def preprocess(self, prompts: List[str], **generate_kwargs):
        pass

    def format_output(self, model_outputs, inputs, preprocess_time, generation_time, **generate_kwargs) -> List[Response]:
        st = time.monotonic()
        decoded: List[Response] = []
        num_generated_tokens_batch = 0
        num_input_tokens_batch = 0
        for output in model_outputs:
            num_generated_tokens = len(self.tokenizer(output["generated_text"]).input_ids)
            num_input_tokens = len(self.tokenizer(inputs[0]).input_ids)
            response = Response(
                generated_text=output["generated_text"],
                num_generated_tokens=num_generated_tokens,
                num_input_tokens=num_input_tokens,
            )
            num_generated_tokens_batch += num_generated_tokens
            num_input_tokens_batch += num_input_tokens
            decoded.append(response)
        et = time.monotonic() - st
        for response in decoded:
            response.num_generated_tokens_batch = num_generated_tokens_batch
            response.num_input_tokens_batch = num_input_tokens_batch
            response.preprocessing_time = preprocess_time
            response.generation_time = generation_time
            response.postprocessing_time = et

        return decoded

    def forward(self, model_inputs, **generate_kwargs):
        pass
