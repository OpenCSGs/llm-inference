from typing import TYPE_CHECKING, List, Optional, Union
import copy

import torch
import time
import json
from transformers import Pipeline as TransformersPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline

from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import Prompt, Response

from ._base import BasePipeline
from .utils import construct_prompts
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

    @torch.inference_mode()
    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[Response]:
        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)
        forward_params = self._sanitize_gen_parameters(forward_params)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        self.pipeline.tokenizer = self.tokenizer
        if isinstance(self.pipeline, transformers.pipelines.text_generation.TextGenerationPipeline):
            forward_params = self._add_default_generate_kwargs(
                forward_params, model_inputs)     

        model_outputs = self.forward(model_inputs, **forward_params)

        outputs = self.postprocess(model_outputs, **postprocess_params)
        return [
            Response(generated_text=text) if isinstance(text, str) else text
            for text in outputs
        ]


    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_id: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> "DefaultTransformersPipeline":
        logger.info(
            f"DefaultTransformersPipeline.from_initializer on device {device} for {model_id}")
        model_from_pretrained_kwargs = initializer.get_model_from_pretrained_kwargs()
        default_kwargs = dict(
            model=model_id,
            **kwargs,
            **model_from_pretrained_kwargs
        )
        extral_kwargs = initializer.get_model_init_kwargs()
        logger.info(
            f"DefaultTransformersPipeline default_kwargs {default_kwargs}")
        logger.info(f"DefaultTransformersPipeline model_kwargs {extral_kwargs}")

        transformers_pipe = pipeline(
            **default_kwargs,
            **extral_kwargs,
        )

        # use initializer to handle "use_bettertransformer" and "torch_compile"
        transformers_pipe.model = initializer.postprocess_model(transformers_pipe.model)
        pipe = cls(
            model=transformers_pipe.model,
            tokenizer=transformers_pipe.tokenizer,
            prompt_format=prompt_format,
            device=device,
            **kwargs,
        )
        pipe.pipeline = transformers_pipe

        logger.info(
            f"pipe.device: {pipe.device}, transformers_pipe.device: {transformers_pipe.device}")
        if "task" in kwargs:
            pipeline_info = render_gradio_params(kwargs["task"])
            pipe.preprocess_extra = pipeline_info["preprocess"]
            pipe.postprocess_extra = pipeline_info["postprocess"]

        return pipe

    def preprocess(self, prompts: List[str], **generate_kwargs):
        # in preprocess, will:
        #   - reconfig the tokenizer
        #   - construct the prompt

        st = time.monotonic()
        inputs = None
        logger.info(f"input from pipeline: ****** {prompts}")
        prompt_text = construct_prompts(
            prompts, prompt_format=self.prompt_format)
        instruction_text = construct_prompts(prompts, prompt_format="")
        logger.info(f"input from pipeline: ****** {prompt_text}")   

        if isinstance(self.pipeline, transformers.pipelines.text_generation.TextGenerationPipeline):
            try:
                prompt_text_bak = prompt_text
                prompt_text = [json.loads(prompt, strict=False) for prompt in prompt_text]
                prompt_text = [self.tokenizer.apply_chat_template(prompt_obj, tokenize=False, add_generation_prompt=True) for prompt_obj in prompt_text]
            except:
                logger.info("Seems no chat template from user or the model donot has a 'chat template'")
                prompt_text = prompt_text_bak

            inputs = self.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens = generate_kwargs.get("add_special_tokens", True), padding=True
            )

            if generate_kwargs.get("eos_token", False):
                self.tokenizer.eos_token = generate_kwargs.get("eos_token")

            if generate_kwargs.get("pad_token", False):
                self.tokenizer.pad_token = generate_kwargs.get("pad_token")

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        if self.preprocess_extra:
            prompt_text = self.preprocess_extra(prompt_text)

        et = time.monotonic() - st
        return {
            "inputs": inputs,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "preprocessing_time": et,
        }

    def forward(self, model_inputs, **generate_kwargs):
        st = time.monotonic()
        inputs = model_inputs["inputs"]
        instruction_text = model_inputs["instruction_text"]
        prompt_text = model_inputs["prompt_text"]
        preprocessing_time = model_inputs["preprocessing_time"]
        logger.info(
            f"Call model.generate with generate_kwargs: {generate_kwargs}")
        
        logger.info(f"self.pipeline.device: {self.pipeline.device}")
        if isinstance(
            self.pipeline,
            (
                pipelines.text_classification.TextClassificationPipeline,
                pipelines.text2text_generation.Text2TextGenerationPipeline,
                pipelines.text2text_generation.TranslationPipeline,
            ),
        ):
            logger.info(
                f"TextClassificationPipeline|Text2TextGenerationPipeline|TranslationPipeline : {prompt_text}")
            logger.info(
                f"TextClassificationPipeline|Text2TextGenerationPipeline|TranslationPipeline : {generate_kwargs}")
            generated_sequence = self.pipeline(*prompt_text, **generate_kwargs)
        else:
            logger.info(f"Call {self.pipeline} : {prompt_text}")
            logger.info(f"Call {self.pipeline} : {generate_kwargs}")
            generated_sequence = self.pipeline(**prompt_text, **generate_kwargs)
        et = time.monotonic() - st

        return {
            "inputs": inputs,
            "generated_sequence": generated_sequence,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "preprocessing_time": preprocessing_time,
            "generation_time": et,
            "generate_kwargs": generate_kwargs,
        }

    def postprocess(self, model_outputs, **postprocess_kwargs) -> List[Response]:
        st = time.monotonic()
        generated_sequence = model_outputs["generated_sequence"]
        logger.info(generated_sequence)
        if self.postprocess_extra:
            generated_sequence = self.postprocess_extra(generated_sequence)

        logger.info(generated_sequence)
        if not isinstance(self.pipeline, transformers.pipelines.text_generation.TextGenerationPipeline):
            return generated_sequence

        inputs = model_outputs["prompt_text"]["text_inputs"]
        decoded: List[Response] = []
        num_generated_tokens_batch = 0
        num_input_tokens_batch = 0
        for index, output in enumerate(generated_sequence):
            num_generated_tokens = len(self.tokenizer(
                output).input_ids)
            num_input_tokens = len(self.tokenizer(inputs[index]))
            response = Response(
                generated_text=output[len(inputs[index]):],
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
            response.preprocessing_time = model_outputs["preprocessing_time"]
            response.generation_time = model_outputs["generation_time"]
            response.postprocessing_time = et

        return decoded


    def _sanitize_gen_parameters(
        self,
        generate_params: dict[str, str]
        ):
        generate_params = copy.deepcopy(generate_params)
        if "max_tokens" in generate_params:
            generate_params["max_new_tokens"] = generate_params["max_tokens"]
            generate_params.pop("max_tokens")
        
        return generate_params