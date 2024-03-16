import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, Iterator
from queue import Queue

import torch
from transformers import (
    LogitsProcessorList,
    MaxTimeCriteria,
    MinNewTokensLengthLogitsProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.pipelines.text_generation import ReturnType
from transformers.utils import ModelOutput

from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import Prompt, Response

from .processors import StopOnTokens
from .utils import tokenize_stopping_sequences_where_needed

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer

logger = get_logger(__name__)


class BasePipeline(ABC):
    """Stripped down version of Transformers pipeline.

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
    ) -> None:
        logger.info(f"BasePipeline init parameter device : {device}")
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_format: str = prompt_format or ""

        if device is None:
            # `accelerate` device map
            hf_device_map = getattr(self.model, "hf_device_map", None)
            if hf_device_map is not None:
                # Take the first device used by `accelerate`.
                device = next(iter(hf_device_map.values()))
            else:
                device = model.device

        logger.info(f"BasePipeline init set device : {device}")
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        logger.info(f"BasePipeline init device on {self.device}")

    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_id: str,
        prompt_format: Optional[str] = None,
        # device: Optional[Union[str, int, torch.device]] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> "BasePipeline":
        logger.info(
            f"BasePipeline initializer loading model {model_id} for device {device}")
        model, tokenizer = initializer.load(model_id)
        logger.info(
            f"BasePipeline loaded model {model_id} done for device {device}")
        if "task" in kwargs:
            del kwargs["task"]

        return cls(
            model,
            tokenizer,
            prompt_format=prompt_format,
            device=device,
            **kwargs,
        )

    def _get_stopping_criteria(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> StoppingCriteriaList:
        lst = []
        stopping_sequences = None
        if generate_kwargs.get("stopping_sequences", None) is not None:
            stopping_sequences = tokenize_stopping_sequences_where_needed(
                self.tokenizer, generate_kwargs["stopping_sequences"]
            )
        stopping_sequences = stopping_sequences or []
        stopping_sequences += [self.tokenizer.eos_token_id]
        lst.append(StopOnTokens(stopping_sequences))

        if generate_kwargs.get("max_time_criteria", None) is not None:
            max_time, initial_time = generate_kwargs.pop("max_time_criteria")
            lst.append(MaxTimeCriteria(max_time, initial_time))

        return StoppingCriteriaList(lst)

    def _get_logits_processors(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> LogitsProcessorList:
        lst = []
        stopping_sequences = None
        if generate_kwargs.get("stopping_sequences", None) is not None:
            stopping_sequences = tokenize_stopping_sequences_where_needed(
                self.tokenizer, generate_kwargs["stopping_sequences"]
            )

        if stopping_sequences and model_inputs is not None:
            min_new_tokens_stopping_sequences = []
            for sequence in stopping_sequences:
                if isinstance(sequence, list):
                    min_new_tokens_stopping_sequences.extend(sequence)
                else:
                    min_new_tokens_stopping_sequences.append(sequence)
            lst.append(
                MinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=model_inputs["inputs"]["input_ids"].shape[1],
                    min_new_tokens=generate_kwargs.pop("min_new_tokens", 4),
                    eos_token_id=min_new_tokens_stopping_sequences
                    + [self.tokenizer.eos_token_id],
                )
            )

        return LogitsProcessorList(lst)

    @abstractmethod
    def preprocess(self, prompts: List[str], **generate_kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, model_inputs, **generate_kwargs):
        raise NotImplementedError

    def postprocess(self, model_outputs, **generate_kwargs) -> List[Response]:
        return model_outputs

    @torch.inference_mode()
    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[Response]:
        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_inputs = self._ensure_tensor_on_device(
            model_inputs, device=self.device)
        
        forward_params = self._add_default_generate_kwargs(
            forward_params, model_inputs)
        logger.info(f"Forward params: {forward_params}")
        model_outputs = self.forward(model_inputs, **forward_params)
        model_outputs = self._ensure_tensor_on_device(
            model_outputs, device=torch.device("cpu")
        )

        outputs = self.postprocess(model_outputs, **postprocess_params)
        return [
            Response(generated_text=text) if isinstance(text, str) else text
            for text in outputs
        ]

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.

        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device: torch.device):
        if isinstance(inputs, ModelOutput):
            return ModelOutput(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, dict):
            return {
                name: self._ensure_tensor_on_device(tensor, device)
                for name, tensor in inputs.items()
            }
        elif isinstance(inputs, UserDict):
            return UserDict(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple(
                [self._ensure_tensor_on_device(item, device)
                 for item in inputs]
            )
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype in {
                torch.float16,
                torch.bfloat16,
            }:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def _add_default_generate_kwargs(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> Dict[str, Any]:
        stopping_criteria = self._get_stopping_criteria(
            generate_kwargs, model_inputs)
        if stopping_criteria:
            if generate_kwargs.get("stopping_criteria", None):
                generate_kwargs["stopping_criteria"].extend(stopping_criteria)
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    generate_kwargs["stopping_criteria"]
                )
            else:
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    stopping_criteria
                )

        logits_processor = self._get_logits_processors(
            generate_kwargs, model_inputs)
        if logits_processor:
            if generate_kwargs.get("logits_processor", None):
                generate_kwargs["logits_processor"].extend(logits_processor)
                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    generate_kwargs["logits_processor"]
                )
            else:
                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    logits_processor
                )

        generate_kwargs.pop("stopping_sequences", None)
        return generate_kwargs

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        # New llmserve arguments
        return_token_type_ids=None,
        return_attention_mask=None,
        stopping_sequences=None,
        add_special_tokens=None,
        timeout_s=None,
        start_timestamp=None,
        # Latest add arguments
        eos_token = None,
        pad_token = None,
        **generate_kwargs,
    ):
        preprocess_params = {}

        if return_token_type_ids is not None:
            preprocess_params["return_token_type_ids"] = return_token_type_ids

        if add_special_tokens is not None:
            preprocess_params["add_special_tokens"] = add_special_tokens

        if return_attention_mask is not None:
            preprocess_params["return_attention_mask"] = return_attention_mask
            
        if eos_token is not None:
            preprocess_params["eos_token"] = eos_token

        if pad_token is not None:
            preprocess_params["pad_token"] = pad_token

        # for StoppingCriteria
        if stopping_sequences is not None:
            generate_kwargs["stopping_sequences"] = stopping_sequences
        if timeout_s is not None and start_timestamp is not None:
            generate_kwargs["max_time_criteria"] = (timeout_s, start_timestamp)

        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError(
                    "`return_text` is mutually exclusive with `return_full_text`"
                )
            if return_tensors is not None:
                raise ValueError(
                    "`return_full_text` is mutually exclusive with `return_tensors`"
                )
            return_type = (
                ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
            )
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError(
                    "`return_text` is mutually exclusive with `return_tensors`"
                )
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params[
                "clean_up_tokenization_spaces"
            ] = clean_up_tokenization_spaces

        return preprocess_params, forward_params, postprocess_params


class StreamingPipeline(BasePipeline):
    def stream(
        self,
        inputs: List[str],
        queue: Queue,
        **kwargs,
    ) -> Iterator[List[Response]]:
        raise NotImplementedError()
