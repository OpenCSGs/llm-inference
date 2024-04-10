import time
from typing import List, Optional, Union, TYPE_CHECKING

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer

from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import Prompt, Response
import json

from ._base import StreamingPipeline
from .processors import StopOnTokens
from .utils import construct_prompts, truncate_to_first_stop_token

from typing import AsyncGenerator, Generator, Iterator
import asyncio
from transformers import TextIteratorStreamer
from threading import Thread
from queue import Empty

logger = get_logger(__name__)

class BatchTextIteratorStreamer(TextIteratorStreamer):
    def __init__(self, batch_size:int, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.batch_size = batch_size
        self.token_cache = [[] for _ in range(batch_size)]
        self.print_len = [0 for _ in range(batch_size)]
        self.generate_exception = None

    def put(self, value):
        if len(value.shape) != 2:
            value = torch.reshape(value, (self.batch_size, value.shape[0] // self.batch_size))

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        printable_texts = list()
        for idx in range(self.batch_size):
            self.token_cache[idx].extend(value[idx].tolist())
            text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)

            if text.endswith("\n"):
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
                # If the last token is a CJK character, we print the characters.
            elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
                printable_text = text[self.print_len[idx] :]
                self.print_len[idx] += len(printable_text)
            else:
                printable_text = text[self.print_len[idx] : text.rfind(" ") + 1]
                self.print_len[idx] += len(printable_text)
            printable_texts.append(printable_text)

        self.on_finalized_text(printable_texts)

    def end(self):
        printable_texts = list()
        for idx in range(self.batch_size):
            if len(self.token_cache[idx]) > 0:
                text = self.tokenizer.decode(self.token_cache[idx], **self.decode_kwargs)
                printable_text = text[self.print_len[idx] :]
                self.token_cache[idx] = []
                self.print_len[idx] = 0
            else:
                printable_text = ""
            printable_texts.append(printable_text)

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_texts, stream_end=True)

    def on_finalized_text(self, texts: List[str], stream_end: bool = False):
        self.text_queue.put(texts, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

class DefaultPipeline(StreamingPipeline):
    """Default text generation pipeline.

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
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            device=device,
        )

    def preprocess(self, prompts: List[str], **generate_kwargs):
        st = time.monotonic()

        prompt_text = construct_prompts(prompts, prompt_format=self.prompt_format)
        instruction_text = construct_prompts(prompts, prompt_format="")

        if generate_kwargs.get("eos_token", False):
            self.tokenizer.eos_token = generate_kwargs.get("eos_token")

        if generate_kwargs.get("pad_token", False):
            self.tokenizer.pad_token = generate_kwargs.get("pad_token")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            prompt_text_bak = prompt_text
            logger.info(f"call json.loads")
            prompt_text = [json.loads(prompt, strict=False) for prompt in prompt_text]
            logger.info(f"call tokenizer.apply_chat_template")
            prompt_text = [self.tokenizer.apply_chat_template(prompt_obj, tokenize=False, add_generation_prompt=True) for prompt_obj in prompt_text]
        except Exception as ex:
            logger.error(f"Exception apply_chat_template: {ex}")
            logger.info("Seems no chat template from user or the model donot has a 'chat template'")
            prompt_text = prompt_text_bak

        logger.info(f"Call model.generate with input: {prompt_text}")

        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens = generate_kwargs.get("add_special_tokens", True), padding=True
        ).to(self.model.device if hasattr(self.model, 'device') else self.device)

        if not generate_kwargs.get("return_token_type_ids", True):
            inputs.pop("token_type_ids", None)

        if not generate_kwargs.get("return_attention_mask", True):
            inputs.pop("attention_mask", None)

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
        
        batch_size = inputs["input_ids"].size(dim=0)
        logger.info(f"batch size is: {batch_size}")
        streamer = BatchTextIteratorStreamer(batch_size=batch_size, tokenizer= self.tokenizer,
                                        # timeout=0,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
            
        generation_kwargs = dict(inputs, streamer=streamer, **generate_kwargs)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        while True:
            try:
                for token in streamer:
                    et = time.monotonic() - st
                    st = et
                    if token:
                        yield {
                            "inputs": inputs,
                            "generated_sequence": [token],
                            "instruction_text": instruction_text,
                            "prompt_text": prompt_text,
                            "preprocessing_time": preprocessing_time,
                            "generation_time": et,
                            "generate_kwargs": generate_kwargs,
                        }
                break
            except Empty:
                asyncio.sleep(0.001)

    def postprocess(self, model_outputs, **postprocess_kwargs) -> List[Response]:
        st = time.monotonic()
        tokens = model_outputs["generated_sequence"]
        tokens = tokens[0]
        input_ids = model_outputs["inputs"]["input_ids"]

        decoded: List[Response] = []
        num_generated_tokens_batch = 0
        num_input_tokens_batch = 0
        for token_unwrapped, inputs_unwrapped in zip(tokens, input_ids):
            tokens = token_unwrapped

            for i in range(len(inputs_unwrapped)):
                if inputs_unwrapped[i] != self.tokenizer.pad_token_id:
                    break
            num_input_tokens = len(inputs_unwrapped[i:])
            num_generated_tokens = len(tokens)
            response = Response(
                generated_text=tokens,
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

    @torch.inference_mode()
    def stream(
        self,
        inputs: List[Union[str, Prompt]],
        **kwargs,
    ) -> Iterator[List[Response]]:
        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
        forward_params = self._add_default_generate_kwargs(forward_params, model_inputs)

        logger.info(
            f"Forward params: {forward_params}, batch size: {len(inputs)} model_inputs {model_inputs}"
        )
        for batch in self.forward(model_inputs, **forward_params):
            yield self.postprocess(batch)

    def __call__(
        self,
        inputs: List[Union[str, Prompt]],
        **kwargs,
    ) -> List[Response]:
        streams = [list() for _ in range(len(inputs))]
        for batch_response in self.stream(inputs, **kwargs):
            for i, response in enumerate(batch_response):
                streams[i].append(response)

        return [Response.merge_stream(*stream) for stream in streams]
    
