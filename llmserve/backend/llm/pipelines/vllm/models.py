from typing import Any, Dict, List, Optional, Set, TypeVar, Type
import os
from pydantic import BaseModel, validator
from .error_handling import TooManyStoppingSequences
from llmserve.backend.server.models import BaseModelExtended

ModelT = TypeVar("ModelT", bound=BaseModel)
MAX_NUM_STOPPING_SEQUENCES = os.getenv("MAX_NUM_STOPPING_SEQUENCES", 8)
class SamplingParams(BaseModelExtended):
    """
    Args:
        max_tokens: The maximum number of tokens to generate. Defaults to inf.
        temperature: What sampling temperature to use.
        top_p: An alternative to sampling with temperature, called nucleus sampling.
        logprobs: Include the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens.
        stop: Up to 4 sequences where the API will stop generating further tokens.
            The returned text will not contain the stop sequence. List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        presence_penalty: Number between -2.0 and 2.0.
            Positive values penalize new tokens based on whether they appear in
            the text so far, increasing the model's likelihood to talk about
            new topics. Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
            new tokens based on their existing frequency in the text so far,
            decreasing the model's likelihood to repeat the same line verbatim.
            Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        n: How many completions to generate for each prompt.
        best_of: Generates `best_of` completions server-side and returns the "best".
        ?logit_bias: Modify the likelihood of specified tokens appearing in
            the completion.
    """

    _ignored_fields: Set[str] = set()

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    logprobs: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: int = 1

    def dict(self, **kwargs):
        if kwargs.get("exclude", None) is None:
            kwargs["exclude"] = self._ignored_fields
        return super().dict(**kwargs)

    @validator("stop", always=True)
    def validate_stopping_sequences(cls, values):
        if not values:
            return values

        unique_val = sorted(list(set(values)))

        if len(unique_val) > MAX_NUM_STOPPING_SEQUENCES:
            TooManyStoppingSequences(
                len(unique_val), MAX_NUM_STOPPING_SEQUENCES
            ).raise_exception()

        return unique_val

    @classmethod
    def merge_generation_params(
        cls: Type[ModelT], prompt: Any, kwargs: dict
    ) -> ModelT:
        # Extract parameters object from prompt
        # parameters = prompt.parameters or {}
        # if not isinstance(parameters, dict):
        #     parameters = parameters.dict()
        def _merge_dicts(overwrite: dict, base: dict) -> dict:
            """
            Merge two dictionaries recursively, with keys from overwrite taking precedence.
            """
            base = base.copy()
            # for key, value in overwrite.items():
            #     if isinstance(value, dict):
            #         # get node or create one
            #         node = base.setdefault(key, {})
            #         _merge_dicts(value, node)
            #     else:
            #         base[key] = value

            return base
        
        # Merge in the generate kwargs
        generate_kwargs = _merge_dicts(
            [],
            kwargs,
        )

        # The stoppping sequence needs to be merged manually
        # generate_kwargs["stop"] = (parameters.get("stop") or []) + (
        #     generation.stopping_sequences or []
        # )
        # generate_kwargs["stop"] = (
        #     kwargs.stopping_sequences or []
        # )
        generate_kwargs["stop"] = []
        return cls.parse_obj(generate_kwargs)

class VLLMSamplingParams(SamplingParams):
    """
    Args:
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    """

    _ignored_fields = {"best_of", "n", "logit_bias", "logprobs"}

    top_k: Optional[int] = None