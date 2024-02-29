from typing import List, Union, Tuple

import torch
from transformers import PreTrainedTokenizer

from llmserve.backend.server.models import Prompt


def tokenize_string(tokenizer: PreTrainedTokenizer, key: str) -> Union[int, List[int]]:
    """Tokenize a string using a tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        key (str): String to tokenize.
    """
    token_ids = tokenizer.encode(key, add_special_tokens=False)
    return token_ids[0] if len(token_ids) == 1 else token_ids


def decode_tokens(tokenizer: PreTrainedTokenizer, tokens: Union[int, List[int]]) -> str:
    tokens = tokens if isinstance(tokens, list) else [tokens]
    text = tokenizer.decode(tokens)
    return text


def truncate_to_first_stop_token(
    tokens: torch.LongTensor,
    stop_ids: List[Union[int, List[int]]],
) -> torch.LongTensor:
    """Truncate tokens up to the first stop_id.

    Args:
        tokens (torch.LongTensor): Tokens to truncate.
        stop_ids (List[Union[int, List[int]]]): Stop ids to truncate at. Can be
            composed of single stop ids or sequences of ids.
    """
    if not stop_ids:
        return tokens
    stop_ids: List[torch.LongTensor] = [
        torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
        for stop_id in stop_ids
    ]
    for i in range(len(tokens)):
        for stop_id_index, _ in enumerate(stop_ids):
            stop_id = stop_ids[stop_id_index].to(tokens.device)
            if len(tokens) - i >= len(stop_id) and tokens[i : len(stop_id) + i].equal(
                stop_id
            ):
                return tokens[:i]
    return tokens



def _construct_prompt(prompt: Union[str, Prompt], prompt_format: str) -> str:
    if isinstance(prompt, Prompt):
        if prompt.use_prompt_format and prompt_format:
            return prompt_format.format(instruction=prompt.prompt)
        else:
            return prompt.prompt
    return prompt_format.format(instruction=prompt) if prompt_format else prompt

def construct_prompts(
    prompts: Union[str, Prompt, List[str], List[Prompt], Tuple[str]],
    prompt_format: str,
) -> List[str]:
    """Construct prompts from a prompt string or list of prompts."""
    if not isinstance(prompts, list):
        prompts = [prompts]
    return [_construct_prompt(prompt, prompt_format) for prompt in prompts]

def construct_prompts_experimental(
    prompts: Union[str, Prompt, List[str], List[Prompt], Tuple[str]],
    prompt_format: str,
) -> List[str]:
    """Construct prompts from a prompt string or list of prompts."""
    if not isinstance(prompts, list):
        prompts = [prompts]
    
    params = []
    for prompt in prompts:
        if isinstance(prompt, Prompt) and isinstance(prompt.prompt, Tuple):
            params += [_construct_prompt(prompt, prompt_format) for prompt in prompt.prompt]
        else:
            params.append(_construct_prompt(prompt, prompt_format)) 
    return params


def tokenize_stopping_sequences_where_needed(
    tokenizer: PreTrainedTokenizer,
    stopping_sequences: List[Union[str, int, List[int]]],
) -> List[Union[List[int], int]]:
    """If any sequence is a string, tokenize it.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        stopping_sequences (List[Union[str, int, List[int]]]): Stopping sequences to
            tokenize. Can be ids, sequences of ids or strings.
    """
    if not stopping_sequences:
        return None
    return [
        tokenize_string(tokenizer, sequence) if isinstance(sequence, str) else sequence
        for sequence in stopping_sequences
    ]


def decode_stopping_sequences_where_needed(
    tokenizer: PreTrainedTokenizer,
    stopping_sequences: List[Union[str, int, List[int]]],
) -> List[str]:
    """If any sequence is a string, tokenize it."""
    if not stopping_sequences:
        return None
    return [
        decode_tokens(tokenizer, sequence)
        if not isinstance(sequence, str)
        else sequence
        for sequence in stopping_sequences
    ]
