import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
)
import transformers
from llmserve.backend.logger import get_logger

from .._base import LLMInitializer

logger = get_logger(__name__)


class TransformersInitializer(LLMInitializer):
    """Initialize model and tokenizer and place them on the correct device.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for ``torch.compile``. Defaults to None.
        **from_pretrained_kwargs: Keyword arguments for ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        **from_pretrained_kwargs,
    ):
        self.device = device
        self.world_size = world_size
        self.dtype = dtype
        self.from_pretrained_kwargs = from_pretrained_kwargs
        self.use_bettertransformer = use_bettertransformer
        self.torch_compile = torch_compile

    def _get_model_from_pretrained_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for AutoModel.from_pretrained."""
        return self.from_pretrained_kwargs

    def load(self, model_id: str) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
        """Load model and tokenizer.

        Args:
            model_id (str): Hugging Face model ID.
        """
        logger.info(f"TransformersInitializer begin load model {model_id}")
        model = self.load_model(model_id)
        logger.info(f"TransformersInitializer begin load tokenizer {model_id}")
        tokenizer = self.load_tokenizer(model_id)
        return self.postprocess_model(model), self.postprocess_tokenizer(tokenizer)

    def _get_model_location_on_disk(self, model_id: str) -> str:
        """Get the location of the model on disk.

        Args:
            model_id (str): Hugging Face model ID.
        """
        from transformers.utils.hub import TRANSFORMERS_CACHE

        path = os.path.expanduser(
            os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")
        )
        model_id_or_path = model_id

        if os.path.exists(path):
            with open(os.path.join(path, "refs", "main"), "r") as f:
                snapshot_hash = f.read().strip()
            if os.path.exists(
                os.path.join(path, "snapshots", snapshot_hash)
            ) and os.path.exists(
                os.path.join(path, "snapshots", snapshot_hash, "config.json")
            ):
                model_id_or_path = os.path.join(path, "snapshots", snapshot_hash)
        return model_id_or_path

    def load_model(self, model_id: str, loader: transformers.AutoModel = AutoModelForCausalLM) -> "PreTrainedModel":
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        model_id_or_path = self._get_model_location_on_disk(model_id)
        from_pretrained_kwargs = self._get_model_from_pretrained_kwargs()

        logger.info(f"TransformersInitializer: load model from_pretrained_kwargs {from_pretrained_kwargs}")
        try:
            # logger.info(f"TransformersInitializer loader {loader}")
            logger.info(f"TransformersInitializer: Loading model {model_id_or_path} by AutoModelForCausalLM")
            model = loader.from_pretrained(model_id_or_path, **from_pretrained_kwargs)
            logger.info(f"TransformersInitializer: Load model {model_id_or_path} done")
        except OSError:
            if model_id_or_path != model_id:
                logger.warning(
                    f"Couldn't load model from derived path {model_id_or_path}, "
                    f"trying to load from model_id {model_id}"
                )
                model = loader.from_pretrained(model_id, **from_pretrained_kwargs)
            else:
                raise
        model.eval()
        return model
        
    def load_tokenizer(self, tokenizer_id: str) -> "PreTrainedTokenizer":
        """Load tokenizer.

        Args:
            tokenizer_id (str): Hugging Face tokenizer name.
        """
        tokenizer_id_or_path = self._get_model_location_on_disk(tokenizer_id)
        from_pretrained_kwargs = self._get_model_from_pretrained_kwargs()
        
        logger.info(f"TransformersInitializer: load tokenizer from_pretrained_kwargs {from_pretrained_kwargs}")
        param_trust_remote_code = from_pretrained_kwargs.get("trust_remote_code")
        # TODO make this more robust
        try:
            logger.info(f"TransformersInitializer: Loading tokenizer by AutoTokenizer from {tokenizer_id_or_path}, trust_remote_code={param_trust_remote_code}")
            return AutoTokenizer.from_pretrained(tokenizer_id_or_path, trust_remote_code=param_trust_remote_code)
            # return AutoTokenizer.from_pretrained(
            #     tokenizer_id_or_path,
            #     # padding_side="left", #TODO have no idea why
            #     trust_remote_code=from_pretrained_kwargs.get(
            #         "trust_remote_code", False
            #     ),
            # )
        except Exception:
            logger.warning(
                f"Couldn't load tokenizer from derived path {tokenizer_id_or_path}, "
                f"trying to load from model_id {tokenizer_id}"
            )
            return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=param_trust_remote_code)
            # return AutoTokenizer.from_pretrained(
            #     tokenizer_id,
            #     # padding_side="left",
            #     trust_remote_code=from_pretrained_kwargs.get(
            #         "trust_remote_code", False
            #     ),
            # )

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """Postprocess model.

        First, transform the model with BetterTransformer if use_bettertransformer is True.
        Then, compile the model with torch.compile() if torch_compile is not None, using
        the provided parameters.

        Args:
            model (PreTrainedModel): Model to postprocess.
        """
        if self.use_bettertransformer:
            from optimum.bettertransformer import BetterTransformer

            logger.info("Transforming the model with BetterTransformer...")
            model = BetterTransformer.transform(model)

        if self.torch_compile and self.torch_compile["backend"]:
            logger.info("Compiling the model with torch.compile()...")
            model = torch.compile(model, **self.torch_compile)

        return model

    def postprocess_tokenizer(
        self, tokenizer: "PreTrainedTokenizer"
    ) -> "PreTrainedTokenizer":
        """Postprocess tokenizer.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to postprocess.
        """
        return tokenizer

class FinetuneInitializer(TransformersInitializer):
    """Initialize model and tokenizer and place them on the correct device(s).

    Uses Hugging Face Transformer's ``device_map`` argument.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for torch.compile. Defaults to None.
        device_map (str, optional): Device map to use (same as in AutoModel.from_pretrained). Defaults to "auto".
        **from_pretrained_kwargs: Keyword arguments for AutoModel.from_pretrained.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        **from_pretrained_kwargs,
    ):
        logger.info("FinetuneInitializer init")
        super().__init__(
            device=device,
            world_size=world_size,
            dtype=dtype,
            use_bettertransformer=use_bettertransformer,
            torch_compile=torch_compile,
            **from_pretrained_kwargs,
        )

    def load_model(self, model_id: str, loader: transformers.AutoModel = AutoModel, **additional_load_kwargs) -> "PreTrainedModel":
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        model_id_or_path = self._get_model_location_on_disk(model_id)
        from_pretrained_kwargs = self._get_model_from_pretrained_kwargs()

        if additional_load_kwargs:
            from_pretrained_kwargs = {
                **from_pretrained_kwargs,
                **additional_load_kwargs,
            }

        logger.info(f"FinetuneInitializer Loading model from {model_id_or_path}")
        logger.info(f"FinetuneInitializer from_pretrained_kwargs {from_pretrained_kwargs}")

        try:
            model = loader.from_pretrained(
                model_id_or_path, **from_pretrained_kwargs
            )
        except OSError:
            if model_id_or_path != model_id:
                logger.warning(
                    f"Couldn't load model from derived path {model_id_or_path}, "
                    f"trying to load from model_id {model_id}"
                )
                model = loader.from_pretrained(
                    model_id, **from_pretrained_kwargs
                )
            else:
                raise
        model.eval()
        return model
    
class DeviceMapInitializer(TransformersInitializer):
    """Initialize model and tokenizer and place them on the correct device(s).

    Uses Hugging Face Transformer's ``device_map`` argument.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for torch.compile. Defaults to None.
        device_map (str, optional): Device map to use (same as in AutoModel.from_pretrained). Defaults to "auto".
        **from_pretrained_kwargs: Keyword arguments for AutoModel.from_pretrained.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        device_map: str = "auto",
        **from_pretrained_kwargs,
    ):
        super().__init__(
            device=device,
            world_size=world_size,
            dtype=dtype,
            use_bettertransformer=use_bettertransformer,
            torch_compile=torch_compile,
            **from_pretrained_kwargs,
        )
        self.device_map = device_map

    def _get_model_from_pretrained_kwargs(self):
        return dict(
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            **self.from_pretrained_kwargs,
        )
    def get_model_from_pretrained_kwargs(self):
        return self._get_model_from_pretrained_kwargs()


class SingleDeviceInitializer(TransformersInitializer):
    """Initialize model and tokenizer and place them on the correct device.

    Uses Hugging Face Transformer's ``device`` argument.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for ``torch.compile``. Defaults to None.
        **from_pretrained_kwargs: Keyword arguments for ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        **from_pretrained_kwargs,
    ):
        super().__init__(
            device=device,
            world_size=world_size,
            dtype=dtype,
            use_bettertransformer=use_bettertransformer,
            torch_compile=torch_compile,
            **from_pretrained_kwargs,
        )

    def _get_model_from_pretrained_kwargs(self):
        return dict(
            # low_cpu_mem_usage=True,   //should move to config yaml file of mode
            torch_dtype=self.dtype,
            **self.from_pretrained_kwargs,
        )

    def get_model_from_pretrained_kwargs(self):
        return self._get_model_from_pretrained_kwargs()

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        logger.info(f"SingleDeviceInitializer postprocess_model to device {self.device}")
        return super().postprocess_model(model.to(device=self.device))


class TransformersPipelineInitializer(LLMInitializer):
    """Initialize model and tokenizer and place them on the correct device.

    actually do nothing, need further research TODO

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for ``torch.compile``. Defaults to None.
        **from_pretrained_kwargs: Keyword arguments for ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        **from_pretrained_kwargs,
    ):
        self.device = device
        self.world_size = world_size
        self.dtype = dtype
        self.from_pretrained_kwargs = from_pretrained_kwargs
        self.use_bettertransformer = use_bettertransformer
        self.torch_compile = torch_compile

    def load_model(self, model_id: str) -> "PreTrainedModel":
        pass

    def load_tokenizer(self, tokenizer_id: str) -> "PreTrainedTokenizer":
        pass

    def _get_model_from_pretrained_kwargs(self):
        return dict(
            # low_cpu_mem_usage=True,   //should move to config yaml file of mode
            torch_dtype=self.dtype,
            **self.from_pretrained_kwargs,
        )

    def get_model_from_pretrained_kwargs(self):
        return self._get_model_from_pretrained_kwargs()



# should be remove soon
class AutoModelInitializer(TransformersInitializer):
    """Initialize model and tokenizer and place them on the correct device(s).

    Uses Hugging Face Transformer's ``device_map`` argument.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for torch.compile. Defaults to None.
        device_map (str, optional): Device map to use (same as in AutoModel.from_pretrained). Defaults to "auto".
        **from_pretrained_kwargs: Keyword arguments for AutoModel.from_pretrained.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        **from_pretrained_kwargs,
    ):
        super().__init__(
            device=device,
            world_size=world_size,
            dtype=dtype,
            use_bettertransformer=use_bettertransformer,
            torch_compile=torch_compile,
            **from_pretrained_kwargs,
        )

    def load_model(self, model_id: str, loader: transformers.AutoModel = AutoModel, **additional_load_kwargs) -> "PreTrainedModel":
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        model_id_or_path = self._get_model_location_on_disk(model_id)
        from_pretrained_kwargs = self._get_model_from_pretrained_kwargs()

        if additional_load_kwargs:
            from_pretrained_kwargs = {
                **from_pretrained_kwargs,
                **additional_load_kwargs,
            }


        logger.info(f"Loading model {model_id_or_path}...")
        print("++++++++++++++++++ AutoModel")
        print(from_pretrained_kwargs)

        try:
            model = loader.from_pretrained(
                model_id_or_path, **from_pretrained_kwargs
            ).half().cuda()
        except OSError:
            if model_id_or_path != model_id:
                logger.warning(
                    f"Couldn't load model from derived path {model_id_or_path}, "
                    f"trying to load from model_id {model_id}"
                )
                model = loader.from_pretrained(
                    model_id, **from_pretrained_kwargs
                )
            else:
                raise
        model.eval()
        return model