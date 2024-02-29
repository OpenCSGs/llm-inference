from abc import ABC, abstractmethod
from llmserve.backend.logger import get_logger
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from typing import Union, TYPE_CHECKING, List
from transformers import PreTrainedModel, PreTrainedTokenizer
from llmserve.backend.server.models import FTApp
import torch
from llmserve.backend.llm.initializers import get_initializer_cls_by_name

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer


logger = get_logger(__name__)
class BaseFT(ABC):
    """base fine tune class.

    Args:
    """

    def __init__(
        self,
        ftapp: FTApp,
        ) -> None:
        self.ftapp = ftapp
        self.data_conf = ftapp.ft_config.data_config
        self.train_conf = ftapp.ft_config.train_config
        self.model_config = ftapp.model_config
        self.ft_task = ftapp.ft_config.ft_task
        self.scale_config = ftapp.scaling_config

        # Lazy import so that the new cache location is used
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
        initializer_name = self.model_config.initialization.initializer
        if not isinstance(initializer_name, str):
            initializer_name = initializer_name.type
        
        initializer = get_initializer_cls_by_name(initializer_name)(
            device=device,
            world_size=1, # fake
            **self.model_config.initialization.initializer.get_initializer_kwargs(),
        )

        self.initializer = initializer

    @abstractmethod
    def train(self):
        pass
    


    

        