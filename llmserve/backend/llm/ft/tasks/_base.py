import abc
from abc import abstractmethod
from typing import Any
from llmserve.backend.server.models import DataConfig
from datasets import load_dataset
from datasets import load_metric
import transformers
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Any, Dict
from llmserve.backend.server.models import FTConfig
from llmserve.backend.logger import get_logger

logger = get_logger(__name__)

class Task(abc.ABC): 
    AUTO_MODEL_CLASS: transformers.AutoModel = None

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    # kwargs when build model with transformer's "from_pretrained"
    FROM_PRETRAINED_KWARGS: Dict[str, Any] = None

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        ft_config: "FTConfig",
    ) -> None:
        self.tokenizer = tokenizer
        self.ft_config = ft_config
        self.download_dataset()
        self._pre()

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: "PreTrainedTokenizer",
        ft_config: "FTConfig",
    ) -> "Task":
        fac = cls(
            tokenizer = tokenizer,
            ft_config = ft_config
        )

        return fac

    @abstractmethod
    def get_data_proprocess(self) -> Any:
        """Change trainning data to tensor model can accepted"""
        pass

    @abstractmethod
    def get_compute_metrics(self) -> Any:
        pass
    
    @abstractmethod
    def get_data_collator(self) -> Any:
        pass

    def _pre(self) -> Any:
        pass

    @abstractmethod
    def training_key(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        pass

    @abstractmethod
    def validation_key(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        pass
    
    @abstractmethod
    def getTrainDataSet(self):
        pass

    @abstractmethod
    def getEvalDataSet(self):
        pass

    @abstractmethod
    def getSmallTrainDataSet(self, len: int):
        pass

    @abstractmethod
    def getSmallEvalDataSet(self, len: int):
        pass
   
    def get_dataset(self):
        return self.dataset

    def download_dataset(self):
        # Downloading and loading a dataset from the hub.
        logger.info("Start loading dataset")
        if self.ft_config.data_config.local_path:
            logger.info(f"Loading dataset from local path {self.ft_config.data_config.local_path}")
            raw_datasets = load_dataset(self.ft_config.data_config.local_path)
        else:
            if self.DATASET_NAME:
                logger.info(f"Downloading dataset {self.DATASET_NAME} from {self.DATASET_PATH}")
                raw_datasets = load_dataset(self.DATASET_PATH, self.DATASET_NAME)
            else:
                logger.info(f"Downloading dataset from {self.DATASET_PATH}")
                raw_datasets = load_dataset(self.DATASET_PATH)
        logger.info("Done load dataset")
        logger.info(f"{raw_datasets}")
        self.dataset = raw_datasets

    def set_model(self, model: PreTrainedModel):
        self.model = model

    def get_model(self):
        return self.model