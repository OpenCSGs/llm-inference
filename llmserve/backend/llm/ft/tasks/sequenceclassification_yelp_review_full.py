from ._base import Task
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from typing import Any
import pandas as pd
import evaluate
import numpy as np


class SequenceclassificationYelpReviewFull(Task):
    AUTO_MODEL_CLASS = AutoModelForSequenceClassification

    DATASET_PATH = "yelp_review_full"
    DATASET_NAME = ""
    FROM_PRETRAINED_KWARGS = {
        "num_labels": 5
    }

    def get_data_proprocess(self) -> Any:
        tokenizer = self.tokenizer

        # adopt python decorator TODO
        def preprocess_function(examples: pd.DataFrame):            
            examples = examples.to_dict("list")
            ret = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
                
            # Add back the original columns
            ret = {**examples, **ret}
            return pd.DataFrame.from_dict(ret)
        
        return preprocess_function

    def get_compute_metrics(self) -> Any:
        DATASET_PATH = self.DATASET_PATH
        DATASET_NAME = self.DATASET_NAME

        def compute_metrics(eval_preds):
            # metric = evaluate.load(DATASET_PATH, DATASET_NAME)
            metric = evaluate.load("accuracy")
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        
        return compute_metrics

    def get_data_collator(self) -> Any:
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return data_collator
    
    def training_key(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return "train"

    def validation_key(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return "validation"
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.ft_config.train_config.max_length)
    
    def getTrainDataSet(self):
        return self.dataset[self.training_key()].map(self.tokenize_function, batched=True)

    def getEvalDataSet(self):
        return self.dataset[self.validation_key()].map(self.tokenize_function, batched=True)

    def getSmallTrainDataSet(self, len: int):
        return self.dataset[self.training_key()].select(range(len)).map(self.tokenize_function, batched=True)

    def getSmallEvalDataSet(self, len: int):
        return self.dataset[self.validation_key()].select(range(len)).map(self.tokenize_function, batched=True)
