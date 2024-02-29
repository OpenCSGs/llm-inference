from ._base import Task
from transformers import AutoModelForTokenClassification
from typing import Any
import pandas as pd
import evaluate
import numpy as np
from transformers import DataCollatorForTokenClassification

class TokenclassificationConll2003(Task):
    AUTO_MODEL_CLASS = AutoModelForTokenClassification

    DATASET_PATH = "conll2003"
    FROM_PRETRAINED_KWARGS = {
        "num_labels": 9
    }

    def _pre(self) -> Any:
        label_names = self.get_dataset()[self.training_key()].features["ner_tags"].feature.names
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        self.FROM_PRETRAINED_KWARGS["id2label"] = id2label
        self.FROM_PRETRAINED_KWARGS["label2id"] = label2id

    def get_data_proprocess(self) -> Any:
        tokenizer = self.tokenizer
        def align_labels_with_tokens(labels, word_ids):
            new_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id != current_word:
                    # Start of a new word!
                    current_word = word_id
                    label = -100 if word_id is None else labels[word_id]
                    new_labels.append(label)
                elif word_id is None:
                    # Special token
                    new_labels.append(-100)
                else:
                    # Same word as previous token
                    label = labels[word_id]
                    # If the label is B-XXX we change it to I-XXX
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)

            return new_labels
        
        # adopt python decorator TODO
        def preprocess_function(examples: pd.DataFrame):            
            # examples = examples.to_dict("list")
            # inputs = [i.tolist() for i in examples["tokens"]]
            inputs = [i for i in examples["tokens"]]
            tokenized_inputs = tokenizer(
                inputs, truncation=True, is_split_into_words=True
            )
            all_labels = examples["ner_tags"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                word_ids = tokenized_inputs.word_ids(i)
                new_labels.append(align_labels_with_tokens(labels, word_ids))

            tokenized_inputs["labels"] = new_labels
                
            # Add back the original columns
            ret = {**examples, **tokenized_inputs}
            return pd.DataFrame.from_dict(ret)
        
        return preprocess_function

    def get_data_collator(self) -> Any:
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        return data_collator

    def get_compute_metrics(self) -> Any:
        label_names = self.get_dataset()[self.training_key()].features["ner_tags"].feature.names
        metric = evaluate.load("seqeval")

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            # Remove ignored index (special tokens) and convert to labels
            true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
            true_predictions = [
                [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": all_metrics["overall_precision"],
                "recall": all_metrics["overall_recall"],
                "f1": all_metrics["overall_f1"],
                "accuracy": all_metrics["overall_accuracy"],
            }
        
        return compute_metrics

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
    
    def getTrainDataSet(self):
        return self.dataset[self.training_key()].map(self.get_data_proprocess(), batched=True)

    def getEvalDataSet(self):
        return self.dataset[self.validation_key()].map(self.get_data_proprocess(), batched=True)

    def getSmallTrainDataSet(self, len: int):
        return self.dataset[self.training_key()].select(range(len)).map(self.get_data_proprocess(), batched=True)

    def getSmallEvalDataSet(self, len: int):
        return self.dataset[self.validation_key()].select(range(len)).map(self.get_data_proprocess(), batched=True)
