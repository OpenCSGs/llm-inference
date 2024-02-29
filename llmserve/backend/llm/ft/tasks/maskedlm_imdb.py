from ._base import Task
from transformers import AutoModelForMaskedLM
from typing import Any
import pandas as pd
import numpy as np



class MaskedLMImdb(Task):
    AUTO_MODEL_CLASS = AutoModelForMaskedLM

    DATASET_PATH = "imdb"

    def get_data_proprocess(self) -> Any:
        tokenizer = self.tokenizer

        def group_texts(examples):
            # Concatenate all texts
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            # Compute length of concatenated texts
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the last chunk if it's smaller than chunk_size
            total_length = (total_length // chunk_size) * chunk_size
            # Split by chunks of max_len
            result = {
                k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
                for k, t in concatenated_examples.items()
            }
            # Create a new labels column
            result["labels"] = result["input_ids"].copy()
            return result

        
        chunk_size = 128
        # adopt python decorator TODO
        def preprocess_function(examples: pd.DataFrame):            
            # examples = examples.to_dict("list")
            result = tokenizer(examples["text"])
            if tokenizer.is_fast:
                result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

            tokenized_inputs = group_texts(result)
                
            # Add back the original columns
            ret = {**tokenized_inputs}
            return pd.DataFrame.from_dict(ret)
        
        return preprocess_function

    def get_data_collator(self) -> Any:
        import collections
        import numpy as np
        from transformers import default_data_collator

        wwm_probability = 0.2
        tokenizer = self.tokenizer
        def whole_word_masking_data_collator(features):
            for feature in features:
                word_ids = feature.pop("word_ids")

                # Create a map between words and corresponding token indices
                mapping = collections.defaultdict(list)
                current_word_index = -1
                current_word = None
                for idx, word_id in enumerate(word_ids):
                    if word_id is not None:
                        if word_id != current_word:
                            current_word = word_id
                            current_word_index += 1
                        mapping[current_word_index].append(idx)

                # Randomly mask words
                mask = np.random.binomial(1, wwm_probability, (len(mapping),))
                input_ids = feature["input_ids"]
                labels = feature["labels"]
                new_labels = [-100] * len(labels)
                for word_id in np.where(mask)[0]:
                    word_id = word_id.item()
                    for idx in mapping[word_id]:
                        new_labels[idx] = labels[idx]
                        input_ids[idx] = tokenizer.mask_token_id
                feature["labels"] = new_labels

            return default_data_collator(features)
        
        return whole_word_masking_data_collator

    def get_compute_metrics(self) -> Any:        
        return None

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
        return "test"

    def getTrainDataSet(self):
        return self.dataset[self.training_key()].map(self.get_data_proprocess(), batched=True)

    def getEvalDataSet(self):
        return self.dataset[self.validation_key()].map(self.get_data_proprocess(), batched=True)

    def getSmallTrainDataSet(self, len: int):
        return self.dataset[self.training_key()].select(range(len)).map(self.get_data_proprocess(), batched=True)

    def getSmallEvalDataSet(self, len: int):
        return self.dataset[self.validation_key()].select(range(len)).map(self.get_data_proprocess(), batched=True)