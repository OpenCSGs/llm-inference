from ._base import Task
from transformers import AutoModel, DataCollatorForSeq2Seq
from typing import Any
import pandas as pd
import numpy as np
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class NoheaderAdvertiseGen(Task):
    AUTO_MODEL_CLASS = AutoModel

    DATASET_PATH = "shibing624/AdvertiseGen"

    def get_data_proprocess(self) -> Any:
        tokenizer = self.tokenizer
        max_length = self.ft_config.train_config.max_length
        # adopt python decorator TODO
        def preprocess_function(examples: pd.DataFrame):            
            examples = examples.to_dict("list")
            #-- start
            max_source_length = max_length / 2
            max_target_length = max_length - max_source_length
            # max_seq_length = data_args.max_source_length + data_args.max_target_length

            model_inputs = {
                "input_ids": [],
                "labels": [],
            }
            for i in range(len(examples["content"])):
                if examples["content"][i] and examples["summary"][i]:
                    prompt, answer = examples["content"][i], examples["summary"][i]
                        
                    a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                    b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                    if len(a_ids) > max_source_length - 1:
                        a_ids = a_ids[: max_source_length - 1]

                    if len(b_ids) > max_target_length - 2:
                        b_ids = b_ids[: max_target_length - 2]

                    input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                    context_length = input_ids.index(tokenizer.bos_token_id)
                    mask_position = context_length - 1
                    labels = [-100] * context_length + input_ids[mask_position+1:]
                    
                    # pad_len = max_length - len(input_ids)
                    # input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    # labels = labels + [tokenizer.pad_token_id] * pad_len
                    # if data_args.ignore_pad_token_for_loss:
                    #     labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                    model_inputs["input_ids"].append(input_ids)
                    model_inputs["labels"].append(labels)

                
            # Add back the original columns
            ret = {**examples, **model_inputs}
            return pd.DataFrame.from_dict(ret)
        
        return preprocess_function

    def get_compute_metrics(self) -> Any:
        tokenizer = self.tokenizer

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            score_dict = {
                "rouge-1": [],
                "rouge-2": [],
                "rouge-l": [],
                "bleu-4": []
            }
            for pred, label in zip(decoded_preds, decoded_labels):
                hypothesis = list(jieba.cut(pred))
                reference = list(jieba.cut(label))
                rouge = Rouge()
                scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
                result = scores[0]
                
                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))
                bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            for k, v in score_dict.items():
                score_dict[k] = float(np.mean(v))
            return score_dict
        
        return compute_metrics

    def get_data_collator(self) -> Any:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=True
        )
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