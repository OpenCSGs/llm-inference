# Adapted from Hugging Face tutorial: https://huggingface.co/docs/transformers/training

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

num_labels = 5
modelPath = "bert-base-cased"
modelPath = "/Users/hub/models/bert-base-cased"
dsPath = "yelp_review_full"
dsPath = "/Users/hub/models/yelp_review_full/1.0.0"

# Datasets
dataset = load_dataset(dsPath)
print('Loaded dataset', dataset)

tokenizer = AutoTokenizer.from_pretrained(modelPath)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

count = 10
small_train_dataset = dataset["train"].select(range(count)).map(tokenize_function, batched=True)
small_eval_dataset = dataset["test"].select(range(count)).map(tokenize_function, batched=True)
print('small train dataset', small_train_dataset)
print('small eval dataset', small_eval_dataset)

# Model
model = AutoModelForSequenceClassification.from_pretrained(modelPath, num_labels=num_labels)

# Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Hugging Face Trainer
training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch", 
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Start Training
trainer.train()
