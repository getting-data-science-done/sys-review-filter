import pandas as pd
import projit as pit
import numpy as np
import evaluate
import sys
from datasets import ClassLabel, Value
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from transformers import TrainingArguments, Trainer

#import os
#os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(1, "./")
import src.eval as eval
import src.explain2 as xpl

experiment_name = "04a : BERT - 01e - BioBERT:PubMed"
target = "Decision"
prediction = "predicted"
hf_model = "pritamdeka/BioBert-PubMed200kRCT"
max_length = 512

project = pit.projit_load()
exec_id = project.start_experiment(experiment_name, sys.argv[0], params={})

dataset = load_dataset("csv", data_files=project.get_dataset("train"), sep=",")
dataset = dataset.rename_column(target, "labels")
new_features = dataset['train'].features.copy()
new_features["labels"] = Value('int64')
dataset = dataset.cast(new_features)

dataset.set_format("pandas")
df = dataset['train'][:]
dataset.reset_format()

def clean_none(example):
    example["TEXT"] = str(example["TEXT"])
    example["Question"] = str(example["Question"])
    return example

cleaned_dataset = dataset.map(clean_none)

tokenizer = AutoTokenizer.from_pretrained(hf_model)
def tokenize_function(examples):
    return tokenizer(examples["TEXT"], examples["Question"], padding='max_length', truncation=True, max_length=max_length)

tokenized_dataset = cleaned_dataset.map(tokenize_function, batched=True)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

"""
Helper function to convert the prediction from HF into a single probability
"""
def extract_probs(pred_dict_list):
    for elem in pred_dict_list:
        if elem['label']=="LABEL_1":
            return elem['score']
    return 0.5

studies = ["CARGEL","CIDP","EMR", "ESG", "IORT", "LANB", "LMTA", "PBRT", "VERT", "PID"]

for study in studies:
    print(f"*** TEST SET [{study}] ")
    train_dataset = tokenized_dataset.filter(lambda x: x['Project'] not in [study])
    eval_dataset = tokenized_dataset.filter(lambda x: x['Project'] in [study])

    model = AutoModelForSequenceClassification.from_pretrained(hf_model, num_labels=2, ignore_mismatched_sizes=True)

    training_args = TrainingArguments(
        output_dir='./checkpoints/BioBERT_'+study+'_1e',          # output directory
        num_train_epochs=1,                     # total number of training epochs
        warmup_steps=100,                       # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                      # strength of weight decay
        logging_dir='./logs',                   # directory for storing logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        eval_dataset=eval_dataset['train'],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    eval_dataset.set_format("pandas")
    test_df = dataset['train'][:]
    test_df = pd.DataFrame(test_df)
    eval_dataset.reset_format()

    x_test = test_df["TEXT"].to_list()
    tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None, **tokenizer_kwargs)
    preds = pipe(x_test)
    probs = [extract_probs(x) for x in preds]
    test_df[prediction] = probs
    metrics = eval.calc_model_metrics(test_df, "labels", prediction)
    for key in metrics:
        project.add_result(experiment_name, key, metrics[key], study)

project.end_experiment(experiment_name, exec_id)

