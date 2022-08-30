from argparse import ArgumentParser
from functools import partial

import pandas as pd
from transformers import AutoModelForSequenceClassification, BertTokenizer, AutoTokenizer, DataCollatorWithPadding, \
    TrainingArguments, Trainer

from finetune_mlm import (
    MODEL_TO_HUB_NAME as MLM_MODEL_TO_HUB_NAME,
    preprocess_examples,
)
from tqdm import tqdm
from utils import read_splits


def main(models):
    for model_name in tqdm(models):
        if "rubert" in model_name:
            tokenizer = BertTokenizer.from_pretrained(MLM_MODEL_TO_HUB_NAME[model_name])
        else:
            tokenizer = AutoTokenizer.from_pretrained(MLM_MODEL_TO_HUB_NAME[model_name])

        splits = read_splits(as_datasets=True)

        tokenized_splits = splits.map(
            partial(preprocess_examples, tokenizer=tokenizer),
            batched=True,
            remove_columns=["sentence"],
            keep_in_memory=True,
        )

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        model = AutoModelForSequenceClassification.from_pretrained(f"checkpoints_best/{model_name}")

        training_args = TrainingArguments(
            output_dir=f"temp",
            evaluation_strategy="epoch",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=0,
            weight_decay=0,
            num_train_epochs=1,
            warmup_ratio=0.1,
            optim="adamw_torch",
            save_strategy="epoch",
            save_total_limit=1,
            seed=0,
            fp16=True,
            tf32=True,
            dataloader_num_workers=4,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        predictions = trainer.predict(test_dataset=tokenized_splits["test"]).predictions

        sample = pd.read_csv("../data/sample_submission.csv", index_col="id")

        predictions = predictions.argmax(axis=1)

        sample["acceptable"] = predictions
        sample.to_csv(f"preds_{model_name}_from_model.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--models", choices=MLM_MODEL_TO_HUB_NAME.keys(), nargs="+", required=True)
    args = parser.parse_args()
    main(args.models)
