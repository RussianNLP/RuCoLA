import os
from argparse import ArgumentParser
from functools import partial
from shutil import rmtree

import numpy as np
from datasets import load_metric
from razdel import tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)

from utils import read_splits

ACCURACY = load_metric("accuracy", keep_in_memory=True)
MCC = load_metric("matthews_correlation", keep_in_memory=True)
MODEL_TO_HUB_NAME = {
    "rubert-base": "sberbank-ai/ruBert-base",
    "rubert-large": "sberbank-ai/ruBert-large",
    "ruroberta-large": "sberbank-ai/ruRoberta-large",
    "xlmr-base": "xlm-roberta-base",
    "rembert": "google/rembert",
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

N_SEEDS = 10
N_EPOCHS = 5
LR_VALUES = (1e-5, 3e-5, 5e-5)
DECAY_VALUES = (1e-4, 1e-2, 0.1)
BATCH_SIZES = (32, 64)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    preds = np.argmax(preds, axis=1)

    acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
    mcc_result = MCC.compute(predictions=preds, references=p.label_ids)

    result = {"accuracy": acc_result["accuracy"], "mcc": mcc_result["matthews_correlation"]}

    return result


def preprocess_examples(examples, tokenizer):
    result = tokenizer(examples["sentence"], padding=False)

    if "acceptable" in examples:
        result["label"] = examples["acceptable"]

    result["length"] = [len(list(tokenize(sentence))) for sentence in examples["sentence"]]
    return result


def main(model_name):
    if "rubert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(MODEL_TO_HUB_NAME[model_name])
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_HUB_NAME[model_name])

    splits = read_splits(as_datasets=True)

    tokenized_splits = splits.map(
        partial(preprocess_examples, tokenizer=tokenizer),
        batched=True,
        remove_columns=["sentence"],
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # seed, lr, wd, bs
    dev_metrics_per_run = np.empty((N_SEEDS, len(LR_VALUES), len(DECAY_VALUES), len(BATCH_SIZES), 2))

    for i, learning_rate in enumerate(LR_VALUES):
        for j, weight_decay in enumerate(DECAY_VALUES):
            for k, batch_size in enumerate(BATCH_SIZES):
                for seed in range(N_SEEDS):
                    set_seed(seed)

                    if "rubert" in model_name:
                        model = BertForSequenceClassification.from_pretrained(MODEL_TO_HUB_NAME[model_name])
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(MODEL_TO_HUB_NAME[model_name])

                    run_base_dir = f"{model_name}_{learning_rate}_{weight_decay}_{batch_size}"

                    training_args = TrainingArguments(
                        output_dir=f"checkpoints/{run_base_dir}",
                        overwrite_output_dir=True,
                        evaluation_strategy="epoch",
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        num_train_epochs=N_EPOCHS,
                        warmup_ratio=0.1,
                        save_strategy="epoch",
                        save_total_limit=1,
                        seed=seed,
                        fp16=True,
                        tf32=True,
                        dataloader_num_workers=4,
                        group_by_length=True,
                        report_to="none",
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_mcc",
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_splits["train"],
                        eval_dataset=tokenized_splits["dev"],
                        compute_metrics=compute_metrics,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                    )

                    train_result = trainer.train()
                    print(f"{run_base_dir}_{seed}")
                    print("train", train_result.metrics)

                    os.makedirs(f"results/{run_base_dir}_{seed}", exist_ok=True)

                    dev_predictions = trainer.predict(test_dataset=tokenized_splits["dev"])
                    print("dev", dev_predictions.metrics)
                    dev_metrics_per_run[seed, i, j, k] = (
                        dev_predictions.metrics["test_accuracy"],
                        dev_predictions.metrics["test_mcc"],
                    )

                    predictions = trainer.predict(test_dataset=tokenized_splits["test"])

                    np.save(f"results/{run_base_dir}_{seed}/preds.npy", predictions.predictions)

                    rmtree(f"checkpoints/{run_base_dir}")

    os.makedirs("results_agg", exist_ok=True)
    np.save(f"results_agg/{model_name}_dev.npy", dev_metrics_per_run)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name", choices=MODEL_TO_HUB_NAME.keys(), required=True)
    args = parser.parse_args()
    main(args.model_name)
