import os
from argparse import ArgumentParser

import numpy as np
from datasets import load_metric
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import read_splits

ACCURACY = load_metric("accuracy")
MCC = load_metric("matthews_correlation")

MODEL_TO_CLASS = {
    "majority": DummyClassifier,
    "logreg": LogisticRegression,
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

N_SEEDS = 5
C_VALUES = (0.01, 0.1, 1)


def train_and_evaluate(model, X_train, X_dev, X_test, y_train, y_dev, run_base_dir):
    model.fit(X_train, y_train)

    dev_preds = model.predict(X_dev)
    test_preds = model.predict(X_test)

    dev_acc = ACCURACY.compute(predictions=dev_preds, references=y_dev)["accuracy"]
    dev_mcc = MCC.compute(predictions=dev_preds, references=y_dev)["matthews_correlation"]

    print(run_base_dir)
    print("dev", dev_acc, dev_mcc)

    os.makedirs(f"results/{run_base_dir}", exist_ok=True)
    np.save(f"results/{run_base_dir}/preds.npy", test_preds)

    return dev_acc, dev_mcc


def main(model_name):
    train, dev, test = read_splits(as_datasets=False)

    if model_name == "majority":
        dev_metrics_per_run = np.empty((2,))

        model = DummyClassifier(strategy="prior")
        run_base_dir = model_name

        dev_acc, dev_mcc = train_and_evaluate(
            model,
            train["sentence"],
            dev["sentence"],
            test["sentence"],
            train["acceptable"],
            dev["acceptable"],
            run_base_dir,
        )

        dev_metrics_per_run[:] = (dev_acc, dev_mcc)

    elif model_name == "logreg":
        tfidf = TfidfVectorizer(lowercase=False, min_df=5, max_df=0.9, ngram_range=(1, 3))

        X_train = tfidf.fit_transform(train["sentence"])
        X_dev = tfidf.transform(dev["sentence"])
        X_test = tfidf.transform(test["sentence"])

        # seed, C
        dev_metrics_per_run = np.empty((N_SEEDS, len(C_VALUES), 2))

        for i, reg_coef in enumerate(C_VALUES):
            for seed in range(N_SEEDS):
                run_base_dir = f"{model_name}_{reg_coef}_{seed}"

                if model_name == "logreg":
                    model = LogisticRegression(random_state=seed, C=reg_coef)
                else:
                    model = MODEL_TO_CLASS[model_name](random_state=seed)

                dev_acc, dev_mcc = train_and_evaluate(
                    model, X_train, X_dev, X_test, train["acceptable"], dev["acceptable"], run_base_dir
                )

                dev_metrics_per_run[seed, i] = (dev_acc, dev_mcc)

    else:
        # can't happen because of argparse, leaving it simply to avoid the warning for dev_metrics_per_run
        raise KeyError(f"{model_name} not in MODEL_TO_CLASS.keys()")

    os.makedirs("results_agg", exist_ok=True)
    np.save(f"results_agg/{model_name}_dev.npy", dev_metrics_per_run)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name", choices=MODEL_TO_CLASS.keys(), required=True)
    args = parser.parse_args()
    main(args.model_name)
