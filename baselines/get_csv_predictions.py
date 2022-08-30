from argparse import ArgumentParser

import numpy as np
import pandas as pd

from finetune_mlm import (
    MODEL_TO_HUB_NAME as MLM_MODEL_TO_HUB_NAME,
    LR_VALUES as MLM_LR_VALUES,
    DECAY_VALUES as MLM_DECAY_VALUES,
    BATCH_SIZES as MLM_BATCH_SIZES,
)
from finetune_t5 import (
    MODEL_TO_HUB_NAME as T5_MODEL_TO_HUB_NAME,
    LR_VALUES as T5_LR_VALUES,
    DECAY_VALUES as T5_DECAY_VALUES,
    BATCH_SIZES as T5_BATCH_SIZES,
)
from train_sklearn_baselines import MODEL_TO_CLASS, C_VALUES

MODEL_NAMES = sum(
    map(list, (MLM_MODEL_TO_HUB_NAME.keys(), T5_MODEL_TO_HUB_NAME.keys(), MODEL_TO_CLASS.keys())), start=[]
)


def main(models):
    for model_name in models:
        sample = pd.read_csv("../data/sample_submission.csv", index_col="id")

        if model_name == "majority":
            best_dir = model_name
        else:
            dev_metrics = np.load(f"results_agg/{model_name}_dev.npy")

            # Use MCC to find the best hyperparameters
            max_ind = np.unravel_index(dev_metrics[..., 1].argmax(), shape=dev_metrics[..., 1].shape)

            if model_name == "logreg":
                seed, reg_coef = max_ind
                best_dir = f"{model_name}_{C_VALUES[reg_coef]}_{seed}"
            elif "t5" in model_name:
                seed, lr, wd, bs = max_ind
                best_dir = f"{model_name}_{T5_LR_VALUES[lr]}_{T5_DECAY_VALUES[wd]}_{T5_BATCH_SIZES[bs]}_{seed}"
            else:
                seed, lr, wd, bs = max_ind
                best_dir = f"{model_name}_{MLM_LR_VALUES[lr]}_{MLM_DECAY_VALUES[wd]}_{MLM_BATCH_SIZES[bs]}_{seed}"

        preds = np.load(f"results/{best_dir}/preds.npy")

        if preds.ndim > 1:
            preds = preds.argmax(axis=1)

        sample["acceptable"] = preds
        sample.to_csv(f"preds_{model_name}.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--models", choices=MODEL_NAMES, nargs="+", required=True)
    args = parser.parse_args()
    main(args.models)
