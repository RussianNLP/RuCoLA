from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR.parent / "data"

TRAIN_FILE = DATA_DIR / "in_domain_train.csv"
IN_DOMAIN_DEV_FILE = DATA_DIR / "in_domain_dev.csv"
OUT_OF_DOMAIN_DEV_FILE = DATA_DIR / "out_of_domain_dev.csv"
TEST_FILE = DATA_DIR / "test.csv"


def read_splits(*, as_datasets):
    train_df, in_domain_dev_df, out_of_domain_dev_df, test_df = map(
        pd.read_csv, (TRAIN_FILE, IN_DOMAIN_DEV_FILE, OUT_OF_DOMAIN_DEV_FILE, TEST_FILE)
    )

    # concatenate datasets to get aggregate metrics
    dev_df = pd.concat((in_domain_dev_df, out_of_domain_dev_df))

    if as_datasets:
        train, dev, test = map(Dataset.from_pandas, (train_df, dev_df, test_df))
        return DatasetDict(train=train, dev=dev, test=test)
    else:
        return train_df, dev_df, test_df
