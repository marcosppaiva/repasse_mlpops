import glob
import logging
import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "/raw/imovirtdataual.parquet")
DATA_PROCESSED_PATH = "data/processed/"

logging.basicConfig(
    level=logging.INFO,
    format="PREPROCESSOR_APP - %(asctime)s - %(levelname)s - %(message)s",
)


def get_last_data(filename: str) -> str:
    list_of_files = glob.glob(filename)
    print(list_of_files)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    return df


def clean_and_settypes(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = data_frame.loc[data_frame.price != "Preço sob consulta"]
    data_frame.price = (
        data_frame.price.str.replace(",", ".").str.replace(" ", "").astype(int)
    )
    data_frame.metric = data_frame.metric.str.replace(",", ".")
    data_frame.metric = (
        data_frame.metric.str.replace(" m²", "").str.replace(" ", "").astype(float)
    )
    data_frame.rooms = (
        data_frame.rooms.str.replace("T", "")
        .str.replace(" ou superior", "")
        .astype(int)
    )

    return data_frame


def remove_outlier(
    data_frame: pd.DataFrame, column: str, min_value, max_value
) -> pd.DataFrame:  # pylint: disable=unused-argument
    return data_frame.query(f"({column} >= {min_value}) and ({column} <= {max_value})")


def save_data_split(data_frame: pd.DataFrame, data_path: str) -> Tuple:
    train, val = train_test_split(
        data_frame, test_size=0.3, train_size=0.7, random_state=42
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    train_filename = os.path.join(data_path, "train.parquet")
    val_filename = os.path.join(data_path, "val.parquet")

    train.to_parquet(train_filename)
    val.to_parquet(val_filename)

    return (train.shape, val.shape)


def preprocessor_flow():
    # filename = get_last_data(os.path.join(DATA_RAW_PATH, '*'))
    data_frame = read_data(RAW_DATA_PATH)

    data_frame = clean_and_settypes(data_frame)

    T = ["T1", "T2", "T3", "T4", "T5", "T6"]

    for t in T:
        indexs = data_frame[
            (data_frame.rooms == 0) & (data_frame.description.str.contains(t))
        ].index
        data_frame.loc[indexs, "rooms"] = int(t.replace("T", ""))

    data_frame.condition = data_frame.condition.fillna("Usado")
    data_frame.bathroom = data_frame.bathroom.fillna(1)

    data_frame = remove_outlier(
        data_frame,
        "bathroom",
        np.percentile(data_frame.bathroom, 0.02),
        np.percentile(data_frame.bathroom, 99),
    )
    data_frame = remove_outlier(
        data_frame,
        "price",
        np.percentile(data_frame.price, 0.02),
        np.percentile(data_frame.price, 99.1),
    )
    data_frame = remove_outlier(
        data_frame,
        "metric",
        np.percentile(data_frame.metric, 0.02),
        np.percentile(data_frame.metric, 99.9),
    )
    data_frame = remove_outlier(
        data_frame,
        "rooms",
        np.percentile(data_frame.rooms, 0.02),
        np.percentile(data_frame.rooms, 99),
    )
    data_frame = data_frame.drop("extract_date", axis=1)

    save_data_split(data_frame, DATA_PROCESSED_PATH)


if __name__ == "__main__":
    preprocessor_flow()
