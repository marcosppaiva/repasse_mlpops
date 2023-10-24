import os
import pickle
from typing import Tuple, Union

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel
from scipy.sparse._csr import csr_matrix
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine

from bucket_utils import download_obj
from entities import Imovel

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "final-project-mlops")
DATABASE_URI = os.getenv("DATABASE_URI", "")
LOGGED_MODEL = os.getenv("LOGGED_MODEL", "")
PREPROCESSOR = os.getenv("PREPROCESSOR", "")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "portugal-rent-price")


def model_prod_id(mlflow_tracking_uri):
    client = MlflowClient(mlflow_tracking_uri)
    registered_model = client.search_registered_models(
        filter_string=f"name='{EXPERIMENT_NAME}'"
    )
    run_id = [
        model.run_id
        for model in registered_model[0].latest_versions
        if model.current_stage == "Production"
    ][0]

    return run_id


def load_model(
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI, aws_s3_bucket: str = AWS_S3_BUCKET
) -> Tuple[PyFuncModel, ColumnTransformer, str]:
    run_id = model_prod_id(mlflow_tracking_uri)
    logged_model = LOGGED_MODEL.format(aws_s3_bucket=aws_s3_bucket, run_id=run_id)

    obj = download_obj(aws_s3_bucket, PREPROCESSOR.format(run_id=run_id))

    preprocessor = pickle.load(obj)

    model = mlflow.pyfunc.load_model(logged_model)

    return (model, preprocessor, run_id)


def prepare_data(
    data: Union[Imovel, pd.DataFrame], preprocessor: ColumnTransformer
) -> Tuple[csr_matrix, pd.DataFrame]:
    if isinstance(data, Imovel):
        imovel_json = {
            "district": data.district,
            "property_type": data.property_type.value,
            "bathroom": data.bathroom,
            "metric": data.metric,
            "rooms": data.room,
            "energy_certify": data.energy_certify.value,
            "condition": data.condition.value,
        }
        data = pd.DataFrame.from_dict(imovel_json, orient="index").T

    data_transformed = preprocessor.transform(data)

    return (data_transformed, data)  # type: ignore


def predict(
    data: Union[Imovel, pd.DataFrame],
    model: PyFuncModel,
    preprocessor: ColumnTransformer,
):
    data_prep, df_data = prepare_data(data, preprocessor)
    predicted_price = model.predict(data_prep)

    return (predicted_price, df_data)


def save_predictions(predictions: pd.DataFrame) -> None:
    """Save predictions to database.

    Args:
        predictions (pd.DataFrame): Pandas dataframe with predictions column.
    """

    engine = create_engine(DATABASE_URI)

    predictions.to_sql(name="prediction", con=engine, if_exists="append", index=False)


def load_predictions(widow_size: int) -> pd.DataFrame:
    engine = create_engine(DATABASE_URI)

    query = f"""
    SELECT *
    FROM prediction
    ORDER BY created_time DESC
    LIMIT {widow_size};
    """

    df_pred = pd.read_sql(sql=query, con=engine)

    return df_pred


def save_metrics(metrics: pd.DataFrame) -> None:
    """Save evidentlyAI metrics to database

    Args:
        metrics (pd.DataFrame): Pandas dataframe with metrics column
    """

    engine = create_engine(DATABASE_URI)

    metrics.to_sql(name="drift_metrics", con=engine, if_exists="append", index=False)
