import os
import pickle
import warnings
from datetime import datetime
from typing import Tuple

import mlflow
import pandas as pd
import scipy
import xgboost as xgb
from dotenv import load_dotenv
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow import MlflowClient
from mlflow.entities import ViewType
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
load_dotenv()

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "data/processed/train.parquet")
VAL_DATA_PATH = os.getenv("VAL_DATA_PATH", "data/processed/val.parquet")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "portugal-rent-price")

SEARCH_ITERATIONS = int(os.getenv("SEARCH_ITERATIONS", "50"))
BOOST_ROUND = int(os.getenv("BOOST_ROUND", "1000"))


def read_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame

    Args:
        filename (str): path of data

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Dataframe with data and pd.Series with target
    """
    data_frame = pd.read_parquet(filename)

    data_frame["property_ads"] = data_frame.company.apply(
        lambda property: "private" if property == "Anúncio Particular" else "private"
    )

    return data_frame


def preprocess_data(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> Tuple[
    scipy.sparse._csr.csr_matrix,
    scipy.sparse._csr.csr_matrix,
    pd.Series,
    pd.Series,
    ColumnTransformer,
]:
    property_map = {"apartamento": 0, "moradia": 1}
    numerical_columns = ["metric", "rooms", "bathroom"]
    categorical_columns = ["energy_certify", "property_type", "district", "condition"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                categorical_columns,
            ),
            ("scaler", StandardScaler(), numerical_columns),
        ]
    )
    X_train = df_train.drop(["price"], axis=1)
    X_train["property_type"] = X_train["property_type"].map(property_map)

    X_val = df_val.drop(["price"], axis=1)
    X_val["property_type"] = X_val["property_type"].map(property_map)

    y_train = df_train["price"]
    y_val = df_val["price"]

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    return (X_train_processed, X_val_processed, y_train, y_val, preprocessor)


def search_hyperparameter_xboost(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
):
    """_summary_

    Args:
        X_train (scipy.sparse._csr.csr_matrix): _description_
        X_val (scipy.sparse._csr.csr_matrix): _description_
        y_train (pd.Series): _description_
        y_val (pd.Series): _description_
        preprocessor (ColumnTransformer): _description_

    Returns:
        _type_: _description_
    """

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xboostRegressor")
            mlflow.set_tag("runtype", "hyperparameter_tuning")

            mlflow.log_params(params)

            booster = xgb.XGBRegressor(
                params=params,
            )

            evaluation = [(X_train, y_train), (X_val, y_val)]
            booster.fit(
                X_train,
                y_train,
                eval_set=evaluation,
                eval_metric="rmse",
                early_stopping_rounds=50,
            )
            y_pred = booster.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            r2 = r2_score(y_val, y_pred)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        "max_depth": scope.int(hp.uniform("max_depth", 1, 20)),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "reg_alpha": hp.loguniform("reg_alpha", -5, 0),
        "reg_lambda": hp.loguniform("reg_lambda", -6, 1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "seed": 42,
    }

    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=SEARCH_ITERATIONS,
        trials=Trials(),
    )


def train_best_model(
    X_train: scipy.sparse._csr.csr_matrix,
    X_val: scipy.sparse._csr.csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    preprocessor: ColumnTransformer,
) -> None:
    client = MlflowClient()
    query = 'tags.runtype = "hyperparameter_tuning"'

    with mlflow.start_run():
        model_tunnings = client.search_runs(
            experiment_ids=["1"],
            filter_string=query,
            order_by=["metrics.rmse ASC", "attributes.start_time DESC"],
            max_results=5,
        )
        best_param = model_tunnings[0].data.params

        mlflow.set_tag("model", "XGBRegressor")
        mlflow.set_tag("runtype", "best_model")

        mlflow.log_params(best_param)

        model = xgb.XGBRegressor(
            params=best_param,
        )

        evaluation = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train,
            y_train,
            eval_set=evaluation,
            eval_metric="rmse",
            early_stopping_rounds=50,
        )

        y_pred = model.predict(X_val)

        mlflow.log_metric("rmse", mean_squared_error(y_val, y_pred, squared=False))

        mlflow.log_metric("r2", r2_score(y_val, y_pred))

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(preprocessor, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(model, artifact_path="models")


def register_best_model():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        filter_string='tags.runtype = "best_model"',
        order_by=["metrics.rmse DESC"],
    )[0]

    best_model_rmse = best_run.data.metrics["rmse"]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/models"

    model_details = mlflow.register_model(model_uri=model_uri, name=EXPERIMENT_NAME)
    version = model_details.version

    client.update_registered_model(
        name=model_details.name, description=f"Current rmse: {best_model_rmse}"
    )

    latest_prod_version = client.get_latest_versions(
        EXPERIMENT_NAME, stages=["Production"]
    )

    if latest_prod_version:
        prod_model = client.get_run(latest_prod_version[0].run_id)
        prod_rmse = prod_model.data.metrics["rmse"]

        if best_model_rmse < prod_rmse:
            stage = "Production"
            archive = True
        else:
            stage = "Archived"
            archive = False
    else:
        stage = "Production"
        archive = False

    date = datetime.today().date()
    client.transition_model_version_stage(
        EXPERIMENT_NAME, version, stage, archive_existing_versions=archive
    )

    description = f"The model version {version} was transitioned to {stage} on {date}"
    client.update_model_version(EXPERIMENT_NAME, version, description=description)


def train_flow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_train = read_data(TRAIN_DATA_PATH)
    df_val = read_data(VAL_DATA_PATH)

    X_train, X_val, y_train, y_val, preprocessor = preprocess_data(df_train, df_val)
    search_hyperparameter_xboost(X_train, X_val, y_train, y_val)
    train_best_model(X_train, X_val, y_train, y_val, preprocessor)

    register_best_model()


if __name__ == "__main__":
    train_flow()
