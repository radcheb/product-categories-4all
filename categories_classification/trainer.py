import logging
from typing import List

from mlflow import set_experiment, start_run, log_params
from mlflow.models import infer_signature
from mlflow.sklearn import eval_and_log_metrics, log_model
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from categories_classification.paths import get_client_train_data_uri

LOGGER = logging.getLogger(__name__)
LABEL_COLUMN = "category_id"
SEED = 1234
TEST_SET_SIZE = 0.1


def train_model(client_id: str, features: List[str], model_params: dict, training_date: str):
    """
    Train a Random Frest model, evaluate it and save it to data directory.
    :param client_id: id of client.
    :param features: input features to be used in training.
    :param model_params: model params. Must be with in RandomForestClassifier parameters.
    :param training_date: date of training
    :return:
    """
    # Build input data path based on client_id.
    input_data_uri = get_client_train_data_uri(client_id=client_id)
    # Load data as csv
    input_data = read_csv(input_data_uri)
    LOGGER.info("Loaded input data with shape %s", input_data.shape)
    # Split data into train set and test set
    x_train, x_test, y_train, y_test = train_test_split(input_data[features], input_data[LABEL_COLUMN],
                                                        test_size=TEST_SET_SIZE,
                                                        random_state=SEED)
    # Log size of data
    LOGGER.info("Training on %s examples", len(x_train))
    LOGGER.info("Testing on %s examples", len(x_test))

    set_experiment(f'ProductCategoriesClassification_{client_id}')
    tags = {
        "client_id": client_id,
    }
    with start_run(run_name=f"training_{training_date}", tags=tags):
        # Create a RandomForestClassifier model and fit it on training set.
        # Set verbose to 2 to follow training status.
        model_kwargs = {"random_state": SEED}
        if model_params:
            model_kwargs.update(model_params)
        model = RandomForestClassifier(**model_kwargs, verbose=2)
        model.fit(x_train, y_train)
        eval_and_log_metrics(model, x_train, y_train, prefix="train_")

        # Log parameters and metrics using the MLflow APIs
        log_params(model_params)

        # Score model on test set and log accuracy
        eval_and_log_metrics(model, x_test, y_test, prefix="val_")

        model_signature = infer_signature(x_test, y_test)
        # Build a model path, fow now it's fixed and based on client_id
        # Dump model with joblib
        log_model(sk_model=model,
                  artifact_path="best_model",
                  signature=model_signature,
                  registered_model_name=f"{client_id}_model")
