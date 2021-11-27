import logging
from typing import Iterator, List, Callable

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import RestException
from mlflow.models import Model
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient
from numpy import argsort, take_along_axis
from pandas import DataFrame
from pyspark import Broadcast
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

from categories_classification.paths import get_client_inference_data_uri, get_products_categories_dir

LOGGER = logging.getLogger(__name__)
# Prediction output Schema
OUTPUT_SCHEMA = StructType([
    StructField("product_id", StringType(), True),
    StructField("pred_label", StringType(), False),
    StructField("pred_score", FloatType(), False),
    StructField("rank", IntegerType(), False),
])
# Number of top categories to predict
NBR_TOP_LABELS = 5
# Name of product identifier column
PRODUCT_ID = "product_id"
# Output comumn names
PRED_LABEL = "pred_label"
PRED_SCORE = "pred_score"
RANK = "rank"


def _get_compute_top_categories_pudf(spark_model: Broadcast, nbr_top_labels: int, input_names: List[str]) -> Callable:
    """
    Build an return compute_top_categories user defined function.
    :param spark_model: broadcasted scikit-learn model.
    :param nbr_top_labels: number of top categories to output.
    :param input_names: name of input columns to pass to scikit-learn model.
    :return:
    """

    def compute_top_categories(iterator: Iterator[DataFrame]) -> Iterator[DataFrame]:
        # Access broadcasted model value property once to load the model one time.
        loaded_sk_model = spark_model.value
        for pdf in iterator:
            # Initialize an 'empty' DataFrame with only product_id column multiplied with the number
            # of categories per product to predict. Will not reset index as it's not important.
            result = pdf[[PRODUCT_ID]].loc[pdf.index.repeat(nbr_top_labels)]
            # For each product, compute each category score (probability)
            labels_scores = loaded_sk_model.predict_proba(pdf[input_names])
            # For each product, get categories indices sorted by score descending and take top 5 (nbr_top_labels)
            labels_scores_argsort = argsort(-labels_scores, axis=1)[:, :nbr_top_labels]
            # For each product take top categories using previous indices.
            top_scores = take_along_axis(labels_scores, labels_scores_argsort, axis=1)
            # Set categories labels after flattening them to match a new line per product and category.
            result[PRED_LABEL] = spark_model.value.classes_[labels_scores_argsort].flatten()
            # Set categories scores after flattening them to match a new line per product and category.
            result[PRED_SCORE] = top_scores.flatten()
            # Add rank column, which is constant over products since categories already sorted by score.
            result[RANK] = argsort(-top_scores, axis=1).flatten()
            yield result

    return compute_top_categories


def _get_latest_model_version(model_name: str) -> ModelVersion:
    """
    Use Mlflow client to get latest model registered version.
    :param model_name:
    :return:
    """
    # Init Mlflow client
    client = MlflowClient()
    try:
        # Get latest version for the given model, stage must be set to "None" as it was not set in training.
        latest_version = client.get_latest_versions(model_name, stages=["None"])
        if latest_version:
            # get_latest_versions result is a list of version, not as function type hint.
            return latest_version[0]
        raise ValueError(f"No versions found for model {model_name}")
    except RestException as err:
        raise ValueError(f"No model found with name {model_name}") from err


def _get_model_input_names(model_version: ModelVersion) -> List[str]:
    """
    Get model input columns names.
    :param model_version: Mlflow model version.
    :return:
    """
    # Load model as Mlflow Model object, required to get the model signature.
    mlflow_model = Model.load(model_version.source)
    return mlflow_model.signature.inputs.input_names()


def predict_categories(client_id, inference_date):
    """
    Predict top categories for products using latest trained model.
    :param client_id: Id of client
    :param inference_date: Date of inference used to store the prediction result.
    :return:
    """
    # Get spark session, usually configured with pyspark command.
    spark = SparkSession.builder.getOrCreate()

    # Build inference data uri and load inference dataset
    inference_uri = get_client_inference_data_uri(client_id=client_id)
    inference_df = spark.read.option("header", "true").csv(inference_uri)

    # Using client model name, load latest version
    model_name = f"{client_id}_model"
    latest_model_version = _get_latest_model_version(model_name)
    LOGGER.warning("Latest model %s version: %s", model_name, latest_model_version)

    # Load latest scikit-learn model and model signature from Mlflow.
    sk_model = load_model(model_uri=f"models:/{model_name}/{latest_model_version.version}")
    input_names = _get_model_input_names(latest_model_version)
    LOGGER.warning("Using input columns: %s", input_names)

    # Broadcast scikit-learn model to made it available in spark workers.
    broadcasted_model = spark.sparkContext.broadcast(sk_model)
    # Build prediction function
    compute_top_categories_pudf = _get_compute_top_categories_pudf(broadcasted_model, NBR_TOP_LABELS, input_names)
    # Select only required columns to avoid copying useless data in pandas dataframe.
    inference_df = inference_df.select(PRODUCT_ID, *input_names)
    # mapInPandas is an experimental features in Spark 3.0 but very powerful to work with python models
    # while going beyond simple map function. Here we are performing a flatmap like operation.
    inference_df = inference_df.mapInPandas(compute_top_categories_pudf, OUTPUT_SCHEMA)
    # Build output uri and save prediction result as parquet dataset.
    output_uri = get_products_categories_dir(client_id, inference_date)
    inference_df.write.parquet(output_uri)
