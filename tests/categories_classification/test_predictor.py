import tempfile
from glob import glob
from os.path import join
from unittest.mock import patch

from dateutil.utils import today
from mlflow.entities.model_registry import ModelVersion
from pyspark.sql import SparkSession

from categories_classification.predictor import predict_categories


@patch("categories_classification.predictor.load_model")
@patch("categories_classification.predictor._get_model_input_names")
@patch("categories_classification.predictor._get_latest_model_version")
@patch("categories_classification.predictor.NBR_TOP_LABELS", 2)
@patch("categories_classification.predictor.get_products_categories_dir")
@patch("pyspark.sql.DataFrameReader.csv")
def test_predict_products_success(read_csv_mck,
                                  get_products_categories_dir_mck,
                                  get_latest_model_version_mck,
                                  get_model_input_names_mck,
                                  load_model_mck,
                                  dummy_model,
                                  inference_dataset):
    # Given
    with tempfile.TemporaryDirectory() as tmpdirname:
        client_id = "some_client"
        get_model_input_names_mck.return_value = ["f0", "f1"]
        get_latest_model_version_mck.return_value = ModelVersion(name="", version=1, creation_timestamp="some_ts")
        # Mock read_csv to use custom inference dataset.
        spark = SparkSession.builder.getOrCreate()
        read_csv_mck.return_value = spark.createDataFrame(inference_dataset)
        # Mock joblib load to use custom model
        load_model_mck.return_value = dummy_model
        # Mock get_products_categories_dir to write predictions to temporary directory.
        get_products_categories_dir_mck.return_value = join(tmpdirname, "predictions")

        # When
        predict_categories(client_id, today().isoformat())

        # Then
        prediction_results = spark.read.parquet(join(tmpdirname, "predictions")).toPandas()
        assert len(prediction_results) == 4
