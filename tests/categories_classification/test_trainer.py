from unittest.mock import patch

from categories_classification.trainer import train_model


@patch("categories_classification.trainer.log_model")
@patch("mlflow.tracking.client.MlflowClient")
@patch("categories_classification.trainer.read_csv")
def test_train_model_success(read_csv_mck, _, log_model_mck, training_dataset):
    # Given
    client_id = "some_client"
    features = ["f0", "f1"]
    model_params = {"n_estimators": 10, "random_state": 1234}
    # Mock read_csv to use our training_dataset fixture.
    read_csv_mck.return_value = training_dataset

    # When
    train_model(client_id, features, model_params, "some_date")

    # Then
    assert log_model_mck.call_count == 1
    print(log_model_mck.call_args_list)
    trained_model = log_model_mck.call_args_list[0][1]["sk_model"]
    # trained_model = load(join(tmpdirname, "model.joblib"))
    # Assert model better than random
    assert trained_model.score(training_dataset[["f0", "f1"]], training_dataset["category_id"]) > 0.5
    # Check all model params are set
    for k, v in model_params.items():
        assert getattr(trained_model, k) == v
    # Check model properties are compatible with training dataset
    assert trained_model.n_features_ == 2
    assert trained_model.n_classes_ == 2
