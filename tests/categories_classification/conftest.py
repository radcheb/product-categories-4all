import numpy as np
from pandas import DataFrame
from pytest import fixture
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification


@fixture
def training_dataset():
    X, y = make_classification(n_samples=100,
                               n_features=3,
                               n_informative=2,
                               n_redundant=1,
                               random_state=1234)
    return DataFrame({
        "f0": X[:, 0],
        "f1": X[:, 0],
        "category_id": y
    })


class TemplateClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.classes_ = np.array(["c0", "c1"], dtype="object")

    def predict(self, X):
        return np.array(["c0" for i in X[0]])

    def predict_proba(self, X):
        if isinstance(X, DataFrame):
            return X.values
        return X

@fixture
def dummy_model():
    return TemplateClassifier()


@fixture
def inference_dataset():
    return DataFrame({
        "product_id": ["p1", "p2"],
        "f0": [0.1, 0.5],
        "f1": [0.2, 0.3]
    })
