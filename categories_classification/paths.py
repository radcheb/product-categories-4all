from os import getenv
from os.path import join

BASE_URI = getenv("DATA_PREFIX", "data")
TRAIN_DATA_DIR = "train"
TRAIN_DATA_FILENAME = "data_train.csv.gz"
INFERENCE_DATA_DIR = "prediction"
INFERENCE_DATA_FILENAME = "data_predict.csv.gz"
PRODUCTS_CATEGORIES_DIR = "products_categories"


def get_client_train_data_uri(client_id):
    """
    Given a client, forge training data path.
    :param client_id:
    :return:
    """
    return join(BASE_URI, TRAIN_DATA_DIR, f"client_id={client_id}", TRAIN_DATA_FILENAME)


def get_client_inference_data_uri(client_id):
    """
    Given a client, forge inference inout data path.
    :param client_id:
    :return:
    """
    return join(BASE_URI, INFERENCE_DATA_DIR, f"client_id={client_id}", INFERENCE_DATA_FILENAME)


def get_products_categories_dir(client_id, inference_date):
    """
    Given a client, forge product predicted categories path.
    :param client_id:
    :param inference_date:
    :return:
    """
    return join(BASE_URI, PRODUCTS_CATEGORIES_DIR, f"client_id={client_id}", f"date={inference_date}")
