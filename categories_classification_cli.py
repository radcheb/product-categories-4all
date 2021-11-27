import json
from json import JSONDecodeError

import click

from categories_classification.predictor import predict_categories
from categories_classification.trainer import train_model


class JsonOption(click.Option):
    """
    A simple Click library class for json arguments. Can be used for python dict or lists.
    """

    def type_cast_value(self, ctx, value):
        """
        Parse input as JSON.
        :param ctx:
        :param value:
        :return:
        """
        try:
            return json.loads(value)
        except JSONDecodeError as err:
            raise click.BadParameter(value) from err


@click.command()
@click.option('--client_id', help='The id of the client.', required=True)
@click.option('--features', cls=JsonOption, help='The list of input features.', required=True)
@click.option('--model_params', cls=JsonOption, help='Params to be passed to model.', required=True)
@click.option('--training_date', help='The training date.', required=True)
def trainer(client_id, features, model_params, training_date):
    """

    :param training_date:
    :param client_id:
    :param features:
    :param model_params:
    :return:
    """
    train_model(client_id, features, model_params, training_date)


@click.command()
@click.option('--client_id', help='The id of the client.', required=True)
@click.option('--inference_date', help='The inference date.', required=True)
def predictor(client_id, inference_date):
    """

    :param client_id:
    :param inference_date:
    :return:
    """
    predict_categories(client_id, inference_date)


@click.group()
def main():
    """Better Categories 4All Demo"""


if __name__ == '__main__':
    main.add_command(trainer)
    main.add_command(predictor)
    main()
