import click
from numpy import argmax
from achilles.model import AchillesModel
from achilles.utils import get_dataset_labels
from colorama import Fore
from pathlib import Path
Y = Fore.YELLOW
G = Fore.GREEN
RE = Fore.RESET

@click.command()
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model file HD5.",
    show_default=True,
    metavar="",
)
@click.option(
    "--evaluation",
    "-e",
    default=None,
    help="Evaluation file HD5 sampled from AchillesModel.",
    show_default=True,
    metavar="",
)
@click.option(
    "--batch_size",
    "-b",
    default=500,
    help="Evaluation batch size.",
    show_default=True,
    metavar="",
)
def evaluate(model, evaluation, batch_size):

    achilles = AchillesModel(evaluation)
    achilles.load_model(model_file=model)

    print(f'{Y}Evaluating model: {G}{Path(model).name}{RE}')
    print(f'{Y}Using evaluation data from: {G}{Path(evaluation).name}{RE}')

    predicted = achilles.predict_generator(
            data_type="data", batch_size=batch_size
    )

    print(predicted)

    predicted = argmax(predicted, -1)

    labels = get_dataset_labels(evaluation)

    correct_labels = 0
    false_labels = 0
    for i, label in enumerate(predicted):
        if int(label) == int(argmax(labels[i])):
            correct_labels += 1
        else:
            false_labels += 1

    print(f'False predictions in evaluation data: '
          f'{correct_labels/false_labels:.2f}%')
