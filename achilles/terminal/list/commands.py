import click

from achilles.achilles import Achilles
from pathlib import Path


@click.command()
@click.option(
    "--collections",
    "-c",
    is_flag=True,
    help="List all collections available in cache.",
    metavar="",
)
@click.option(
    "--models",
    "-m",
    is_flag=True,
    help="List all models available in cache.",
    metavar="",
)
def list(models, collections):

    achilles = Achilles()

    if collections:
        achilles.list_collections()
