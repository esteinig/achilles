import click
import h5py

from achilles.achilles import Achilles

@click.command()
@click.option(
    "--model",
    "-m",
    metavar="",
    default=None,
    required=False,
    show_default=True,
    help="Model file to inspect.",
)
@click.option(
    "--collection",
    "-c",
    metavar="",
    default=None,
    required=False,
    show_default=True,
    help="Name or UUID of model collection in local cache.",
)
@click.option(
    "--params",
    "-p",
    metavar="",
    is_flag=True,
    show_default=True,
    help="Show detailed collection parameters for sampling and training stages.",
)
def inspect(model, collection, params):

    if collection:
        achilles = Achilles()

        achilles.inspect_collection(collection, params=params)
