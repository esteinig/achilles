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

    """ Inspect a model or model collection """

    achilles = Achilles()

    if collection and not model:
        achilles.inspect_collection(collection, params=params)
    elif collection and model:
        achilles.inspect_model(collection, model=model, params=params)
    else:
        click.echo(
            'Please specify a collection to inspect a model.', color='red'
        )