import click

from achilles.achilles import Achilles

from colorama import Fore

R = Fore.RED
RE = Fore.RESET


@click.command()
@click.option(
    "--collections",
    "-c",
    is_flag=True,
    help="List all collections available in cache.",
    metavar="",
)
def list(collections):

    achilles = Achilles()

    if collections:
        achilles.list_collections()
    else:
        print(f'{R}Use the -c flag, other options not yet available.{RE}')