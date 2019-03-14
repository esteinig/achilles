import click

from achilles.achilles import Achilles


@click.command()
def pull():

    """ Pull model collections into a local cache """

    achilles = Achilles()
    achilles.pull_collections()
