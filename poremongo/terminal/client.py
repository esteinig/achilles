import click

from .sample import sample
from .test import test
from .index import index
from .query import query
from .drop import drop
from .display import display

VERSION = '0.3'


@click.group()
@click.version_option(version=VERSION)
def terminal_client():
    """ PoreMongo: Fast5 management in MongoDB """
    pass


terminal_client.add_command(sample)
terminal_client.add_command(test)
terminal_client.add_command(index)
terminal_client.add_command(query)
terminal_client.add_command(drop)
terminal_client.add_command(display)