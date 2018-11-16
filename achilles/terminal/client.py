import click

from .app import app
from .create import create

VERSION = '0.2'

@click.group()
@click.version_option(version=VERSION)
def entry_point():
    pass


entry_point.add_command(app)
entry_point.add_command(create)