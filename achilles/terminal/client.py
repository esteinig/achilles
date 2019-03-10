import click

from .app import app
from .create import create
from .config import config
from .train import train
from .predict import predict

VERSION = "0.3"


@click.group()
@click.version_option(version=VERSION)
def terminal_client():
    pass


terminal_client.add_command(app)
terminal_client.add_command(create)
terminal_client.add_command(config)
terminal_client.add_command(train)
terminal_client.add_command(predict)
