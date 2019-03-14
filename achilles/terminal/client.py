import click

from .app import app
from .create import create
from .config import config
from .train import train
from .predict import predict
from .lab import lab
from .inspect import inspect
from .evaluate import evaluate
from .list import list
from .pull import pull

VERSION = "0.3-alpha"


@click.group()
@click.version_option(version=VERSION)
def terminal_client():
    pass


terminal_client.add_command(app)
terminal_client.add_command(create)
terminal_client.add_command(config)
terminal_client.add_command(train)
terminal_client.add_command(predict)
terminal_client.add_command(lab)
terminal_client.add_command(inspect)
terminal_client.add_command(evaluate)
terminal_client.add_command(list)
terminal_client.add_command(pull)