import click
import uuid

from pathlib import Path
from functools import partial
from poremongo.poremongo import PoreMongo

from ont_fast5_api.fast5_interface import get_fast5_file

# Monkey patching to show all default options
click.option = partial(click.option, show_default=True)


@click.command()
@click.option(
    '--tags', '-t', type=str, default=None, required=True,
    help='Tags to apply to all reads in Fast5 files; comma-separated'
)
@click.option(
    '--fast5', '-f', type=Path, default=None,
    help='Path to single Fast5 or directory of Fast5 (.fast5)'
)
@click.option(
    '--db', '-d', type=str, default='poremongo',
    help='Database name to create or connect to'
)
@click.option(
    '--uri', '-u', type=str, default='local',
    help='MongoDB URI, "local" for localhost:27017 or custom Mongo URI'
)
@click.option(
    '--config', '-c', type=Path, default=None,
    help='Path to config file for database connection'
)
def index(uri, config, fast5, db, tags):

    """ Index signal reads from Fast5 files """

    if uri == 'local':
        uri = f'mongodb://localhost:27017/{db}'

    pongo = PoreMongo(
        config=config if config else dict(),
        uri=uri if uri else None
    )
    pongo.connect()

    if tags:
        tags = tags.split(",")
    else:
        tags = []

    if fast5.is_dir():
        files = fast5.glob("*.fast5")
    elif fast5.is_file():
        files = [fast5]
    else:
        raise ValueError(f'Fast5 input is neither directory nor file: {fast5}')

    pongo.index_fast5(files=files, tags=tags, store_signal=False)

    pongo.disconnect()








