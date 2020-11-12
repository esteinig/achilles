import click
import logging

from pathlib import Path
from functools import partial
from poremongo.poremongo import PoreMongo
from poremongo.poremodels import Read
# Monkey patching to show all default options
click.option = partial(click.option, show_default=True)


@click.command()
@click.option(
    '--uri', '-u', type=str, default='local',
    help='MongoDB connection: "local" or URI'
)
@click.option(
    '--db', '-d', type=str, default="poremongo",
    help='Name of database to connect to [poremongo]'
)
@click.option(
    '--config', '-c', type=Path, default=None,
    help='Path to JSON config file for MongoDB connection.'
)
@click.option(
    '--limit', '-l', type=int, default=100,
    help='Limit the number of unique tags on display.'
)
@click.option(
    '--total', '-t', is_flag=True,
    help='Add a total count of reads in --db [false]'
)
@click.option(
    '--quiet', '-q', is_flag=True,
    help='Suppress logging output'
)
def display(uri, config, limit, db, quiet, total):

    """ Display most common tags in the database """

    if uri == 'local':
        uri = f'mongodb://localhost:27017/{db}'

    pongo = PoreMongo(
        config=config if config else dict(),
        uri=uri if uri else None
    )

    if quiet:
        pongo.logger.setLevel(logging.ERROR)

    pongo.connect()

    counts = pongo.get_tag_counts(limit=limit)

    print("tag\tread_count")
    for tag in counts:
        print(f'{tag["_id"]}\t{tag["count"]}')

    if total:
        print(f'total\t{len(list(Read.objects))}')
    pongo.disconnect()


