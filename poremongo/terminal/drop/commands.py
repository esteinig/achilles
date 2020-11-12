import click

from pathlib import Path
from functools import partial
from poremongo.poremongo import PoreMongo

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
    '--force', '-f', is_flag=True,
    help='Force confirm database drop without user prompt'
)
def drop(uri, config, force, db):

    """ Drop the database at the given URI """

    if uri == 'local':
        uri = f'mongodb://localhost:27017/{db}'

    db = Path(uri).stem

    pongo = PoreMongo(
        config=config if config else dict(),
        uri=uri if uri else None
    )

    if not force:
        try:
            click.confirm(
                f'Drop database: {db} at {pongo.decompose_uri()}'
            )
        except click.Abort:
            print('Drop terminated. Exiting.')
            exit(0)

    pongo.connect()
    pongo.client.drop_database(db)

    pongo.logger.info(
        f'Dropped database at {pongo.decompose_uri()}'
    )

    pongo.disconnect()


