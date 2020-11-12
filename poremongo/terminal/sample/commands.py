import click
import json as js

import logging
from pathlib import Path
from poremongo.poremodels import Read
from functools import partial
from poremongo.poremongo import PoreMongo

# Monkey patching to show all default options
click.option = partial(click.option, show_default=True)


@click.command()

@click.option(
    '--tags', '-t', type=str, default=None, required=True,
    help='Comma separated string for list of tags to query'
)
@click.option(
    '--db', '-d', type=str, default='poremongo',
    help='Database name to create or connect to'
)
@click.option(
    '--uri', '-u', type=str, default='local',
    help='MongoDB connection: "local" or URI'
)
@click.option(
    '--config', '-c', type=Path, default=None,
    help='Path to config for MongoDB connection [none]'
)
@click.option(
    '--sample', '-s', type=int, default=10,
    help='Number of reads to sample'
)
@click.option(
    '--proportion', '-p', type=str, default=None,
    help='Proportion to sample across tags'
)
@click.option(
    '--display', '-d', is_flag=True,
    help='Print query results in human readable row summary format to STDOUT'
)
@click.option(
    '--quiet', '-q', is_flag=True,
    help='Suppress logging output'
)
@click.option(
    '--unique', is_flag=True,
    help='Force sampled documents to be unique by their ObjectID'
)
@click.option(
    '--json_in', type=Path, default=None,
    help='Process query results (in memory): input query results from JSON'
)
@click.option(
    '--json_out', type=Path, default=None,
    help='Process query results (in memory): output query results as JSON'
)
def sample(
    uri,
    config,
    tags,
    db,
    proportion,
    json_in,
    json_out,
    unique,
    display,
    quiet,
    sample
):

    """ Query a Fast5 collection with PoreMongo """

    if uri == 'local':
        uri = f'mongodb://localhost:27017/{db}'

    pongo = PoreMongo(
        config=config if config else dict(),
        uri=uri if uri else None
    )

    if quiet:
        pongo.logger.setLevel(logging.ERROR)

    pongo.connect()

    if json_in:
        with open(json_in, 'r') as infile:
            data = js.load(infile)
            read_objects = [Read(**entry) for entry in data]
    else:
        read_objects = Read.objects

    read_objects = pongo.sample(
        objects=read_objects, tags=tags.split(','), unique=unique, limit=sample,
        proportion=proportion.split(',') if proportion else '',
    )

    if display:
        for o in read_objects:
            print(o)

    if json_out:
        if isinstance(read_objects, list):
            data_dict = [
                js.loads(
                    o.to_json()
                ) for o in read_objects
            ]
        else:
            data_dict = js.loads(
                read_objects.to_json()
            )
        with open(json_out, 'w') as outfile:
            js.dump(data_dict, outfile)

    pongo.disconnect()



