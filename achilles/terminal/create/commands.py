import click

from pathlib import Path
from poremongo import PoreMongo
from achilles.terminal.utils import OptionEatAll
from achilles.dataset import AchillesDataset


@click.command()
@click.option(
    "--uri",
    "-u",
    default="local",
    help="PoreMongo connection 'local' or URI ",
    show_default=True,
    metavar="",
)
@click.option(
    "--db",
    "-db",
    default="poremongo",
    help="PoreMongo database to sample from",
    show_default=True,
    metavar="",
)
@click.option(
    "--tags",
    "-t",
    cls=OptionEatAll,
    default=None,
    metavar="",
    help="Tags (labels) to sample from, comma separated args",
)
@click.option(
    '--config',
    '-c', type=Path, default=None,
    help='Path to config file for database connection'
)
@click.option(
    "--dataset",
    "-d",
    default="dataset.hd5",
    metavar="",
    show_default=True,
    help="Dataset HDF5 file containing sampled tensors and labels",
)
@click.option(
    "--max_windows",
    "-mw",
    default=100000,
    metavar="",
    show_default=True,
    help="Maximum number of sampled " "signal value windows per tag / label",
)
@click.option(
    "--max_windows_per_read",
    "-mwr",
    default=50,
    metavar="",
    show_default=True,
    help="Maximum number of windows sampled from" " read / diversity of input data",
)
@click.option(
    "--window_size",
    "-wsz",
    default=200,
    metavar="",
    show_default=True,
    help="Length of sliding window to sample from signal read",
)
@click.option(
    "--window_step",
    "-wsp",
    default=0.1,
    metavar="",
    show_default=True,
    help="Step of sliding window to sample from signal read",
)
@click.option(
    "--sample_reads_per_tag",
    "--sample",
    "-s",
    default=10000,
    metavar="",
    show_default=True,
    help="Number of random Fast5 models to sample from database per tag / label",
)
@click.option(
    "--proportion",
    '-p',
    default="equal",
    metavar="",
    show_default=True,
    help="Proportion of Fast5 models to sample per tag / label",
)
@click.option(
    "--exclude",
    "-e",
    default=None,
    metavar="",
    show_default=True,
    help="Comma separated list of HDF5 datasets to exclude from sampling",
)
@click.option(
    "--global_tags",
    "-g",
    default=None,
    metavar="",
    show_default=True,
    help="Global tags to apply to sample, comma-separated, e.g. to force pore version: R9.4",
)
@click.option(
    "--validation",
    "-v",
    default=0.3,
    metavar="",
    help="Proportion of data to be split into validation",
)
def create(
    uri,
    db,
    tags,
    dataset,
    config,
    max_windows,
    max_windows_per_read,
    window_size,
    window_step,
    sample_reads_per_tag,
    proportion,
    exclude,
    global_tags,
    validation
):
    """Sample and compile datasets with PoreMongo"""

    tags = [tag.split(",") for tag in tags]

    if uri == 'local':
        uri = f'mongodb://localhost:27017/{db}'

    pongo = PoreMongo(uri=uri, config=config if config else dict())
    pongo.connect()

    ds = AchillesDataset(poremongo=pongo)

    ds.write(
        tags=tags,
        data_file=dataset,
        max_windows=max_windows,
        max_windows_per_read=max_windows_per_read,
        window_size=window_size,
        window_step=window_step,
        window_random=True,  # sample a sequence of signal windows from random start point
        window_recover=False,  # do not recover incomplete sample of contiguous windows (mimick real seq end)
        sample_reads_per_tag=sample_reads_per_tag,
        sample_proportions=proportion,
        sample_unique=False,  # can be used as safe guard
        exclude_datasets=exclude,
        global_tags=global_tags.split() if global_tags else None,
        validation=validation,
        chunk_size=10000
    )

    pongo.disconnect()
