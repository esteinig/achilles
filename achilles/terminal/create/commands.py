import click

from poremongo import PoreMongo
from achilles.terminal.utils import get_uri, OptionEatAll
from achilles.dataset import AchillesDataset


@click.command()
@click.option(
    "--pmid",
    "-i",
    default=None,
    help="PoreMongo connection ID.",
    show_default=True,
    metavar="",
)
@click.option(
    "--config",
    "-c",
    default=None,
    metavar="",
    help="YAML configuration file for creating Datasets.",
)
@click.option(
    "--tags",
    "-t",
    cls=OptionEatAll,
    default=None,
    metavar="",
    help="Tags (labels) to sample from, comma separated args.",
)
@click.option(
    "--output",
    "-o",
    "--dataset",
    default="dataset.hd5",
    metavar="",
    show_default=True,
    help="Output HDF5 file containing sampled tensors and labels.",
)
@click.option(
    "--max_windows",
    "-m",
    default=100000,
    metavar="",
    show_default=True,
    help="Maximum number of sampled " "signal value windows per tag / label.",
)
@click.option(
    "--max_windows_per_read",
    "-r",
    default=50,
    metavar="",
    show_default=True,
    help="Maximum number of windows sampled from" " read / diversity of input data.",
)
@click.option(
    "--window_size",
    "-w",
    default=200,
    metavar="",
    show_default=True,
    help="Length of sliding window to sample from signal read.",
)
@click.option(
    "--window_step",
    "-s",
    default=0.1,
    metavar="",
    show_default=True,
    help="Step of sliding window to sample from signal read.",
)
@click.option(
    "--sample_reads_per_tag",
    "--sample",
    default=10000,
    metavar="",
    show_default=True,
    help="Number of random Fast5 models to " "sample from database per tag / label",
)
@click.option(
    "--proportion",
    default="equal",
    metavar="",
    show_default=True,
    help="Proportion of Fast5 models to sample per tag / label",
)
@click.option(
    "--exclude",
    default=None,
    metavar="",
    show_default=True,
    help="Comma separated list of HDF5 datasets to exclude from sampling",
)
@click.option(
    "--global_tags",
    default="R9.4",
    metavar="",
    show_default=True,
    help="Global tags to apply to sample, comma-separated, "
    "e.g. to force pore version: R9.4",
)
@click.option(
    "--validation",
    default=0.3,
    metavar="",
    help="Proportion of data to be split into validation",
)
@click.option(
    "--scale",
    is_flag=True,
    metavar="",
    help="Scale the raw signal (data acquisition values) to pA.",
)
@click.option(
    "--display",
    is_flag=True,
    help="Display tags in database and exit.",
    show_default=True,
    metavar="",
)
def create(
    pmid,
    config,
    display,
    tags,
    output,
    max_windows,
    max_windows_per_read,
    window_size,
    window_step,
    sample_reads_per_tag,
    proportion,
    exclude,
    global_tags,
    validation,
    scale,
):
    """Sample and compile datasets with PoreMongo"""

    tags = [tag.split(",") for tag in tags]

    uri = None
    if pmid:
        uri = get_uri(pmid)
    elif config:
        pass  # uri = get_uri_from_config(config)
    else:
        click.echo(
            "You must enter a PoreMongo ID or " "configuration file to create Datasets."
        )
        exit(1)

    pongo = PoreMongo(uri=uri, connect=True)

    if display:
        pongo.display("tags")
        exit(0)

    ds = AchillesDataset(poremongo=pongo)

    ds.write(
        tags=tags,
        data_file=output,
        max_windows=max_windows,
        max_windows_per_read=max_windows_per_read,
        window_size=window_size,
        window_step=window_step,
        window_random=True,
        window_recover=False,
        sample_files_per_tag=sample_reads_per_tag,
        sample_proportions=proportion,
        sample_unique=False,
        exclude_datasets=exclude,
        global_tags=global_tags,
        validation=validation,
        scale=scale,
        chunk_size=10000,
        ssh=False,
    )

    pongo.disconnect()
