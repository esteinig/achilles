import click
import h5py


@click.command()
@click.option(
    "--file",
    "-f",
    metavar="",
    default=None,
    required=True,
    show_default=True,
    help="HD5 file to inspect.",
)
def inspect(file):

    with h5py.File(file, "r") as data_file:

        if 'data/data' in data_file.keys():
            print(
                data_file['data/data'][:5]
            )
            print(
                data_file['data/labels'][:5]
            )

        print(
            data_file['data/files'][:5]
        )
        print(
            data_file['data/decoded'][:5]
        )