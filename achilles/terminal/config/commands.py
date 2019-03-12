import click

from achilles.terminal.utils import read_config_path, write_config_path


def display_uri():

    config = read_config_path(config_file="poremongo.json")
    for pmid, uri in config.items():
        print(f"{pmid}\t{uri}")


def delete_uri(pmid):

    config = read_config_path(config_file="poremongo.json")

    try:
        _ = config[pmid]
        if click.confirm(f"Do you want to delete PMID {pmid}?"):
            del config[pmid]
            write_config_path(config, config_file="poremongo.json")
            click.echo(f"Deleted PMID: {pmid}")
    except KeyError:
        click.echo(f"Could not find PMID: {pmid}")


def save_uri(pmid, user, host, port, db):
    # TODO: Plain text storage of URIs,
    # TODO: need to have master password for file encryption

    config = read_config_path(config_file="poremongo.json")

    pw = click.prompt("Enter password for MongoDB", type=str)

    uri = f"mongodb://{user}:{pw}@{host}:{port}/{db}"

    try:
        _ = config[pmid]
        if click.confirm(f"Do you want to overwrite PMID {pmid}?"):
            config[pmid] = uri
        config[pmid] = uri
        click.echo(f"Saved connection with PMID: {pmid}")
    except KeyError:
        config[pmid] = uri
        click.echo(f"New connection saved with PMID: {pmid}")

    write_config_path(config, config_file="poremongo.json")


@click.command()
@click.option(
    "--save", is_flag=True, help="Save a connection with PMID (--pmid)."
)
@click.option(
    "--delete", is_flag=True, help="Delete connection with PMID (--pmid)."
)
@click.option(
    "--display", is_flag=True, help="Display saved connections for PoreMongo."
)
@click.option(
    "--pmid",
    "-i",
    default=None,
    help="PoreMongo connection (URI) ID.",
    show_default=True,
    metavar="",
)
@click.option(
    "--user",
    "-u",
    default="",
    help="PoreMongo user name.",
    show_default=True,
    metavar="",
)
@click.option(
    "--host",
    "-h",
    default="localhost",
    help="PoreMongo host address.",
    show_default=True,
    metavar="",
)
@click.option(
    "--port",
    "-p",
    default="27017",
    help="PoreMongo host port.",
    show_default=True,
    metavar="",
)
@click.option(
    "--db",
    "-d",
    default="poremongo",
    help="PoreMongo database name.",
    show_default=True,
    metavar="",
)
def config(pmid, user, host, port, db, save, delete, display):
    """Configure and save connections to PoreMongo"""

    if display:
        display_uri()

    if delete:
        if pmid:
            delete_uri(pmid)
        else:
            click.echo("Enter PMID (--pmid) to delete connection to PoreMongo.")

    if save:
        if pmid:
            save_uri(pmid, user, host, port, db)
        else:
            click.echo("Enter PMID (--pmid) to save connection to PoreMongo.")