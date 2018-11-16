import yaml
import click

from pathlib import Path
from poremongo import PoreMongo
from achilles.terminal.utils import OptionPromptNull


def get_password(config):
    """ Get password if the config parameter is the prompt (returns tuple from cls OptionPromptNull) """
    if isinstance(config, tuple):
        return config[0]
    else:
        return None


def get_poremongo_uri(pwd, user, host, port, db, config):
    """ Read PoreMongo URI from configuratin file or construct it from parameters. """
    if pwd is None:
        return get_uri_from_config(config)
    else:
        return f'mongodb://{user}:{pwd}@{host}:{port}/{db}'


def get_uri_from_config(config):
    """ Load URI from configuration file (YAML). """
    config_path = Path(config)
    if not config_path.is_file():
        print(f'Configuration file could not be found at: {config_path.absolute()}')
        exit(1)

    with open(config, "r") as config_file:
        config_dict = yaml.load(config_file)
        try:
            return config_dict['uri']
        except IOError:
            print('URI not specified in configuration file.')
            exit(1)


@click.command()
@click.option('--config', '-c', default=None, cls=OptionPromptNull, prompt='PoreMongo password', hide_input=True,
              required=False, help='YAML configuration file for Achilles parameters and connection to PoreMongo. '
                                   'Enable password prompt if argument is not passed.', metavar='')
@click.option('--user', '-u', default='esteinig', help='PoreMongo user name.', show_default=True, metavar='')
@click.option('--host', '-h', default='206.189.91.28', help='PoreMongo host address.', show_default=True, metavar='')
@click.option('--port', '-p', default='27017', help='PoreMongo host port.', show_default=True, metavar='')
@click.option('--database', '-db', default='poremongo', help='PoreMongo database name.', show_default=True, metavar='')
@click.option('--display', '-d', is_flag=True, help='Display tags and exit.', show_default=True, metavar='')
def create(config, user, host, port, database, display):
    """ Create datasets with PoreMongo. """

    pwd = get_password(config)
    uri = get_poremongo_uri(pwd, user, host, port, database, config)

    pongo = PoreMongo(uri=uri, connect=True)

    if display:
        pongo.display('tags')
        exit(0)

    # if config:
    #
    # else:
    #     uri =
    #
    # pongo = PoreMongo(uri=uri)
    # pongo.connect()
    #
    # if args["ssh"]:
    #     pongo.open_ssh(config_file=args["config"])
    #     pongo.open_scp()
    #
    # ds = AchillesDataset(poremongo=pongo)
    #
    # if args["display"]:
    #     ds.poremongo.display_tags()
    #
    # ds.write(tags=args["tags"], data_file=args["data_file"], max_windows=args["max_windows"],
    #          max_windows_per_read=args["max_windows_per_read"], window_size=args["window_size"],
    #          window_step=args["window_step"], window_random=args["random"], window_recover=args["recover"],
    #          sample_files_per_tag=args["sample_files_per_tag"], sample_proportions=args["sample_proportions"],
    #          sample_unique=args["sample_unique"], exclude_datasets=args["exclude"], global_tags=args["global_tags"],
    #          validation=args["validation"], scale=args["scale"], chunk_size=args["chunk_size"], ssh=args["ssh"])
    #
    # pongo.disconnect()
    # if args["ssh"]:
    #     pongo.close_ssh()
    #     pongo.close_scp()