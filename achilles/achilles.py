"""
=====================
Achilles Base Module
====================

"""

from textwrap import dedent
from achilles.utils import TableFormatter
from achilles.templates import get_param_template
from pathlib import Path

import delegator
import wget
import yaml

from colorama import Fore

C = Fore.CYAN
R = Fore.RED
G = Fore.GREEN
LB = Fore.LIGHTBLUE_EX
RE = Fore.RESET
M = Fore.MAGENTA
Y = Fore.YELLOW


class Achilles:
    """ Achilles base management class """

    def __init__(self):

        self.path: Path = Path().home() / '.achilles'
        self.collections: Path = self.path / 'collections'

        self.collections_yaml = 'https://raw.githack.com/esteinig/achilles/' \
                                'master/models/collections.yaml'

        self.models_template = 'https://github.com/esteinig/achilles/' \
                               'trunk/models/{collection}'

    def inspect_collection(self, collection, params=False):

        collection_yaml = self.collections / collection / f'{collection}.yaml'

        data = self.read_yaml(collection_yaml)

        ds = data['config']['create']
        tr = data['config']['train']

        model_data = []
        for name, attr in data['models'].items():
            val_acc = attr['accuracy']['validation']
            labels = ", ".join(attr['labels'])
            model_data.append(
                [name, f'{val_acc*100:.2f}', labels]
            )

        with TableFormatter(
            header=['Model', "Validation", "Labels"],
            row_template=f"{Y}{{0:15}} {G}{{1:^15}} {LB}{{2:21}}{RE}",
            header_template="{0:15} {1:15} {2:21}",
            header_color=R,
        ) as table:

            print(
                dedent(f"""
                {R}Collection Inspection{RE}
                ====================== 
                
                {M}Name{RE}     {C}{collection}{RE}
                {M}Date{RE}     {C}{data['date']}{RE}
                {M}Author{RE}   {C}{data['author']}{RE}
                {M}Note{RE}     {C}{data['description']}{RE}
                """)
            )

            print(table.head)
            for row_data in model_data:
                table.format_row(data=row_data, color=M)
                print(table.row)

            if params:
                param_temp = get_param_template(ds=ds, tr=tr)
                print(param_temp)

    def list_collections(self, remote: bool = False):
        """ List local or remote model collections """

        with TableFormatter(
            header=['Collection', 'Date', 'Author', 'Description'],
            row_template="{0:15} {1:15} {2:15} {3:21}"
        ) as table:

            if remote:
                pass
            else:
                collections = list(
                    self.collections.glob('*')
                )
                if not collections:
                    print(
                        f'{Y} No collections found in local cache,'
                        f'use {R}achilles pull{RE}'
                    )
                    exit(1)
                else:
                    self._list_collections(
                        collections=collections, table=table
                    )

    def _list_collections(self, collections, table, color: Fore = RE):

        row_num = 0
        print(table.head)
        for collection in collections:
            if collection.is_dir():
                yml = self.read_yaml(
                    yaml_file=self.collections / collection /
                    f'{collection.name}.yaml'
                )

                data = [
                    collection.name,
                    yml["date"],
                    yml["author"],
                    yml["description"]
                ]

                table.format_row(
                    data, color=color if row_num % 2 == 0 else color
                )

                row_num += 1

                print(table.row)

    def pull_collections(self):
        """ Pull collections into local cache """

        collection_yaml = self.pull_collections_yaml()

        cnames = self.read_yaml(collection_yaml)
        print(cnames)
        for collection in cnames:
            cpath = self.collections / collection
            giturl = self.models_template.format(collection=collection)

            print(giturl, cpath)

            delegator.run(
                f'git-svn export {giturl} {cpath}'
            )

            print(f'{Y}Downloaded collection: {G}{collection}{RE}.')

    def pull_collections_yaml(self) -> Path:
        """ Pull the collections.yaml file from Github """
        fpath = self.collections / 'collections.yaml'
        wget.download(
            self.collections_yaml, str(fpath)
        )
        return fpath

    @staticmethod
    def read_yaml(yaml_file: Path):

        with yaml_file.open('r') as fstream:
            return yaml.load(fstream)
