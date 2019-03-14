"""
=====================
Achilles Base Module
====================

"""

from textwrap import dedent
from achilles.utils import TableFormatter
from achilles.templates import get_param_template
from pathlib import Path

import shutil
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

    def __init__(self, verbose=False):

        self.path: Path = Path().home() / '.achilles'
        self.collections: Path = self.path / 'collections'

        self.vprint = print if verbose else lambda *a, **k: None

        # TODO: Change to production CDN:
        self.collections_yaml = 'https://raw.githack.com/esteinig/achilles/' \
                                'master/models/collections.yaml'

        self.collection_template = 'https://raw.githack.com/esteinig/achilles/'\
                                   'master/models/{collection}/{file}'

        if not self.collections.exists():
            self.collections.mkdir(parents=True)

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
            row_template=f"{Y}{{0:21}} {G}{{1:15}} {LB}{{2:21}}{RE}",
            header_template="{0:21} {1:15} {2:21}",
            header_color=R,
        ) as table:

            print(
                dedent(f"""
                {Y}Collection Inspection{RE}
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
                        f'{Y}No collections found in local cache,'
                        f'use: {R}achilles pull{RE}'
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
                    collection.name, yml["date"],
                    yml["author"], yml["description"]
                ]

                table.format_row(
                    data, color=color if row_num % 2 == 0 else color
                )

                row_num += 1

                print(table.row)

    def pull_collections(self):
        """ Pull collections into local cache """

        print(f'{Y}Pulling model collections from GitHub.{RE}')

        collection_yaml = self.pull_collections_yaml()

        collections = self.read_yaml(collection_yaml)

        for collection, data in collections.items():
            cpath = self.collections / collection

            # Model files
            file_urls = [
                self.collection_template.format(
                    collection=collection, file=file
                )
                for file in data['models']
            ]

            # Configuration YAML
            file_urls += [
                self.collection_template.format(
                    collection=collection, file=data['config']
                )
            ]

            # Check if collection exists:
            if cpath.exists():
                self.vprint(f'{Y}Updating collection: {G}{collection}{RE}')
                shutil.rmtree(cpath)

            cpath.mkdir(parents=True)

            # Download collection files:
            for url in file_urls:
                fpath = str(
                    self.collections / collection / Path(url).name
                )
                wget.download(url, fpath, bar=None)

            self.vprint(f'{Y}Downloaded collection: {G}{collection}{RE}.')

    def pull_collections_yaml(self) -> Path:
        """ Pull the collections.yaml file from Github """
        fpath = self.collections / 'collections.yaml'

        if fpath.exists():
            self.vprint(f'{Y}Updating collection YAML.{RE}')
            fpath.unlink()

        wget.download(
            str(self.collections_yaml), str(fpath), bar=None
        )

        return fpath

    @staticmethod
    def read_yaml(yaml_file: Path):

        with yaml_file.open('r') as fstream:
            yml = yaml.load(fstream)

        return yml