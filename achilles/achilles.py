"""
=====================
Achilles Base Module
====================

"""

from textwrap import dedent, wrap
from achilles.utils import TableFormatter

from achilles.templates import get_param_template
from achilles.templates import get_collection_template
from achilles.templates import get_collection_yaml_template

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
    """ AchillesModel base management class """

    def __init__(self, verbose=False):

        self.path: Path = Path().home() / '.achilles'
        self.collections: Path = self.path / 'collections'

        self.vprint = print if verbose else lambda *a, **k: None

        # TODO: Change to production CDN:
        self.collections_yaml = get_collection_yaml_template()

        self.collection_template = get_collection_template()

        if not self.collections.exists():
            self.collections.mkdir(parents=True)

    def inspect_model(self, collection, model, params=False):

        data = self.get_collection_yaml(collection=collection)

        models = data['models']

        try:
            model_data = models[model]
        except KeyError:
            print(f'{R}Could not find model: {Y}{model}{RE}')
            raise

        descr = '\n'.join(wrap(
            model_data['description'], width=60
        ))

        labels = ', '.join([
            f'{M}Label {i}{RE}: {C}{label}{RE}'
            for i, label in enumerate(model_data['labels'])
        ])

        header = dedent(
            f"""
            
            {Y}Model Inspection{RE}
            =================
            
            {M}Name{RE}      {C}{model}{RE}
            {M}Date{RE}      {C}{data['date']}{RE}
            {M}Author{RE}    {C}{data['author']}{RE}
            
            {Y}Description{RE}
            ============
            
            {labels}

            """
        )

        header += f'{Y}{descr}{RE}'

        print(header)

        if params:
            with TableFormatter(
                header=['Stage', 'Loss', 'Accuracy'],
                header_color=R,
                header_template='{0:15} {1:10} {2:10}',
                row_template=f'{M}{{0:15}} {LB}{{1:10}} {G}{{2:10}}{RE}'
            ) as table:

                print(table.head)

                train = [
                    'Training',
                    model_data['loss']['training'],
                    model_data['accuracy']['training'],
                ]

                val = [
                    'Validation',
                    model_data['loss']['validation'],
                    model_data['accuracy']['validation']
                ]

                table.format_row(train)
                print(table.row)
                table.format_row(val)
                print(table.row)
        else:
            print()

    def get_collection_yaml(self, collection) -> dict:

        collection_yaml = self.collections / collection / f'{collection}.yaml'

        try:
            return self.read_yaml(collection_yaml)
        except FileNotFoundError:
            print(f'Could not find collection file: {collection_yaml}.')
            raise

    def inspect_collection(self, collection, params=False):

        data = self.get_collection_yaml(collection=collection)

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
            row_template="{0:15} {1:15} {2:15} {3:21}",
            header_color=Y,
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

    def _find_local_model(self, collection, name):

        cyaml = self.read_yaml(
            self.collections / 'collections.yaml'
        )

        try:
            models = cyaml[collection]['models']
        except KeyError:
            print(f'{R}Could not find collection: {collection}.{RE}')
            raise

        model_names = [m.strip('.h5') for m in models]

        if name not in model_names:
            print(f'{R}Could not find model {name} '
                  f'in collection {collection}.{RE}')
            exit(1)
        else:
            return (self.collections / collection / name).with_suffix('.h5')

    def get_model(self, model_name) -> Path:

        """ Name should always be `collection/model` """

        if model_name.endswith('.h5'):
            model_name = model_name.strip('.h5')

        try:
            collection, model = model_name.split('/')
        except ValueError:
            print(f'{R}Model name ({Y}{model_name}{R}) must be '
                  f'in format: {Y}<collection>/<model>{RE}')
            raise

        return self._find_local_model(
            collection=collection, name=model
        )



    @staticmethod
    def read_yaml(yaml_file: Path):

        with yaml_file.open('r') as fstream:
            yml = yaml.load(fstream)

        return yml