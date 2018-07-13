import os
import h5py
import random
import shutil
import json
import tarfile
import pandas
import textwrap
from colorama import Fore, Style

from tqdm import tqdm
import multiprocessing
from achilles.utils import read_signal, chunk

from pandas.errors import EmptyDataError


test_config = {
    "training": {
        "host": {
            "data": {
                "minimal": ["chr_2_part03", "chr_4_part06"],
                "diverse": ["chr_2_part03", "chr_4_part06", "chr_8_part07",
                            "chr_14_part06", "chr_20_part05"]
            },
            "params": {
                "exclude": None,
                "include": ["FAB23716"],
                "limit": 20000,
                "basedir": "/rdsi/vol08/Q0220/Eike/nanopore/human",
                "shuffle": True,
                "min_signal": None
                }
            },
        "pathogen": {
            "data": {
                "zibra_1": ["zika_library2"],
            },
            "params": {
                "exclude": None,
                "include": None,
                "limit": 10000,
                "basedir": "/rdsi/vol08/Q0220/Eike/nanopore/zibra",
                "shuffle": True,
                "min_signal": None
            }
        }
    },
}

evaluate_config = {"evaluation": {
            "host": {"data": {
                            "minimal": ["chr_3_part03", "chr_5_part06"],
                            "diverse": ["chr_3_part03", "chr_5_part06", "chr_9_part07",
                                        "chr_15_part06", "chr_21_part05"]
                    }, "params": {
                            "exclude": None,
                            "include": ["FAB23716"],
                            "limit": 1000,
                            "basedir": "/rdsi/vol08/Q0220/Eike/nanopore/human",
                            "shuffle": True,
                            "min_signal": None
                     }},
            "pathogen": {"data": {
                            "day2": ["zika_library2"],
                            "day3": ["zika_library3"],
                            "day4": ["zika_library4"],
                            "day5": ["zika_library5"]
                    }, "params": {
                            "exclude": None,
                            "include": None,
                            "limit": 1000,
                            "basedir": "/rdsi/vol08/Q0220/Eike/nanopore/zibra",
                            "shuffle": True,
                            "min_signal": None
                     }
            }
    }
}



class DataSelector:
    """ Mix and match samples from Fast5 directories to generate DataSets """

    def __init__(self, outdir, basedir=None, config=None, config_file=None, ncpu=1, chunk_size=100):

        self.config = config
        self.config_file = config_file

        self.ncpu = ncpu
        self.chunk_size = chunk_size

        if config_file is not None:
            with open(self.config_file, "r") as cfg:
                self.config = json.load(cfg)

        self.outdir = os.path.abspath(outdir)

        self.default_params = {
                "exclude": None,
                "include": None,
                "limit": 1000,
                "basedir": "" if basedir is None else basedir,
                "shuffle": True,
                "min_signal": None
            }

    @staticmethod
    def print_params(params, out_dir, sample_dirs, base_dir, data="training"):

        print(textwrap.dedent(f"""
        PARAMS FOR {Fore.RED}{data.upper()}{Style.RESET_ALL} CONFIG
        =================================
        Limit:      {params["limit"]}
        Shuffle:    {params["shuffle"]}
        Signal:     {params["min_signal"]}
        Exclude:    {", ".join([f'{Fore.RED}{e}{Style.RESET_ALL}' for e in params["exclude"]])
                                if params["exclude"] is not None else None}
        Include:    {", ".join([f'{Fore.GREEN}{e}{Style.RESET_ALL}' for e in params["include"]])
                                if params["include"] is not None else None}
        Sampling:   {", ".join([os.path.basename(d) for d in sample_dirs])}
        From:       {base_dir}
        To:         {out_dir}
        """))

    def run_training_config(self):

        # Training data set generation:

        training_host_data, training_pathogen_data, training_host_params, training_pathogen_params = \
            self._get_data_params(self.config["training"])

        host_basedir = training_host_params.pop("basedir")
        pathogen_basedir = training_pathogen_params.pop("basedir")

        if host_basedir is None or pathogen_basedir is None:
            raise ValueError("Please specify data_path or include basedir in host and pathogen configurations.")

        # Data selection combinations:
        host_pathogen_data = [(host, pathogen) for pathogen in training_pathogen_data.keys()
                              for host in training_host_data.keys()]

        for host_id, pathogen_id in host_pathogen_data:
            print(f"\n{Fore.YELLOW}Preparing data set for processing.\n"
                  f"=================================={Style.RESET_ALL}")

            data_set_id = host_id + "_" + pathogen_id
            data_set_path = os.path.join(self.outdir, data_set_id)

            os.makedirs(data_set_path, exist_ok=False)

            host_path = os.path.join(data_set_path, "host")
            pathogen_path = os.path.join(data_set_path, "pathogen")

            host_dirs = [os.path.join(host_basedir, fast5_dir)
                         for fast5_dir in training_host_data[host_id]]

            pathogen_dirs = [os.path.join(pathogen_basedir, fast5_dir)
                             for fast5_dir in training_pathogen_data[pathogen_id]]

            def _print_progress(label):
                print(f"\n{Fore.GREEN}Sampling {Fore.YELLOW}{label}{Fore.GREEN} reads in progress for"
                      f" dataset {Fore.YELLOW}{data_set_id}{Style.RESET_ALL}")

            _print_progress("host")
            self.print_params(training_host_params, host_path, host_dirs, host_basedir)
            self.sample_mix(host_dirs, host_path, ncpu=self.ncpu, chunk_size=self.chunk_size,
                            **training_host_params)

            _print_progress("pathogen")
            self.print_params(training_pathogen_params, pathogen_path, pathogen_dirs, pathogen_basedir)
            self.sample_mix(pathogen_dirs, pathogen_path, ncpu=self.ncpu, chunk_size=self.chunk_size,
                            **training_pathogen_params)

        print("\n")

    def _get_data_params(self, config):

        training_pathogen, training_host = config["pathogen"], config["host"]

        return training_host["data"], training_pathogen["data"],\
            {**self.default_params, **training_host["params"]}, \
            {**self.default_params, **training_pathogen["params"]}

    @staticmethod
    def sample_mix(dirs, outdir, shuffle=True, limit=12000, min_signal=None, include=None,
                   exclude=None, ncpu=1, chunk_size=100):

        # TODO: Might want to add recursive option?
        sample_limit = limit//len(dirs)
        files = []

        for d in dirs:
            fast5 = select_fast5(input_dir=d, output_dir=outdir, limit=sample_limit, min_signal=min_signal,
                                 shuffle=shuffle, exclude=exclude, include=include, ncpu=ncpu, chunk_size=chunk_size)

            nb_target = f"{Fore.YELLOW}{sample_limit}{Style.RESET_ALL}"
            nb_sampled = f"{Fore.RED}{len(fast5)}{Style.RESET_ALL}" if len(fast5) < sample_limit \
                else f"{Fore.GREEN}{len(fast5)}{Style.RESET_ALL}"
            print(f"{os.path.basename(d)}: {nb_sampled} ({nb_target})")
            files.append(fast5)

        return files


def select_fast5(input_dir, output_dir=None, limit=None, min_signal=None, symlink=False, shuffle=True,
                 exclude=None, include=None, recursive=True, ncpu=1, chunk_size=100):

    fast5_paths = filter_fast5(input_dir, include=include, min_signal=min_signal, shuffle=shuffle,
                               limit=limit, exclude=exclude, recursive=recursive)

    # Copy / link files to output directory:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        if ncpu == 1:
            with tqdm(total=len(fast5_paths)) as pbar:
                pbar.set_description("Copying files")
                copy_link_files(fast5_paths, output_dir=output_dir, pbar=pbar, symlink=symlink)
        else:
            # Multiprocessing file copies, chunk paths into
            # processor chunks with limit:

            fast5_chunks = list(chunk(fast5_paths, chunk_size))
            nb_chunks = len(fast5_chunks)

            def cbk_update(*args):
                pass

            pool = multiprocessing.Pool(processes=ncpu)
            for i in range(nb_chunks):
                pool.apply_async(copy_link_files, args=(fast5_chunks[i], output_dir, None, symlink, ),
                                 callback=cbk_update)
            pool.close()
            pool.join()

    return fast5_paths


def copy_link_files(fast5_paths, output_dir, pbar=None, symlink=False, overwrite=False):

    for file_path in fast5_paths:
        target_path = os.path.join(output_dir, os.path.basename(file_path))
        if os.path.exists(target_path):
            if not overwrite:
                return

        if symlink:
            # If not copy, symlink:
            file_name = os.path.basename(file_path)
            target_link = os.path.join(output_dir, file_name)

            os.symlink(file_path, target_link)
        else:
            # Copy files to target directory
            # TODO: skip if paths exist?
            shutil.copy(file_path, output_dir)

        if pbar:
            pbar.update(n=1)


def filter_fast5(input_dir, min_signal=None, shuffle=True, limit=1000, include=None, exclude=None, recursive=True):

    # Check for datasets in include or exclude lists, unpack
    # file basenames for inclusion or exclusion:

    include, exclude = check_include_exclude(include, exclude, recursive)

    fast5 = get_recursive_files(input_dir, include=include, exclude=exclude,
                                extension=".fast5", recursive=recursive)

    if shuffle:
        random.shuffle(fast5)

    if min_signal:
        if limit is None:
            raise ValueError("Selecting Fast5 with minimum signal length requires specifying a limit.")

        # Filter for minimum signal length:
        lim = 0
        filtered = []
        with tqdm(total=limit) as pbar:
            pbar.set_description("Fast5 filtering")
            for file in fast5:
                _, signal_length = read_signal(file, normalize=False, scale=False, return_signal=True)
                if signal_length >= min_signal:
                    filtered.append(file)
                    lim += 1
                    pbar.update(n=1)
                    if lim == limit:
                        break
        return filtered
    else:
        return fast5[:limit]


def get_tarred_fast5(input_dir, shuffle=True,  include=None, exclude=None, limit=1000):

    tar = tarfile.open(input_dir)

    extract = [path for path in tar if path.name.endswith(".fast5")]

    if include:
        extract = [path for path in extract for incl in include if incl in path.name]
    if exclude:
        extract = [path for path in extract for excl in exclude if excl not in path.name]

    if shuffle:
        random.shuffle(extract)

    if limit:
        extract = extract[:limit]

    # Extract tarred Fast5 into their path:
    extracted = []
    with tqdm(total=len(extract)) as pbar:
        pbar.set_description("Extract TAR")
        for tar_info in extract:
            if not os.path.exists(tar_info.name):
                tar.extract(tar_info)
                extracted.append(tar_info.name)
            pbar.update(n=1)

    # Return only the file paths that have actually been extracted
    # and are not duplicates
    return extracted


def check_include_exclude(include, exclude, recursive, verbose=False):

    if include is None:
        include = []
    if exclude is None:
        exclude = []

    # If string is empty ot
    if isinstance(include, str) and include != "":
        include = [incl for incl in include.split(",")]
    if isinstance(include, str) and include == "":
        include = []

    if isinstance(exclude, str) and exclude != "":
        exclude = [excl for excl in exclude.split(",")]
    if isinstance(exclude, str) and exclude == "":
        exclude = []

    include_datasets = [os.path.abspath(item) for item in include if item.endswith(".h5")]
    exclude_datasets = [os.path.abspath(item) for item in exclude if item.endswith(".h5")]

    include_strings = [item for item in include if not item.endswith(".h5")]
    exclude_strings = [item for item in exclude if not item.endswith(".h5")]

    include_dirs = [item for item in include if os.path.isdir(item)]
    exclude_dirs = [item for item in exclude if os.path.isdir(item)]

    include_ds = get_dataset_file_names(include_datasets)
    exclude_ds = get_dataset_file_names(exclude_datasets)

    # Added if given is directory, then get all files from that dir
    # recursively and do not use, i.e. to get only unique resamples
    # from dir into other collection
    include_df = get_dir_file_names(include_dirs, recursive)
    exclude_df = get_dir_file_names(exclude_dirs, recursive)

    if verbose:
        print("Excluding {} files from {} data sets + including {} files from {} data sets."
              .format(len(exclude_ds), len(exclude_datasets), len(include_ds), len(include_datasets)))
        print("Excluding {} strings in file names + including {} strings in file names from user specified inputs"
              .format(len(exclude_strings), len(include_strings)))
        print("Excluding {} files from dirs + including {} files from dirs."
              .format(len(exclude_strings), len(include_strings)))

    # [Include files, include strings], [Exclude files, exclude strings], BASENAMES
    return [include_ds + include_df, include_strings], [exclude_ds + exclude_df, exclude_strings]


def get_dir_file_names(dirs, recursive):

    files = []
    for d in dirs:
        fast5 = get_recursive_files(d, recursive=recursive)
        files += [os.path.basename(path) for path in fast5]
    return files


def get_dataset_file_names(datasets):

    """ If we sample from the same (random) subset of reads as the training data, this function
    makes sure that we are not using the same files used in training for evaluation / prediction. """

    file_names = []
    for data_file in datasets:
        with h5py.File(data_file, "r") as data:
            file_names += [os.path.basename(file) for file in data["data/files"]]

    return file_names


def get_recursive_files(directory, include=None, exclude=None, recursive=True, extension=".fast5"):

    def _init_index():
        if "achilles.index" in os.listdir(directory):
            try:
                df = pandas.read_csv(os.path.join(directory, "achilles.index"), header=None)
            except EmptyDataError:
                return None
            else:
                return df.ix[:, 0].values.tolist()
        else:
            return None

    if recursive:
        # Check if there is index:
        file_paths = _init_index()
        if file_paths is None:
            # Get file paths recursively:
            file_paths = []
            for root, directories, fnames in os.walk(directory):
                for fname in fnames:
                    if fname.endswith(extension):
                        file_paths.append(os.path.join(root, fname))

            print(f"Writing index to achilles.index in {directory}.")
            pandas.DataFrame(file_paths).to_csv(os.path.join(directory, "achilles.index"), index=False, header=False)
    else:
        file_paths = [os.path.join(directory, path) for path in os.listdir(directory)]

    # TODO: Try set difference on file names (exact matches) and fast loops for within string matches.
    file_paths = retain_after_include(file_paths, include)
    file_paths = retain_after_exclude(file_paths, exclude)

    return file_paths


def retain_after_exclude(file_paths, exclude):

    if exclude is None or (not exclude[0] and not exclude[1]):
        return file_paths

    exclude_files, exclude_strings = exclude[0], exclude[1]

    if exclude_files:
        file_retains = [path for path in file_paths if not os.path.basename(path) in exclude_files]
    else:
        file_retains = []

    if exclude_strings:
        string_retains = [path for path in file_paths if not any([string in path for string in exclude_strings])]
    else:
        string_retains = []

    return list(set(string_retains + file_retains))


def retain_after_include(file_paths, include):

    if include is None or (not include[0] and not include[1]):
        return file_paths

    include_files, include_strings = include[0], include[1]

    if include_files:
        file_retains = [path for path in file_paths if os.path.basename(path) in include_files]
    else:
        file_retains = []

    if include_strings:
        string_retains = [path for path in file_paths if any([string in path for string in include_strings])]
    else:
        string_retains = []

    return list(set(string_retains + file_retains))
