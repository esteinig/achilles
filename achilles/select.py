import os
import h5py
import random
import shutil
import json
import tarfile

from tqdm import tqdm
import multiprocessing
from achilles.utils import read_signal, chunk


test_config = {
    "training": {

        "params": {
            "limit": 12000,
            "shuffle": True,
            "min_signal": None,
        },

        "host": {
            "data": {
                "minimal": ["chr_2_part03", "chr_4_part06"],
                "diverse": ["chr_2_part03", "chr_4_part06", "chr_8_part07",
                            "chr_14_part06", "chr_20_part05", "chr_X_part02"],
            },
            "params": {
                "exclude": None,
                "include": None
            }
        },


        "pathogen": {
            "data": {
                "zibra_1": ["zika_library1"],
            },
            "params": {
                "exclude": None,
                "include": None  # Also possible, add param "basedir"
            }
        }

    }
}


class DataSelector:
    """ Mix and match samples from Fast5 directories to generate DataSets """

    def __init__(self, outdir, data_path="select_data", config=None, config_file=None, ncpu=1, chunk_size=100):

        self.config = config
        self.config_file = config_file

        self.ncpu = ncpu
        self.chunk_size = chunk_size

        if config_file is not None:
            with open(self.config_file, "r") as cfg:
                self.config = json.load(cfg)

        self.outdir = os.path.abspath(outdir)
        self.data_path = os.path.abspath(data_path)

    def run_config(self):

        # Training data set generation:

        training_config = self.config["training"]
        training_global_params, training_pathogen, training_host = \
            training_config["params"], training_config["pathogen"], training_config["host"]

        training_host_data, training_pathogen_data = training_host["data"], training_pathogen["data"]
        training_host_params, training_pathogen_params = training_host["params"], training_pathogen["params"]

        host_basedir = training_host_params.pop("basedir")\
            if "basedir" in training_host_params.keys() else self.data_path
        pathogen_basedir = training_pathogen_params.pop("basedir")\
            if "basedir" in training_pathogen_params.keys() else self.data_path

        if host_basedir is None or pathogen_basedir is None:
            raise ValueError("Please specify data_path or include basedir in host and pathogen configurations.")

        host_pathogen_data = [(host, pathogen) for pathogen in training_pathogen_data.keys()
                              for host in training_host_data.keys()]

        print("Initiated data set combinations.")
        for host_id, pathogen_id in host_pathogen_data:
            host_dirs = [self.get_dir_path_recursively(fast5_dir, host_basedir)
                         for fast5_dir in training_host_data[host_id]]

            pathogen_dirs = [self.get_dir_path_recursively(fast5_dir, pathogen_basedir)
                             for fast5_dir in training_pathogen_data[pathogen_id]]

            data_set_id = host_id + "_" + pathogen_id
            data_set_path = os.path.join(self.outdir, data_set_id)

            os.makedirs(data_set_path, exist_ok=False)
            host_path = os.path.join(data_set_path, "host")
            pathogen_path = os.path.join(data_set_path, "pathogen")

            print("Sampling host reads in progress for dataset {}...".format(data_set_id))
            host_fast5 = self.sample_mix(host_dirs, host_path, ncpu=self.ncpu, chunk_size=self.chunk_size,
                                         **training_global_params, **training_host_params)

            print("Sampling pathogen reads in progress for dataset {}...".format(data_set_id))
            pathogen_fast5 = self.sample_mix(pathogen_dirs, pathogen_path, ncpu=self.ncpu, chunk_size=self.chunk_size,
                                             **training_global_params, **training_pathogen_params)

            host_files = os.listdir(host_path)
            pathogen_files = os.listdir(pathogen_path)

            if len(host_files) < len(host_fast5):
                print("Warning: there are less host files selected {} than expected {} in data set {}. "
                      "This may be because only unique files are considered from across the sampled data directories."
                      .format(len(host_files), len(host_fast5), data_set_id))
            if len(pathogen_files) < len(pathogen_fast5):
                print("Warning: there are less host files selected ({}) than expected ({}) in data set {}. "
                      "This may be because only unique files are considered from across the sampled data directories."
                      .format(len(pathogen_files), len(pathogen_fast5), data_set_id))

    def get_dir_path_recursively(self, target_dir, data_path):

        target_dir = os.path.basename(target_dir)

        for root, dirs, files in os.walk(data_path):
            if target_dir in dirs:
                return os.path.abspath(os.path.join(root, target_dir))

        raise ValueError("Could not find path of directory {} in data path {}."
                         .format(target_dir, self.data_path))

    @staticmethod
    def sample_mix(dirs, outdir, shuffle=True, limit=12000, min_signal=None, include=None,
                   exclude=None, ncpu=1, chunk_size=100):

        sample_limit = limit//len(dirs)
        print("Sampling maximum of {} Fast5 files per directory.".format(sample_limit))
        fast5 = []
        for d in dirs:
            fast5 += select_fast5(input_dir=d, output_dir=outdir, limit=sample_limit, min_signal=min_signal,
                                  shuffle=shuffle, exclude=exclude, include=include, ncpu=ncpu, chunk_size=chunk_size)

        return fast5


def select_fast5(input_dir, output_dir=None, limit=None, min_signal=None, symlink=False, shuffle=True,
                 exclude=None, include=None, ncpu=1, chunk_size=100):

    fast5_paths = filter_fast5(input_dir, include=include, min_signal=min_signal, shuffle=shuffle,
                               limit=limit, exclude=exclude)

    # Copy / link files to output directory:
    if output_dir:
        if os.path.exists(output_dir):
            print("Warning: output directory for copying files exist, files will be copied.")

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
            pbar = tqdm(total=nb_chunks)

            def cbk_update(*args):
                pbar.update()

            pool = multiprocessing.Pool(processes=ncpu)
            for i in range(nb_chunks):
                pool.apply_async(copy_link_files, args=(fast5_chunks[i], output_dir, None, symlink, ),
                                 callback=cbk_update)
            pool.close()
            pool.join()

    return fast5_paths


def copy_link_files(fast5_paths, output_dir, pbar=None, symlink=False):

    for file_path in fast5_paths:
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

def filter_fast5(input_dir, min_signal=None, shuffle=True, limit=1000, include=None, exclude=None):

    # Check for datasets in include or exclude lists, unpack
    # file basenames for inclusion or exclusion:

    include, exclude = check_include_exclude(include, exclude)

    tar_ext = (".tar", ".tar.gz", ".tgz")

    if input_dir.endswith(tar_ext):
        if min_signal is not None:
            raise ValueError("Selecting Fast5 of minimum signal length from tarred files is currently not possible.")

        return get_tarred_fast5(input_dir, shuffle=shuffle, limit=limit, include=include, exclude=exclude)

    else:
        # Always recursive, always a limit:
        fast5 = get_recursive_files(input_dir, include=include, exclude=exclude, extension=".fast5")

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


def check_include_exclude(include, exclude, verbose=False):

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
    include_df = get_dir_file_names(include_dirs)
    exclude_df = get_dir_file_names(exclude_dirs)

    if verbose:
        print("Excluding {} files from {} data sets + including {} files from {} data sets."
              .format(len(exclude_ds), len(exclude_datasets), len(include_ds), len(include_datasets)))
        print("Excluding {} strings in file names + including {} strings in file names from user specified inputs"
              .format(len(exclude_strings), len(include_strings)))
        print("Excluding {} files from dirs + including {} files from dirs."
              .format(len(exclude_strings), len(include_strings)))

    return include_ds + include_strings + include_df, exclude_ds + exclude_strings + exclude_df


def get_dir_file_names(dirs):

    files = []
    for d in dirs:
        fast5 = get_recursive_files(d)
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


def get_recursive_files(directory, include=None, exclude=None, extension=".fast5"):

    file_paths = []
    for root, directories, fnames in os.walk(directory):
        for fname in fnames:
            fpath = os.path.join(root, fname)

            if extension:
                if fpath.endswith(extension):
                    file_paths.append(fpath)
            else:
                file_paths.append(fpath)

    if include:
        retain = []
        for f in file_paths:
            if any(i in f for i in include) and f not in retain:
                retain.append(f)
        file_paths = retain.copy()

    if exclude:
        retain = []
        for f in file_paths:
            if not any(x in f for x in exclude) and f not in retain:
                retain.append(f)
        file_paths = retain.copy()

    return file_paths
