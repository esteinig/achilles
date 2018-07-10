import os
import h5py
import random
import shutil
import tarfile

from tqdm import tqdm
from achilles.utils import read_signal


def select_fast5(input_dir, output_dir=None, limit=None, min_signal=None, symlink=False, shuffle=True,
                 exclude=None, include=None):

    fast5_paths = filter_fast5(input_dir, include=include, min_signal=min_signal, shuffle=shuffle,
                               limit=limit, exclude=exclude)

    # Copy / link files to output directory:
    if output_dir:
        if os.path.exists(output_dir):
            print("Warning: output directory for copying files exist, files will be copied.")

        os.makedirs(output_dir, exist_ok=True)
        with tqdm(total=len(fast5_paths)) as pbar:
            pbar.set_description("Copying files")
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
                pbar.update(n=1)

    return fast5_paths


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

    if isinstance(include, str):
        include = [include]
    if isinstance(exclude, str):
        exclude = [exclude]

    include_datasets = [item for item in include if item.endswith(".h5")]
    exclude_datasets = [item for item in exclude if item.endswith(".h5")]

    include_strings = [item for item in include if not item.endswith(".h5")]
    exclude_strings = [item for item in exclude if not item.endswith(".h5")]

    include_ds = get_dataset_file_names(include_datasets)
    exclude_ds = get_dataset_file_names(exclude_datasets)

    if verbose:
        print("Excluding {} files from {} data sets + including {} files from {} data sets."
              .format(len(exclude_ds), len(exclude_datasets), len(include_ds), len(include_datasets)))
        print("Excluding {} strings in file names + including {} strings in file names from user specified inputs"
              .format(len(exclude_strings), len(include_strings)))

    return include_ds + include_strings, exclude_ds + exclude_strings


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
                retain.append(retain)
        file_paths = retain
    if exclude:
        retain = []
        for f in file_paths:
            if not any(x in f for x in exclude) and f not in retain:
                retain.append(f)
        file_paths = retain

    return file_paths
