import random
import datetime
import pandas
import itertools
import numpy as np
import tarfile
from matplotlib import style
from matplotlib import pyplot as plt

from collections import deque
from itertools import tee, islice

from skimage.util import view_as_windows
from ont_fast5_api.ont_fast5_api.fast5_file import Fast5File
import h5py
import os
from tqdm import tqdm
import shutil

import seaborn


style.use("ggplot")


def percentage_split(seq, percentages):

    """ Helper function splitting window list into training, testing and evaluation proportions

    https://stackoverflow.com/a/14281094

    """

    prv = 0
    size = len(seq)
    cum_percentage = 0
    for p in percentages:
        cum_percentage += p
        nxt = int(cum_percentage * size)
        yield seq[prv:nxt]
        prv = nxt


def chunk(seq, size):

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_tarred_fast5(input_dir, shuffle=True,  include="", exclude="", limit=1000):

    tar = tarfile.open(input_dir)

    extract = [path for path in tar if path.name.endswith(".fast5")]

    if include:
        extract = [path for path in extract if include in path.name]
    if exclude:
        extract = [path for path in extract if exclude not in path.name]

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


def filter_fast5(input_dir, min_signal=None, shuffle=True, limit=1000, include="", exclude=""):

    tar_ext = (".tar", ".tar.gz", ".tgz")

    if input_dir.endswith(tar_ext):
        if min_signal is not None:
            raise ValueError("Selecting Fast5 of minimum signal length from tarred files is currently not possible.")

        return get_tarred_fast5(input_dir, shuffle=shuffle, limit=limit, include=include, exclude=exclude)

    else:
        # Always recursive, always a limit:
        fast5 = get_recursive_files(input_dir, include=include, extension=".fast5")

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


# TODO
def exclude_training_files(selection_files, data_file):

    """ If we sample from the same (random) subset of reads as the training data, this function
    makes sure that we are not using the same files used in training for prediction. """

    with h5py.File(data_file, "r") as data:
        training_file_paths = data["data/files"]

    # Exclude files based on file name, not paths:
    training_file_names = [os.path.basename(file) for file in training_file_paths]
    fast5_file_names = [os.path.basename(file) for file in selection_files]

    retained_files = [fast5 for fast5 in fast5_file_names if fast5 not in training_file_names]

    print("Excluded", len(set()))

    return retained_files


def select_fast5(input_dir, output_dir=None, limit=1000, min_signal=None, symlink=False, shuffle=True,
                 exclude="", include=""):

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


def plot_signal(signal_windows):

    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    selection = select_random_windows(signal_windows, n=4)

    ax1.plot(selection[0])
    ax2.plot(selection[1])
    ax3.plot(selection[2])
    ax4.plot(selection[3])

    plt.show()


def select_random_windows(signal_windows, n=4):

    return [signal_windows[random.randrange(len(signal_windows))][:] for _ in range(n)]


def get_recursive_files(directory, include="", exclude="", extension=".fast5"):

    file_paths = []
    for root, directories, fnames in os.walk(directory):
        for fname in fnames:
            fpath = os.path.join(root, fname)

            if extension:
                if fpath.endswith(extension):
                    file_paths.append(fpath)
                else:
                    continue
            else:
                file_paths.append(fpath)

    if include:
        file_paths = [f for f in file_paths if include in f]
    if exclude:
        file_paths = [f for f in file_paths if exclude not in f]

    return file_paths


def transform_signal_to_tensor(array):

    """ Transform data (nb_windows,window_size) to (nb_windows, 1, window_size, 1)
    for input into Conv2D layer: (samples, height, width, channels),

    TODO: this is slow, is there a better way?

    """

    # Rshape 2D array (samples, width) to 4D array (samples, 1, width, 1)
    return np.reshape(array, (array.shape[0], 1, array.shape[1], 1))


def timeit(micro=False):
    def decorator(func):
        """ Timing decorator for functions and methods """
        def timed(*args, **kw):
            start_time = datetime.datetime.now()
            result = func(*args, **kw)
            time_delta = datetime.datetime.now()-start_time
            seconds = time_delta.total_seconds()
            if micro:
                seconds = int(seconds * 1000000)  # Microseconds
            # print("Runtime:", seconds, "seconds")
            # Flatten output if the output of a function is a tuple with multiple items
            # if this is the case, seconds are at index -1
            return [num for item in [result, seconds]
                    for num in (item if isinstance(item, tuple) else (item,))]
        return timed
    return decorator


def read_signal(fast5: str, normalize: bool=False, scale: bool=True, window_size: int=400, window_step: int=400,
                window_max: int=10, window_random: bool=True, window_recover: bool=True,
                return_signal: bool=False) -> np.array:

    """ Read scaled raw signal in pA (float32) if scaling is enabled or raw (DAC, int16) values from Fast5 using ONT API

    :param fast5            str     path to .fast5 file
    :param normalize        bool    normalize signal by subtracting mean and dividing by standard deviation
    :param window_size      int     run sliding window along signal with size, pass None to return all signal values
    :param window_step      int     sliding window stride, usually 10% of window_size, but appears good on as well
                                    on non-overlapping window slides where window_step = window_size
    :param window_max       int
    :param window_random    bool
    :param window_recover   bool

    :returns tuple of window_max signal windows (np.array) and number of total signal windows before window_max
    """

    try:
        fast5 = Fast5File(fname=fast5)
    except OSError:
        # If the file can't be opened:
        print("Could not open Fast5 file:", fast5)
        return None, 0

    # Scale for array of float(pA values)
    signal = fast5.get_raw_data(scale=scale)

    if normalize:
        signal = (signal - signal.mean()) / signal.std()

    if return_signal:
        # Here, we only return the signal array (1D) and number of signals,
        # used in select function:
        return signal, len(signal)

    # Window processing part:

    signal_windows = view_as_windows(signal, window_size, window_step)

    # Select a random index to extract signal slices, subtract window_max
    # to generate a suitable index where the total number of windows is larger
    # than the requested number of windows then proceed to take a sequence
    # of windows from the random index or from start:
    nb_windows_total = len(signal_windows)
    max_index = nb_windows_total - window_max

    if max_index >= 0:
        if window_random:
            # If max_windows_per_read can be extracted...select random index:
            rand_index = random.randint(0, max_index)
            # ... and extract them:
            signal_windows = signal_windows[rand_index:rand_index + window_max, :]
        else:
            signal_windows = signal_windows[:window_max]
    else:
        # If there are fewer signal windows in the file than window_max...
        if window_recover:
            # If recovery is on, take all windows from this file,
            # as to not bias for longer reads in the sampling process for
            # generating training data:
            signal_windows = signal_windows[:]
        else:
            # Otherwise, return None and skip read, this is more
            # useful in live prediction if a read has not enough
            # (window_max) signal values written to it yet
            signal_windows = None

    return signal_windows, nb_windows_total


# Test this out instead of view_as_windows:
def window_generator(it, size=3):
    yield from zip(*[islice(it, s, None) for s, it in enumerate(tee(it, size))])


def mem_usage(pandas_obj):
    """ https://www.dataquest.io/blog/pandas-big-data/ """
    if isinstance(pandas_obj,pandas.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def plot_confusion_matrix(cm, class_labels,
                          title='Confusion Matrix',
                          normalize=True,
                          cmap="Blues",
                          save=""):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    From Scikit-learn examples:

    http://scikit-learn.org/stable/auto_examples/model_selection/
    plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)  # Remove white gridlines when also importing Seaborn

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()

    if save:
        plt.savefig(save)
    else:
        plt.show()

    plt.close()


def plot_pevaluate_runner(results, class_labels=(0, 1)):

    """ Currently only for binary classification! """

    # Results is dictionary of dictionaries with keys:
    #          keys: prefix
    #          values: {"confusion_matrix": np.array, "average_prediction_time": float}

    data_frame = []
    for prefix, data in results.items():
        cm = data["confusion_matrix"]
        ms = data["average_prediction_time"]
        batches = data["batches"]

        # First deconstruct prefix as {model}:{signal_type}:{sample}:{number_windows}
        model, signal_type, sample, nb_windows = prefix.split(":")

        # Binary classification, needs to be extended later:
        for i, label in enumerate(class_labels):
            if i == 0:
                error, acc = cm[0, 1], cm[0, 0]
            else:
                error, acc = cm[1, 0], cm[1, 1]

            row = [model, signal_type, sample, int(nb_windows), label, error, acc, ms, batches]
            data_frame.append(row)

    df = pandas.DataFrame(data_frame, columns=["model", "signal", "sample", "windows", "label",
                                               "error", "accuracy", "mu", "per_batch"])

    # Setup a plot for each model:

    for model in df["model"].unique():
        # TODO: Make this dynamic (columns for start, random)
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 10))

        model_df = df[df["model"] == model]

        for col, sample in enumerate(sorted(model_df["sample"].unique(), reverse=True)):
            model_sample_df = model_df[model_df["sample"] == sample]
            for row, metric in enumerate(("error", "accuracy")):
                ax = axes[row, col]
                plot = seaborn.pointplot(x="windows", y=metric, hue="label", data=model_sample_df, ci=None, ax=ax)
                plot.set_title("sampling: " + sample)

            model_sample_df["x_labels"] = model_sample_df["windows"].astype(str).str.cat(
                model_sample_df["per_batch"].astype(str), sep="_")

            time_ax = axes[2, col]
            time_plot = seaborn.pointplot(x="x_labels", y="mu", data=model_sample_df,
                                          ci=None, ax=time_ax, color="green")

            time_plot.set_title("Mean prediction speed per batch {} (mu)".format(sample))
            time_plot.set_xlabel("windows _ files)")

        plt.suptitle("Model: {}".format(model), size=16)
        plt.tight_layout()
        plt.savefig(model+"_summary.pdf")

        plt.close()


def sliding_window(iterable, size=2, step=1, fillvalue=None):

    """ https://stackoverflow.com/a/13408251 """

    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration: # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))

# From Mako

def med_mad(data, factor=1.4826, axis=None):

    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median.

    :param data: A :class:`ndarray` object
    :param axis: For multidimensional arrays, which axis to calculate over
    :returns: a tuple containing the median and MAD of the data
    .. note :: the default `factor` scales the MAD for asymptotically normal
        consistency as in R.

    """
    dmed = np.median(data, axis=axis)
    if axis is not None:
        dmed1 = np.expand_dims(dmed, axis)
    else:
        dmed1 = dmed

    dmad = factor * np.median(
        np.abs(data - dmed1),
        axis=axis
    )
    return dmed, dmad


def _scale_data(data):
    if data.ndim == 3:
        # (batches, timesteps, features)
        med, mad = med_mad(data, axis=1)
        med = med.reshape(med.shape + (1,))
        mad = mad.reshape(mad.shape + (1,))
        data = (data - med) / mad
    elif data.ndim == 1:
        med, mad = med_mad(data)
        data = (data - med) / mad
    else:
        raise AttributeError("'data' should have 3 or 1 dimensions.")
    return data

def norm(prediction):

    """ Probability normalization to 1 for predictions along multiple windows of signal """

    return [float(i)/sum(prediction) for i in prediction]
