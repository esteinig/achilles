import random
import datetime
import pandas
import itertools
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt

from itertools import tee, islice

from skimage.util import view_as_windows
from ont_fast5_api.ont_fast5_api.fast5_file import Fast5File

import os
from tqdm import tqdm
import shutil


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


def filter_fast5(input_dir, min_signal=None, shuffle=True, limit=1000):

    # Always recursive, always a limit:
    fast5 = get_recursive_files(input_dir, extension=".fast5")

    if shuffle:
        random.shuffle(fast5)

    if min_signal:
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


def select_fast5(input_dir, output_dir=None, limit=1000, min_signal=None, symlink=False, shuffle=True):

    fast5_paths = filter_fast5(input_dir, min_signal=min_signal, shuffle=shuffle, limit=limit)

    if limit and limit > len(fast5_paths):
        raise ValueError("Number of Fast5 files is smaller than specified limit, please "
                         "select from a sufficient pool of files.")

    # Copy / link files to output directory:
    if output_dir:
        os.makedirs(output_dir)
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


def get_recursive_files(directory, extension=".fast5"):

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
    # to generate a suitable index for desired size, if the maximum possible
    # index with the slice size is >= 0 (when number of windows >= slice size)
    # then proceed to either randomly take a slice or from start:
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
def window(it, size=3):
    yield from zip(*[islice(it, s, None) for s, it in enumerate(tee(it, size))])


def mem_usage(pandas_obj):
    """ https://www.dataquest.io/blog/pandas-big-data/ """
    if isinstance(pandas_obj,pandas.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def plot_confusion_matrix(cm, classes,
                          title='Confusion Matrix',
                          normalize=False,
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
        print("Normalized confusion matrix.")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

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

