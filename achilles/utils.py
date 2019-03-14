import os
import h5py
import random
import pandas
import datetime
import itertools
import numpy as np

from collections import deque
from itertools import tee, islice
from pandas.errors import EmptyDataError

from ont_fast5_api.fast5_file import Fast5File


from colorama import Fore

Y = Fore.YELLOW
R = Fore.RED
G = Fore.GREEN
C = Fore.CYAN
M = Fore.MAGENTA
LR = Fore.LIGHTRED_EX
LC = Fore.LIGHTCYAN_EX
LY = Fore.LIGHTYELLOW_EX
RE = Fore.RESET

import matplotlib

matplotlib.use("agg")


from matplotlib import pyplot as plt
from matplotlib import style
from skimage.util import view_as_windows

style.use("ggplot")

# Data IO and Transformation


def read_signal(
    fast5: str,
    normalize: bool = False,
    scale: bool = False,
    window_size: int = 400,
    window_step: int = 400,
    window_max: int = 10,
    window_random: bool = True,
    window_recover: bool = True,
    return_signal: bool = False,
) -> np.array:

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
            signal_windows = signal_windows[rand_index : rand_index + window_max, :]
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


def transform_signal_to_tensor(array):

    """ Transform data (nb_windows,window_size) to (nb_windows, 1, window_size, 1)
    for input into Conv2D layer: (samples, height, width, channels),
    """

    # Reshape 2D array (samples, width) to 4D array (samples, 1, width, 1)
    return np.reshape(array, (array.shape[0], 1, array.shape[1], 1))


# Data Plots


def plot_training(file="epochs.log"):

    df = pandas.read_csv(file)
    df = df.drop(axis=1, labels="epoch")
    df.plot()

    plt.show()


def plot_signal(fast5):

    fig, axes = plt.subplots(ncols=2, nrows=2)
    ax1, ax2, ax3, ax4 = axes.ravel()

    signal_windows, _ = read_signal(
        fast5=fast5,
        normalize=False,
        scale=False,
        window_size=200,
        window_step=200,
        window_max=10,
        window_random=True,
        window_recover=False,
        return_signal=False,
    )

    selection = select_random_windows(signal_windows, n=4)

    ax1.plot(selection[0], "go")
    ax2.plot(selection[1], "go")
    ax3.plot(selection[2], "mo")
    ax4.plot(selection[3], "mo")

    plt.show()


def plot_confusion_matrix(
    cm, class_labels, title="Confusion Matrix", normalize=True, cmap="Blues", save=""
):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    From Scikit-learn examples:

    http://scikit-learn.org/stable/auto_examples/model_selection/
    plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.grid(None)  # Remove white gridlines when also importing Seaborn

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.tight_layout()

    if save:
        plt.savefig(save)
    else:
        plt.show()

    plt.close()


# Helper Functions


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
        except StopIteration:  # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))


def chunk(seq, size):

    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def select_random_windows(signal_windows, n=4):

    return [signal_windows[random.randrange(len(signal_windows))][:] for _ in range(n)]


def timeit(micro=False):
    def decorator(func):
        """ Timing decorator for functions and methods """

        def timed(*args, **kw):
            start_time = datetime.datetime.now()
            result = func(*args, **kw)
            time_delta = datetime.datetime.now() - start_time
            seconds = time_delta.total_seconds()
            if micro:
                seconds = int(seconds * 1000000)  # Microseconds
            # print("Runtime:", seconds, "seconds")
            # Flatten output if the output of a function is a
            # tuple with multiple items if this is the case,
            # seconds are at index -1
            return [
                num
                for item in [result, seconds]
                for num in (item if isinstance(item, tuple) else (item,))
            ]

        return timed

    return decorator


# Test this out instead of view_as_windows:
def window_generator(it, size=3):
    yield from zip(*[islice(it, s, None) for s, it in enumerate(tee(it, size))])


def mem_usage(pandas_obj):
    """ https://www.dataquest.io/blog/pandas-big-data/ """
    if isinstance(pandas_obj, pandas.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2  # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


# From Mako (ONT) - GitHub site here


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

    dmad = factor * np.median(np.abs(data - dmed1), axis=axis)
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
    return [float(i) / sum(prediction) for i in prediction]


def find(key, dictionary):
    """ https://gist.github.com/douglasmiranda/5127251 """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result


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

    include_datasets = [item for item in include if item.endswith(".h5")]
    exclude_datasets = [item for item in exclude if item.endswith(".h5")]

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
        print(
            "Excluding {} files from {} data sets + including {} files from {} data sets.".format(
                len(exclude_ds),
                len(exclude_datasets),
                len(include_ds),
                len(include_datasets),
            )
        )
        print(
            "Excluding {} strings in file names + including {} strings in file names from user specified inputs".format(
                len(exclude_strings), len(include_strings)
            )
        )
        print(
            "Excluding {} files from dirs + including {} files from dirs.".format(
                len(exclude_strings), len(include_strings)
            )
        )

    # [Include files, include strings], [Exclude files, exclude strings], BASENAMES
    return (
        [include_ds + include_df, include_strings],
        [exclude_ds + exclude_df, exclude_strings],
    )


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


def get_dataset_labels(dataset):

    """ If we sample from the same (random) subset of reads as the training data, this function
    makes sure that we are not using the same files used in training for evaluation / prediction. """

    with h5py.File(dataset, "r") as data:
        labels = data["data/labels"]
        return np.array(labels)


def get_dataset_dim(dataset):

    """ If we sample from the same (random) subset of reads as the training data, this function
    makes sure that we are not using the same files used in training for evaluation / prediction. """

    with h5py.File(dataset, "r") as data:
        return np.array(data["training/data"]).shape


def get_recursive_files(
    directory, include=None, exclude=None, recursive=True, extension=".fast5"
):

    # TODO: Index file in Dataset - make index a parameter to disable for testing!

    def _init_index():
        if "achilles.index" in os.listdir(directory):
            try:
                df = pandas.read_csv(
                    os.path.join(directory, "achilles.index"), header=None
                )
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

            pandas.DataFrame(file_paths).to_csv(
                os.path.join(directory, "achilles.index"), index=False, header=False
            )
    else:
        file_paths = [
            os.path.join(directory, path)
            for path in os.listdir(directory)
            if path.endswith(extension)
        ]

    # TODO: Try set difference on file names (exact matches) and fast loops for within string matches.
    file_paths = retain_after_include(file_paths, include)
    file_paths = retain_after_exclude(file_paths, exclude)

    return file_paths


def retain_after_exclude(file_paths, exclude):

    if exclude is None or (not exclude[0] and not exclude[1]):
        return file_paths

    exclude_files, exclude_strings = exclude[0], exclude[1]

    if exclude_files:
        file_retains = [
            path for path in file_paths if not os.path.basename(path) in exclude_files
        ]
    else:
        file_retains = []

    if exclude_strings:
        string_retains = [
            path
            for path in file_paths
            if not any([string in path for string in exclude_strings])
        ]
    else:
        string_retains = []

    return list(set(string_retains + file_retains))


def retain_after_include(file_paths, include):

    if include is None or (not include[0] and not include[1]):
        return file_paths

    include_files, include_strings = include[0], include[1]

    if include_files:
        file_retains = [
            path for path in file_paths if os.path.basename(path) in include_files
        ]
    else:
        file_retains = []

    if include_strings:
        string_retains = [
            path
            for path in file_paths
            if any([string in path for string in include_strings])
        ]
    else:
        string_retains = []

    return list(set(string_retains + file_retains))


class TableFormatter:

    def __init__(
            self,
            header: list = None,
            row_template: str = None,
            header_template: str = None,
            header_color: str = None,
            header_gap: bool = True
    ):

        self.header = header

        if header_color is None:
            self.header_color = RE
        else:
            self.header_color = header_color

        self.header_gap = header_gap

        self.row_template = row_template

        if not header_template:
            self.header_template = row_template
        else:
            self.header_template = header_template

        self.head = self.format_header()  # Current header formatted
        self.row = None  # Current row formatted
        self.table = None  # Current table formatted

    def __enter__(self):

        print()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        print()

    def format_header(self):

        head = f"{self.header_color}{self.header_template.format(*self.header)}{RE}" + "\n"

        divider = [len(colname) * '=' + '=' for colname in self.header]

        head += self.header_template.format(*divider)

        if self.header_gap:
            head += '\n'

        return head

    def format_row(self, data: list, color=G):

        data = [self._shorten(d) for d in data]

        self.row = f"{color}{self.row_template.format(*data)}{RE}"

    @staticmethod
    def _shorten(string, max_len=32):

        if len(string) > max_len:
            return string[:max_len-3] + '...'
        else:
            return string


