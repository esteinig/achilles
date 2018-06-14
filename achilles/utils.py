import random
import time
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt

from skimage.util import view_as_windows
from ont_fast5_api.ont_fast5_api.fast5_file import Fast5File

import heapq
import os

import shutil
import operator

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


def select_fast5(input_dir, output_dir, n=3000, largest_files=False):

    """ Copy n largest files from recursive input directory (e.g. Fast5 files) """

    os.makedirs(output_dir)

    def file_sizes(directory):
        for path, _, filenames in os.walk(directory):
            for name in filenames:
                full_path = os.path.join(path, name)
                yield full_path, os.path.getsize(full_path)

    if largest_files:
        files = heapq.nlargest(n, file_sizes(input_dir), key=operator.itemgetter(1))
    else:
        # Default mode random_files:
        files_and_sizes = [item for item in file_sizes(input_dir)]

        indices = np.arange(len(files_and_sizes))
        # Shuffle indices for random files:
        np.random.shuffle(indices)
        # Assumes that there are > n files
        files = [files_and_sizes[i] for i in indices[:n]]

    for file, size in files:
        shutil.copy(file, os.path.join(output_dir, os.path.basename(file)))

    return files


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


def transform_signal_to_tensor(array):

    """ Transform data (nb_windows,window_size) to (nb_windows, 1, window_size, 1)
    for input into Conv2D layer: (samples, height, width, channels),

    TODO: this is slow, is there a better way?

    """

    # Rshape 2D array (samples, width) to 4D array (samples, 1, width, 1)
    return np.reshape(array, (array.shape[0], 1, array.shape[1], 1))



def read_signal(fast5: str, normalize: bool = False, window_size: int = 4000, window_step: int = 400) -> np.array:

    """ Read scaled raw signal in pA (float) from Fast5 using ONT API

    :param fast5        str     path to .fast5 file
    :param normalize    bool    normalize signal by subtracting mean and dividing by standard deviation
    :param window_size  int     run sliding window along signal with size, pass None to return all signal values
    :param window_step  int     sliding window stride, usually 10% of window_size, but appears good on as well
                                on non-overlapping window slides where window_step = window_size

    """

    fast5 = Fast5File(fname=fast5)

    # Scale for array of float(pA values)
    signal = fast5.get_raw_data(scale=True)

    if normalize:
        signal = (signal - signal.mean()) / signal.std()

    if window_size:
        return view_as_windows(signal, window_size, window_step)
    else:
        return signal


def timeit(func):
    """ Timing decorator for functions and methods """
    def timed(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        minutes, seconds = divmod(time.time()-start_time, 60)
        print("Runtime:", round(minutes, 2), "minutes")
        return result
    return timed
