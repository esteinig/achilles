from skimage.util import view_as_windows

import os
import random
import numpy as np

from textwrap import dedent

from matplotlib import style
from matplotlib import pyplot as plt

style.use("ggplot")


from ont_fast5_api.ont_fast5_api.fast5_file import Fast5File

# BASE = r"C:\Users\jc225327\PycharmProjects\achilles"

BASE = r"/home/esteinig/code/achilles"


def print_data_extraction_message(signal_data, files, step, size):

    windows = signal_data.shape[0]
    shape = signal_data.shape

    msg = dedent("""
    Signal data extracted {}:
    
        Files:                  {}
        Windows:                {}
        Window Step:            {}
        Signals per window:     {}
        
        """).format(shape, files, windows, step, size)

    print(msg)


def get_signal_windows(dirs, split_data, max_reads=10, normalize=True, window_size=3000, window_step=300,
                       check_plot=False):

    nb_files = 0

    final_data = {
        "train": list(),
        "test": list(),
        "val": list()
    }

    # For each directory of two classes (labels)
    # with Bp and Human reads...
    for label, path in enumerate(dirs):
        path = os.path.join(BASE, path)
        # Get a list of Fast5 files...
        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".fast5")]
        # ... and pick first <max_reads>
        files = files[:max_reads]  # TO DO: make this random?

        data = []
        # Extract the normalized signal windows into nd array(num_windows, window_size)
        for fast5 in files:
            signal_windows = read_signal(fast5, normalize=normalize, window_size=window_size, window_step=window_step)
            # Testing purpose only,
            # check signal plot per file:
            if check_plot:
                plot_signal(signal_windows)

            data += [(window, label) for window in signal_windows]

        train, test, val = tuple(percentage_split(data, split_data))

        # Add training, testing and validation data (nb_windows, label) of each class
        # to combined class data set:

        final_data["train"] += train
        final_data["test"] += test
        final_data["val"] += val

        nb_files += len(files)

    final_data = {dataset: np.array(data) for dataset, data in final_data.items()}

    return final_data, nb_files


def percentage_split(seq, percentages) -> iter:

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


def read_signal(fast5: str, normalize: bool=True, window_size: int=4000, window_step: int=400) -> np.array:

    """ Read scaled raw signal in pA (float) from Fast5 using ONT API

    :param fast5        str     path to .fast5 file
    :param normalize    bool    normalize signal by subtracting mean and dividing by standard deviation
    :param window_size  int     run sliding window along signal with size, pass None to return all signal values
    :param window_step  int     sliding window stride, usually 10% of window_size

    """

    fast5 = Fast5File(fname=fast5)

    # Scale for array of float(pA values)
    signal = fast5.get_raw_data(scale=True)

    if normalize:
        signal = (signal - signal.mean()) / signal.std()

    # Overlapping windows (n, size)

    if window_size:
        return view_as_windows(signal, window_size, window_step)
    else:
        return signal


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

    return [signal_windows[random.randrange(len(signal_windows))][:] for i in range(n)]
