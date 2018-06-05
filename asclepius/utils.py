import random
import pandas
import numpy as np
from matplotlib import style
from matplotlib import pyplot as plt

import heapq
import os

import shutil
import operator

from keras import callbacks

style.use("ggplot")


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


# TODO: Figure out what's going on in calculating average mini-batch loss - cumulative?
def plot_batch_loss_accuracy(fname, outname="plot.pdf", sep="\t", error=False):

    df = pandas.read_csv(fname, sep=sep, names=["batches", "loss", "acc"], index_col=0)

    zero_batch_count = df.index.value_counts()
    max_batch = max(df.index)

    epoch = 0
    epochs = []
    for i in df.index.values:
        if int(i) == 0:
           epoch += 1

        epochs.append(epoch)

    print(len(epochs))
    print(len(df))

    epoch_lines = [max_batch*i for i in range(zero_batch_count[0])]

    df = df.expanding(min_periods=1).mean().reset_index()

    df["epoch"] = epochs

    print(df)

    if error:
        df["acc"] = 1 - df["acc"]

    batch_size = df.index[1] - df.index[0]

    batch_index = [i for i in range(0, len(df.index)*batch_size, batch_size)]

    print(batch_index)

    df.index = batch_index

    print(df)

    df.plot()
    plt.show()
    plt.savefig(outname)


def select_fast5(input_dir, output_dir, n=3000, largest_files=True):

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
        # Defaul mode random_files:
        files_and_sizes = [item for item in file_sizes(input_dir)]
        # Assumes that there are > n files
        files = np.random.shuffle(files_and_sizes)[:n]
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


class BatchLogger(callbacks.Callback):

    """
    A Logger that log average performance per `display` steps.
    """

    def __init__(self, output_file="metrics.log", log_interval=10):

        super().__init__()

        self.output_file = output_file
        self.log_interval = log_interval

    def on_batch_end(self, batch, logs={}):

        if batch % self.log_interval == 0:

            try:
                loss = logs["loss"]
            except KeyError:
                loss = "error"

            try:
                acc = logs["acc"]
            except KeyError:
                acc = "none"

            metrics = "{},{},{}\n".format(batch, loss, acc)

            with open(self.output_file, "a") as logfile:
                logfile.write(metrics)
