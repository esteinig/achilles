import random
from textwrap import dedent

from matplotlib import style
from matplotlib import pyplot as plt

from keras import callbacks

style.use("ggplot")


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

            metrics = "{}\t{}\t{}".format(batch, loss, acc)
            with open(self.output_file, "w") as logfile:
                logfile.write(metrics)
