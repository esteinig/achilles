import os
import numpy as np

from skimage.util import view_as_windows
from keras.utils.np_utils import to_categorical
from ont_fast5_api.ont_fast5_api.fast5_file import Fast5File

import asclepius.utils as utils


class Dataset:

    def __init__(self, *dirs):

        # Binary classification format for now
        # where length of dirs = number of classes
        self.dirs = dirs
        self.classes = len(dirs)

        # Data proportions into Train, Validation, Evaluation
        self.split_data = (0.7, 0.2, 0.1)

    def get_data(self, max_reads_per_class=10, normalize=True, window_size=4000, window_step=400):

        """ Get data from directories for labelling and splitting into training, validation and evaluation sets """

        signal_data, n_files = self.get_signal_windows(self.dirs, split_data=self.split_data,
                                                       max_reads=max_reads_per_class,
                                                       normalize=normalize, window_size=window_size,
                                                       window_step=window_step)

        datasets = {
            "data": dict(),
            "labels": dict()
        }

        for dataset, data in signal_data.items():
            achilles_input = np.take(data, indices=0, axis=1)
            achilles_labels = np.take(data, indices=1, axis=1)

            # Manual reshape into 4D array (samples, height, width, channels)
            datasets["data"][dataset] = self.transform_signal_to_tensor(achilles_input)
            # Categorical one-hot encoded labels, currently two classes:
            datasets["labels"][dataset] = to_categorical(achilles_labels, num_classes=self.classes)

        return datasets

    @staticmethod
    def transform_signal_to_tensor(vector):

        """ Transform data (nb_windows,) to (nb_windows, 1, window_size, 1)
        for input into Conv2D layer: (samples, height, width, channels),

        TODO: this is slow, is there a better way?

        """

        # Return 4D array (samples, 1, width, 1)
        return np.array([[[[signal] for signal in data]] for data in vector[:]])

    def get_signal_windows(self, dirs, split_data, max_reads=10, normalize=True, window_size=3000, window_step=300,
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
            # path = os.path.join(BASE, path)
            # Get a list of Fast5 files...
            files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".fast5")]
            # ... and pick first <max_reads>
            files = files[:max_reads]  # TO DO: make this random?

            data = []
            # Extract the normalized signal windows into nd array(num_windows, window_size)
            for fast5 in files:
                signal_windows = self.read_signal(fast5, normalize=normalize, window_size=window_size,
                                                  window_step=window_step)
                # Testing purpose only,
                # check signal plot per file:
                if check_plot:
                    utils.plot_signal(signal_windows)

                data += [(window, label) for window in signal_windows]

            train, test, val = tuple(utils.percentage_split(data, split_data))

            # Add training, testing and validation data (nb_windows, label) of each class
            # to combined class data set:

            final_data["train"] += train
            final_data["test"] += test
            final_data["val"] += val

            nb_files += len(files)

        final_data = {dataset: np.array(data) for dataset, data in final_data.items()}

        return final_data, nb_files

    @staticmethod
    def read_signal(fast5: str, normalize: bool = True, window_size: int = 4000, window_step: int = 400) -> np.array:

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
