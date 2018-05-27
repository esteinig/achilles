
import numpy as np

from keras.utils.np_utils import to_categorical
from asclepius.reader import get_signal_windows

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

        signal_data, n_files = get_signal_windows(self.dirs, split_data=self.split_data, max_reads=max_reads_per_class,
                                                  normalize=normalize, window_size=window_size, window_step=window_step)

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
