import os
import h5py
import operator
import numpy as np

from textwrap import dedent
from keras.utils import Sequence
from skimage.util import view_as_windows
from keras.utils.np_utils import to_categorical
from ont_fast5_api.ont_fast5_api.fast5_file import Fast5File


class Dataset:

    def __init__(self, data_file="data.h5"):

        self.data_file = data_file

    def get_signal_generator(self, data_type="training", batch_size=15, shuffle=True):

        """ Main function to generate signal window training and validation data generators
        from directories of Fast5 files, generate data in batches """

        return DataGenerator(self.data_file, data_type=data_type, batch_size=batch_size, shuffle=shuffle)

    def write_data(self, *dirs, classes=2, max_per_class=20000, proportions=(0.7, 0.3),
                   window_size=4000, window_step=400, normalize=True):

        with h5py.File(self.data_file, "w") as f:

            # Save data, labels in /training and /validation HDF5
            for i, data_type in enumerate(("training", "validation")):
                # Proportion of training / validation data signal window limit:
                max_per_type = max_per_class*proportions[i]
                # HDF5 file dataset creation:
                data = f.create_dataset(data_type + "/data", shape=(0, 1, window_size, 1),
                                        maxshape=(None, 1, window_size, 1))
                labels = f.create_dataset(data_type + "/labels", shape=(0, classes),
                                          maxshape=(None, classes))

                # each dir corresponds to label (0, 1)
                for label, path in enumerate(dirs):
                    # All Fast5 files in dir:
                    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".fast5")]
                    # Sort by largest (assume longest):
                    files_and_sizes = ((path, os.path.getsize(path)) for path in files)
                    files = sorted(files_and_sizes, key=operator.itemgetter(1))

                    # Main loop for reading through Fast5 files and extracting overlapping windows of signal
                    # (window_size, window_step) of normalized (pA) signal until limit for proportioned data
                    # set is reached. Write each signal windows to HDF5 (self.output) after transforming to
                    # 4D input tensor to residual blocks in Achilles model.
                    total = 0
                    # Extract the normalized signal windows into nd array(num_windows, window_size)
                    for fast5, _ in files:
                        signal_windows = self.read_signal(fast5, normalize=normalize, window_size=window_size,
                                                          window_step=window_step)

                        # 4D input tensor (nb_samples, 1, signal_length, 1) for Residual Blocks
                        input_tensor = self.transform_signal_to_tensor(signal_windows)

                        if total < max_per_type:
                            if input_tensor.shape[0] > max_per_type-total:
                                input_tensor = input_tensor[:max_per_type-total]
                            self.write_chunk(data, input_tensor)

                            total += input_tensor.shape[0]

                    # Writing all training labels to HDF5
                    encoded_labels = to_categorical(np.array([label for _ in range(total)]), classes)
                    self.write_chunk(labels, encoded_labels)

    def get_data_summary(self, data_type):

        with h5py.File(self.data_file, "r") as f:
            return f[data_type+"/data"].shape, f[data_type+"/labels"].shape

    def print_data_summary(self):

        with h5py.File(self.data_file, "r") as f:

            print(dedent("""
                HDF5 file: {}
                Training data: {}
                Training labels: {}
                """
            ).format(self.data_file, f["training/data"].shape, f["training/labels"].shape))

    @staticmethod
    def write_chunk(dataset, data):

        dataset.resize(dataset.shape[0]+data.shape[0], axis=0)
        dataset[-data.shape[0]:] = data

        return dataset

    @staticmethod
    def transform_signal_to_tensor(vector):

        """ Transform data (nb_windows,window_size) to (nb_windows, 1, window_size, 1)
        for input into Conv2D layer: (samples, height, width, channels),

        TODO: this is slow, is there a better way?

        """

        # Return 4D array (samples, 1, width, 1)
        return np.array([[[[signal] for signal in data]] for data in vector[:]])

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


class DataGenerator(Sequence):

    def __init__(self, data_file, data_type="training", batch_size=15, shuffle=True):

        self.data_file = data_file
        self.data_type = data_type

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = []

        self.data_shape, self.label_shape = self._get_data_shapes()

        self.on_epoch_end()

    def _get_data_shapes(self, ):

        with h5py.File(self.data_file, "r") as f:
            return f[self.data_type + "/data"].shape, f[self.data_type + "/labels"].shape

    def __len__(self):

        """ Number of batches per epoch """

        return int(np.floor(self.data_shape[0]) / self.batch_size)

    def __getitem__(self, index):

        """ Generate one batch of data """

        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        data, labels = self.__data_generation(indices)

        # Testing print statements:

        # print("Training data batch:", data.shape)
        # print("Training label batch:", labels.shape)
        # print("Generated data for indices:", indices)

        return data, labels

    def on_epoch_end(self):

        """ Updates indexes after each epoch """

        self.indices = np.arange(self.data_shape[0])

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):

        """ Generates data containing batch_size samples """

        with h5py.File(self.data_file, "r") as f:

            file_data = f[self.data_type + "/data"]
            data = np.take(file_data, indices, axis=0)

            file_labels = f[self.data_type + "/labels"]
            labels = np.take(file_labels, indices, axis=0)

            return data, labels


def test():

    """ Local dataset test function on Windows """

    dir1 = r"C:\Users\jc225327\PycharmProjects\asclepius\dir1"
    dir2 = r"C:\Users\jc225327\PycharmProjects\asclepius\dir2"

    ds = Dataset(data_file="../data.h5")

    ds.write_data(dir1, dir2, classes=2, max_per_class=200000, window_size=4000,
                  window_step=400, normalize=True)

    ds.print_data_summary()
    ds.get_signal_generator()