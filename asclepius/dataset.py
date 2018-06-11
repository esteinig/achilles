import os
import h5py
import random
import logging
import numpy as np

from tqdm import tqdm
from asclepius.utils import read_signal
from textwrap import dedent
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical


class Dataset:

    def __init__(self, data_file="data.h5"):

        self.data_file = data_file

    def get_signal_generator(self, data_type="training", batch_size=15, shuffle=True):

        """Access function to generate signal window training and validation data generators
        from directories of Fast5 files, generate data in batches

        :param data_type:
        :param batch_size:
        :param shuffle:
        :return:
        """

        return DataGenerator(self.data_file, data_type=data_type, batch_size=batch_size, shuffle=shuffle)

    def write_data(self, *dirs, classes=2, max_windows_per_class=20000, max_windows_per_read=100,
                   window_size=4000, window_step=400, random_consecutive_windows=True, normalize=False):

        """ Primary function to extract windows (slices) at random indices in the arrays that hold
        nanopore signal values in the (shuffled) sequencing files (.fast5) located in directories that contain
        the files for each class (label). Signal arrays are sliced by consecutive windows of window_size and
        window_step and a maximum of max_windows_per_read of random signal windows can be read from a single file.
        (until max_windows_per_class are reached). If the signal array is smaller than the maximum number of possible
        slices, the whole array is extracted so as not to bias for selecting longer reads.

        This is followed by reshaping each slice into an nd-array of dimensions  (nb_slices, 1, window_size, 1)
        for input to the residual blocks in the model. The sliced signal array is then written chunk-wise (append)
        to the path 'data/data' (nb_slices, 1, window_size, 1) and its corresponding labels one-hot encoded
        to the path to 'data/labels' (nb_slices, nb_classes).

        For summary purposes, the file paths for all files from which signal slices were extracted are stored
        in 'data/files' and integer encoded labels are kept in 'data/decoded'

        :param dirs:
        :param classes:
        :param max_windows_per_class:
        :param max_windows_per_read:
        :param window_size:
        :param window_step:
        :param random_consecutive_windows:
        :param normalize:
        :return:
        """

        with h5py.File(self.data_file, "w") as f:

            # Each input directory corresponds to label (0, 1)
            for label, path in enumerate(dirs):

                # Create data paths for storing all extracted data:
                data, labels, decoded, extracted = \
                    self.create_data_paths(file=f, window_size=window_size, classes=classes)

                # All Fast5 files in directory:
                files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".fast5")]

                # Randomize:
                random.shuffle(files)

                # Main loop for reading through Fast5 files and extracting windows (slices) of signal
                # (window_size, window_step) of normalized (pA) signal until limit for proportioned data
                # set is reached. Write each signal windows to HDF5 (self.output) after transforming to
                # 4D input tensor to residual blocks in Achilles model.
                total = 0
                n_files = []
                # Logging message:
                class_summary = "Extracting {} signal windows (size: {}, step: {}) from each of {} " \
                                "Fast5 files for label {}".format(max_windows_per_read, window_size,
                                                                  window_step, len(files), label)
                logging.debug(class_summary)
                # The progress bar is just a feature for reference, this loop will be stopped as soon
                # as the maximum number of signal arrays per class is reached (progress bar is therefore not
                # accurate but just looks good and gives the user an overestimate of when extraction is finished.
                with tqdm(total=max_windows_per_class) as pbar:
                    for fast5 in files:

                        # Slice whole signal array into windows. May be more efficient to index first:
                        signal_windows = read_signal(fast5, normalize=normalize, window_size=window_size,
                                                     window_step=window_step)

                        # TODO: Evaluate what happens when constructing data from beginning of read (probably not good
                        # TODO: as it captures the adapters) - at the moment use random index + consecutive windows
                        if random_consecutive_windows:
                            # Select a random index to extract signal windows
                            rand_index = random.randint(0, signal_windows.shape[0])
                            # If max_windows_per_read can be extracted...
                            if rand_index + max_windows_per_read <= len(signal_windows):
                                # ...extract them:
                                signal_windows = signal_windows[rand_index:rand_index+max_windows_per_read, :]
                            else:
                                # If there are fewer signal windows in the file than max_windows_per_read, take all
                                # windows from this file, as to not bias for longer reads, then continue:
                                signal_windows = signal_windows[:]
                        else:
                            # From beginning of read:
                            signal_windows = signal_windows[:max_windows_per_read]

                        # Proceed if the maximum number of windows per class has not been reached,
                        # and if there are windows extracted from the Fast5:
                        if total < max_windows_per_class and signal_windows.size > 0:
                            # If the number of extracted signal windows exceeds the difference between
                            # current total and max_windows_per_class is reached, cut off the signal window
                            # array and write it to file, to complete the loop for generating data for this label:
                            if signal_windows.size > max_windows_per_class-total:
                                signal_windows = signal_windows[:max_windows_per_class-total]

                            # 4D input tensor (nb_samples, 1, signal_length, 1) for input to Residual Blocks
                            input_tensor = self.transform_signal_to_tensor(signal_windows)

                            # Write this tensor to file instead of storing in memory
                            # otherwise might raise OOM:
                            self.write_chunk(data, input_tensor)

                            # Operations for update to total number of windows processed for this label,
                            # tracking files from which signal is extracted, and updating progress bar:
                            nb_windows = input_tensor.shape[0]
                            total += nb_windows
                            pbar.update(nb_windows)
                            n_files.append(fast5)

                            # If the maximum number of signals for this class (label) has
                            # been reached, break the Fast5-file loop and proceed to writing
                            # label stored in memory (al at once)
                            if total == max_windows_per_class:
                                break

                # Writing all training labels to HDF5, as categorical (one-hot) encoding:
                encoded_labels = to_categorical(np.array([label for _ in range(total)]), classes)
                self.write_chunk(labels, encoded_labels)

                # Decoded (label-based) encoding for dataset summary:
                decoded_labels = np.array([label for _ in range(total)])
                self.write_chunk(decoded, decoded_labels)
                # Fast5 file paths from which signal arrays were extracted for dataset summary:
                file_labels = np.array([np.string_(fast5_file) for fast5_file in n_files])
                self.write_chunk(extracted, file_labels)

    def training_validation_split(self, validation: float=0.3, shuffle: bool=True):

        """ This function takes a complete data set generated with write_data,
        randomizes the data and splits it into training and validation under the paths
        training/data, training/label, validation/data, validation/label
        Work with attributes in HDF5.

        :param validation   proportion of data to be split into validation set
        :param shuffle      randomize indices of data in data/data and data/labels
        """

        pass

    @staticmethod
    def create_data_paths(file, window_size=400, classes=2):

        # HDF5 file dataset creation:
        data = file.create_dataset("data/data", shape=(0, 1, window_size, 1), maxshape=(None, 1, window_size, 1))
        labels = file.create_dataset("data/labels", shape=(0, classes), maxshape=(None, classes))

        # For data set summary only:
        decoded = file.create_dataset("data/decoded", shape=(0,), maxshape=(None,))
        extracted = file.create_dataset("data/files", shape=(0,), maxshape=(None,), dtype="S10")

        return data, labels, decoded, extracted

    @staticmethod
    def create_training_validation_paths(file, window_size=400, classes=2):

        data_paths = {
            "training": [],
            "validation": []
        }

        for data_type in data_paths.keys():

            # HDF5 file dataset creation:
            data = file.create_dataset(data_type + "/data", shape=(0, 1, window_size, 1),
                                       maxshape=(None, 1, window_size, 1))

            labels = file.create_dataset(data_type + "/labels", shape=(0, classes),
                                         maxshape=(None, classes))

            data_paths[data_type] += [data, labels]

        return data_paths["training"][0], data_paths["training"][1],\
               data_paths["validation"][0], data_paths["validation"][1]

    def get_data_summary(self, data_type):

        with h5py.File(self.data_file, "r") as f:
            return f[data_type+"/data"].shape, f[data_type+"/labels"].shape

    def print_data_summary(self):

        with h5py.File(self.data_file, "r") as f:

            msg = dedent("""
                HDF5 file: {}
                Training data: {}
                Training labels: {}
                
                Number of training samples per class:
                
                """
                         ).format(self.data_file, f["training/data"].shape, f["training/labels"].shape)

            for label in f["training/decoded/"]:

                # Signal window count:
                window_count = str(f["training/decoded/{label}".format(label=int(label))].shape[0])
                # File count:
                file_count = str(f["training/files/{label}".format(label=int(label))].shape[0])

                msg += "Encoded class: {} = {} from {} files\n".format(int(label), window_count, file_count)

            print(msg)

    @staticmethod
    def write_chunk(dataset, data):

        dataset.resize(dataset.shape[0]+data.shape[0], axis=0)
        dataset[-data.shape[0]:] = data

        return dataset

    @staticmethod
    def transform_signal_to_tensor(vector):

        """ Transform data (nb_windows, window_size) to (nb_windows, 1, window_size, 1)
        for input into Conv2D layer: (samples, height, width, channels),
        """

        # Return 4D array (samples, 1, width, 1)
        # TODO: This can be done better, look at numpy.reshape
        return np.array([[[[signal] for signal in data]] for data in vector[:]])


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
