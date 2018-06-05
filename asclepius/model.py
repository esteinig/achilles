# Keras model of Chiron (Asclepius)

import os
import re
import shutil
import pickle
import numpy as np

from keras import backend as K
from keras import layers, Model
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint

from asclepius.utils import BatchLogger
from asclepius.dataset import Dataset


class Asclepius:

    def __init__(self, data_file=None):

        self.data_file = data_file
        self.model = None

    def build(self, signal_length=4000, activation="softmax", nb_channels=256, _nb_classes=2, _lstm_units=200,
              nb_residual_block=1, nb_lstm=1, dropout=0.0, rc_dropout=0.0, summary=True):

        # Need to talk to Micheal, how to convert the signal sequence to input Conv2D
        # with dimensions (height, width, depth) - since it is a signal sequence:
        # height = 1, depth = 1

        shape = (1, signal_length, 1)

        # Input data shape for residual block (Conv2D)
        inputs = layers.Input(shape=shape)

        ######################
        # Residual Block CNN #
        ######################

        # Residual block stack, see config
        # There must always be one residual block for input dimensions
        # by data generator:
        x = self.residual_block(inputs, nb_channels, input_shape=shape)

        if nb_residual_block > 1:
            for i in range(nb_residual_block-1):
                x = self.residual_block(x, nb_channels)

        # Reshape the output layer of residual blocks from 4D to 3D,
        # have changed this from (1, signal_length * _nb_channels, 1)
        # which crashed the LSTM with OOM to (1, signal_length, _nb_channels)
        # which might work?

        # Check dimensins here!
        x = layers.Reshape((1 * signal_length, nb_channels))(x)

        ######################
        # Bidirectional LSTM #
        ######################

        # Add two Bidirectional LSTM layers where sequences returned,
        # then into last layer with standard LSTM output into Dense

        if nb_lstm > 0:
            # Deep bidirectional LSTM layers must return sequences for stacking
            if nb_lstm > 1:
                for i in range(nb_lstm-1):
                    x = layers.Bidirectional(layers.LSTM(_lstm_units, return_sequences=True, dropout=dropout,
                                                         recurrent_dropout=rc_dropout))(x)

            x = layers.Bidirectional(layers.LSTM(_lstm_units, dropout=dropout, recurrent_dropout=rc_dropout))(x)
        else:
            # If no RNN layers, flatten shape for Dense
            x = layers.Flatten()(x)

        outputs = layers.Dense(_nb_classes, activation=activation)(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        if summary:
            self.model.summary()

        return self.model

    def save(self, file):

        self.model.save(file)

    def compile(self, optimizer="adam", loss="binary_crossentropy"):

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return self.model

    def train(self, batch_size=15, epochs=10, workers=2, run_id="run_1", log_interval=10):

        # Reads data from HDF5 data file:
        dataset = Dataset(data_file=self.data_file)

        training_generator = dataset.get_signal_generator(data_type="training", batch_size=batch_size, shuffle=True)
        validation_generator = dataset.get_signal_generator(data_type="validation", batch_size=batch_size, shuffle=True)

        log_file = self.init_logs(run_id=run_id)

        # Callbacks
        csv = CSVLogger(run_id + ".epochs.log")
        log = BatchLogger(log_file, log_interval=log_interval)
        chk = ModelCheckpoint(run_id + ".checkpoint.val_loss.h5", monitor="val_loss", verbose=0,
                              save_best_only=False, save_weights_only=False,
                              mode="auto", period=1)

        print("Running on batch size {} for {} epochs with {} worker processes --> run ID: {}"
              .format(batch_size, epochs, workers, run_id))

        data_shape, label_shape = dataset.get_data_summary("training")

        print("Training data dimensions: {}".format(data_shape))
        print("Training label dimensions: {}".format(label_shape))

        # TODO: Implement TensorBoard
        history = self.model.fit_generator(training_generator, use_multiprocessing=True, workers=workers, epochs=epochs,
                                           validation_data=validation_generator, callbacks=[log, csv, chk])

        with open("{}.model.history".format(run_id), "wb") as history_out:
            pickle.dump(history.history, history_out)

    def load_model(self, model_file):

        """ Load model from HDF5 output file with model layers and weights """

        self.model = load_model(model_file)

    def evaluate(self, batch_size=10, workers=2, data_path="training"):

        """ Evaluate model against presented dataset """

        dataset = Dataset(data_file=self.data_file)

        # For evaluation get the training generator from the HDF5 data file prepared with asclepius make,
        # this way different input data can be evaluated (70% of it as split into training) - it is
        # easier to integrate this right now with the task make, but if user supplied format the data must
        # be in the paths HDF5 paths: /training/data,/training/labels or /data_path/data, /data_path/labels
        eval_generator = dataset.get_signal_generator(data_type=data_path, batch_size=batch_size, shuffle=True)

        loss, acc = self.model.evaluate_generator(eval_generator, workers=workers, use_multiprocessing=True)

        msg = "Loss: {} - Accuracy: {}".format(loss, acc)

        print(msg)

    def predict(self):

        """ Predict signal arrays using model test function, might implement in class later"""

        pass

    @staticmethod
    def init_logs(run_id):

        # Make log directory:
        os.makedirs(run_id, exist_ok=True)

        # Log file path:
        log_path = os.path.join(run_id, run_id + ".log")

        # TODO: Fix error None type has no group attribute in regex
        if os.path.exists(log_path):
            # Extract trailing number from log file strings:
            log_numbers = [int(re.match('.*?([0-9]+)$', file).group(1))
                           for file in os.listdir(run_id) if ".log" in file]

            # Get trailing number
            if not log_numbers:
                nb_log = 1
            else:
                nb_log = max(log_numbers)+1

            # Move the current log (run_id.log) to consecutively numbered log files:
            # > run_id/run_id.log.1 (first run after current)
            # > run_id/run_id.log.2 (second one after current)

            shutil.move(log_path, log_path + "." + str(nb_log))

        return log_path

    @staticmethod
    def residual_block(y, nb_channels, input_shape=None, _strides=(1, 1), _project_shortcut=True, _leaky=False):

        """ Residual block adapted from https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64

        Added one more convolution filter and changed kernel sizes to those described in Chiron.
        Also set _project_shortcut to default True for activating condition for sum of shortcut and layers, see Chiron.

        """

        shortcut = y

        # Stack 1
        if input_shape:
            y = layers.Conv2D(nb_channels, input_shape=input_shape, kernel_size=(1, 1),
                              strides=_strides, padding='same')(y)
        else:
            y = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(y)

        y = layers.BatchNormalization()(y)

        if _leaky:
            y = layers.LeakyReLU()(y)
        else:
            y = layers.Activation('relu')(y)

        # Stack 2
        y = layers.Conv2D(nb_channels, kernel_size=(1, 3), strides=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)

        if _leaky:
            y = layers.LeakyReLU()(y)
        else:
            y = layers.Activation('relu')(y)

        # Stack 3
        y = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)

        # ... with shortcut for concatenation before ReLU
        # nb: identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        if _leaky:
            y = layers.LeakyReLU()(y)
        else:
            y = layers.Activation('relu')(y)

        return y

    def estimate_memory_usage(self, batch_size):

        """ https://stackoverflow.com/a/46216013 """

        shapes_mem_count = 0
        for l in self.model.layers:
            single_layer_mem = 1
            for s in l.output_shape:
                if s is None:
                    continue
                single_layer_mem *= s
            shapes_mem_count += single_layer_mem

        trainable_count = np.sum([K.count_params(p) for p in set(self.model.trainable_weights)])
        non_trainable_count = np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)])

        total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)

        return gbytes