# Keras model of Chiron (Asclepius)

import uuid
from keras import layers, Model
from keras.callbacks import TensorBoard, ProgbarLogger, CSVLogger

import numpy as np
from keras import backend as K


class Asclepius:

    def __init__(self):

        self.model = None

    def build(self, signal_length=4000, _nb_channels=256, _nb_classes=2, _lstm_units=200,
              _nb_residual_block_layers=5, _nb_lstm_layers=3, rnn=True, deep=False, summary=True):

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
        x = self.residual_block(inputs, _nb_channels, input_shape=shape)

        if deep:
            for i in range(_nb_residual_block_layers-1):
                x = self.residual_block(x, _nb_channels)


        # Reshape the output layer of residual blocks from 4D to 3D,
        # have changed this from (1, signal_length * _nb_channels, 1)
        # which crashed the LSTM with OOM to (1, signal_length, _nb_channels)
        # which might work?

        x = layers.Reshape((1 * signal_length, _nb_channels))(x)

        ######################
        # Bidirectional LSTM #
        ######################

        # Add two Bidirectional LSTM layers where sequences returned,
        # then into last layer with standard LSTM output into Dense

        if rnn:

            if deep:
                for i in range(_nb_lstm_layers-1):
                    x = layers.Bidirectional(layers.LSTM(_lstm_units, return_sequences=True))(x)  # recurrent_dropout=0.3

            x = layers.Bidirectional(layers.LSTM(_lstm_units))(x)
        else:
            # If no RNN layers, flatten shape for Dense
            x = layers.Flatten()(x)

        outputs = layers.Dense(_nb_classes, activation="softmax")(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        if summary:
            self.model.summary()

        return self.model

    def compile(self, optimizer="Adam", loss="binary_crossentropy"):

        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return self.model

    def train(self, dataset, batch_size=32, epochs=10):

        # For clarity extract the training and test data from dataset object:
        x_train, y_train, x_test, y_test = dataset["data"]["train"], dataset["labels"]["train"],\
                                           dataset["data"]["test"], dataset["labels"]["test"]

        print("Input shape:", x_train.shape)

        csv = CSVLogger('{}.csv'.format(uuid.uuid4()), append=True)

        # TODO: Implement TensorBoard
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=(x_test, y_test), callbacks=[csv])

    @staticmethod
    def residual_block(y, nb_channels, input_shape=None, _strides=(1, 1), _project_shortcut=True):

        """ Residual block adapted from https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64

        Added one more convolution filter and changed kernel sizes to those described in Chiron.
        Also set _project_shortcut to default True for activating condition for sum of shortcut and layers, see Chiron.

        """

        shortcut = y

        if input_shape:
            y = layers.Conv2D(nb_channels, input_shape=input_shape, kernel_size=(1, 1), strides=_strides, padding='same')(y)
        else:
            y = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(y)

        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        y = layers.Conv2D(nb_channels, kernel_size=(1, 3), strides=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        y = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])
        y = layers.LeakyReLU()(y)

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