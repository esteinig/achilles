import os
import pickle
import numpy as np

from keras import backend as K
from keras import layers, Model
from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import CSVLogger, ModelCheckpoint

from keras import callbacks

from achilles.utils import timeit
from achilles.dataset import AchillesDataset


class Achilles:
    def __init__(self, data_file=None, log_dir=""):

        self.data_file = data_file
        self.log_dir = log_dir
        self.model = None

    def build(
        self,
        window_size=400,
        activation="softmax",
        bidirectional=True,
        nb_channels=256,
        rnn_units=200,
        _nb_classes=2,
        nb_residual_block=1,
        nb_rnn=1,
        dropout=0.0,
        rc_dropout=0.0,
        gru=False,
        gpus=1,
        summary=True,
    ):

        # Kernel size and strides are only used for single convolutional layers (1D, or 2D)

        # Default for resisdual block or Conv2D:
        shape = (1, window_size, 1)

        # Input data shape for residual block (Conv2D)
        inputs = layers.Input(shape=shape)

        ######################
        # Residual Block CNN #
        ######################

        # Residual block stack, see config
        x = self.residual_block(inputs, nb_channels, input_shape=shape)

        if nb_residual_block > 1:
            for i in range(nb_residual_block - 1):
                x = self.residual_block(x, nb_channels)

        # Reshape the output layer of residual blocks from 4D to 3D
        x = layers.Reshape((1 * window_size, nb_channels))(x)

        ######################
        # Bidirectional RNN  #
        ######################

        if gru:
            # GRU does not appear to be as good as LSTM!
            rnn_layer = layers.GRU
        else:
            rnn_layer = layers.LSTM

        dropout_params = {"dropout": dropout, "recurrent_dropout": rc_dropout}

        # Add two Bidirectional RNN layers where sequences returned,
        # then into last layer with standard RNN output into Dense
        if nb_rnn > 0:
            # Deep bidirectional RNN layers must return sequences for stacking
            if nb_rnn > 1:
                for i in range(nb_rnn - 1):
                    # The following structure adds GRU or LSTM cells to the
                    # model, and depending on whether the net is
                    # trained / executed exclusively on GPU, standard cells
                    # are replaced by CuDNN variants, these do
                    # currently not support DROPOUT!
                    if bidirectional:
                        x = layers.Bidirectional(
                            rnn_layer(
                                rnn_units, return_sequences=True, **dropout_params
                            )
                        )(x)
                    else:
                        x = rnn_layer(
                            rnn_units, return_sequences=True, **dropout_params
                        )(x)
            if bidirectional:
                x = layers.Bidirectional(rnn_layer(rnn_units, **dropout_params))(x)
            else:
                x = rnn_layer(rnn_units, **dropout_params)(x)
        else:
            # If no RNN layers, flatten shape for Dense
            x = layers.Flatten()(x)

        outputs = layers.Dense(_nb_classes, activation=activation)(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        if summary:
            self.model.summary()

        if gpus <= 1:
            print("Built model for training on 1 GPU.")
            return self.model
        else:
            print(f"Building model for distributed training on {gpus} GPUs.")
            return multi_gpu_model(self.model, gpus=gpus)

    def save(self, run_id, file):

        self.model.save(os.path.join(run_id, file))

    def compile(self, optimizer="adam", loss="binary_crossentropy"):

        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        return self.model

    def train(
        self,
        batch_size=15,
        epochs=10,
        workers=2,
        run_id="run_1",
        outdir="run_1",
        verbose=True,
        gpu=None,
    ):

        if gpu:
            print(f"CUDA_VISIBLE_DEVICES environment variable set to {gpu}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        # Estimated memory for dimensions and
        # batch size of model, before adjustment:
        memory = self.estimate_memory_usage(batch_size=batch_size)
        print("Estimated GPU memory for Achilles model: {} GB".format(memory))

        # Reads data from HDF5 data file:
        dataset = AchillesDataset()

        # Get training and validation data generators
        training_generator = dataset.get_signal_generator(
            self.data_file, data_type="training", batch_size=batch_size, shuffle=True
        )
        validation_generator = dataset.get_signal_generator(
            self.data_file, data_type="validation", batch_size=batch_size, shuffle=True
        )

        # Make log directory:
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        else:
            outdir = os.getcwd()

        # Callbacks
        csv = CSVLogger(os.path.join(outdir, run_id + ".epochs.log"))
        chk = ModelCheckpoint(
            os.path.join(outdir, run_id + ".checkpoint.val_loss.h5"),
            monitor="val_loss",
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode="auto",
            period=1,
        )

        print(
            f"Running on batch size {batch_size} for {epochs} epochs "
            f"with {workers} worker processes --> run ID: {run_id}"
        )

        # TODO: Enable NCPU

        history = self.model.fit_generator(
            training_generator,
            use_multiprocessing=False,
            workers=workers,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[csv, chk],
            verbose=verbose,
        )

        with open(
            os.path.join(outdir, "{}.model.history".format(run_id)), "wb"
        ) as history_out:
            pickle.dump(history.history, history_out)

    def adjust_batch_size(self, batch_size):

        """ Function for adjusting batch size to live GPU memory; this is not an accurate estimation
        but rather aims at conservatively estimating available GPU memory and adjusting the batch size
        so that training does not raise out-of-memory errors, particularly when using training as part
        of Nextflow workflows where the underlying data dimensions (and therefore memory occupancy) may
        differ between runs or across a grid search.
        """

        # TODO
        mem = self.estimate_memory_usage(batch_size)

    def load_model(self, model_file, summary=True):

        """ Load model from HDF5 output file with model layers and weights """

        # Read model stats

        self.model = load_model(model_file)
        if summary:
            self.model.summary()

    def evaluate(self, eval_generator, workers=2):

        """ Evaluate model against presented dataset """

        loss, acc = self.model.evaluate_generator(
            eval_generator, workers=workers, use_multiprocessing=False, verbose=True
        )

        return loss, acc

    @timeit(micro=True)
    def predict(
        self, signal_tensor: np.array = None, batch_size=10,
        null_pass: np.shape = None
    ):

        """ Predict signal arrays using model test function,
         might implement in class later"""

        # Read Fast5 and extract windows from signal array:

        if null_pass:
            # Warmup pass to allocate memory
            signal_tensor = np.empty(shape=null_pass)

        # Select random or beginning consecutive windows
        return self.model.predict(x=signal_tensor, batch_size=batch_size)

    def predict_generator(self, data_type="data", batch_size=1000):

        # Reads data from HDF5 data file:
        dataset = AchillesDataset()

        # Get training and validation data generators
        prediction_generator = dataset.get_signal_generator(
            self.data_file,
            data_type=data_type,
            batch_size=batch_size,
            shuffle=False,
            no_labels=True,
        )

        return self.model.predict_generator(prediction_generator)

    @staticmethod
    def residual_block(
        y,
        nb_channels,
        input_shape=None,
        _strides=(1, 1),
        _project_shortcut=True,
        _leaky=False,
    ):

        """ Residual block adapted from https://gist.github.com/mjdietzx/5319e42637ed7ef095d430cb5c5e8c64

        Added one more convolution filter and changed kernel sizes to those described in Chiron.
        Also set _project_shortcut to default True for activating condition for sum of shortcut and layers, see Chiron.

        """

        shortcut = y

        # Stack 1
        if input_shape:
            y = layers.Conv2D(
                nb_channels,
                input_shape=input_shape,
                kernel_size=(1, 1),
                strides=_strides,
                padding="same",
            )(y)
        else:
            y = layers.Conv2D(
                nb_channels, kernel_size=(1, 1), strides=_strides, padding="same"
            )(y)

        y = layers.BatchNormalization()(y)

        if _leaky:
            y = layers.LeakyReLU()(y)
        else:
            y = layers.Activation("relu")(y)

        # Stack 2
        y = layers.Conv2D(
            nb_channels, kernel_size=(1, 3), strides=(1, 1), padding="same"
        )(y)
        y = layers.BatchNormalization()(y)

        if _leaky:
            y = layers.LeakyReLU()(y)
        else:
            y = layers.Activation("relu")(y)

        # Stack 3
        y = layers.Conv2D(
            nb_channels, kernel_size=(1, 1), strides=(1, 1), padding="same"
        )(y)
        y = layers.BatchNormalization()(y)

        # ... with shortcut for concatenation before ReLU
        # nb: identity shortcuts used directly when the input
        # and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            shortcut = layers.Conv2D(
                nb_channels, kernel_size=(1, 1), strides=_strides, padding="same"
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        if _leaky:
            y = layers.LeakyReLU()(y)
        else:
            y = layers.Activation("relu")(y)

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

        trainable_count = np.sum(
            [K.count_params(p) for p in set(self.model.trainable_weights)]
        )
        non_trainable_count = np.sum(
            [K.count_params(p) for p in set(self.model.non_trainable_weights)]
        )

        total_memory = (
            4.0
            * batch_size
            * (shapes_mem_count + trainable_count + non_trainable_count)
        )
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)

        return gbytes


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
