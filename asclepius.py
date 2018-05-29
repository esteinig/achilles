import json
import datetime

from asclepius.model import Asclepius
from asclepius.dataset import Dataset

# Fast5 raw signal per read
signal_length = 4000
signal_stride = 400

# Constructing training / test data from n reads per class (dir)
max_reads = 10

# Normalizing signal: (s - mean(s)) / std(s)
normalize = True

# Configure activation function for classification output layer,
# activation function and optimizer for compilation

activation = "signmoid"
loss = "binary_crossentropy"
optimizer = "adam"

# Multi-layer residual blocks and Bi-LSTMs

rnn = False

deep = False
nb_residual_block = 5
nb_lstm = 3

# Training with adam and binary_crossentropy
batch_size = 10
epochs = 3

# Config

paperspace = False
cheetah = False

if paperspace:
    dir1 = r"/home/paperspace/asclepius/dir1"
    dir2 = r"/home/paperspace/asclepius/dir2"
elif cheetah:
    dir1 = r"/home/esteinig/code/asclepius/dir1"
    dir2 = r"/home/esteinig/code/asclepius/dir2"
else:
    dir1 = r"C:\Users\jc225327\PycharmProjects\asclepius\dir1"
    dir2 = r"C:\Users\jc225327\PycharmProjects\asclepius\dir2"

###


def main():

    # Run ID
    run_id = datetime.datetime.now()

    # Read data:
    ds = Dataset(dir1, dir2)
    dataset = ds.get_data(max_reads_per_class=max_reads, normalize=normalize,
                          window_size=signal_length, window_step=signal_stride)

    # Build model
    asclep = Asclepius()

    asclep.build(signal_length=signal_length, activation=activation,
                 nb_residual_block=nb_residual_block,
                 nb_lstm=nb_lstm, deep=deep, rnn=rnn)

    asclep.compile(optimizer=optimizer, loss=loss)

    memory = asclep.estimate_memory_usage(batch_size=batch_size)

    print("Estimated GPU memory for Asclepius model: {} GB".format(deep, memory))

    asclep.train(dataset, epochs=epochs, batch_size=batch_size, run_id=run_id)

    asclep.save("model.h5")


def config():

    with open(r"/home/esteinig/.keras/keras.json", "r") as keras_config:
        config = json.load(keras_config)

        config["image_dim_ordering"] = 'channel_last'
        config["backend"] = "tensorflow"

    with open(r"/home/esteinig/.keras/keras.json", "w") as keras_config:
        json.dump(config, keras_config)

main()