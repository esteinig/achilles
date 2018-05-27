import json

from asclepius.model import Asclepius
from asclepius.dataset import Dataset

###

# Config

dir1 = r"/home/paperspace/asclepius/dir1"
dir2 = r"home/paperspace/asclepius/dir2"

signal_length = 4000
signal_stride = 400

# Constructing training / test data from n reads per class (dir)
max_reads = 1

# Normalizing signal: (s - mean(s)) / std(s)
normalize = True

# Multi-layer residual blocks and Bi-LSTMs
deep = False

# Training with adam and binary_crossentropy
batch_size = 10
epochs = 2

###

# 1. 40, 4
# 2. 4000, 400
# 3. -"- epochs 128 batch_size 64
# 4. Reduce reads (1 per class), epochs 10 batch size 10, TensorBoard and history, print to out STDOUT (run4.log)
# 5. Same as before but point TensorBoard to UUID directory in /logs
# 6. Increased RAM to 128 GB, jobs before getting killed. Reason not clear.
# 7. No TensorBoard callback, checking for model build and training output? Should print shape before training.
# 8. Not writing to log file with >
# 9. - 12. RAM tests

###


def main():

    ds = Dataset(dir1, dir2)

    dataset = ds.get_data(max_reads_per_class=max_reads, normalize=normalize,
                          window_size=signal_length, window_step=signal_stride)

    # dataset.plot_random_sample()

    asclep = Asclepius()
    asclep.build(signal_length=signal_length, deep=deep)

    print("Built Asclepius model (deep = {}).".format(deep))

    asclep.compile()

    print("Compiled Asclepius model (deep = {}).".format(deep))

    memory = asclep.estimate_memory_usage(batch_size=batch_size)

    print("Estimated memory for Asclepius model (deep = {}): {} GB".format(deep, memory))

    asclep.train(dataset, epochs=epochs, batch_size=batch_size)

    asclep.save("model.h5")


def config():

    with open(r"/home/esteinig/.keras/keras.json", "r") as keras_config:
        config = json.load(keras_config)

        config["image_dim_ordering"] = 'channel_last'
        config["backend"] = "tensorflow"

    with open(r"/home/esteinig/.keras/keras.json", "w") as keras_config:
        json.dump(config, keras_config)

main()