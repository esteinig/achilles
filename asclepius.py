import json

from asclepius.model import Asclepius
from asclepius.dataset import Dataset
from asclepius.terminal import Terminal

import asclepius.utils as utils

def main():

    # Terminal input
    args = Terminal().args

    if args["subparser"] == "prep":
        # Prepping data for batch-wise input into Achilles
        # Writes trianing and validation data to HDF5 file
        ds = Dataset(data_file=args["data_file"])

        if args["print"]:
            ds.print_data_summary()
        else:
            ds.write_data(*args["dirs"], classes=len(args["dirs"]), max_per_class=args["signal_max"],
                          window_size=args["signal_length"], window_step=args["signal_stride"],
                          normalize=args["normalize"])

            ds.print_data_summary()

    if args["subparser"] == "train":

        # Build model
        asclep = Asclepius(data_file=args["data_file"])

        asclep.build(signal_length=args["signal_length"], activation=args["activation"],
                     nb_residual_block=args["nb_residual_blocks"], nb_channels=args["nb_channels"],
                     nb_lstm=args["nb_lstm"], minimal=args["minimal"], rnn=args["rnn"])

        asclep.compile(optimizer=args["optimizer"], loss=args["loss"])

        memory = asclep.estimate_memory_usage(batch_size=args["batch_size"])

        print("Estimated GPU memory for Asclepius model: {} GB".format(memory))

        asclep.train(epochs=args["epochs"], batch_size=args["batch_size"],
                     workers=args["threads"], run_id=args["run_id"], log_interval=args["log_interval"])

        asclep.save(args["output_file"])

    if args["subparser"] == "plot":

        utils.plot_batch_loss_accuracy(fname=args["log_file"], outname=args["plot_file"], error=args["error"])

    if args["subparser"] == "select":

        utils.select_largest_files(input_dir=args["input_dir"], output_dir=args["output_dir"], n=args["n"])


def config():

    with open(r"/home/esteinig/.keras/keras.json", "r") as keras_config:
        config = json.load(keras_config)

        config["image_dim_ordering"] = 'channel_last'
        config["backend"] = "tensorflow"

    with open(r"/home/esteinig/.keras/keras.json", "w") as keras_config:
        json.dump(config, keras_config)


main()