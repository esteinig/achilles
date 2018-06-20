import json

import matplotlib.pyplot as plt

import achilles.utils as utils

from achilles.model import Achilles
from achilles.dataset import Dataset
from achilles.terminal import Terminal
from achilles.analysis import predict, evaluate, evaluate_predictions


def main():

    # Terminal input
    args = Terminal().args

    if args["agg"]:
        plt.switch_backend('agg')

    if args["subparser"] == "make":

        # Generate data for batch-wise input into Achilles,
        # write training and validation data to HDF5 file
        ds = Dataset(data_file=args["data_file"], log_file=args["log_file"])

        if args["print"]:
            ds.print_data_summary(data_file=args["data_file"])
        else:
            ds.write_data(*args["dirs"], classes=len(args["dirs"]), max_windows_per_class=args["signal_max"],
                          window_size=args["signal_length"], window_step=args["signal_stride"],
                          normalize=args["normalize"], max_windows_per_read=args["window_max"],
                          window_random=args["window_random"], window_recover=True, scale=args["scale"])

            if args["validation"] > 0:
                ds.training_validation_split(validation=args["validation"], window_size=args["signal_length"],
                                             classes=len(args["dirs"]), chunk_size=args["chunk_size"])

    if args["subparser"] == "train":

        # Build model
        achilles = Achilles(data_file=args["data_file"])

        achilles.build(signal_length=args["signal_length"], activation=args["activation"],
                       nb_residual_block=args["nb_residual_blocks"], nb_channels=args["nb_channels"],
                       nb_rnn=args["nb_rnn"], rnn_units=args["rnn_units"], gru=args["gru"], gpu=args["gpu"],
                       dropout=args["dropout"], rc_dropout=args["rc_dropout"])

        # Compile model with loss function and optimizer
        achilles.compile(optimizer=args["optimizer"], loss=args["loss"])

        # Compute estimated memory for dimensions and batch size of model:
        memory = achilles.estimate_memory_usage(batch_size=args["batch_size"])

        print("Estimated GPU memory for Achilles model by layers : {} GB".format(memory))

        achilles.train(epochs=args["epochs"], batch_size=args["batch_size"], workers=args["threads"],
                       run_id=args["run_id"], log_interval=args["log_interval"])

        achilles.save(args["output_file"])

    if args["subparser"] == "evaluate":

        evaluate(data_files=args["data_files"], models=args["model_files"], batch_size=args["batch_size"],
                 workers=args["threads"], data_path=args["data_path"], write=args["output_file"])

    if args["subparser"] == "predict":

        predict(fast5=args["input_files"], model=args["model_file"], window_max=args["windows"],
                window_size=args["window_size"], window_step=args["window_step"],
                batch_size=args["batch_size"], window_random=args["window_random"])

    if args["subparser"] == "pevaluate":

        evaluate_predictions(dirs=args["dirs"], model=args["model_file"], window_max=args["windows"],
                             window_size=args["window_size"], window_step=args["window_step"],
                             batch_size=args["batch_size"], window_random=args["window_random"])

    if args["subparser"] == "plot":

        pass

        # utils.plot_batch_loss_accuracy(fname=args["log_file"], outname=args["plot_file"], error=args["error"])

    if args["subparser"] == "select":

        utils.select_fast5(input_dir=args["input_dir"], output_dir=args["output_dir"], n=args["n"],
                           largest_files=args["largest"])


def config():

    with open(r"/home/esteinig/.keras/keras.json", "r") as keras_config:
        config = json.load(keras_config)

        config["image_dim_ordering"] = 'channel_last'
        config["backend"] = "tensorflow"

    with open(r"/home/esteinig/.keras/keras.json", "w") as keras_config:
        json.dump(config, keras_config)


main()
