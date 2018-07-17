import json

import matplotlib.pyplot as plt

from achilles.model import Achilles
from achilles.dataset import Dataset
from achilles.terminal import Terminal

from achilles.analysis import predict, evaluate, evaluate_predictions
from achilles.runners import pevaluate_runner, plot_runner_results

from achilles.select import select_fast5

def main():

    # Terminal input
    args = Terminal().args

    if args["agg"]:
        plt.switch_backend('agg')

    if args["subparser"] == "make":

        # Generate data for batch-wise input into Achilles,
        # write training and validation data to HDF5 file
        ds = Dataset(data_file=args["output_file"])

        ds.write_data(*args["dirs"], classes=len(args["dirs"]), max_windows_per_class=args["signal_max"],
                      window_size=args["signal_length"], window_step=args["signal_stride"],
                      normalize=args["normalize"], max_windows_per_read=args["window_max"],
                      window_random=args["window_random"], window_recover=True, scale=args["scale"],
                      include=args["include"], exclude=args["exclude"])

        if args["validation"] > 0:
            ds.training_validation_split(validation=args["validation"], window_size=args["signal_length"],
                                         classes=len(args["dirs"]), chunk_size=args["chunk_size"])

    if args["subparser"] == "train":

        # Build model
        achilles = Achilles(data_file=args["data_file"])

        if args["load"]:
            achilles.load_model(args["load"])
        else:
            achilles.build(signal_length=args["signal_length"], activation=args["activation"],
                           nb_residual_block=args["nb_residual_blocks"], nb_channels=args["nb_channels"],
                           nb_rnn=args["nb_rnn"], rnn_units=args["rnn_units"], gru=args["gru"], gpu=args["gpu"],
                           dropout=args["dropout"], rc_dropout=args["rc_dropout"], bidirectional=args["bi"],
                           conv_2d=args["conv_2d"])

            # Compile model with loss function and optimizer
            achilles.compile(optimizer=args["optimizer"], loss=args["loss"])

        # Compute estimated memory for dimensions and batch size of model:
        memory = achilles.estimate_memory_usage(batch_size=args["batch_size"])

        print("Estimated GPU memory for Achilles model by layers : {} GB".format(memory))

        achilles.train(epochs=args["epochs"], batch_size=args["batch_size"], workers=args["threads"],
                       run_id=args["run_id"], verbose=args["verbose"])

        achilles.save(args["run_id"], args["output_file"])

    if args["subparser"] == "evaluate":

        evaluate(data_files=args["data_files"], models=args["model_files"], batch_size=args["batch_size"],
                 workers=args["threads"], data_path=args["data_path"], write=args["output_file"])

    if args["subparser"] == "predict":

        predict(fast5=args["input_files"], model=args["model_file"], window_max=args["windows"],
                window_size=args["window_size"], window_step=args["window_step"],
                batches=args["batches"], window_random=args["window_random"])

    if args["subparser"] == "pevaluate":

        evaluate_predictions(dirs=args["dirs"], model=args["model_file"], window_max=args["windows"],
                             window_size=args["window_size"], window_step=args["window_step"],
                             batches=args["batches"], window_random=args["window_random"],
                             prefix=args["prefix"], include=args["include"], exclude=args["exclude"],
                             class_labels=args["labels"])

    if args["subparser"] == "runner":

        if args["plot_pickle"]:
            plot_runner_results(pickle_file=args["plot_pickle"], class_labels=args["labels"], runner=args["runner"])
        else:

            if args["runner"] == "pevaluate":

                pevaluate_runner(config=args["config"], class_labels=args["labels"], outdir=args["output_dir"])

    if args["subparser"] == "select":

        select_fast5(input_dir=args["input_dir"], output_dir=args["output_dir"], limit=args["number"],
                     shuffle=args["random"], min_signal=args["min_signal"], symlink=args["symlink"],
                     include=args["include"], exclude=args["exclude"])


def config():

    with open(r"/home/esteinig/.keras/keras.json", "r") as keras_config:
        config = json.load(keras_config)

        config["image_dim_ordering"] = 'channel_last'
        config["backend"] = "tensorflow"

    with open(r"/home/esteinig/.keras/keras.json", "w") as keras_config:
        json.dump(config, keras_config)


main()
