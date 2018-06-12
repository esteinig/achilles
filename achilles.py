import json
import achilles.utils as utils

from achilles.model import Achilles
from achilles.dataset import Dataset
from achilles.terminal import Terminal
from achilles.predictor import predict


def main():

    # Terminal input
    args = Terminal().args

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
                          random_consecutive_windows=args["random"])

            if args["validation"] > 0:
                ds.training_validation_split(validation=args["validation"], window_size=args["signal_length"],
                                             classes=len(args["dirs"]))

    if args["subparser"] == "train":

        # Build model
        achilles = Achilles(data_file=args["data_file"])

        achilles.build(signal_length=args["signal_length"], activation=args["activation"],
                       nb_residual_block=args["nb_residual_blocks"], nb_channels=args["nb_channels"],
                       nb_lstm=args["nb_lstm"], dropout=args["dropout"], rc_dropout=args["rc_dropout"])

        # Compile model with loss function and optimizer
        achilles.compile(optimizer=args["optimizer"], loss=args["loss"])

        # Compute estimated memory for dimensions and batch size of model:
        memory = achilles.estimate_memory_usage(batch_size=args["batch_size"])

        print("Estimated GPU memory for Achilles model by layers : {} GB".format(memory))

        achilles.train(epochs=args["epochs"], batch_size=args["batch_size"], workers=args["threads"],
                       run_id=args["run_id"], log_interval=args["log_interval"])

        achilles.save(args["output_file"])

    if args["subparser"] == "evaluate":

        achilles = Achilles(data_file=args["data_file"])

        print("Loading model...")
        achilles.load_model(model_file=args["model_file"])

        print("Evaluating model...")
        achilles.evaluate(batch_size=args["batch_size"], workers=args["threads"], data_path=args["data_path"])

    if args["subparser"] == "predict":

        # Options here: predict file (random, beginning, watch dir for new files and predict, ReadUntil API)

        model = Achilles()

        model.load_model(args["model_file"])

        # Is it faster to predict by file or aggregate into batches
        # and predict bacth-wise - probably for live sequencing, it needs
        # to be by file, for retrospective / simulation as batches.
        for file in args["input_files"]:
            prediction = predict(fast5=file, model=model, window_max=args["windows"], window_size=args["window_size"],
                                 window_step=args["window_step"], batch_size=args["batch_size"],
                                 random=args["window_random"])

            print(prediction)

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
