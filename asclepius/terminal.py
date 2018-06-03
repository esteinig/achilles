import os
import argparse


class Terminal:

    def __init__(self, commands=None):

        parser = argparse.ArgumentParser()

        subparsers = parser.add_subparsers(help='Tasks for config of Achilles model')

        prep = subparsers.add_parser("prep", help="Generate dataset from Fast5 raw signal for training.")

        prep.add_argument("--dirs", "-d", required=False, dest="dirs", default="dir1,dir2", type=str,
                          help="Class directories with Fast5 files (max: 2)")
        prep.add_argument("--data_file", "-o", required=False, dest="data_file", default="data.h5", type=str,
                          help="Output HDF5 containing training and validation signal data")

        prep.add_argument("--signal_max_per_class", "-max", "-m", required=False, dest="signal_max", default=40000, type=int,
                           help="Maximum number of signal windows extracted from Fast5 directories per class (dir).")

        prep.add_argument("--max_windows_per_read", "-mw", required=False, dest="window_max", default=100,
                          type=int, help="Maximum number of signal windows extracted per Fast5 file.")
        prep.add_argument("--random_windows_per_read", "-rand", "-r", required=False, dest="random",
                          action="store_true", help="Maximum number of signal windows extracted per Fast5 file.")

        prep.add_argument("--signal_length", "-len", "-l", required=False, default=4000, dest="signal_length", type=int,
                           help="Length of signal windows over each read from Fast5.")
        prep.add_argument("--signal_stride", "-s", required=False, default=400, dest="signal_stride", type=int,
                           help="Length of stride for signal windows over each read from Fast5.")

        # TODO: Try without normalization!
        prep.add_argument("--normalize", "-norm", "-n", required=False, action="store_true", dest="normalize",
                           help="Normalize signal values to pA floats (subtract mean, divide by std)")

        prep.add_argument("--print", "-p", required=False, action="store_true", dest="print",
                          help="Print summary of data file")

        prep.set_defaults(subparser='prep')

        train = subparsers.add_parser("train", help="Train Achilles on prepped raw signal data from HDF5.")

        train.add_argument("--data_file", "--file", "-f", required=False, dest="data_file", default="data.h5", type=str,
                          help="HDF5 prepped data file (achilles prep) for streaming batches.")
        train.add_argument("--output_file", "-o", required=False, dest="output_file", default="model.h5", type=str,
                          help="Output trained model to HDF5 file.")
        train.add_argument("--run_id", "-i", required=False, dest="run_id", default="run_test", type=str,
                          help="Training run ID.")

        train.add_argument("--signal_length", "-s", required=False, dest="signal_length", default=4000, type=int,
                          help="Length of signal windows over each read from Fast5.")

        train.add_argument("--batch_size", "-b", required=False, dest="batch_size", default=15, type=int,
                          help="Training mini batch size.")
        train.add_argument("--threads", "-t", required=False, dest="threads", default=2, type=int,
                           help="CPU threads to feed batches into generator to fit to model.")
        train.add_argument("--epochs", "-e", required=False, dest="epochs", default=3, type=int,
                          help="Training epochs.")
        train.add_argument("--log_interval", "-log", required=False, dest="log_interval", default=1, type=int,
                           help="Log loss and accuracy every batch (default: 1).")

        train.add_argument("--activation", "-a", required=False, dest="activation", default="sigmoid", type=str,
                          help="Activation function (default: sigmoid)")
        train.add_argument("--loss", "-l", required=False, dest="loss", default="binary_crossentropy", type=str,
                          help="Loss function (default: binary_crossentropy)")
        train.add_argument("--optimizer", "-opt", required=False, dest="optimizer", default="adam", type=str,
                          help="Gradient optimizer (default: adam)")

        train.add_argument("--nb_channels", "-ch", required=False, type=int, default=256, dest="nb_channels",
                          help="Number of channels in residual block convolution layers.")
        train.add_argument("--nb_residual_blocks", "-rb", required=False, type=int, default=5, dest="nb_residual_blocks",
                          help="Number of residual blocks in CNN layers.")
        train.add_argument("--nb_lstm", "-lstm", required=False, type=int, default=1, dest="nb_lstm",
                          help="Number of bidirectional LSTMs in RNN layers.")
        train.add_argument("--dropout", "-d", required=False, type=float, default=0, dest="dropout",
                           help="Dropout fraction applied to LSTM between 0 and 1 (default: 0.0)")

        train.add_argument("--minimal", "-m", required=False, action="store_true", dest="minimal",
                           help="Shallow architecture (one layer) for testing model.")
        train.add_argument("--cnn_only", "-c", required=False, action="store_true", dest="rnn",
                           help="Deactivate LSTM layers for testing model.")

        train.set_defaults(subparser='train')

        plot = subparsers.add_parser("plot", help="Plot loss and accuracy for model runs from logs.")

        plot.add_argument("--log_file", "--file", "-f", required=False, dest="log_file", default="test_1.log", type=str,
                           help="Log file from model training run.")
        plot.add_argument("--plot_file", "--plot", "-p", required=False, dest="plot_file", default="test.pdf",
                           type=str, help="Plot of loss and accuracy per batch (default: test.pdf).")
        plot.add_argument("--error", "-e", required=False, action="store_true", dest="error",
                          help="Plot accuracy as error: 1 - accuracy")

        plot.set_defaults(subparser='plot')

        select = subparsers.add_parser("select", help="Utility function for selecting largest reads of recursive dir for"
                                                      "generating data and training model")
        select.add_argument("--input_dir", "--in", "-i", required=False, dest="input_dir", default="dir1", type=str,
                          help="Recursive directory of passing Fast5 file for sorting by file size.")
        select.add_argument("--output_dir", "--out", "-o", required=False, dest="output_dir", default="largest",
                          type=str, help="Output file to copy largest Fast5 into.")
        select.add_argument("--number_of_fast5", "--number", "-n", required=False, dest="n", default=3000,
                          type=int, help="Number of largest Fast5 to copy.")

        select.set_defaults(subparser='select')

        if commands is None:
            self.args = parser.parse_args()
        else:
            # Commands is a list of commands to be parsed
            # by Argparse - for testing only:
            self.args = parser.parse_args(commands)

        self.args = vars(self.args)

        # Real paths:

        if "dirs" in self.args.keys():
            self.args["dirs"] = [os.path.abspath(directory) for directory in self.args["dirs"].split(",")]

        for key in ("data_file", "output_file", "log_file", "plot_file", "input_dir", "output_dir"):
            if key in self.args.keys():
                self.args[key] = os.path.abspath(self.args[key])