import os
import argparse


class Terminal:

    def __init__(self, commands=None):

        parser = argparse.ArgumentParser()

        parser.add_argument("--agg", required=False, dest="agg", action="store_true", default=False,
                            help="Use Agg backend for plotting without X-server.")

        subparsers = parser.add_subparsers(help='Tasks for configuration of Achilles model')

        prep = subparsers.add_parser("make", help="Generate dataset from Fast5 raw signal for training.")

        prep.add_argument("--dirs", "-d", required=False, dest="dirs", default="dir1,dir2", type=str,
                          help="Class directories with Fast5 files (max: 2)")
        prep.add_argument("--data_file", "-o", required=False, dest="data_file", default="data.h5", type=str,
                          help="Output HDF5 containing training and validation signal data")
        prep.add_argument("--log_file", "-log", required=False, dest="log_file", default="data.log", type=str,
                          help="Log file for generating training data.")
        prep.add_argument("--max_windows_per_class", "-max", "-m", required=False, dest="signal_max", default=1000, type=int,
                          help="Maximum number of signal windows extracted from Fast5 directories per class (dir).")
        prep.add_argument("--max_windows_per_read", "-mw", required=False, dest="window_max", default=100,
                          type=int, help="Maximum number of signal windows extracted per Fast5 file.")
        prep.add_argument("--windows_from_start", "--start", required=False, dest="window_start",
                          action="store_true", help="Extract consecutive windows from start of read.")
        prep.add_argument("--window_length", "-len", "-l", required=False, default=400, dest="signal_length", type=int,
                           help="Length of signal windows.")
        prep.add_argument("--raw", "-r", required=False, action="store_true", dest="raw",
                          help="Use raw (DAC) values instead of scaled picoampere (pA).")
        prep.add_argument("--window_step", "-s", required=False, default=400, dest="signal_stride", type=int,
                           help="Length of step for signal windows.")
        prep.add_argument("--normalize", "-norm", "-n", required=False, action="store_true", dest="normalize",
                           help="Normalize signal values to pA floats (subtract mean, divide by std)")
        prep.add_argument("--validation", "-val", "-v", required=False, default=0.3, dest="validation", type=float,
                          help="Proportion of data to randomly split into validation set.")
        prep.add_argument("--chunk_size", "--chunk", "-c", required=False, default=1000, dest="chunk_size", type=int,
                          help="Chunk size for writing training and validation data when splitting dataset.")
        prep.add_argument("--print", "-p", required=False, action="store_true", dest="print",
                          help="Print summary of data file")

        prep.set_defaults(subparser='make')

        train = subparsers.add_parser("train", help="Train Achilles on prepped raw signal data from HDF5.")

        train.add_argument("--data_file", "--file", "-f", required=False, dest="data_file", default="data.h5", type=str,
                          help="HDF5 prepped data file (achilles prep) for streaming batches.")
        train.add_argument("--output_file", "-o", required=False, dest="output_file", default="model.h5", type=str,
                          help="Output trained model to HDF5 file.")
        train.add_argument("--run_id", "-i", required=False, dest="run_id", default="run_test", type=str,
                          help="Training run ID.")
        train.add_argument("--signal_length", "--length", "-s", required=False, dest="signal_length", default=400, type=int,
                          help="Length of signal windows over each read from Fast5.")
        train.add_argument("--batch_size", "-b", required=False, dest="batch_size", default=15, type=int,
                          help="Training mini batch size.")
        train.add_argument("--threads", "-t", required=False, dest="threads", default=2, type=int,
                           help="CPU threads to feed batches into generator to fit to model.")
        train.add_argument("--epochs", "-e", required=False, dest="epochs", default=3, type=int,
                          help="Training epochs.")
        train.add_argument("--log_interval", "-log", required=False, dest="log_interval", default=1, type=int,
                           help="Log loss and accuracy every batch (default: 1).")
        train.add_argument("--activation", "-a", required=False, dest="activation", default="softmax", type=str,
                          help="Activation function (default: softmax)")
        train.add_argument("--loss", "-l", required=False, dest="loss", default="binary_crossentropy", type=str,
                          help="Loss function (default: binary_crossentropy)")
        train.add_argument("--optimizer", "-opt", required=False, dest="optimizer", default="adam", type=str,
                          help="Gradient optimizer (default: adam)")

        train.add_argument("--nb_residual_blocks", "-rb", required=False, type=int, default=1, dest="nb_residual_blocks",
                          help="Number of residual blocks in CNN layers.")
        train.add_argument("--nb_channels", "-ch", required=False, type=int, default=256, dest="nb_channels",
                          help="Number of channels in residual block convolution layers.")

        train.add_argument("--nb_rnn", "-rnn", required=False, type=int, default=1, dest="nb_rnn",
                          help="Number of bidirectional RNN layers (default LSTM).")
        train.add_argument("--rnn_units", "-units", "-u", required=False, type=int, default=200, dest="rnn_units",
                           help="Number of units in bidirectional RNN layers (default 200).")

        train.add_argument("--load", required=False, dest="load", default=None, type=str,
                           help="Load a trained model for transfer training.")

        train.add_argument("--gru", required=False, action="store_true", dest="gru",
                           help="Use GRU layers instead of LSTM in RNN.")
        train.add_argument("--gpu", "-g", required=False, action="store_true", dest="gpu",
                           help="Use CuDNN variants of RNN layers (only when using GPU).")

        # Architecture simplification:
        train.add_argument("--deactivate_bidirectional", "--no_bi", required=False, action="store_false", dest="bi",
                           default=True, help="Deactivate bidirectional RNN layers for parameter reduction.")
        train.add_argument("--conv_2d", "--conv", required=False, action="store_true", dest="conv_2d", default=False,
                           help="Activate simple convolutional layer (2D + ReLU) instead of Residual Block.")

        train.add_argument("--dropout", "-d", required=False, type=float, default=0, dest="dropout",
                           help="Dropout fraction applied to LSTM between 0 and 1 (default: 0.0)")
        train.add_argument("--recurrent_dropout", "--rc_dropout", "-r", required=False, type=float, default=0,
                           dest="rc_dropout", help="Dropout fraction applied to LSTM between 0 and 1 (default: 0.0)")

        train.set_defaults(subparser='train')

        eval = subparsers.add_parser("evaluate", help="Evaluate data with given model file on data paths"
                                                      "data_path/data and data_path/label in HDF5 file.")

        eval.add_argument("--data_files", "--file", "-f", required=True, dest="data_files", type=str,
                          help="HDF5 prepared data files for evaluation.")
        eval.add_argument("--model_files", "--model", "-m", required=True, dest="model_files", type=str,
                          help="HDF5 trained model files for evaluation.")
        eval.add_argument("--batch_size", "-b", required=False, dest="batch_size", default=100, type=int,
                           help="Training mini batch size.")
        eval.add_argument("--threads", "-t", required=False, dest="threads", default=2, type=int,
                           help="CPU threads to feed batches into generator to fit to model.")
        eval.add_argument("--data_path", "-d", required=False, dest="data_path", default="data", type=str,
                          help="HDF5 data path for data_path/training and data_path/labels.")
        eval.add_argument("--output_file", "--out", "-o", required=False, dest="output_file", default="evaluation.csv",
                          type=str, help="CSV output paths for evaluations")
        eval.set_defaults(subparser='evaluate')

        # Prediction
        pred = subparsers.add_parser("predict", help="Run prediction on Fast5 signal file")

        pred.add_argument("--input_files", "--input", "-i", required=False, dest="input_files", default="read.fast5",
                          type=str, help="Fast5 files for prediction.", nargs="+")
        pred.add_argument("--model_file", "--model", "-m", required=False, dest="model_file", default="model.h5",
                          type=str, help="HDF5 prepped trained model file for prediction.")
        pred.add_argument("--windows", "-w", required=False, dest="windows", default=10, type=int,
                          help="Number of consecutive windows to extract for prediction.")
        pred.add_argument("--windows_size", "--size", required=False, dest="window_size", default=400, type=int,
                          help="Window size to extract for prediction.")
        pred.add_argument("--window_step", "--step", required=False, dest="window_step", default=400, type=int,
                          help="Number of consecutive windows to extract for prediction.")
        pred.add_argument("--window_random", "--random", required=False, action="store_true", dest="window_random",
                          help="Number of consecutive windows to extract for prediction.")
        pred.add_argument("--raw", "-r", required=False, action="store_true", dest="raw",
                          help="Use raw (DAC) values instead of scaled picoampere (pA).")
        pred.add_argument("--batches", "-b", required=False, dest="batches", default=10, type=int,
                          help="Number of files to predict for in one pass through model (*windows = batch_size)")
        pred.set_defaults(subparser='predict')

        # Prediction Evaluation
        peval = subparsers.add_parser("pevaluate", help="Run prediction evaluation on Fast5 signal files")
        peval.add_argument("--dirs", "-d", required=True, dest="dirs", default="dir1,dir2", type=str,
                           help="Class directories with Fast5 files same order as training (max: 2)")
        peval.add_argument("--model_file", "--model", "-m", required=False, dest="model_file", default="model.h5",
                           type=str, help="HDF5 prepped trained model file for prediction.")
        peval.add_argument("--windows", "-w", required=False, dest="windows", default=10, type=int,
                          help="Number of consecutive windows to extract for prediction.")
        peval.add_argument("--windows_size", "--size", required=False, dest="window_size", default=400, type=int,
                          help="Window size to extract for prediction.")
        peval.add_argument("--window_step", "--step", required=False, dest="window_step", default=400, type=int,
                          help="Number of consecutive windows to extract for prediction.")
        peval.add_argument("--window_random", "--random", required=False, action="store_true", dest="window_random",
                          help="Number of consecutive windows to extract for prediction.")
        peval.add_argument("--raw", "-r", required=False, action="store_true", dest="raw",
                          help="Use raw (DAC) values instead of scaled picoampere (pA).")
        peval.add_argument("--batches", "-b", required=False, dest="batches", default=10, type=int,
                           help="Number of files for batch-wise prediction (*windows = batch_size for Keras).")
        peval.add_argument("--prefix", "-p", required=False, dest="prefix", default="peval", type=str,
                           help="Prefix for plot and summary outputs.")
        peval.set_defaults(subparser='pevaluate')

        plot = subparsers.add_parser("plot", help="Plot loss and accuracy for model runs from logs.")
        
        plot.add_argument("--log_file", "--file", "-f", required=False, dest="log_file", default="test_1.log", type=str,
                          help="Log file from model training run.")
        plot.add_argument("--plot_file", "--plot", "-p", required=False, dest="plot_file", default="test.pdf",
                          type=str, help="Plot of loss and accuracy per batch (default: test.pdf).")
        plot.add_argument("--error", "-e", required=False, action="store_true", dest="error",
                          help="Plot accuracy as error: 1 - accuracy")
        plot.set_defaults(subparser='plot')

        select = subparsers.add_parser("select", help="Utility function for selecting signal from Fast5 in recursive"
                                                      " directory structure for generating data and training model")

        select.add_argument("--input_dir", "--in", "-i", required=False, dest="input_dir", default="dir1", type=str,
                            help="Recursive directory of Fast5 files to select from randomly.")
        select.add_argument("--output_dir", "--out", "-o", required=False, dest="output_dir", default="out_select",
                            type=str, help="Output file to copy largest Fast5 into.")
        select.add_argument("--number", "-n", required=False, dest="number", default=None, type=int,
                            help="Number of Fast5 files to copy.")
        select.add_argument("--min_signal", "-m", required=False, dest="min_signal", default=None, type=int,
                            help="Minimum length of signal array .")
        select.add_argument("--random", "-r", required=False, dest="random",  action="store_true",
                            help="Shuffle files and select random subset.")
        select.add_argument("--symlink", "-s", required=False, dest="symlink", action="store_true",
                            help="Create symlinks instead of copying files.")
        select.add_argument("--include", required=False, dest="include", default="", type=str,
                            help="Include data file Fast5 (.h5) or single string or list of strings (item1,item2) "
                                 "for selection if string in part of file path.")
        select.add_argument("--exclude", required=False, dest="exclude", default="", type=str,
                            help="Exclude data file Fast5 (.h5) or single string or list of strings (item1,item2) "
                                 "for selection if string in part of file path.")

        select.set_defaults(subparser='select')

        runner = subparsers.add_parser("runner", help="Execute runners for summary evaluations / predictions.")

        runner.add_argument("--runner", "-r", required=True, dest="runner", default="pevaluate",
                            type=str, help="Name of runner to execute.")
        runner.add_argument("--config", "-c", required=False, dest="config", default="config.json",
                            type=str, help="Configuration file for runner (JSON).")
        runner.add_argument("--output_dir", "--out", "-o", required=False, dest="output_dir", default="largest",
                            type=str, help="Output directory for runner results.")
        runner.add_argument("--plot_pickle", "--plot", "-p", required=False, dest="plot_pickle", default="",
                            type=str, help="Pickle file to plot summary, requires name of Runner and Labels.")
        runner.add_argument("--labels", "-l", required=False, dest="labels", default="0,1",
                            type=str, help="Label names, comma-separated")

        runner.set_defaults(subparser='runner')

        if commands is None:
            self.args = parser.parse_args()
        else:
            # Commands is a list of commands to be parsed
            # by Argparse - for testing only:
            self.args = parser.parse_args(commands)

        self.args = vars(self.args)

        # Data checks and conversions

        # For input lists (comma-separated):

        for key in ("labels",):
            if key in self.args.keys():
                self.args[key] = [item for item in self.args[key].split(",")]

        # For input lists (comma-separated) as paths:
        for key in ("dirs", "data_files", "model_files"):
            if key in self.args.keys():
                self.args[key] = [os.path.abspath(directory) for directory in self.args[key].split(",")]

        # For input strings that are paths:
        for key in ("data_file", "output_file", "log_file", "plot_file", "input_dir", "output_dir"):
            if key in self.args.keys():
                self.args[key] = os.path.abspath(self.args[key])

        # For nargs:
        if "input_files" in self.args.keys():
            self.args["input_files"] = [os.path.abspath(file) for file in self.args["input_files"]]

        # For making dataset, random window sampling is default on,
        # to disable with window_start flag for better user comprehension:
        if "window_start" in self.args.keys():
            if self.args["window_start"]:
                self.args["window_random"] = False
            else:
                self.args["window_random"] = True

        if "raw" in self.args.keys():
            if self.args["raw"]:
                self.args["scale"] = False
            else:
                self.args["scale"] = True

        if "include" in self.args.keys():
            # Split input string by comma and transform into list
            # for function checks on what to include or exclude for selection:
            self.args["include"] = [os.path.abspath(include) if include.endswith(".h5") else include
                                    for include in self.args["include"].split(",")]

        if "exclude" in self.args.keys():
            # Split input string by comma and transform into list
            # for function checks on what to include or exclude for selection:
            self.args["exclude"] = [os.path.abspath(exclude) if exclude.endswith(".h5") else exclude
                                    for exclude in self.args["exclude"].split(",")]

