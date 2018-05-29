import argparse


class Terminal:

    def __init__(self, commands=None):

        parser = argparse.ArgumentParser()

        subparsers = parser.add_subparsers(help='Achilles')

        make = subparsers.add_parser("make", help="Make dataset from Fast5 ra signal for reading into training.")

        make.add_argument("--dirs", "-d", required=False, dest="dirs", default="dir1,dir2", type=str,
                           help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--out_file", "-o", required=False, dest="out_file", default="data.h5", type=str,
                          help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--run_id", "-i", required=False, dest="run_id", default="run_test", type=str,
                           help="Surveillance query file: surveillance/query.txt")


        make.add_argument("--max_signals", "-m", required=False, dest="max", default="dir1,dir2", type=str,
                           help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--norm", "-n", required=False, action="store_true", dest="normalize",
                           help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--signal_length", "l", required=False, action=4000, dest="signal_length", type=int,
                           help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--signal_stride", "-s", required=False, action=4000, dest="signal_stride", type=int,
                           help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--min_read_length", "-min", required=False, action=1000, dest="min", type=int,
                           help="Surveillance query file: surveillance/query.txt")

        make.set_defaults(subparser='make')

        make = subparsers.add_parser("train", help="Train Achilles on data.")

        make.add_argument("--data_file", "--file", "-f", required=False, dest="file", default="model.h5", type=str,
                          help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--output_file", "-o", required=False, dest="output", default="model.h5", type=str,
                          help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--run_id", "-i", required=False, dest="run_id", default="run_test", type=str,
                          help="Surveillance query file: surveillance/query.txt")

        make.add_argument("--signal_length", "-s", required=False, dest="signal_length", default=4000, type=int,
                          help="Surveillance query file: surveillance/query.txt")

        make.add_argument("--activation", "-a", required=False, dest="activation", default="sigmoid", type=str,
                          help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--loss", "-l", required=False, dest="loss", default="binary_crossentropy", type=str,
                          help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--optimizer", "-opt", required=False, dest="optimizer", default="adam", type=str,
                          help="Surveillance query file: surveillance/query.txt")

        make.add_argument("--deep", "-d", required=False, action="store_true", dest="deep",
                          help="Deep architecture for multiple layer model.")
        make.add_argument("--cnn_only", "-c", required=False, action="store_true", dest="rnn",
                          help="Deactivate LSTM layers.")

        make.add_argument("--batch_size", "-b", required=False, dest="batch_size", default=15, type=int,
                          help="Surveillance query file: surveillance/query.txt")
        make.add_argument("--epochs", "-e", required=False, dest="epochs", default=3, type=int,
                          help="Surveillance query file: surveillance/query.txt")

        make.set_defaults(subparser='train')

        if commands is None:
            self.args = parser.parse_args()
        else:
            # Commands is a list of commands to be parsed
            # by Argparse - for testing only:
            self.args = parser.parse_args(commands)

        self.args = vars(self.args)
