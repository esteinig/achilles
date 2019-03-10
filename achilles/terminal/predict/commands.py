import click
import os
from colorama import Fore

from achilles.utils import read_signal, transform_signal_to_tensor
from achilles.model import Achilles
from pathlib import Path

RE = Fore.RESET
R = Fore.RED
G = Fore.GREEN
Y = Fore.YELLOW


@click.command()
@click.option(
    "--watch",
    "-w",
    default=None,
    help="Watch directory for incoming Fast5 to classify.",
    show_default=True,
    metavar="",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="H5Py file containing trainedAchilles model for predictions",
    show_default=True,
    metavar="",
)
@click.option(
    "--window_size",
    "-size",
    "-s",
    default=400,
    metavar="",
    show_default=True,
    help="Length fo window, must match trained input model",
)
@click.option(
    "--window_slices",
    "-slice",
    default=50,
    metavar="",
    show_default=True,
    help="Maximum number of window slices sampled from"
    " read / diversity of input data.",
)
@click.option(
    "--window_step",
    "-ws",
    default=0.1,
    metavar="",
    show_default=True,
    help="Step of sliding window to sample from signal read.",
)
@click.option(
    "--batch_size",
    "-b",
    metavar="",
    default=200,
    required=False,
    show_default=True,
    help="Batch size for prediction, major determinant of RAM used on GPU",
)
@click.option(
    "--gpu",
    "-g",
    metavar="",
    default=None,
    type=str,
    required=False,
    show_default=True,
    help="Set CUDA_VISIBLE_DEVICES to train model on specific" " GPU (e.g. 0 or 0,1)",
)
def predict(watch, model, window_slices, window_size, window_step, batch_size, gpu):
    """ Make predictions on a directory of Fast5 """

    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    path = Path(watch)

    fast5 = path.glob("*.fast5")

    # init Keras model and empty warmup pass

    achilles = Achilles()

    print(f"{Y}Loading model: {G}{model}{RE}")
    achilles.load_model(model_file=model, summary=True)

    for i, f5 in enumerate(fast5):

        print(f"{Y}Extracting signal slices from: {G}{f5.name}{RE}")
        signal_array, nb_slices = read_signal(
            fast5=str(f5),
            normalize=False,
            scale=False,
            window_size=window_size,
            window_step=int(window_step * window_size),
            window_max=window_slices,
            window_random=True,
            window_recover=True,
            return_signal=False,
        )

        signal_tensor = transform_signal_to_tensor(signal_array)

        if i == 0:
            print(f"{R}Null pass{Y} for memory allocation in Keras.{RE}")
            _ = achilles.predict(null_pass=signal_tensor.shape)
            print(f"{Y}Null pass complete, getting ready for predictions.{RE}")

        print(f"{G}Predictions fo: {Y}{f5.name}{RE}")
        read_prediction = achilles.predict(signal_tensor, batch_size=batch_size)

        print(read_prediction)
