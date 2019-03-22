import click
import os
import random

from colorama import Fore
from numpy import prod

import numpy as np

from ont_fast5_api.fast5_file import Fast5File, Fast5Info
from achilles.utils import view_as_windows
from achilles.model import AchillesModel
from achilles.achilles import Achilles
from achilles.realtime import watch_path
from pathlib import Path

RE = Fore.RESET
R = Fore.RED
G = Fore.GREEN
Y = Fore.YELLOW
C = Fore.CYAN
M = Fore.MAGENTA


@click.command()
@click.option(
    "--directory",
    "-d",
    default=None,
    help="Directory with Fast5 files to classify.",
    show_default=True,
    metavar="",
)
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
    help="HD5 model file or <collection>/<model> from local model cache.",
    show_default=True,
    metavar="",
)
@click.option(
    "--window_size",
    "--size",
    "-s",
    default=300,
    metavar="",
    show_default=True,
    help="Length fo window, must match trained input model",
)
@click.option(
    "--window_slices",
    '--slices',
    "-c",
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
    help="Set CUDA_VISIBLE_DEVICES to train model"
         " on specific GPUs (e.g. 0 or 0,1)",
)
@click.option(
    "--model_summary",
    "-ms",
    metavar="",
    is_flag=True,
    required=False,
    show_default=True,
    help="Show model layer summary on loading model.",
)
def predict(
        directory,
        watch,
        model,
        window_slices,
        window_size,
        window_step,
        batch_size,
        gpu,
        model_summary,
):
    """ Make model predictions on a directory of Fast5 """

    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # TODO: live watching of Fast5

    base = Achilles()
    achilles = AchillesModel()

    # Find model - if in local collection, the
    # collections should be specified like this: collection/model
    model_path = Path(model)
    if not model_path.exists():
        # Try and see if the name matches a model in
        # the collections, and if so, return the file path:
        model_path = str(base.get_model(model))

    print(f"{Y}Preparing predictions with Achilles:{RE}\n")
    print(f"  {Y}- Loading model: {G}{model}{RE}")
    achilles.load_model(model_file=model_path, summary=model_summary)

    print(f"  {Y}- {G}Null pass {Y}for memory allocation in Keras.{RE}\n")
    _ = achilles.predict(null_pass=(window_slices, 1, window_size, 1))

    if directory:
        path = Path(directory)
        fast5 = path.glob("*.fast5")

        for i, f5 in enumerate(fast5):
            predict_read(
                f5,
                achilles,
                window_size,
                window_step,
                window_slices,
                batch_size
            )
    elif watch:
        watch_path(
            watch,
            callback=predict_read,
            recursive=False,
            achilles=achilles,
            window_size=window_size,
            window_step=window_step,
            window_slices=window_slices,
            batch_size=batch_size
        )


def predict_read(
        f5,
        achilles,
        window_size,
        window_step,
        window_slices,
        batch_size
):
    read_windows = get_reads(
        fast5=f5, window_size=window_size,
        window_step=int(window_step * window_size),
        scale=False, template=True, return_all=False
    )

    random_windows = sample_from_array(
        array=read_windows, sample_size=window_slices,
        random_sample=True, recover=False
    )

    if random_windows is None:
        return

    signal_tensor = sample_to_input(random_windows)

    try:
        read_prediction, time_ms = achilles.predict(
            signal_tensor, batch_size=batch_size
        )
    except IndexError:
        print(f'{R}Prediction failed for read: {f5.name}{RE}')
        return

    scores, score_str = label_product_score(
        read_prediction, product=False
    )

    print(
        f"{f5.name}\t{round(time_ms * 1e-06, 6)}\t{score_str}"
    )

def get_reads(
        fast5,
        window_size: int = None,
        window_step: int = None,
        scale: bool = False,
        template: bool = True,
        return_all=True
) -> np.array:

        """ Scaled pA values (float32) or raw DAQ values (int16),
        return first read (1D) or if all_reads = True return
        array of all reads """

        if template:
            reads = np.array([Fast5File(fast5).get_raw_data(scale=scale)])
        else:
            reads = np.array(
                [
                    Fast5File(fast5).get_raw_data(attr.read_number, scale=scale)
                    for attr in Fast5Info(fast5).read_info
                ]
            )

        # Windows will only return full-sized windows,
        # incomplete windows at end of read are not included -
        # this is necessary for complete tensors in training and prediction:

        if window_size and window_step:
            reads = np.array(
                [
                    view_as_windows(
                        read, window_shape=window_size, step=window_step
                    ) for read in reads
                ]
            )

        if return_all:
            return reads
        else:
            if len(reads) > 0:
                return reads[0]
            else:
                raise ValueError("No reads in array.")


def label_product_score(read_prediction, product=False):

    scores = list()
    for i in range(
            read_prediction.shape[1]
    ):
        if product:
            score = np.prod(
                read_prediction[:, i], axis=0
            )
        else:
            score = read_prediction[:, i].mean()

        scores.append(
            float(score)
        )

    score_str = get_score_str(
        np.array(scores)
    )
    return scores, score_str


def get_score_str(scores: np.array):

    max_index = scores.argmax()
    string = f''
    for i, score in enumerate(scores.tolist()):
        if i == max_index:
            string += f'{G}'
        else:
            string += f'{RE}'

        string += f'{round(float(score), 5):<10}{RE}'

    return string


def sample_from_array(
        array: np.array,
        sample_size: int,
        random_sample: bool = True,
        recover: bool = True,
) -> np.array:

        """Return a contiguous sample from an array of signal windows

        :param array
        :param sample_size
        :param random_sample
        :param recover

        """

        num_windows = array.shape[0]

        if num_windows < sample_size and recover:
            return array

        if num_windows < sample_size and not recover:
            return None

        if num_windows == sample_size:
            return array
        else:
            if random_sample:
                idx_max = num_windows - sample_size
                rand_idx = random.randint(0, idx_max)

                return array[rand_idx: rand_idx + sample_size]
            else:
                return array[:sample_size]


def sample_to_input(array: np.array) -> np.array:

    """ Transform input array of (number_windows, window_size) to:
    (number_windows, 1, window_size, 1)
    for input into convolutional layers:
    (samples, height, width, channels) in AchillesModel
    """

    if array.ndim != 2:
        raise ValueError(
            f"Array of shape {array.shape} must conform to"
            f" shape: (number_windows, window_size)"
        )

    return np.reshape(array, (array.shape[0], 1, array.shape[1], 1))
