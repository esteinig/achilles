import click
import os
import random

from colorama import Fore
from numpy import prod

import numpy as np

from ont_fast5_api.fast5_file import Fast5File, Fast5Info
from achilles.utils import view_as_windows, norm
from achilles.model import Achilles
from pathlib import Path

RE = Fore.RESET
R = Fore.RED
G = Fore.GREEN
Y = Fore.YELLOW
C = Fore.CYAN
M = Fore.MAGENTA

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
    default=100,
    metavar="",
    show_default=True,
    help="Length fo window, must match trained input model",
)
@click.option(
    "--window_slices",
    "-sc",
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
@click.option(
    "--model_summary",
    "-ms",
    metavar="",
    is_flag=True,
    required=False,
    show_default=True,
    help="Show model layer summary on loading model.",
)
@click.option(
    "--normal",
    "-n",
    metavar="",
    is_flag=True,
    required=False,
    show_default=True,
    help="Calculate the average cumulative probability for labels over slices.",
)
@click.option(
    "--cumulative",
    "-cu",
    metavar="",
    is_flag=True,
    required=False,
    show_default=True,
    help="Calculate the average cumulative probability for labels over slices.",
)
def predict(
        watch,
        model,
        window_slices,
        window_size,
        window_step,
        batch_size,
        gpu,
        model_summary,
        normal,
        cumulative
):
    """ Make predictions on a directory of Fast5 """

    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    path = Path(watch)

    fast5 = path.glob("*.fast5")

    achilles = Achilles()

    print(f"{Y}Loading model: {G}{model}{RE}")
    achilles.load_model(model_file=model, summary=model_summary)

    first_pass = True
    for i, f5 in enumerate(fast5):
        # print(f"{Y}Extracting signal slices from: {G}{f5.name}{RE}")

        read_windows = get_reads(
            fast5=f5, window_size=window_size,
            window_step=int(window_step*window_size),
            scale=False, template=True, return_all=False
        )

        random_windows = sample_from_array(
           array=read_windows, sample_size=window_slices,
           random_sample=True, recover=False
        )

        if random_windows is None:
            continue

        signal_tensor = sample_to_input(random_windows)

        if first_pass:
            print(f"\n{R}Null pass {Y}for memory allocation in Keras.{RE}")
            _ = achilles.predict(null_pass=signal_tensor.shape)
            print(f"{G}Null pass complete.{RE}\n")
            first_pass = False

        try:
            read_prediction, time_ms = achilles.predict(
                signal_tensor, batch_size=batch_size
            )
        except IndexError:
            print(f'{R}Prediction failed for read: {f5.name}{RE}')
            continue

        # Slice the predictions by window_max and
        # compute mean over slice of batch:

        scores, score_str = label_product_score(
            read_prediction, normalize=normal, cumsum=cumulative)

        if len(f5.name) > 30:
            name = f5.name[:27] + '...'
        else:
            name = f5.name

        msg = f"{C}{name:<30} " \
              f"{M}{round(time_ms * 1e-06, 4):<10}{Y}" \
              f"{G}{score_str}{RE}"

        print(msg)

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
            reads = np.array([Fast5File(fast5).get_raw_data(attr.read_number, scale=scale)
                              for attr in Fast5Info(fast5).read_info])

        # Windows will only return full-sized windows,
        # incomplete windows at end of read are not included -
        # this is necessary for complete tensors in training and prediction:

        if window_size and window_step:
            reads = np.array(
                [
                    view_as_windows(
                        read, window_shape=window_size, step=window_step
                    )  for read in reads
                ]
            )

        if return_all:
            return reads
        else:
            if len(reads) > 0:
                return reads[0]
            else:
                raise ValueError("No reads in array.")


def label_product_score(read_prediction, cumsum=False, normalize=False):

    scores = list()
    for i in range(
            read_prediction.shape[1]
    ):
        if cumsum:
            score = np.cumsum(
                read_prediction[:, i]
            )[-1] / read_prediction.shape[0]
        elif normalize:
            score = np.prod(
                norm(read_prediction[:, i]), axis=0
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
    (samples, height, width, channels) in Achilles
    """

    if array.ndim != 2:
        raise ValueError(
            f"Array of shape {array.shape} must conform to"
            f" shape: (number_windows, window_size)"
        )

    return np.reshape(array, (array.shape[0], 1, array.shape[1], 1))