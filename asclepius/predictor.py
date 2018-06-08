import numpy

from asclepius.model import Asclepius
from asclepius.utils import read_signal, transform_signal_to_tensor


def predict(fast5: str, model: Asclepius, window_max: int = 10, window_size: int = 400, window_step: int = 400,
            batch_size: int = 10, random: bool = False) -> numpy.array:

    """ Predict from Fast5 using loaded model, either from begiining of signal or randomly sampled """

    # This can be memory consuming and may be too slow to load all windows
    # and then select first or random (signal_max)
    signal_windows = read_signal(fast5, window_size=window_size, window_step=window_step)

    if random:
        numpy.random.shuffle(signal_windows)

    # Select first
    signal_windows = signal_windows[:window_max]

    signal_tensors = transform_signal_to_tensor(signal_windows)

    # Predict with instance of model, batch size is
    # the number of windows extracted for prediction for now:

    return model.predict(signal_tensors, batch_size=batch_size).mean(axis=0)
