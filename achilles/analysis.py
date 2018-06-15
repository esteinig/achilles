import numpy
import pandas

from achilles.dataset import Dataset
from achilles.model import Achilles
from achilles.utils import read_signal, transform_signal_to_tensor


def evaluate(data_files: list, models: list, batch_size: int=100, workers: int=2, data_path: str="data",
             meta_data: dict=None, write: str="") -> pandas.DataFrame:

    results = {}

    for model in models:
        results[model] = {}

        achilles = Achilles()
        achilles.load_model(model_file=model)

        for file in data_files:
            ds = Dataset(data_file=file)
            eval_gen = ds.get_signal_generator(data_type=data_path, batch_size=batch_size, shuffle=True)

            seconds, loss, acc = achilles.evaluate(eval_generator=eval_gen, workers=workers)

            results[model][file] = {
                "seconds": seconds,
                "accuracy": round(acc, 4),
                "loss": round(loss, 4)
            }

            print("""
            Model:      {}
            Data:       {}
            Loss:       {}  
            Accuracy:   {} %
            Time:       {} seconds
            """.format(model, file, loss, acc*100, seconds))

    # Multi-index dataframe Model / File
    df = pandas.DataFrame.from_dict({(i, j): results[i][j] for i in results.keys() for j in results[i].keys()},
                                    orient="index").reset_index()

    # Inplace is necessary for underlying objects:
    df.rename(columns={'level_0': 'model', 'level_1': 'dataset'}, inplace=True)

    if write:
        df.to_csv(write)

    return df


def evaluate_predictions():

    pass


def predict(fast5: str, model: Achilles, window_max: int = 10, window_size: int = 400, window_step: int = 400,
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
