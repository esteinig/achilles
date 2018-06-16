import numpy
import pandas

from achilles.dataset import Dataset
from achilles.model import Achilles
from achilles.utils import read_signal, transform_signal_to_tensor, timeit


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

            loss, acc, seconds = achilles.evaluate(eval_generator=eval_gen, workers=workers)

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


def predict(fast5: str, model: str, window_max: int = 10, window_size: int = 400, window_step: int = 400,
            batch_size: int = 10, window_random: bool = False) -> numpy.array:

    """ Predict from Fast5 using loaded model, either from beginning of signal or randomly sampled """

    achilles = Achilles()
    achilles.load_model(model_file=model)

    for file in fast5:
        # This can be memory consuming and may be too slow to load all windows
        # and then select first or random (signal_max) - need test for None, to get all windows:
        signal_windows, total_windows = read_signal(file, window_size=window_size, window_step=window_step,
                                                    normalize=False, window_random=window_random,
                                                    window_recover=False, window_max=window_max)

        if signal_windows is not None:
            nb_windows = len(signal_windows)
            # Select first
            signal_tensors = transform_signal_to_tensor(signal_windows)
            # Predict with instance of model, batch size is
            # the number of windows extracted for prediction for now:
            # Test if cumulative sum of probabilities is bettert han average?
            prediction = achilles.predict(signal_tensors, batch_size=batch_size).mean(axis=0)
        else:
            nb_windows, prediction = "-", "-"

        # name = os.path.basename(file)
        stdout = "{}\t{}\t{}\t{}\t".format(prediction, nb_windows, total_windows, file)

        print(stdout)
