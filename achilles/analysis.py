import os
import numpy
import pandas

from achilles.dataset import Dataset
from achilles.model import Achilles
from achilles.utils import read_signal, transform_signal_to_tensor, get_recursive_files, plot_confusion_matrix

from sklearn.metrics import confusion_matrix


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


def evaluate_predictions(dirs, model, prefix="peval", **kwargs):

    """ Wrapper for evaluating predictions with analysis.predict() on a set of directories containing
    Fast5 files from the labelled classes (species) used for model training. Fast5 files should be independent of
    the ones used for extraction of signal windows for model training. This function returns a confusion matrix
    for assessing prediction errors. """

    fast5 = []
    labels = []
    for label, directory in enumerate(dirs):
        # Recursively grab a list of Fast5 files:
        files = get_recursive_files(directory, extension=".fast5")

        fast5 += files
        labels += [label for _ in files]

    predictions = predict(fast5=fast5, model=model, **kwargs)

    df = pandas.DataFrame({
        "file_name": [os.path.basename(file) for file in fast5],
        "label": labels,
        "prediction": predictions
    })

    nan = int(df["prediction"].isnull().sum())
    df = df.dropna()
    print("Removed {} failed prediction from final results.".format(nan))

    df.to_csv(prefix+".csv")

    cm = confusion_matrix(df["label"], df["prediction"])

    plot_confusion_matrix(cm, classes=["Bp", "Human"], save=prefix+".pdf", normalize=True)

    return cm


def predict(fast5: str, model: str, window_max: int = 10, window_size: int = 400, window_step: int = 400,
            batch_size: int = 10, window_random: bool = False, scale: bool = True, stdout: bool = True) -> numpy.array:

    """ Predict from Fast5 using loaded model, either from beginning of signal or randomly sampled """

    achilles = Achilles()
    achilles.load_model(model_file=model)

    predictions = []
    for file in fast5:
        # This can be memory consuming and may be too slow to load all windows
        # and then select first or random (signal_max) - need test for None, to get all windows:
        signal_windows, total_windows = read_signal(file, window_size=window_size, window_step=window_step,
                                                    normalize=False, window_random=window_random,
                                                    window_recover=False, window_max=window_max,
                                                    scale=scale)

        if signal_windows is not None:
            # Transform to tensors:
            nb_windows, signal_tensors = len(signal_windows), transform_signal_to_tensor(signal_windows)
            # Predict with instance of model, batch size is
            # the number of windows extracted for prediction for now:
            # Test if cumulative sum of probabilities is better than average?
            prediction_windows, microseconds = achilles.predict(signal_tensors, batch_size=batch_size)

            if len(prediction_windows) > 1:
                prediction = prediction_windows.mean(axis=0)
            else:
                prediction = prediction_windows[0]

            predicted_label = numpy.argmax(prediction)
        else:
            # If no signal windows could be extracted:
            nb_windows, prediction, microseconds = 0, numpy.empty(), 0
            predicted_label = None

        predictions.append(predicted_label)

        # name = os.path.basename(file)
        if stdout:
            print("{}\t{}\t{}\t{}\t{}\t".format(prediction, predicted_label, nb_windows,
                                                total_windows, microseconds, file))

    return predictions
