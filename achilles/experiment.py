import pandas
import yaml
import pickle

import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from pathlib import Path

from poremongo import PoreMongo
from achilles.dataset import AchillesDataset
from achilles.model import AchillesModel
from achilles.utils import get_dataset_labels

# TODO: Check function for lab configuration JSON


class TestTube:
    def __init__(self, config: str = None, outdir: str = None):

        if config:
            with open(config, "r") as config_file:
                self.config = yaml.load(config_file)

            self.parameters = self.config["global"]
            self.datasets = self.config["datasets"]

        self.name_keys = ("window_size", "max_windows")

        self.outdir = outdir

        # self.setup_experiment_datasets(outdir="ctbmb/experiment_3", poremongo=True)

    def setup_experiment_datasets(self, poremongo=True):

        """ Setup the datasets for the AchillesModel create
        task with global and specific parameters """

        paths = self.setup_experiment_paths()

        print("Here are the paths:", paths)

        ssh = False
        if "ssh" in self.parameters.keys():
            ssh = True

        pongo = None
        if poremongo:
            pongo = PoreMongo(
                config=self.parameters
            )  # Parameters contains URI
            pongo.connect()

        ds = AchillesDataset(poremongo=pongo)

        training_datasets = {}
        evaluation_datasets = {}
        for prefix, config in self.datasets.items():

            # Pull dataset specific parameters
            training_params, evaluation_params = self._get_dataset_params(
                config, "create"
            )

            exclude_datasets = []
            for dataset in config["training"]["data"]:
                data_file = paths["training"] / self._create_dataset_name(
                    prefix, dataset, training_params, "training", False
                )
                if not data_file.exists():
                    ds.write(
                        tags=dataset["tags"],
                        data_file=data_file,
                        **training_params,
                    )
                else:
                    print(f"Warning: data set file for training exists at {data_file}.")

                training_data_file = paths["training"] / self._create_dataset_name(
                    prefix, dataset, training_params, "training", True
                )

                training_datasets[f"{prefix}.{dataset['id']}"] = training_data_file

                exclude_datasets.append(data_file)

            if "evaluation" in config.keys():
                for dataset in config["evaluation"]["data"]:
                    data_file = paths["evaluation"] / self._create_dataset_name(
                        prefix, dataset, evaluation_params, "evaluation"
                    )
                    if not data_file.exists():
                        ds.write(
                            tags=dataset["tags"],
                            data_file=data_file,
                            validation=0,
                            exclude_datasets=exclude_datasets,
                            **evaluation_params,
                        )
                    else:
                        print(
                            f"Warning: data set file for evaluation exists at {data_file}."
                        )
                    evaluation_datasets[f"{prefix}.{dataset['id']}"] = data_file

        if poremongo:
            pongo.disconnect(ssh=ssh)

        self._save_dataset_paths(
            self.outdir, training_datasets, evaluation_datasets, paths
        )

        return training_datasets, evaluation_datasets, paths

    def _save_dataset_paths(
        self, outdir, training_datasets, evaluation_datasets, paths
    ):

        out_path = Path(outdir) / "datasets.npy"

        data = {
            "training": training_datasets,
            "evaluation": evaluation_datasets,
            "paths": paths,
        }

        with open(out_path, "wb") as outfile:
            pickle.dump(data, outfile)

        return out_path

    def _read_dataset_paths(self, file):

        with open(file, "rb") as infile:
            return pickle.load(infile)

    def _get_dataset_params(self, dataset_config, key="create"):

        try:
            dataset_params = dataset_config[key]
        except KeyError:
            dataset_params = dict()

        try:
            training_params = dataset_config["training"][key]
        except KeyError:
            training_params = dict()

        try:
            evaluation_params = dataset_config["evaluation"][key]
        except KeyError:
            evaluation_params = dict()

        train_params = {}
        for param in self.parameters[key], dataset_params, training_params:
            train_params.update(param)

        eval_params = {}
        for param in self.parameters[key], dataset_params, evaluation_params:
            eval_params.update(param)

        return train_params, eval_params

    def run_training(self, training_datasets: dir, paths: dir):

        print("Running training")
        print(self.datasets.keys())

        for prefix, config in self.datasets.items():
            for dataset in config["training"]["data"]:
                print(dataset)
                training_params, _ = self._get_dataset_params(config, "train")
                data_file = training_datasets[f"{prefix}.{dataset['id']}"]
                run_id = data_file.name.replace(".training.h5", "")
                outdir = paths["training"] / run_id

                if not outdir.exists():
                    self._launch_model_training(
                        data_file=data_file,
                        run_id=run_id,
                        outdir=outdir,
                        **training_params,
                    )
                else:
                    print(f"Training directory for data file exists at: {outdir}")

    def setup_experiment_paths(self) -> dict:
        """ Setup the subdirectories of the experiments
        for training, evaluation and plotting """

        out_path = Path(self.outdir)
        train_path = out_path / "training"
        eval_path = out_path / "evaluation"

        for path in (out_path, train_path, eval_path):
            path.mkdir(parents=True, exist_ok=True)

        return {"outpath": out_path, "training": train_path, "evaluation": eval_path}

    def _create_dataset_name(
        self,
        prefix: str,
        dataset: dict,
        params: dict,
        stage="training",
        validation=False,
    ) -> str:
        """ Construct the dataset name for joining into Path """
        name_params = [str(params[key]) for key in self.name_keys]

        if validation:
            return (
                f"{prefix}.{dataset['id']}.{stage}."
                + ".".join(name_params)
                + ".training.h5"
            )
        else:
            return f"{prefix}.{dataset['id']}.{stage}." + ".".join(name_params) + ".h5"

    @staticmethod
    def _launch_model_training(
        data_file,
        window_size: int = 200,
        activation: str = "softmax",
        nb_residual_block=1,
        nb_channels=256,
        nb_rnn=1,
        rnn_units=200,
        gru=False,
        gpus=1,
        dropout=0.2,
        recurrent_dropout=0.2,
        bidirectional=False,
        optimizer="adam",
        loss="binary_crossentropy",
        epochs=100,
        batch_size=1000,
        workers=2,
        run_id="run_id",
        outdir="training_output",
        verbose=True,
        nb_classes=2,
    ):

        # Build model
        achilles = AchillesModel(data_file=data_file)

        achilles.build(
            window_size=window_size,
            activation=activation,
            nb_residual_block=nb_residual_block,
            nb_channels=nb_channels,
            nb_rnn=nb_rnn,
            rnn_units=rnn_units,
            gru=gru,
            gpus=gpus,
            _nb_classes=nb_classes,
            dropout=dropout,
            rc_dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )

        # Compile model with loss function and optimizer
        achilles.compile(optimizer=optimizer, loss=loss)

        achilles.train(
            epochs=epochs,
            batch_size=batch_size,
            workers=workers,
            run_id=run_id,
            outdir=outdir,
            verbose=verbose,
        )

    def run_predictions(
        self,
        training_dir,
        evaluation_path,
        prefix="lab",
        mode="pairwise",
        batch_size=1000,
    ):

        evaluation_path = Path(evaluation_path)

        if mode == "pairwise":
            print("Pairwise mode engaged.")
            data = {}
            for item in evaluation_path.iterdir():
                if item.is_file():
                    print(f"Predicting on {item} with models in {training_dir}")
                    labels = get_dataset_labels(item.absolute())

                    predictions = self.predict_models_on_evaluation(
                        training_dir=training_dir, eval_data=item, batch_size=batch_size
                    )

                    data[item.name.split("evaluation")[0]] = {
                        "predictions": predictions,
                        "labels": np.argmax(labels, 1),
                    }

            np.save(f"{prefix}.eval.{mode}.npy", data)

    @staticmethod
    def predict_models_on_evaluation(training_dir, eval_data, batch_size=1000):

        training_path = Path(training_dir)

        data = {}
        for item in training_path.iterdir():
            if item.is_dir():

                try:
                    model = list(Path(item).glob("*.h5"))[0]
                except IndexError:
                    raise

                print(f"Using {model} to predict on {eval_data}")

                achilles = AchillesModel(eval_data)
                achilles.load_model(model_file=str(model))

                # Model name: numpy array shape (1, 2)
                prefix = model.name.split("training")[0]
                data[prefix] = np.argmax(
                    achilles.predict_generator(
                        data_type="data", batch_size=batch_size
                    ),
                    1,
                )

                print(data[prefix])

        return data


def visualize_binary_predictions(
    npy_file, prefix="exp1", mode="pairwise", labels=("pathogen", "human")
):

    eval_data = np.load(npy_file)
    eval_data = eval_data.item()

    decont_states = [f"false_{labels[0]}", f"false_{labels[1]}"]

    #     # decont_states = [f"incorrectly_remove_{labels[0]}", f"incorrectly_retain_{labels[1]}"]

    dfs_class_0 = []
    dfs_class_1 = []
    dfs_class_2 = []
    dfs_class_3 = []
    for eval_prefix, eval_dict in eval_data.items():
        model_names = []
        labels_0 = []
        labels_1 = []
        labels_2 = []
        labels_3 = []
        for model_prefix, data in eval_dict["predictions"].items():
            cfm = confusion_matrix(eval_dict["labels"], data) / (len(data) / 2)
            print(cfm)
            labels_0.append(cfm[0, 0])
            labels_1.append(cfm[1, 1])
            labels_2.append(cfm[0, 1])  # Upper right: predict pathogen, but is human
            labels_3.append(cfm[1, 0])  # Lower left: predict human, but is pathogen
            model_names.append(model_prefix)

        df_0 = pandas.DataFrame(
            data={
                "models": model_names,
                f"{labels[0]}": labels_0,
                "evaluation": [eval_prefix for _ in range(len(model_names))],
            }
        )
        df_1 = pandas.DataFrame(
            data={
                "models": model_names,
                f"{labels[1]}": labels_1,
                "evaluation": [eval_prefix for _ in range(len(model_names))],
            }
        )
        df_2 = pandas.DataFrame(
            data={
                "models": model_names,
                f"{decont_states[0]}": labels_2,
                "evaluation": [eval_prefix for _ in range(len(model_names))],
            }
        )
        df_3 = pandas.DataFrame(
            data={
                "models": model_names,
                f"{decont_states[1]}": labels_3,
                "evaluation": [eval_prefix for _ in range(len(model_names))],
            }
        )
        dfs_class_0.append(df_0)
        dfs_class_1.append(df_1)
        dfs_class_2.append(df_2)
        dfs_class_3.append(df_3)

    df_pathogen, pathogen_name = (
        pandas.concat(dfs_class_0),
        f"{prefix}.eval.{mode}.{labels[0]}.csv",
    )
    df_host, host_name = (
        pandas.concat(dfs_class_1),
        f"{prefix}.eval.{mode}.{labels[1]}.csv",
    )
    df_upper_right, ur_name = (
        pandas.concat(dfs_class_2),
        f"{prefix}.eval.{mode}.{decont_states[0]}.csv",
    )
    df_lower_left, ll_name = (
        pandas.concat(dfs_class_3),
        f"{prefix}.eval.{mode}.{decont_states[1]}.csv",
    )

    df_pathogen.to_csv(pathogen_name)
    df_host.to_csv(host_name)
    df_upper_right.to_csv(ur_name)
    df_lower_left.to_csv(ll_name)

    return df_pathogen, df_upper_right, df_lower_left, df_host


def summarize_training(outdir, prefix="exp1"):

    log_dfs = []
    training_path = Path(outdir)

    for item in training_path.iterdir():
        if item.is_dir():
            logs = list(Path(item).glob("*.log"))

            try:
                df = pandas.read_csv(logs[0])
            except IndexError:
                continue

            df["id"] = [item.name for _ in range(len(df))]
            log_dfs.append(df)

    log_df = pandas.concat(log_dfs)

    df_group = log_df.groupby("id")

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.suptitle("Training - Pathogen vs. Mixed Human Chromosomes")

    df_group["acc"].plot(x="epochs", legend=True, ax=ax1, title="Training Accuracy")
    df_group["val_acc"].plot(
        x="epochs", legend=False, ax=ax2, title="Validation Accuracy"
    )
    df_group["loss"].plot(x="epochs", legend=False, ax=ax3, title="Training Loss")
    df_group["val_loss"].plot(x="epochs", legend=False, ax=ax4, title="Validation Loss")

    plt.tight_layout()
    plt.savefig(f"{prefix}.training.pdf", figsize=(8, 6), bbox_inches="tight")


def make_confusion_superheatmap(confusion_panels):

    labels = ["pathogen", f"false_pathogen", f"false_human", "human"]
    cmaps = ["GnBu", "GnBu", "GnBu", "GnBu"]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col", sharey="row")

    vmin, vmax = _get_vmin_vmax(confusion_panels)

    axes = [ax1, ax2, ax3, ax4]
    for i in range(len(confusion_panels)):
        ax = make_heatmap(
            df=confusion_panels[i],
            label=labels[i],
            cmap=cmaps[i],
            ax=axes[i],
            vmin=vmin,
            vmax=vmax,
        )

    plt.tight_layout()
    plt.show()


def _get_vmin_vmax(confusion_panels):

    maxs = []
    mins = []
    for cp in confusion_panels:
        maxs.append(cp.iloc[:, 1].max())
        mins.append(cp.iloc[:, 1].min())

    print(maxs, mins)

    return min(mins), max(maxs)


def make_heatmap(
    df=None,
    csv_file=None,
    label="class_0",
    cmap="GnBu",
    postfix_split=".200.",
    ax=None,
    **kwargs,
):

    if csv_file:
        df = pandas.read_csv(csv_file)

    header = df["evaluation"].unique()
    cols = []
    indices = []
    for group in header:
        print(df.columns)
        col = df.loc[df["evaluation"] == group, f"{label}"]
        cols.append(col.tolist())
        indices = df.loc[df["evaluation"] == group, "models"]

    heat = pandas.DataFrame(cols).T
    heat.columns = [h.split(f"{postfix_split}")[0] for h in header]
    heat.index = [h.split(f"{postfix_split}")[0] for h in indices]
    heat.sort_index(axis=0, inplace=True)
    heat.sort_index(axis=1, inplace=True)
    ax = sns.heatmap(heat, annot=True, linewidths=0.5, cmap=cmap, ax=ax, **kwargs)
    ax.set_title(label)

    return ax
