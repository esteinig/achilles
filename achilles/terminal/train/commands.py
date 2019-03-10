import click
from achilles.model import Achilles
from achilles.utils import get_dataset_dim


@click.option(
    "--gpu",
    "-g",
    metavar="",
    default=None,
    required=False,
    show_default=True,
    help="SET CUDA_VISIBLE_DEVICES to train model on specific" " GPU (e.g. 0 or 0,1)",
)
@click.option(
    "--gpus",
    metavar="",
    default=1,
    required=False,
    show_default=True,
    help="Build the model for distributed training across multiple GPUs",
)
@click.option(
    "--threads",
    "-t",
    metavar="",
    default=2,
    required=False,
    show_default=True,
    help="Feed batches into training function using multiple processes",
)
@click.option(
    "--batch_size",
    "-b",
    metavar="",
    default=200,
    required=False,
    show_default=True,
    help="Batch size for training, major determinant of RAM used on GPU",
)
@click.option(
    "--epochs",
    "-e",
    metavar="",
    default=100,
    required=False,
    show_default=True,
    help="Number of epochs to train model for",
)
@click.option(
    "--loss_function",
    "--loss",
    metavar="",
    default="binary_crossentropy",
    required=False,
    show_default=True,
    help="Compile model with loss function for training",
)
@click.option(
    "--optimizer",
    metavar="",
    default="adam",
    required=False,
    show_default=True,
    help="Compile model with optimizer for training",
)
@click.option(
    "--recurrent_dropout",
    metavar="",
    default=0.2,
    required=False,
    show_default=True,
    help="Internal dropout applied to LSTM layers",
)
@click.option(
    "--dropout",
    metavar="",
    default=0.2,
    required=False,
    show_default=True,
    help="Dropout applied to LSTM layers",
)
@click.option(
    "--bidirectional",
    metavar="",
    is_flag=True,
    required=False,
    show_default=True,
    help="Bidirectional LSTM",
)
@click.option(
    "--gru",
    metavar="",
    is_flag=True,
    required=False,
    show_default=True,
    help="Simple GRU cell instead of LSTM",
)
@click.option(
    "--units",
    metavar="",
    default=200,
    required=False,
    show_default=True,
    help="Number of units per LSTMs",
)
@click.option(
    "--channels",
    metavar="",
    default=256,
    required=False,
    show_default=True,
    help="Number channels per Residual Block",
)
@click.option(
    "--lstm",
    metavar="",
    default=1,
    required=False,
    show_default=True,
    help="Number of stacked LSTMs connected to Residual Blocks",
)
@click.option(
    "--residual_block",
    metavar="",
    default=1,
    required=False,
    show_default=True,
    help="Number of stacked ResidualBlocks in initial layers",
)
@click.option(
    "--activation",
    "-a",
    metavar="",
    default="softmax",
    required=False,
    show_default=True,
    help="Activation function applied to final fully connected " "classification layer",
)
@click.option(
    "--verbose",
    "-v",
    metavar="",
    is_flag=True,
    required=False,
    show_default=True,
    help="Show training progress output and model architecture in Keras",
)
@click.option(
    "--load",
    "-l",
    metavar="",
    default="",
    required=False,
    show_default=True,
    help="Trained model weights from Keras, HDF5 to"
    " continue training, or re-train model",
)
@click.option(
    "--outdir",
    "-o",
    metavar="",
    default="training_model",
    required=True,
    show_default=True,
    help="Output directory",
)
@click.option(
    "--run_id",
    "-i",
    metavar="",
    default="model",
    required=True,
    show_default=True,
    help="Training run ID",
)
@click.option(
    "--file",
    "-f",
    metavar="",
    default=None,
    required=True,
    show_default=True,
    help="Input training / validation HDF5 dataset",
)
@click.command()
def train(
    file,
    run_id,
    outdir,
    load,
    verbose,
    activation,
    residual_block,
    lstm,
    channels,
    units,
    gru,
    bidirectional,
    dropout,
    recurrent_dropout,
    optimizer,
    loss_function,
    epochs,
    batch_size,
    threads,
    gpus,
    gpu,
):
    """Train neural network classifiers in Achilles"""

    achilles = Achilles(data_file=file)

    window_size = get_dataset_dim(file)[2]  # 4D tensors

    if load:
        achilles.load_model(load)
    else:
        achilles.build(
            window_size=window_size,
            activation=activation,
            gpus=gpus,
            nb_residual_block=residual_block,
            nb_channels=channels,
            nb_rnn=lstm,
            rnn_units=units,
            gru=gru,
            dropout=dropout,
            rc_dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )

        # Compile model with loss function and optimizer
        achilles.compile(optimizer=optimizer, loss=loss_function)

    achilles.train(
        epochs=epochs,
        batch_size=batch_size,
        workers=threads,
        run_id=run_id,
        outdir=outdir,
        verbose=verbose,
        gpu=gpu,
    )
