import click
import os

import numpy as np

from achilles.experiment import TestTube
from achilles.model import AchillesModel
from pathlib import Path

@click.command()
@click.option(
    "--training_dir",
    "-t",
    metavar="",
    default=None,
    required=True,
    show_default=True,
    help="Training directory from which to parse models.",
)
@click.option(
    "--evaluation_dir",
    "-e",
    metavar="",
    default=None,
    required=True,
    show_default=True,
    help="Evaluation HD5 file from AchillesModel.",
)
@click.option(
    "--outdir",
    "-o",
    metavar="",
    default=None,
    required=True,
    show_default=True,
    help="Output directory.",
)
def lab(outdir, training_dir, evaluation_dir):

    """ Run predictions in a test tube, not completed. """

    tube = TestTube(outdir=outdir)

    tube.run_predictions(
        training_dir=training_dir,
        evaluation_path=evaluation_dir,
        prefix="lab",
        mode="pairwise",
        batch_size=800,
    )