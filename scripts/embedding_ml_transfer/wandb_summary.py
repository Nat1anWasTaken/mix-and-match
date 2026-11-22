"""
One W&B run per invocation of a P0-1 script that has no step-wise progress to log
(vban_count, fake_french, audit): its arguments as config, its results as the run's
summary. Same project argument and run naming as prepare.py and the scorers.

On Forerunner1 compute nodes wandb runs offline (scripts/f1-cpu.sbatch); sync the runs
from a login node with scripts/embedding_ml_transfer/sync_wandb.sh.
"""

import argparse
from collections.abc import Mapping

from src.loggers import WandbLogger

DEFAULT_PROJECT = "embedding-ml-transfer"


def add_arguments(parser: argparse.ArgumentParser, run_name_required: bool) -> None:
    parser.add_argument(
        "--run-name", required=run_name_required, help="names the wandb run"
    )
    parser.add_argument("--no-wandb", action="store_true", help="log to console only")
    parser.add_argument("--wandb-project", default=DEFAULT_PROJECT)


def log_summary(arguments: argparse.Namespace, summary: Mapping[str, object]) -> None:
    if arguments.no_wandb or arguments.run_name is None:
        return

    logger = WandbLogger(arguments.wandb_project, arguments.run_name, vars(arguments))

    for key, value in summary.items():
        logger.run.summary[key] = value

    logger.finish()
