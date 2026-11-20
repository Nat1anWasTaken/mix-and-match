"""
Early stops: rules that end a training run before total_tokens.

A stop is a Python file in src/stops/, chosen with `train.py --stop <file name>`. A run
has at most one: its flags configure it, and a rule that combines others is a stop of
its own. The file defines

    add_arguments(parser) -> None
        its own command line flags, named --stop-... so they don't clash with train.py's
    build(arguments) -> StopCondition
        the condition, from the parsed flags. Raise ValueError for flag values argparse
        could not check itself; train.py reports it as a usage error

A StopCondition gets the trainer state and every metric logged this step (train/... and
data/... on a log step, eval/... on an eval step, both when the step is both) and
returns why to stop, or None to go on. The trainer calls it on rank 0 only, after
every log and eval step (metrics is empty on an eval step of a run without an
evaluator), and broadcasts the decision so every rank stops at the same step.
"""

import argparse
import importlib
import pkgutil
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import TYPE_CHECKING

# trainer.py imports this module, so TrainerState is only imported for type checkers
if TYPE_CHECKING:
    from ..trainer import TrainerState

type StopCondition = Callable[[TrainerState, Mapping[str, float]], str | None]


def available_stops() -> list[str]:
    """The stop names --stop takes: the module files in this package"""
    return sorted(module.name for module in pkgutil.iter_modules(__path__))


def load_stop(name: str) -> ModuleType:
    """The stop's module, with its add_arguments and build"""
    return importlib.import_module(f"{__name__}.{name}")


def add_stop_arguments(parser: argparse.ArgumentParser, name: str) -> None:
    """Add the stop's flags to parser, in a help section of its own"""
    load_stop(name).add_arguments(parser.add_argument_group(f"--stop {name}"))


def build_stop(arguments: argparse.Namespace) -> StopCondition | None:
    """The condition --stop chose, None without --stop"""
    if arguments.stop is None:
        return None
    return load_stop(arguments.stop).build(arguments)
