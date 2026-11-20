"""
Stop once a logged metric crosses a threshold, e.g. once Paloma val BPB reaches 1.15:

    --stop threshold --stop-metric eval/paloma@val/macro/bits_per_byte --stop-below 1.15

eval/... metrics are only logged on eval steps, so the stop is seen at the first eval
past the threshold and --eval-every-steps sets how many tokens it can overshoot by.
"""

import argparse
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..trainer import TrainerState


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument(
        "--stop-metric",
        required=True,
        metavar="METRIC",
        help="a logged key, e.g. eval/paloma@val/macro/bits_per_byte or train/loss",
    )
    bound = parser.add_mutually_exclusive_group(required=True)
    bound.add_argument(
        "--stop-below", type=float, metavar="VALUE", help="stop once METRIC <= VALUE"
    )
    bound.add_argument(
        "--stop-above", type=float, metavar="VALUE", help="stop once METRIC >= VALUE"
    )


def build(arguments: argparse.Namespace) -> Threshold:
    if arguments.stop_below is not None:
        return Threshold(arguments.stop_metric, "<=", arguments.stop_below)
    return Threshold(arguments.stop_metric, ">=", arguments.stop_above)


COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
}


@dataclass(frozen=True)
class Threshold:
    metric: str
    op: str
    value: float

    def __call__(self, state: TrainerState, metrics: Mapping[str, float]) -> str | None:
        if self.metric not in metrics:
            # The metric's group (eval/, train/, ...) was logged this step, so the
            # metric should have been: a misspelled one would never stop the run
            group = self.metric.split("/", 1)[0] + "/"
            if any(key.startswith(group) for key in metrics):
                raise KeyError(
                    f"--stop-metric {self.metric!r} is not logged; this step logs "
                    f"{', '.join(sorted(metrics))}"
                )
            # A step that logs only other groups, e.g. a log step for an eval metric
            return None

        current = metrics[self.metric]
        if COMPARISONS[self.op](current, self.value):
            return f"{self.metric}={current:.6g} {self.op} {self.value:g}"
        return None
