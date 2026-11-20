"""
Trainer hooks: code that watches the metrics the trainer logs and can end the run early.

A hook subclasses TrainerHook and overrides the events it cares about. Each event gets
the trainer state and the metrics just logged, and returns a reason string to stop
training after this step, or None to go on. The trainer calls hooks on rank 0 only,
where the metrics are complete, and broadcasts the decision so every rank stops at the
same step (a rank that kept going would hang the others in the next collective).
"""

import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

# trainer.py imports this module, so TrainerState is only imported for type checkers
if TYPE_CHECKING:
    from .trainer import TrainerState


class TrainerHook:
    def on_log(self, state: TrainerState, metrics: dict[str, float]) -> str | None:
        """After every log step, with its train/... and data/... metrics"""
        return None

    def on_eval(self, state: TrainerState, metrics: dict[str, float]) -> str | None:
        """After every eval, with its eval/... metrics"""
        return None


COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
}

# Longest operators first, so "<=" is not read as "<" followed by "=1.15"
CONDITION = re.compile(r"^(?P<metric>[^<>=\s]+)\s*(?P<op><=|>=|<|>)\s*(?P<value>\S+)$")


@dataclass(frozen=True)
class ThresholdStop(TrainerHook):
    """
    Stop once a logged metric crosses a threshold, e.g. Paloma val BPB <= 1.15.
    eval/... metrics are checked after each eval, every other metric after each log step
    """

    metric: str
    op: str
    value: float

    def __post_init__(self) -> None:
        if self.op not in COMPARISONS:
            raise ValueError(
                f"unknown comparison {self.op!r}, expected one of {list(COMPARISONS)}"
            )

    @property
    def is_eval_metric(self) -> bool:
        return self.metric.startswith("eval/")

    def on_log(self, state: TrainerState, metrics: dict[str, float]) -> str | None:
        return None if self.is_eval_metric else self._check(metrics)

    def on_eval(self, state: TrainerState, metrics: dict[str, float]) -> str | None:
        return self._check(metrics) if self.is_eval_metric else None

    def _check(self, metrics: dict[str, float]) -> str | None:
        # A misspelled metric would otherwise never stop the run, silently
        if self.metric not in metrics:
            raise KeyError(
                f"--stop-when {self}: {self.metric!r} is not logged; this event "
                f"logs {', '.join(sorted(metrics))}"
            )
        current = metrics[self.metric]
        if COMPARISONS[self.op](current, self.value):
            return f"{self.metric}={current:.6g} {self.op} {self.value:g}"
        return None

    def __str__(self) -> str:
        return f"{self.metric}{self.op}{self.value:g}"


def parse_stop_condition(text: str) -> ThresholdStop:
    """METRIC<=VALUE (or <, >=, >), METRIC a logged key like eval/paloma@val/macro/bits_per_byte"""
    match = CONDITION.match(text.strip())
    if match is None:
        raise ValueError(
            f"--stop-when {text}: expected METRIC<=VALUE, e.g. "
            "eval/paloma@val/macro/bits_per_byte<=1.15"
        )
    try:
        value = float(match["value"])
    except ValueError:
        raise ValueError(
            f"--stop-when {text}: {match['value']!r} is not a number"
        ) from None
    return ThresholdStop(match["metric"], match["op"], value)
