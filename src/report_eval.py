"""
Write a `flock eval run` result into a training run's wandb summary, for the OJ.

    uv run python -m src.report_eval checkpoints/<run>/paloma-step<step>/results.json \\
        --run-name <run> --eval paloma --key paloma_ppl --step <step> --model <ref>

The score is the one train.py logs for the same --eval (src.evaluation.spec_metrics):
an unscored suite of scored suites, like Paloma, gives the unweighted mean of its member
suites (Paloma's 9-source macro bits per byte), a scored suite its own score, a task its
primary metric. The run is resumed by its id (train.py uses the run name) and finished
again, so it stays a finished run. A result scored with --limit, without every task of
the eval, or in a dtype other than float32 is refused: the OJ reads it as the final score
"""

import argparse
import json
import math
from pathlib import Path

import wandb
from flock.eval.tasks import Suite

from src.evaluation import parse_eval_spec, spec_metrics, suite_metric
from train import WANDB_PROJECT


def eval_score(results: dict, eval_name: str) -> float:
    """
    The eval's score from a results.json, aggregated as train.py aggregates it.
    Raises ValueError if the result can't be the final score
    """
    spec = parse_eval_spec(eval_name)
    dtype = results["provenance"]["dtype"]
    if dtype != "float32":
        raise ValueError(f"scored in {dtype}; the final score is float32")

    by_task = {task["task"]: task for task in results["tasks"]}
    missing = [task.name for task in spec.tasks if task.name not in by_task]
    if missing:
        raise ValueError(
            f"{len(missing)} tasks of {eval_name} not scored: {missing[:5]}"
        )
    limited = [
        task.name for task in spec.tasks if by_task[task.name]["limit"] is not None
    ]
    if limited:
        raise ValueError(f"{len(limited)} tasks scored with --limit: {limited[:5]}")

    metrics = spec_metrics(
        spec, {name: task["metrics"] for name, task in by_task.items()}
    )
    prefix = f"eval/{spec.label}"
    item = spec.item
    if not isinstance(item, Suite):
        key = f"{prefix}/{item.primary_metric}"
    elif item.scored:
        key = f"{prefix}/{item.name}/{suite_metric(item)}"
    else:
        # The macro over member suites; spec_metrics logs it only when they share a metric
        first = item.members[0]
        key = (
            f"{prefix}/macro/{suite_metric(first) if isinstance(first, Suite) else ''}"
        )

    score = metrics.get(key)
    if score is None or not math.isfinite(score):
        raise ValueError(f"{eval_name}: no finite score {key} in the result")

    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, help="results.json from flock eval run")
    parser.add_argument("--run-name", required=True, help="the training run's name/id")
    parser.add_argument("--eval", required=True, help="the suite or task, as --eval")
    parser.add_argument("--key", required=True, help="summary key, e.g. paloma_ppl")
    parser.add_argument("--step", type=int, required=True, help="the checkpoint's step")
    parser.add_argument("--model", required=True, help="the scored flock model ref")
    parser.add_argument("--project", default=WANDB_PROJECT)
    arguments = parser.parse_args()

    score = eval_score(json.loads(arguments.results.read_text()), arguments.eval)
    summary = {
        arguments.key: score,
        f"{arguments.key}_step": arguments.step,
        f"{arguments.key}_model": arguments.model,
    }
    print(f"{arguments.run_name}: {summary}")

    # resume="must": the run has to exist already, so a typo never starts a new run
    run = wandb.init(project=arguments.project, id=arguments.run_name, resume="must")
    run.summary.update(summary)
    run.finish()


if __name__ == "__main__":
    main()
