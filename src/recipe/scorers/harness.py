"""
The harness scorers run in. A scorer is a module in src/recipe/scorers/ that writes one
per-document metadata column, meta/<kind>/, onto a shard store from prepare.py. The
module holds what is particular to the scorer; this module handles the rest:

    from .harness import Context, TextScorer, run

    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", required=True)

    def fetch(context: Context) -> None:       # optional: extra downloads for process
        hf_hub_download(...)

    def setup(context: Context) -> TextScorer:
        return TextScorer(
            info={"scorer": "fasttext", "model": context.arguments.model},
            build=functools.partial(build, context.arguments.model),
        )

    if __name__ == "__main__":
        run(setup, add_arguments, fetch=fetch, kind="fasttext")

setup returns one of:
  - TextScorer: scores each row's text. build is called once per worker process and
    returns the scoring function; it must pickle, since workers are spawned
  - ColumnScorer: reads columns of the source Lance dataset and turns them into one
    value per row with a vectorized transform, on the main process

run() parses the common arguments (--store, --kind, --workers, --run-name, --no-wandb, --wandb-project,
--stage, --mirror, --remote) plus the scorer's own from add_arguments, then by --stage:

    fetch     mirrors the store's source fragments, then calls fetch
    process   calls setup, scans the source fragments (from --mirror) in parallel,
    / all     joins the values onto the shards through row_ids.npy and writes
              meta/<kind>/, logging progress and a summary of the column to wandb
    upload    copies the store, meta/<kind>/ included, to --remote

Next to the column it writes, like a compiler beside its recipe:

    meta/<kind>/info.json     the scorer's settings; a rerun with other settings is
                              refused, so a new scorer or new settings need a new kind
    meta/<kind>/record.json   where it came from: git commit, dirty, argv, arguments.
                              Recorded only, never compared
    meta/<kind>/scorer.py     a copy of the scorer module's code

join_onto_shards rewrites every shard, so record.json, written once the join is done,
describes every shard of the column. A rerun that finds the column complete changes
nothing, opens no wandb run and keeps the old record.
"""

import argparse
import inspect
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy

from ...loggers import ConsoleLogger, WandbLogger
from ...stages import add_stage_arguments, check_stage, upload_store
from ...tokenizer import ShardStore
from ..compile import compiler_record
from ..metadata import (
    META,
    ensure_info,
    fetch_source_fragments,
    is_complete,
    join_onto_shards,
    open_source_dataset,
    read_columns,
    scan_columns,
    source_dataset,
    source_fragment_ids,
)
from .text import ScorerFactory, scan_texts

# prepare.py's project, so a store's prepare and score runs sit side by side
WANDB_PROJECT = "lab7-data-curation"
RECORD_FILE = "record.json"
SCORER_FILE = "scorer.py"

type Logger = ConsoleLogger | WandbLogger


@dataclass
class TextScorer:
    """
    Scores the text of every row: each worker process calls build() once, then scores
    batches of batch_size texts with what it returns, one value per text
    """

    info: Mapping[str, object]
    build: ScorerFactory
    batch_size: int = 1024


@dataclass
class ColumnScorer:
    """
    Reads columns of the source dataset; transform maps them, as one array per column
    over every scanned row, to one value per row
    """

    info: Mapping[str, object]
    columns: list[str]
    transform: Callable[[dict[str, numpy.ndarray]], numpy.ndarray]


type Scorer = TextScorer | ColumnScorer


@dataclass
class Context:
    arguments: argparse.Namespace
    store: ShardStore
    kind: str
    workers: int
    mirror: str | None
    # Kept until run() returns, e.g. for files the workers memory-map
    scratch: Path


class Progress:
    """
    Called once per scanned fragment, in fragment order, with its row count
    """

    def __init__(self, logger: Logger, fragments: int) -> None:
        self.logger = logger
        self.fragments = fragments
        self.done = 0
        self.rows = 0
        self.start_time = time.monotonic()

    def __call__(self, rows: int) -> None:
        self.done += 1
        self.rows += rows
        self.logger.log(
            {
                "score/fragments": self.done,
                "score/fragment_fraction": self.done / self.fragments,
                "score/rows": self.rows,
                "score/rows_per_second": self.rows
                / (time.monotonic() - self.start_time),
            },
            step=self.done,
        )


def open_logger(
    name: str | None, config: Mapping[str, object], project: str = WANDB_PROJECT
) -> Logger:
    if name is None:
        return ConsoleLogger()

    return WandbLogger(project, name, dict(config))


def column_summary(values: numpy.ndarray) -> dict[str, float]:
    """
    Distribution of a column over every stored document. Non-numeric columns only get
    their document count
    """
    summary: dict[str, float] = {"score/documents": values.size}

    numeric = values.dtype == numpy.bool_ or numpy.issubdtype(
        values.dtype, numpy.number
    )

    if not numeric or values.size == 0:
        return summary

    floats = values.astype(numpy.float64)
    finite = floats[numpy.isfinite(floats)]
    summary["score/nan_fraction"] = 1 - finite.size / floats.size

    if finite.size:
        summary["score/mean"] = float(finite.mean())
        summary["score/std"] = float(finite.std())
        summary["score/min"] = float(finite.min())
        summary["score/max"] = float(finite.max())
        # For Paloma, the fraction of contaminated documents
        summary["score/nonzero_fraction"] = (
            float(numpy.count_nonzero(finite)) / floats.size
        )

    return summary


def finish(logger: Logger, store: ShardStore, kind: str, seconds: float) -> None:
    """
    Summarizes meta/<kind> as written, i.e. over the stored documents the compilers
    will read, not the scanned source rows (which include the rows prepare.py dropped)
    """
    values = numpy.concatenate(list(read_columns(store, kind).values()))
    summary = {**column_summary(values), "score/seconds": seconds}
    print(
        f"meta/{kind}: "
        + " ".join(
            f"{key.removeprefix('score/')}={value:.6g}"
            for key, value in summary.items()
        ),
        flush=True,
    )

    if not isinstance(logger, WandbLogger):
        return

    for key, value in summary.items():
        logger.run.summary[key] = value

    if "score/mean" in summary:
        floats = values.astype(numpy.float64)
        logger.run.summary["score/histogram"] = logger.wandb.Histogram(
            floats[numpy.isfinite(floats)]
        )

    logger.finish()


def score(
    context: Context,
    scorer: Scorer,
    record: Mapping[str, object],
    scorer_source: str,
    run_name: str | None,
) -> None:
    store, kind, workers = context.store, context.kind, context.workers
    source, version = source_dataset(store)
    info = {**scorer.info, "dataset": source, "dataset_version": version}
    ensure_info(store, kind, info)

    if is_complete(store, kind):
        print(f"{store.uri} already has every shard of meta/{kind}", flush=True)
        return

    dataset = open_source_dataset(store, context.mirror)
    fragment_ids = source_fragment_ids(store)
    logger = open_logger(
        run_name,
        {**vars(context.arguments), **info, "kind": kind},
        context.arguments.wandb_project,
    )
    progress = Progress(logger, len(fragment_ids))

    if isinstance(scorer, TextScorer):
        addresses, values = scan_texts(
            dataset, fragment_ids, scorer.build, scorer.batch_size, workers, progress
        )
    else:
        addresses, columns = scan_columns(
            dataset, fragment_ids, scorer.columns, workers, progress
        )
        values = numpy.asarray(scorer.transform(columns))

        if values.shape != addresses.shape:
            raise ValueError(
                f"transform returned shape {values.shape} for {addresses.size} rows"
            )

    join_onto_shards(store, kind, addresses, values, workers)
    # After the join, so the record describes every shard
    store.write_json(f"{META}/{kind}/{RECORD_FILE}", dict(record))
    store.write_text(f"{META}/{kind}/{SCORER_FILE}", scorer_source)
    print(f"wrote meta/{kind} for {store.uri}", flush=True)
    finish(logger, store, kind, time.monotonic() - progress.start_time)


def run(
    setup: Callable[[Context], Scorer],
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
    fetch: Callable[[Context], None] | None = None,
    kind: str | Callable[[argparse.Namespace], str] | None = None,
) -> None:
    """
    The entry point of a scorer module. kind is the default --kind, or a function of
    the arguments for one (e.g. the column a scorer copies); None makes --kind
    required. The module's docstring is the --help text
    """
    module = sys.modules[setup.__module__]
    parser = argparse.ArgumentParser(
        description=module.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--store", required=True, help="shard store from prepare.py")
    parser.add_argument(
        "--kind",
        default=kind if isinstance(kind, str) else None,
        required=kind is None,
        help="metadata kind to write",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--run-name",
        required=True,
        help="names the wandb run, e.g. score-<kind>-<store name>",
    )
    parser.add_argument("--no-wandb", action="store_true", help="log to console only")
    parser.add_argument(
        "--wandb-project",
        default=WANDB_PROJECT,
        help=f"wandb project to log to (default {WANDB_PROJECT}, the lab's)",
    )
    add_stage_arguments(parser)

    if add_arguments is not None:
        add_arguments(parser)

    arguments = parser.parse_args()
    check_stage(arguments, "store")

    store = ShardStore(arguments.store)
    resolved_kind: str

    # --kind is set (given or defaulted) unless kind is a function
    if arguments.kind is not None:
        resolved_kind = arguments.kind
    elif kind is not None and not isinstance(kind, str):
        resolved_kind = kind(arguments)
    else:
        raise ValueError("no --kind")

    if arguments.stage == "upload":
        if not is_complete(store, resolved_kind):
            raise FileNotFoundError(f"{store.uri} lacks shards of meta/{resolved_kind}")

        upload_store(store.uri, arguments.remote)
        return

    with tempfile.TemporaryDirectory() as scratch:
        context = Context(
            arguments,
            store,
            resolved_kind,
            arguments.workers,
            arguments.mirror,
            Path(scratch),
        )

        if arguments.stage == "fetch":
            fetch_source_fragments(store, arguments.mirror, min(arguments.workers, 32))

            if fetch is not None:
                fetch(context)

            return

        # Read before scoring, so an edit during a long run can't change the copy
        source_file = inspect.getsourcefile(setup)

        if source_file is None:
            raise ValueError(f"can't find the source of {setup}")

        path = Path(source_file).resolve()
        scorer_source = path.read_text()
        record = compiler_record(setup, path, arguments)

        score(
            context,
            setup(context),
            record,
            scorer_source,
            None if arguments.no_wandb else arguments.run_name,
        )
