"""
The parallel text scan behind TextScorer (see harness.py), the counterpart of
metadata.scan_columns for scorers that read document text, like a fastText classifier
or Paloma decontamination.

A scorer's factory builds the model once per worker process and returns the scoring
function:

    def build() -> Callable[[list[str]], numpy.ndarray]:
        model = fasttext.load_model(...)
        return lambda texts: ...           # one value per text

Each worker scans whole Lance fragments in order, which is the fast way to read them;
the harness joins the scores onto the shards through row_ids.npy afterwards. Workers
are processes, not threads: scoring text is Python and numpy work that holds the GIL.
"""

import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

import lance
import numpy
from lance import LanceDataset

from ...tokenizer import RowIds

type ScoreFunction = Callable[[list[str]], numpy.ndarray]
type ScorerFactory = Callable[[], ScoreFunction]

# One per worker process, built by _initialize
_worker: tuple[LanceDataset, ScoreFunction, int] | None = None


def _initialize(
    dataset_uri: str, dataset_version: int, factory: ScorerFactory, batch_size: int
) -> None:
    global _worker

    # Each process already uses one core; library threads on top would oversubscribe
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _worker = (
        lance.dataset(dataset_uri, version=dataset_version),
        factory(),
        batch_size,
    )


def _score_fragment(fragment_id: int) -> tuple[RowIds, numpy.ndarray]:
    if _worker is None:
        raise RuntimeError("worker is not initialized")

    dataset, score, batch_size = _worker
    fragment = dataset.get_fragment(fragment_id)

    if fragment is None:
        raise ValueError(f"fragment {fragment_id} does not exist")

    address_parts: list[RowIds] = []
    score_parts: list[numpy.ndarray] = []

    for batch in fragment.to_batches(
        columns=["text"], batch_size=batch_size, with_row_address=True
    ):
        texts = [text or "" for text in batch.column("text").to_pylist()]
        scores = numpy.asarray(score(texts))

        if scores.shape != (len(texts),):
            raise ValueError(
                f"scorer returned shape {scores.shape} for {len(texts)} texts"
            )

        address_parts.append(batch.column("_rowaddr").to_numpy().astype(numpy.int64))
        score_parts.append(scores)

    if not address_parts:
        return numpy.empty(0, dtype=numpy.int64), numpy.empty(0)

    return numpy.concatenate(address_parts), numpy.concatenate(score_parts)


def scan_texts(
    dataset: LanceDataset,
    fragment_ids: Iterable[int],
    factory: ScorerFactory,
    batch_size: int = 1024,
    workers: int = os.cpu_count() or 1,
    on_fragment: Callable[[int], None] | None = None,
) -> tuple[RowIds, numpy.ndarray]:
    """
    The row addresses and scores of every row of the given fragments, in scan order.
    on_fragment is called on the main process with each fragment's row count, in
    fragment order, as the fragments finish
    """
    # spawn, not fork: Lance runs background threads, and forking them can deadlock
    context = multiprocessing.get_context("spawn")
    parts: list[tuple[RowIds, numpy.ndarray]] = []

    # ProcessPoolExecutor, not Pool: a Pool respawns workers whose initializer
    # failed forever and map never returns, while this raises BrokenProcessPool
    with ProcessPoolExecutor(
        workers,
        mp_context=context,
        initializer=_initialize,
        initargs=(dataset.uri, dataset.version, factory, batch_size),
    ) as pool:
        # map yields in fragment_ids order however the workers finish
        for part in pool.map(_score_fragment, fragment_ids):
            parts.append(part)

            if on_fragment is not None:
                on_fragment(part[0].size)

    if not parts:
        return numpy.empty(0, dtype=numpy.int64), numpy.empty(0)

    return (
        numpy.concatenate([part[0] for part in parts]),
        numpy.concatenate([part[1] for part in parts]),
    )
