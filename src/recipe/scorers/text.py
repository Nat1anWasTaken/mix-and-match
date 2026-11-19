"""
The parallel text scan behind TextScorer (see harness.py), the counterpart of
metadata.scan_columns for scorers that read document text, like a fastText classifier
or Paloma decontamination.

A scorer's factory builds the model once per worker process and returns the scoring
function:

    def build() -> Callable[[list[str]], numpy.ndarray]:
        model = fasttext.load_model(...)
        return lambda texts: ...           # one value per text

The work is split into chunks: contiguous row ranges of a fragment, scanned in order,
which is the fast way to read them. A store often spans only a few fragments of up to
2^20 rows each, so one task per fragment would leave most workers idle; chunks are
sized so every worker gets about TASKS_PER_WORKER of them. The harness joins the scores
onto the shards through row_ids.npy afterwards. Workers are processes, not threads:
scoring text is Python and numpy work that holds the GIL.
"""

import itertools
import math
import multiprocessing
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor

import lance
import numpy
from lance import LanceDataset

from ...tokenizer import RowIds

type ScoreFunction = Callable[[list[str]], numpy.ndarray]
type ScorerFactory = Callable[[], ScoreFunction]

# Chunks per worker: more balance the load better (the scan waits for the last chunk),
# fewer cost less per-task overhead
TASKS_PER_WORKER = 4

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


def chunk_ranges(rows: int, chunk_rows: int) -> list[tuple[int, int]]:
    """
    (offset, limit) of the chunks of a fragment of rows rows: as few as keep each within
    chunk_rows, all about the same size, so none is a tiny remainder
    """
    if rows == 0:
        return []

    chunks = math.ceil(rows / chunk_rows)
    bounds = [chunk * rows // chunks for chunk in range(chunks + 1)]
    return [(start, stop - start) for start, stop in itertools.pairwise(bounds)]


def plan_chunks(
    fragment_rows: Sequence[tuple[int, int]], workers: int, min_rows: int
) -> list[tuple[int, list[tuple[int, int]]]]:
    """
    Every fragment's chunks as (fragment_id, [(offset, limit), ...]), in fragment
    order. A chunk holds at least min_rows rows (a full scoring batch) unless its
    fragment is smaller
    """
    total = sum(rows for _, rows in fragment_rows)
    chunk_rows = max(min_rows, math.ceil(total / (workers * TASKS_PER_WORKER)), 1)
    return [
        (fragment_id, chunk_ranges(rows, chunk_rows))
        for fragment_id, rows in fragment_rows
    ]


def _score_chunk(chunk: tuple[int, int, int]) -> tuple[RowIds, numpy.ndarray]:
    if _worker is None:
        raise RuntimeError("worker is not initialized")

    fragment_id, offset, limit = chunk
    dataset, score, batch_size = _worker
    fragment = dataset.get_fragment(fragment_id)

    if fragment is None:
        raise ValueError(f"fragment {fragment_id} does not exist")

    address_parts: list[RowIds] = []
    score_parts: list[numpy.ndarray] = []

    for batch in fragment.to_batches(
        columns=["text"],
        batch_size=batch_size,
        offset=offset,
        limit=limit,
        with_row_address=True,
    ):
        texts = [text or "" for text in batch.column("text").to_pylist()]
        scores = numpy.asarray(score(texts))

        if scores.shape != (len(texts),):
            raise ValueError(
                f"scorer returned shape {scores.shape} for {len(texts)} texts"
            )

        address_parts.append(batch.column("_rowaddr").to_numpy().astype(numpy.int64))
        score_parts.append(scores)

    rows = sum(part.size for part in address_parts)

    # The chunks tile the fragment only if each returns exactly its range
    if rows != limit:
        raise ValueError(
            f"fragment {fragment_id} rows {offset}+{limit} returned {rows} rows"
        )

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
    fragment order, as the fragments finish (once all of a fragment's chunks have)
    """
    fragment_rows = []

    for fragment_id in fragment_ids:
        fragment = dataset.get_fragment(fragment_id)

        if fragment is None:
            raise ValueError(f"fragment {fragment_id} does not exist")

        # From the fragment's metadata, no data read. Counts rows after deletions,
        # the rows a scan returns and offset and limit index into
        fragment_rows.append((fragment_id, fragment.count_rows()))

    plan = plan_chunks(fragment_rows, workers, batch_size)
    chunks = [
        (fragment_id, offset, limit)
        for fragment_id, ranges in plan
        for offset, limit in ranges
    ]

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
        # map yields in chunks order, i.e. fragment by fragment and each fragment's
        # chunks by offset, however the workers finish: the same order as scanning
        # whole fragments
        results = pool.map(_score_chunk, chunks)

        for _, ranges in plan:
            rows = 0

            for _ in ranges:
                part = next(results)
                parts.append(part)
                rows += part[0].size

            if on_fragment is not None:
                on_fragment(rows)

    if not parts:
        return numpy.empty(0, dtype=numpy.int64), numpy.empty(0)

    return (
        numpy.concatenate([part[0] for part in parts]),
        numpy.concatenate([part[1] for part in parts]),
    )
