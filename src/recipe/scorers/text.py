"""
A harness for scorers that read document text, like a fastText classifier or Paloma
decontamination.

A scorer module defines a top-level factory that builds the model once per worker
process and returns the scoring function:

    def build() -> Callable[[list[str]], numpy.ndarray]:
        model = fasttext.load_model(...)
        return lambda texts: ...           # one value per text

and calls score_texts(store, kind, build, info). Each worker scans whole Lance fragments
in order, which is the fast way to read them; the scores are joined onto the shards
through row_ids.npy afterwards.
"""

import multiprocessing
import os
from collections.abc import Callable, Mapping

import lance
import numpy
from lance import LanceDataset

from ...tokenizer import RowIds, ShardStore
from ..metadata import (
    ensure_info,
    is_complete,
    join_onto_shards,
    open_source_dataset,
    source_dataset,
    source_fragment_ids,
)

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


def score_texts(
    store: ShardStore,
    kind: str,
    factory: ScorerFactory,
    info: Mapping[str, object],
    workers: int = os.cpu_count() or 1,
    batch_size: int = 1024,
    mirror: str | None = None,
) -> None:
    """
    Scores every row of the store's source fragments and writes meta/<kind>/. info
    names the scorer and its settings; a different info needs a different kind. The
    fragments are read from mirror if given
    """
    source, version = source_dataset(store)
    ensure_info(store, kind, {**info, "dataset": source, "dataset_version": version})

    if is_complete(store, kind):
        print(f"{store.uri} already has every shard of meta/{kind}", flush=True)
        return

    dataset = open_source_dataset(store, mirror)

    # spawn, not fork: Lance runs background threads, and forking them can deadlock
    context = multiprocessing.get_context("spawn")

    with context.Pool(
        workers,
        initializer=_initialize,
        initargs=(dataset.uri, dataset.version, factory, batch_size),
    ) as pool:
        parts = pool.map(_score_fragment, source_fragment_ids(store))

    addresses = numpy.concatenate([part[0] for part in parts])
    scores = numpy.concatenate([part[1] for part in parts])
    join_onto_shards(store, kind, addresses, scores)
    print(f"wrote meta/{kind} for {store.uri}", flush=True)
