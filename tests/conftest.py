from pathlib import Path

import numpy
import pytest

from src.tokenizer import ShardStore

TOKENIZER = "test-tokenizer"


def document_code(shard_index: int, document: int) -> int:
    """
    Every token of a synthetic document is this code, so a yielded document says
    exactly which stored document it is
    """
    return shard_index * 10_000 + document


def make_store(
    root: Path,
    shard_lengths: list[list[int]],
    tokenizer: str = TOKENIZER,
) -> ShardStore:
    """
    A prepared store with one shard per entry of shard_lengths, each document's tokens
    filled with its document_code and its row id equal to that code too
    """
    store = ShardStore(str(root))
    shards = []

    for shard_index, lengths in enumerate(shard_lengths):
        name = f"{shard_index:06d}"
        codes = [document_code(shard_index, d) for d in range(len(lengths))]
        tokens = numpy.concatenate(
            [numpy.full(n, code, dtype=numpy.int32) for code, n in zip(codes, lengths)]
        )
        offsets = numpy.zeros(len(lengths) + 1, dtype=numpy.int64)
        numpy.cumsum(lengths, out=offsets[1:])
        store.write_shard(
            name, tokens, offsets, numpy.asarray(codes, dtype=numpy.int64)
        )
        shards.append({"name": name, "tokens": int(offsets[-1])})

    store.write_json("metadata.json", {"tokenizer": tokenizer, "layout": 2})
    store.write_json("manifest.json", {"shards": shards, "fragments": []})
    return store


@pytest.fixture
def random_generator() -> numpy.random.Generator:
    return numpy.random.default_rng(42)
