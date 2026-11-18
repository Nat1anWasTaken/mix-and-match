from pathlib import Path
from types import SimpleNamespace

import numpy
import pyarrow
import pytest

from src.recipe.metadata import RowJoin, join_onto_shards, read_column
from src.tokenizer import BucketGatherer, FragmentTokenizer, ShardStore

from .conftest import make_store

BOS = 1
EOS = 2


class FakeFragment:
    """
    Rows whose text is their own row address, so the tokens say which row they came
    from. Every fourth row is empty and gets filtered out, which shifts positions
    """

    def __init__(self, fragment_id: int, num_rows: int) -> None:
        self.addresses = (fragment_id << 32) | numpy.arange(num_rows, dtype=numpy.int64)

    def to_batches(self, **_: object) -> list[pyarrow.RecordBatch]:
        texts = [
            "" if offset % 4 == 3 else str(address)
            for offset, address in enumerate(self.addresses)
        ]
        ids = [f"id-{address}" for address in self.addresses]
        batch = pyarrow.record_batch(
            {
                "id": ids,
                "text": texts,
                "_rowaddr": pyarrow.array(self.addresses, pyarrow.uint64()),
            }
        )
        # Two batches, to cover concatenating the row ids
        return [batch.slice(0, 7), batch.slice(7)]


class FakeTokenizer:
    bos_token_id = BOS
    eos_token_id = EOS

    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        # Encode the address in two tokens, low and high 32 bits
        return {"input_ids": [[int(t) & 0xFFFFFFFF, int(t) >> 32] for t in texts]}


def decode_address(document: numpy.ndarray) -> int:
    return int(document[1]) | (int(document[2]) << 32)


def test_row_ids_follow_documents_through_the_shuffle(tmp_path: Path) -> None:
    fragment_ids = [3, 5, 8]
    fragments = {
        fragment_id: FakeFragment(fragment_id, 20) for fragment_id in fragment_ids
    }
    dataset = SimpleNamespace(get_fragment=fragments.get)
    fragment_store = ShardStore(str(tmp_path / "fragments"))
    store = ShardStore(str(tmp_path / "output"))
    num_buckets = 4

    tokenizer = FragmentTokenizer(
        dataset,  # ty: ignore[invalid-argument-type]
        FakeTokenizer(),  # ty: ignore[invalid-argument-type]
        frozenset({"id-12884901889"}),  # fragment 3, row 1
        fragment_store,
        seed=42,
        num_buckets=num_buckets,
    )
    bucket_offsets = [tokenizer.tokenize_fragment(f)[1] for f in fragment_ids]

    gatherer = BucketGatherer(
        fragment_store,
        store,
        [f"{f:06d}" for f in fragment_ids],
        numpy.stack(bucket_offsets),
    )
    for bucket in range(num_buckets):
        gatherer.gather_bucket(bucket)

    seen = []
    for bucket in range(num_buckets):
        name = f"{bucket:06d}"
        tokens, offsets = store.read_tokens(name), store.read_offsets(name)
        row_ids = store.read_row_ids(name)
        assert row_ids.size == offsets.size - 1

        for document, row_id in enumerate(row_ids):
            assert (
                decode_address(tokens[offsets[document] : offsets[document + 1]])
                == row_id
            )

        seen.extend(row_ids.tolist())

    expected = [
        int(address)
        for fragment in fragments.values()
        for offset, address in enumerate(fragment.addresses)
        if offset % 4 != 3 and address != 12884901889
    ]
    assert sorted(seen) == sorted(expected)


def test_write_shard_rejects_misaligned_row_ids(tmp_path: Path) -> None:
    store = ShardStore(str(tmp_path))
    with pytest.raises(ValueError):
        store.write_shard(
            "000000",
            numpy.zeros(4, dtype=numpy.int32),
            numpy.array([0, 2, 4]),
            numpy.array([7]),
        )


def test_join_onto_shards_aligns_by_row_id(tmp_path: Path) -> None:
    store = make_store(tmp_path, [[3, 1, 2], [5, 4]])
    # Values keyed by row id (= document code in make_store), in scrambled order
    addresses = numpy.array([10_001, 0, 10_000, 2, 1, 99], dtype=numpy.int64)
    values = addresses.astype(numpy.float32) * 0.5

    join_onto_shards(store, "half", addresses, values)

    assert read_column(store, "half", "000000").tolist() == [0.0, 0.5, 1.0]
    assert read_column(store, "half", "000001").tolist() == [5000.0, 5000.5]


def test_row_join_rejects_missing_rows() -> None:
    join = RowJoin(numpy.array([5, 1, 3], dtype=numpy.int64))
    assert join.positions(numpy.array([3, 5])).tolist() == [2, 0]
    with pytest.raises(ValueError):
        join.positions(numpy.array([4]))
