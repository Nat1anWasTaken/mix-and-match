"""
Per-document metadata columns of a shard store, e.g. quality scores.

    <store>/meta/<kind>/info.json       who made the column and with which settings
    <store>/meta/<kind>/<shard>.npy     one value per document, in the shard's stored order

The stored order is the order of data/<shard>/offsets.npy and row_ids.npy, so a
column lines up with a shard by position. Scorers write columns once per (store,
kind); compilers read them, which is cheap: a float32 column of a 30M-document store is
120 MB in total.

Scores that need the source dataset (its columns or its text) come from a sequential
scan of the source's Lance fragments, joined onto the shards through row_ids.npy. A
shard's documents come from every fragment, so random takes per shard would be slow.
"""

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor

import lance
import numpy
from lance import LanceDataset

from ..tokenizer import RowIds, ShardStore
from .recipe import read_shard_names

META = "meta"


def column_path(kind: str, shard: str) -> str:
    return f"{META}/{kind}/{shard}.npy"


def ensure_info(store: ShardStore, kind: str, info: Mapping[str, object]) -> None:
    """
    Columns are skipped by name, so refuse to mix columns made with different settings.
    A new scorer or new settings need a new kind
    """
    path = f"{META}/{kind}/info.json"
    existing = store.read_json(path)

    if existing is None:
        store.write_json(path, dict(info))
    elif existing != dict(info):
        raise ValueError(
            f"{store.uri} {path} holds {existing}, not {dict(info)}; use a new kind"
        )


def has_column(store: ShardStore, kind: str, shard: str) -> bool:
    return store.has_array(column_path(kind, shard))


def write_column(
    store: ShardStore, kind: str, shard: str, values: numpy.ndarray
) -> None:
    documents = store.read_offsets(shard).size - 1

    if values.ndim != 1 or values.size != documents:
        raise ValueError(
            f"column {kind} for shard {shard} has shape {values.shape}, "
            f"but the shard holds {documents} documents"
        )

    store.write_array(column_path(kind, shard), values)


def read_column(store: ShardStore, kind: str, shard: str) -> numpy.ndarray:
    return store.read_array(column_path(kind, shard))


def read_columns(
    store: ShardStore, kind: str, workers: int = 16
) -> dict[str, numpy.ndarray]:
    names = read_shard_names(store)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        columns = pool.map(lambda name: read_column(store, kind, name), names)
        return dict(zip(names, columns, strict=True))


def open_source_dataset(store: ShardStore) -> LanceDataset:
    """
    The Lance dataset the store was prepared from, at the version it was prepared from.
    Row addresses are only valid within that version
    """
    metadata = store.read_json("metadata.json")

    if metadata is None:
        raise FileNotFoundError(f"no metadata.json in {store.uri}")

    if metadata.get("layout") != 2:
        raise ValueError(f"{store.uri} has no row_ids; rerun prepare.py")

    version = metadata["dataset_version"]

    if not isinstance(version, int):
        raise TypeError(f"{store.uri} metadata.json has no integer dataset_version")

    return lance.dataset(str(metadata["dataset"]), version=version)


def source_fragment_ids(store: ShardStore) -> list[int]:
    manifest = store.read_json("manifest.json")

    if manifest is None:
        raise FileNotFoundError(f"no manifest.json in {store.uri}")

    fragments = manifest["fragments"]

    if not isinstance(fragments, list):
        raise TypeError(f"manifest.json in {store.uri} lists no fragments")

    return [int(fragment["fragment_id"]) for fragment in fragments]


def scan_columns(
    dataset: LanceDataset,
    fragment_ids: Iterable[int],
    columns: list[str],
    workers: int = 16,
) -> tuple[RowIds, dict[str, numpy.ndarray]]:
    """
    The row addresses and values of every row of the given fragments, in scan order.
    Lance reads release the GIL, so threads overlap the fragment reads
    """

    def scan(fragment_id: int) -> tuple[RowIds, dict[str, numpy.ndarray]]:
        fragment = dataset.get_fragment(fragment_id)

        if fragment is None:
            raise ValueError(f"fragment {fragment_id} does not exist")

        table = fragment.to_table(columns=columns, with_row_address=True)
        addresses = table.column("_rowaddr").to_numpy().astype(numpy.int64)
        values = {
            column: table.column(column).to_numpy(zero_copy_only=False)
            for column in columns
        }
        return addresses, values

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(scan, fragment_ids))

    addresses = numpy.concatenate([part[0] for part in parts])
    values = {
        column: numpy.concatenate([part[1][column] for part in parts])
        for column in columns
    }
    return addresses, values


class RowJoin:
    """
    Maps row addresses to values: sorts the addresses once, then each shard's row_ids
    are looked up with one searchsorted
    """

    def __init__(self, addresses: RowIds) -> None:
        self.order = numpy.argsort(addresses, kind="stable")
        self.sorted_addresses = addresses[self.order]

        if numpy.any(numpy.diff(self.sorted_addresses) == 0):
            raise ValueError("row addresses are not unique")

    def positions(self, row_ids: RowIds) -> numpy.ndarray:
        """
        Index into the original addresses array for every row id
        """
        found = numpy.searchsorted(self.sorted_addresses, row_ids)
        found = numpy.minimum(found, self.sorted_addresses.size - 1)

        if not numpy.array_equal(self.sorted_addresses[found], row_ids):
            raise ValueError("some row ids were not in the scanned fragments")

        return self.order[found]


def join_onto_shards(
    store: ShardStore,
    kind: str,
    addresses: RowIds,
    values: numpy.ndarray,
    workers: int = 16,
) -> None:
    """
    Writes meta/<kind>/<shard>.npy for every shard, from values keyed by row address
    """
    join = RowJoin(addresses)

    def write(name: str) -> None:
        positions = join.positions(store.read_row_ids(name))
        write_column(store, kind, name, values[positions])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(write, read_shard_names(store)))


def is_complete(store: ShardStore, kind: str) -> bool:
    return all(has_column(store, kind, name) for name in read_shard_names(store))
