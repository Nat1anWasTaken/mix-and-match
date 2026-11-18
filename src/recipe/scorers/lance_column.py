"""
Copy a column of the source Lance dataset onto a shard store as meta/<kind>/.

    uv run python -m src.recipe.scorers.lance_column \\
        --store s3://<bucket>/sources/fineweb-edu-llama3 --column int_score

FineWeb-Edu ships its classifier's `score` (float) and `int_score` (0-5), so this is a
quality score for free. The source rows are scanned fragment by fragment and joined
onto the shards through row_ids.npy.
"""

import argparse
import os

import numpy

from ...tokenizer import ShardStore
from ..metadata import (
    ensure_info,
    is_complete,
    join_onto_shards,
    open_source_dataset,
    scan_columns,
    source_fragment_ids,
)


def copy_column(
    store: ShardStore, column: str, kind: str, dtype: str | None, workers: int
) -> None:
    dataset = open_source_dataset(store)
    info = {
        "scorer": "lance_column",
        "column": column,
        "dtype": dtype,
        "dataset": dataset.uri,
        "dataset_version": dataset.version,
    }
    ensure_info(store, kind, info)

    if is_complete(store, kind):
        print(f"{store.uri} already has every shard of meta/{kind}", flush=True)
        return

    addresses, values = scan_columns(
        dataset, source_fragment_ids(store), [column], workers
    )
    column_values = values[column]

    if dtype is not None:
        column_values = column_values.astype(numpy.dtype(dtype))

    join_onto_shards(store, kind, addresses, column_values, workers)
    print(f"wrote meta/{kind} for {store.uri}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", required=True, help="shard store from prepare.py")
    parser.add_argument("--column", required=True, help="column of the Lance dataset")
    parser.add_argument("--kind", help="metadata kind to write; default: --column")
    parser.add_argument("--dtype", help="cast the values, e.g. float32")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    arguments = parser.parse_args()

    copy_column(
        ShardStore(arguments.store),
        arguments.column,
        arguments.kind or arguments.column,
        arguments.dtype,
        arguments.workers,
    )


if __name__ == "__main__":
    main()
