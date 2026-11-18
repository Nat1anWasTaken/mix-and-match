"""
Copy a column of the source Lance dataset onto a shard store as meta/<kind>/.

    uv run python -m src.recipe.scorers.lance_column \\
        --store s3://<bucket>/sources/fineweb-edu-llama3 --column int_score

FineWeb-Edu ships its classifier's `score` (float) and `int_score` (0-5), so this is a
quality score for free. The source rows are scanned fragment by fragment and joined
onto the shards through row_ids.npy.

Takes --stage like prepare.py (see src/stages.py): fetch mirrors the store's source
fragments, if prepare.py --stage fetch has not, process scores from --mirror, upload
copies the store, new column included, to --remote.
"""

import argparse
import os

import numpy

from ...stages import add_stage_arguments, check_stage, upload_store
from ...tokenizer import ShardStore
from ..metadata import (
    ensure_info,
    fetch_source_fragments,
    is_complete,
    join_onto_shards,
    open_source_dataset,
    scan_columns,
    source_dataset,
    source_fragment_ids,
)


def copy_column(
    store: ShardStore,
    column: str,
    kind: str,
    dtype: str | None,
    workers: int,
    mirror: str | None = None,
) -> None:
    source, version = source_dataset(store)
    info = {
        "scorer": "lance_column",
        "column": column,
        "dtype": dtype,
        "dataset": source,
        "dataset_version": version,
    }
    ensure_info(store, kind, info)

    if is_complete(store, kind):
        print(f"{store.uri} already has every shard of meta/{kind}", flush=True)
        return

    dataset = open_source_dataset(store, mirror)

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
    add_stage_arguments(parser)
    arguments = parser.parse_args()
    check_stage(arguments, "store")

    store = ShardStore(arguments.store)
    kind = arguments.kind or arguments.column

    if arguments.stage == "fetch":
        fetch_source_fragments(store, arguments.mirror, min(arguments.workers, 32))
    elif arguments.stage == "upload":
        if not is_complete(store, kind):
            raise FileNotFoundError(f"{store.uri} lacks shards of meta/{kind}")

        upload_store(store.uri, arguments.remote)
    else:
        copy_column(
            store,
            arguments.column,
            kind,
            arguments.dtype,
            arguments.workers,
            arguments.mirror,
        )


if __name__ == "__main__":
    main()
