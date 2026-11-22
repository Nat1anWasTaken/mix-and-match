"""
Convert the Parquet files fetch_hf_parquet.py picked into one local Lance dataset, the
form prepare.py and the scorers read. Run on a compute node (it only reads /work1):

    sbatch scripts/f1-cpu.sbatch scripts/embedding_ml_transfer/parquet_to_lance.py \\
        --picks /work1/$USER/hf-raw/fineweb-2/picks/data--spa_Latn--train.json \\
        --output /work1/$USER/mirrors/fineweb-2-spa_Latn-train

Every row keeps the source's columns and gains two that say where it came from, as the
Flock export of FineWeb-2 has them: _hf_source_file (the file's repo path) and
_hf_source_row (the row's index in that file). With the picks record's revision, any
document of any store can be traced back to the exact Hugging Face row.

Each Parquet file becomes the fragments of one worker (at most --rows-per-fragment rows
each, so prepare.py's shuffled fragment order mixes files at a fine grain), written in
parallel, then committed once in file order. Nothing is visible until that commit, so
an interrupted run leaves no half-written dataset; rerun it with --overwrite.
"""

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lance
import pyarrow
import pyarrow.parquet

ROWS_PER_FRAGMENT = 1 << 18


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--picks", required=True, type=Path, help="picks/<...>.json")
    parser.add_argument("--output", required=True, help="Lance dataset to create")
    parser.add_argument("--rows-per-fragment", type=int, default=ROWS_PER_FRAGMENT)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_schema(source: pyarrow.Schema) -> pyarrow.Schema:
    """
    The source columns, dictionaries decoded to plain values, plus the lineage columns
    """
    fields = [
        field.with_type(field.type.value_type)
        if pyarrow.types.is_dictionary(field.type)
        else field
        for field in source
    ]
    return pyarrow.schema(
        [
            *fields,
            pyarrow.field("_hf_source_file", pyarrow.string()),
            pyarrow.field("_hf_source_row", pyarrow.int64()),
        ]
    )


def convert_file(
    task: tuple[Path, str, str, pyarrow.Schema, int],
) -> list[lance.FragmentMetadata]:
    local, repo_path, output, schema, rows_per_fragment = task
    parquet = pyarrow.parquet.ParquetFile(local)

    def batches():  # noqa: ANN202
        start = 0

        for batch in parquet.iter_batches(batch_size=8192):
            rows = batch.num_rows
            columns = [
                batch.column(name).cast(schema.field(name).type)
                for name in schema.names[:-2]
            ]
            columns.append(pyarrow.array([repo_path] * rows, pyarrow.string()))
            columns.append(pyarrow.array(range(start, start + rows), pyarrow.int64()))
            start += rows
            yield pyarrow.RecordBatch.from_arrays(columns, schema=schema)

    reader = pyarrow.RecordBatchReader.from_batches(schema, batches())
    return lance.fragment.write_fragments(
        reader, output, schema=schema, max_rows_per_file=rows_per_fragment
    )


def main() -> None:
    arguments = parse_arguments()
    picks = json.loads(arguments.picks.read_text())
    root = arguments.picks.parent.parent
    files = [(root / path, path) for path in picks["files"]]

    for local, _ in files:
        if not local.exists():
            raise FileNotFoundError(f"{local} is missing; rerun fetch_hf_parquet.py")

    if Path(arguments.output).exists() and not arguments.overwrite:
        raise SystemExit(f"{arguments.output} exists; pass --overwrite to replace it")

    schema = output_schema(pyarrow.parquet.read_schema(files[0][0]))
    tasks = [
        (local, path, arguments.output, schema, arguments.rows_per_fragment)
        for local, path in files
    ]

    # One process per file: each reads and compresses its own file. Spawned, as lance
    # is not fork-safe
    with ProcessPoolExecutor(
        max_workers=min(arguments.workers, len(tasks)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        per_file = list(pool.map(convert_file, tasks))

    fragments = [fragment for part in per_file for fragment in part]
    dataset = lance.LanceDataset.commit(
        arguments.output,
        lance.LanceOperation.Overwrite(schema, fragments),
        commit_message=f"{picks['repo']}@{picks['revision']} {picks['prefix']}",
    )
    print(
        f"{arguments.output}: {dataset.count_rows()} rows in "
        f"{len(dataset.get_fragments())} fragments from {len(files)} files",
        flush=True,
    )


if __name__ == "__main__":
    main()
