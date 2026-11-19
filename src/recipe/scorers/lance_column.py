"""
Copy a column of the source Lance dataset onto a shard store as meta/<kind>/.

    uv run python -m src.recipe.scorers.lance_column \\
        --store s3://<bucket>/sources/fineweb-edu-llama3 --column int_score \\
        --run-name score-int_score-fineweb-edu-llama3

FineWeb-Edu ships its classifier's `score` (float) and `int_score` (0-5), so this is a
quality score for free. The source rows are scanned fragment by fragment and joined
onto the shards through row_ids.npy.

Takes --stage like prepare.py (see src/stages.py and harness.py): fetch mirrors the
store's source fragments, if prepare.py --stage fetch has not, process scores from
--mirror, upload copies the store, new column included, to --remote.
"""

import argparse

import numpy

from .harness import ColumnScorer, Context, run


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--column", required=True, help="column of the Lance dataset")
    parser.add_argument("--dtype", help="cast the values, e.g. float32")


def setup(context: Context) -> ColumnScorer:
    column: str = context.arguments.column
    dtype: str | None = context.arguments.dtype

    def transform(values: dict[str, numpy.ndarray]) -> numpy.ndarray:
        return values[column] if dtype is None else values[column].astype(dtype)

    return ColumnScorer(
        info={"scorer": "lance_column", "column": column, "dtype": dtype},
        columns=[column],
        transform=transform,
    )


if __name__ == "__main__":
    # --kind defaults to the column
    run(setup, add_arguments, kind=lambda arguments: arguments.column)
