"""
Copy FineWeb-2's Chinese columns onto a store from Flock's fineweb-2 export, as codes.

    uv run python -m src.recipe.scorers.fineweb2_chinese \\
        --store /work1/$USER/stores/fineweb-2-cmn_Hani-qwen3 \\
        --mirror /work1/$USER/mirrors/fineweb-2 \\
        --run-name score-fineweb2_chinese-fineweb-2-cmn_Hani-qwen3 --no-wandb

Flock's s3://flock-datasets/fineweb-2 holds every Han-script config of FineWeb-2 (cmn,
yue, wuu, lzh, hak) with their train and test splits in one Lance dataset, plus an
opencc-based simplified/traditional label per document. The embedding-ml-transfer
Chinese data is cmn_Hani train, simplified only (FLORES+ evaluates cmn_Hans), and the
test split is kept for evaluation, so the compiler needs all three. They are strings in
the source; meta/ holds numbers, so this writes one structured column meta/<kind>/:

    config   1 for cmn_Hani, 0 for any other config
    split    0 train, 1 test
    script   SCRIPTS index of chinese_script.label
"""

import numpy

from .harness import ColumnScorer, Context, run

DTYPE = numpy.dtype(
    [("config", numpy.uint8), ("split", numpy.uint8), ("script", numpy.uint8)]
)
SPLITS = ["train", "test"]
SCRIPTS = ["undetermined", "simplified", "traditional", "mixed"]
COLUMNS = ["_hf_config", "_hf_split", "chinese_script.label"]


def codes(values: numpy.ndarray, names: list[str]) -> numpy.ndarray:
    known = set(names)
    unknown = set(values.tolist()) - known

    if unknown:
        raise ValueError(
            f"unexpected values {sorted(map(str, unknown))}; known {names}"
        )

    lookup = {name: index for index, name in enumerate(names)}
    return numpy.array([lookup[value] for value in values], dtype=numpy.uint8)


def transform(columns: dict[str, numpy.ndarray]) -> numpy.ndarray:
    values = numpy.empty(columns["_hf_config"].size, dtype=DTYPE)
    values["config"] = columns["_hf_config"] == "cmn_Hani"
    values["split"] = codes(columns["_hf_split"], SPLITS)
    values["script"] = codes(columns["chinese_script.label"], SCRIPTS)
    return values


def setup(context: Context) -> ColumnScorer:
    return ColumnScorer(
        info={
            "scorer": "fineweb2_chinese",
            "columns": COLUMNS,
            "splits": SPLITS,
            "scripts": SCRIPTS,
        },
        columns=COLUMNS,
        transform=transform,
    )


if __name__ == "__main__":
    run(setup, kind="fineweb2_chinese")
