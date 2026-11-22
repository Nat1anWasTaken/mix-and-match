"""
P0-1's evaluation text (plan §6), as jsonl, one {"id", "text"} per line:

    uv run python scripts/embedding_ml_transfer/eval_sets.py \\
        --mirrors /work1/$USER/mirrors/embedding-ml-transfer \\
        --output /work1/$USER/p0-1/eval

    fineweb-2-<lang>-test   FineWeb-2's own test split, the documents its train split
                            (and so every store here) leaves out: spa_Latn, deu_Latn,
                            rus_Cyrl, fra_Latn from the local test mirrors, and cmn_Hani
                            from Flock's export, simplified only, as the training
                            Chinese is
    flores-<lang>-devtest   FLORES+ devtest (openlanguagedata/flores_plus, gated: accept
                            its terms on the Hub first), eng_Latn, spa_Latn, deu_Latn,
                            rus_Cyrl, fra_Latn, cmn_Hans; one sentence per line, the
                            same 1,012 sentences in every language, in the same order

Runs on a login node: FLORES+ comes from the Hub and the Chinese test split from S3.
It only moves a few MB. The Fake-French versions (shifted token ids, with the original
French byte counts for BPB) are written by fake_french.py build, once the table exists.
"""

import argparse
import json
from pathlib import Path

import lance

FINEWEB2_LOCAL = ["spa_Latn", "deu_Latn", "rus_Cyrl", "fra_Latn"]
FLORES = ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl", "fra_Latn", "cmn_Hans"]
FLORES_REPO = "openlanguagedata/flores_plus"
# Flock's fineweb-2 export puts each source file in its own fragments; this one is
# data/cmn_Hani/test/000_00000.parquet (checked below, not assumed)
FLOCK_FINEWEB2 = "s3://flock-datasets/fineweb-2"
CMN_TEST_FILE = "data/cmn_Hani/test/000_00000.parquet"


def write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{path}: {len(rows)} rows", flush=True)


def fineweb2_local(mirrors: Path, output: Path) -> None:
    for language in FINEWEB2_LOCAL:
        table = lance.dataset(str(mirrors / f"fineweb-2-{language}-test")).to_table(
            columns=["id", "text"]
        )
        write(output / f"fineweb-2-{language}-test.jsonl", table.to_pylist())


def fineweb2_chinese(output: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.tokenizer import open_dataset

    dataset = open_dataset(FLOCK_FINEWEB2)
    rows = []

    for fragment in dataset.get_fragments():
        files = fragment.to_table(columns=["_hf_source_file"]).column(0)

        if files[0].as_py() != CMN_TEST_FILE:
            # The test file's fragments are contiguous: stop at the first one after
            if rows:
                break

            continue

        table = fragment.to_table(
            columns=["id", "text", "_hf_source_file", "chinese_script.label"]
        )
        rows += [
            {"id": row["id"], "text": row["text"]}
            for row in table.to_pylist()
            if row["_hf_source_file"] == CMN_TEST_FILE
            and row["chinese_script.label"] == "simplified"
        ]

    if not rows:
        raise SystemExit(f"no fragment of {FLOCK_FINEWEB2} holds {CMN_TEST_FILE}")

    write(output / "fineweb-2-cmn_Hani-test.jsonl", rows)


def flores(output: Path) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    for language in FLORES:
        try:
            path = hf_hub_download(
                FLORES_REPO, f"devtest/{language}.jsonl", repo_type="dataset"
            )
        except GatedRepoError:
            print(
                f"FLORES+ is gated: accept its terms at huggingface.co/datasets/"
                f"{FLORES_REPO}, then rerun with --only flores",
                flush=True,
            )
            return

        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        write(
            output / f"flores-{language}-devtest.jsonl",
            [{"id": str(row["id"]), "text": row["text"]} for row in rows],
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mirrors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--only", choices=["fineweb2", "fineweb2-chinese", "flores"], action="append"
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    only = set(arguments.only or ["fineweb2", "fineweb2-chinese", "flores"])

    if "fineweb2" in only:
        fineweb2_local(arguments.mirrors, arguments.output)

    if "fineweb2-chinese" in only:
        fineweb2_chinese(arguments.output)

    if "flores" in only:
        flores(arguments.output)


if __name__ == "__main__":
    main()
