"""
Download a random subset of a Hugging Face dataset's Parquet files, for P0-1.

Run on a login node: compute nodes have no internet. The full FineWeb-2 configs are
0.5-2 TB each, and P0-1 only needs ~10B tokens per language, so this picks --files of
a split's Parquet files at random (seeded, so a rerun picks the same ones, and a larger
--files keeps the earlier picks and adds more) instead of downloading the whole config.
A random pick rather than the first N files, because the files are ordered by crawl
dump, and the first N would cover only a few dumps.

    uv run python scripts/embedding_ml_transfer/fetch_hf_parquet.py \\
        --repo HuggingFaceFW/fineweb-2 --prefix data/spa_Latn/train --files 16 \\
        --to /work1/$USER/hf-raw/fineweb-2

Writes the chosen files under --to with their repo paths, and <--to>/picks/<prefix>.json
recording the repo, revision, seed and files, so parquet_to_lance.py knows exactly what
it converts. Rerunning resumes interrupted downloads.
"""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True, help="dataset repo, e.g. owner/name")
    parser.add_argument(
        "--prefix", required=True, help="folder in the repo, e.g. data/spa_Latn/train"
    )
    parser.add_argument(
        "--files", type=int, help="how many Parquet files to pick (default: all)"
    )
    parser.add_argument("--to", required=True, type=Path, help="local folder")
    parser.add_argument("--revision", help="commit to pin (default: main's current)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    api = HfApi()

    # Pin the commit, so the record says exactly which files were read
    revision = (
        arguments.revision or api.dataset_info(arguments.repo, revision="main").sha
    )
    files = sorted(
        entry.path
        for entry in api.list_repo_tree(
            arguments.repo,
            path_in_repo=arguments.prefix,
            repo_type="dataset",
            revision=revision,
            recursive=True,
        )
        if entry.path.endswith(".parquet")
    )

    if not files:
        raise SystemExit(f"no Parquet files under {arguments.prefix}")

    # A seeded permutation of all files, cut to --files: a larger --files extends the pick
    order = list(files)
    random.Random(arguments.seed).shuffle(order)
    picked = sorted(order[: arguments.files] if arguments.files else order)

    print(
        f"{arguments.repo}@{revision[:12]} {arguments.prefix}: {len(picked)}/{len(files)} files"
    )

    snapshot_download(
        arguments.repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=picked,
        local_dir=arguments.to,
        max_workers=arguments.max_workers,
    )

    record = (
        arguments.to
        / "picks"
        / f"{arguments.prefix.strip('/').replace('/', '--')}.json"
    )
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(
            {
                "repo": arguments.repo,
                "revision": revision,
                "prefix": arguments.prefix,
                "seed": arguments.seed,
                "available": len(files),
                "files": picked,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {record}")


if __name__ == "__main__":
    main()
