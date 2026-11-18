"""
Paloma decontamination (Paloma App. C.1.1): count, per document, the paragraphs that
also appear in Paloma's evaluation data.

    uv run python -m src.recipe.scorers.paloma \\
        --store s3://<bucket>/sources/fineweb-edu-llama3

Paloma is gated: accept its terms on Hugging Face and log in (`hf auth login`) first.

Following the paper:
  - text is matched at the paragraph level, i.e. newline-separated spans
  - evaluation paragraphs shorter than 13 Unicode-segmented tokens are ignored, which
    also drops paragraphs of only punctuation, spaces and emoji
  - a document is contaminated if any of its paragraphs matches; the compiler removes
    the whole document

The paper matches with a Bloom filter. This matches exactly, by a 64-bit hash of each
paragraph against the sorted hashes of every val and test paragraph: no false positives,
and the hashes of a few million paragraphs take a few tens of MB per worker. Paragraphs
are compared after stripping surrounding whitespace, so a trailing "\\r" or indent does
not hide a match.

The column is an int32 count of matching paragraphs, so 0 means clean.
"""

import argparse
import functools
import gzip
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy
from huggingface_hub import HfApi, snapshot_download

from ...tokenizer import ShardStore
from .text import ScoreFunction, score_texts

PALOMA_REPOSITORY = "allenai/paloma"
MIN_PARAGRAPH_WORDS = 13

# Approximates Unicode word segmentation (UAX #29): a word is a run of letters, digits
# or underscores, so punctuation, spaces and emoji count as nothing. A run of CJK
# ideographs counts as one word here, where UAX #29 counts each ideograph
WORD = re.compile(r"\w+")


def paragraphs(text: str) -> Iterator[str]:
    for line in text.split("\n"):
        paragraph = line.strip()

        if paragraph:
            yield paragraph


def paragraph_hash(paragraph: str) -> int:
    # Stable across processes, unlike hash()
    digest = hashlib.blake2b(paragraph.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def download_paloma(revision: str | None) -> tuple[Path, str]:
    """
    The val and test files of every Paloma source, and the commit they are from
    """
    commit = HfApi().dataset_info(PALOMA_REPOSITORY, revision=revision).sha

    if commit is None:
        raise ValueError(f"cannot resolve {PALOMA_REPOSITORY}@{revision}")

    path = snapshot_download(
        PALOMA_REPOSITORY,
        repo_type="dataset",
        revision=commit,
        allow_patterns=["*/val/*.jsonl.gz", "*/test/*.jsonl.gz"],
    )
    return Path(path), commit


def evaluation_hashes(root: Path, sources: list[str] | None) -> numpy.ndarray:
    """
    Sorted unique hashes of every long enough val and test paragraph
    """
    files = sorted(
        path
        for path in root.rglob("*.jsonl.gz")
        if {"val", "test"} & set(path.relative_to(root).parts)
        and (sources is None or path.relative_to(root).parts[0] in sources)
    )

    if not files:
        raise FileNotFoundError(f"no Paloma val or test files under {root}")

    hashes: list[int] = []

    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                for paragraph in paragraphs(json.loads(line)["text"]):
                    if len(WORD.findall(paragraph)) >= MIN_PARAGRAPH_WORDS:
                        hashes.append(paragraph_hash(paragraph))

    print(f"{len(hashes):,} Paloma paragraphs from {len(files)} files", flush=True)
    return numpy.unique(numpy.array(hashes, dtype=numpy.uint64))


def build(hashes_path: str) -> ScoreFunction:
    contaminated = numpy.load(hashes_path, mmap_mode="r")

    def score(texts: list[str]) -> numpy.ndarray:
        document_of: list[int] = []
        hashes: list[int] = []

        for document, text in enumerate(texts):
            for paragraph in paragraphs(text):
                document_of.append(document)
                hashes.append(paragraph_hash(paragraph))

        if not hashes:
            return numpy.zeros(len(texts), dtype=numpy.int32)

        queries = numpy.array(hashes, dtype=numpy.uint64)
        found = numpy.searchsorted(contaminated, queries)
        found = numpy.minimum(found, contaminated.size - 1)
        matches = contaminated[found] == queries
        counts = numpy.bincount(numpy.array(document_of)[matches], minlength=len(texts))
        return counts.astype(numpy.int32)

    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", required=True, help="shard store from prepare.py")
    parser.add_argument("--kind", default="paloma", help="metadata kind to write")
    parser.add_argument(
        "--revision", help=f"{PALOMA_REPOSITORY} revision; default: the latest commit"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Paloma sources to match against, e.g. c4_en ptb; default: all of them",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    arguments = parser.parse_args()

    root, commit = download_paloma(arguments.revision)
    hashes = evaluation_hashes(root, arguments.sources)
    print(f"{hashes.size:,} unique Paloma paragraphs", flush=True)

    if hashes.size == 0:
        raise ValueError("no Paloma paragraph is long enough to match")

    info = {
        "scorer": "paloma",
        "repository": PALOMA_REPOSITORY,
        "revision": commit,
        "sources": sorted(arguments.sources) if arguments.sources else None,
        "min_paragraph_words": MIN_PARAGRAPH_WORDS,
        "paragraphs": int(hashes.size),
    }

    with tempfile.TemporaryDirectory() as directory:
        # Workers memory-map the hashes instead of each receiving a pickled copy
        hashes_path = os.path.join(directory, "hashes.npy")
        numpy.save(hashes_path, hashes)
        score_texts(
            ShardStore(arguments.store),
            arguments.kind,
            functools.partial(build, hashes_path),
            info,
            arguments.workers,
        )


if __name__ == "__main__":
    main()
