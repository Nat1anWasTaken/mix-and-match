"""
Exact document deduplication, the scorer half: a 64-bit hash of each document's
normalized text. The compiler does the dedup itself, keeping one document per hash:

    uv run python -m src.recipe.scorers.text_hash \\
        --store s3://<bucket>/sources/pile-uncopyrighted-llama3 \\
        --run-name score-text_hash-pile-uncopyrighted-llama3

Takes --stage like prepare.py (see src/stages.py and harness.py): fetch mirrors the
store's source fragments, process hashes offline from --mirror, upload copies the
store to --remote.

Dolma's document dedup (Dolma App. N.4) compares the raw text. Here the text is
normalized first, so copies that differ only in Unicode form, case or whitespace count
as duplicates too:
  1. Unicode NFKC, so e.g. full-width and composed/decomposed forms compare equal
  2. lowercase
  3. every run of whitespace (spaces, tabs, newlines, "\\r") becomes one space, and the
     ends are stripped
Punctuation and digits are kept: two documents that differ in them are different
documents, not copies.

The hash is per document, so the column says nothing about duplicates by itself: two
documents are copies when their hashes are equal. That keeps this a plain per-document
scorer, and lets a compiler dedup across stores too, by comparing the columns of
several stores. With n documents, the chance of any 64-bit collision is about
n^2 / 2^65: 5e-5 for 43M documents.

The column is the uint64 hash. Its score/* summary in wandb (mean, max, ...) means
nothing; count duplicates with numpy.unique in a compiler.
"""

import hashlib
import re
import unicodedata

import numpy

from .harness import Context, TextScorer, run
from .text import ScoreFunction

WHITESPACE = re.compile(r"\s+")
# Recorded in info.json: change it whenever normalize changes, so a rerun with the new
# normalization is refused onto an old column instead of mixing two kinds of hash
NORMALIZATION = "nfkc-lower-whitespace-v1"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return WHITESPACE.sub(" ", text).strip()


def text_hash(text: str) -> int:
    # Stable across processes, unlike hash()
    digest = hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def build() -> ScoreFunction:
    def score(texts: list[str]) -> numpy.ndarray:
        return numpy.array([text_hash(text) for text in texts], dtype=numpy.uint64)

    return score


def setup(context: Context) -> TextScorer:
    return TextScorer(
        info={
            "scorer": "text_hash",
            "hash": "blake2b-64",
            "normalization": NORMALIZATION,
        },
        build=build,
    )


if __name__ == "__main__":
    run(setup, kind="text_hash")
