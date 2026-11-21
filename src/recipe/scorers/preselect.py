"""
PreSelect (Shum et al., 2025, arXiv:2503.00808): the probability that a document has
high "predictive strength", from the fastText classifier the authors released.

    uv run python -m src.recipe.scorers.preselect \\
        --store s3://<bucket>/sources/dolma-v1_7-falcon-refinedweb-filtered-llama3-150b \\
        --run-name score-preselect-dolma-v1_7-falcon-refinedweb-filtered-llama3-150b

Takes --stage like prepare.py (see src/stages.py and harness.py): fetch downloads the
classifier into the Hugging Face cache and mirrors the store's source fragments, process
scores offline from the cache and --mirror, upload copies the store to --remote. Pass
the same --revision to every stage, or none to use what fetch cached as the latest.

What the paper did: a document's predictive strength is how well its normalized loss
under six Llama models ranks them in the order of their benchmark scores. The
documents with perfect rankings became __label__1 and as many low-ranked ones
__label__0, and a fastText classifier (wordNgrams=2) learned to tell them apart. The
paper keeps the top 10% of a RefinedWeb pool by this classifier.

How this scores, following the authors' code (data_processing/fasttext/ in
github.com/hkust-nlp/PreSelect):
  - every "\\n" becomes " ", as in their training file (train_fasttext.py). fastText
    reads one line per document and refuses newlines; the other whitespace is split
    on anyway
  - the value is P(__label__1), the probability the authors' datatrove filter keeps or
    thresholds on (find_threshold.py takes the top 10% of characters by it)

The column is that probability as float32, in [0, 1]. The selection itself (top
fraction, by characters or tokens) belongs in a compiler, e.g. Docs.top(..., by=kind).

Memory: fastText loads the whole model (2.7 GB) into every worker process, so allow
about 3 GB per --workers.
"""

import argparse
import functools
import re
from pathlib import Path

import fasttext
import numpy
from huggingface_hub import hf_hub_download

from .harness import Context, TextScorer, run
from .text import ScoreFunction

REPOSITORY = "hkust-nlp/preselect-fasttext-classifier"
MODEL_FILE = "PreSelect-classifier.bin"
POSITIVE_LABEL = "__label__1"
LABELS = {"__label__0", POSITIVE_LABEL}
# What "\n" becomes before predicting: the authors' training data used " "
NEWLINE_REPLACEMENT = " "


def download_model(revision: str | None) -> tuple[Path, str]:
    """
    The classifier file and the commit it is from. With HF_HUB_OFFLINE=1 the revision
    (default: main) resolves from the local cache
    """
    path = Path(hf_hub_download(REPOSITORY, MODEL_FILE, revision=revision))
    # The cache keeps each snapshot in a directory named by its commit
    commit = path.parent.name

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"cannot tell the commit of {REPOSITORY} from {path}")

    return path, commit


def build(model_path: str) -> ScoreFunction:
    model = fasttext.load_model(model_path)
    labels = set(model.get_labels())

    if labels != LABELS:
        raise ValueError(
            f"{model_path} has labels {sorted(labels)}, not {sorted(LABELS)}"
        )

    def score(texts: list[str]) -> numpy.ndarray:
        lines = [text.replace("\n", NEWLINE_REPLACEMENT) for text in texts]
        # A list goes through multilinePredict, one C++ call per batch. k=-1 returns
        # both labels, most probable first, so the positive one's position varies
        predicted_labels, probabilities = model.predict(lines, k=-1)
        scores = numpy.empty(len(texts), dtype=numpy.float32)

        for row, (row_labels, row_probabilities) in enumerate(
            zip(predicted_labels, probabilities, strict=True)
        ):
            scores[row] = row_probabilities[row_labels.index(POSITIVE_LABEL)]

        return scores

    return score


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--revision", help=f"{REPOSITORY} revision; default: the latest commit"
    )


def fetch(context: Context) -> None:
    _, commit = download_model(context.arguments.revision)
    print(f"{REPOSITORY} at {commit}", flush=True)


def setup(context: Context) -> TextScorer:
    # Offline (--stage process), the revision resolves from what fetch cached
    path, commit = download_model(context.arguments.revision)
    print(f"{REPOSITORY} at {commit}", flush=True)

    return TextScorer(
        info={
            "scorer": "preselect",
            "repository": REPOSITORY,
            "revision": commit,
            "file": MODEL_FILE,
            "label": POSITIVE_LABEL,
            "newline_replacement": NEWLINE_REPLACEMENT,
        },
        build=functools.partial(build, str(path)),
    )


if __name__ == "__main__":
    run(setup, add_arguments, fetch=fetch, kind="preselect")
