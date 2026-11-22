"""
Paragraph-level language ID with GlotLID, for the embedding-ml-transfer leakage filter.

    uv run python -m src.recipe.scorers.glotlid \\
        --store /work1/$USER/stores/fineweb-2-spa_Latn-qwen3 \\
        --model /work1/$USER/models/lid/glotlid/model_v3.bin \\
        --run-name score-glotlid-fineweb-2-spa_Latn-qwen3 --no-wandb

The plan (§3, "雙層防洩漏濾網") drops a training document when any of its paragraphs
reads as a banned language (P > 0.5), or when the whole document has P(L_ban) > 0.01.
Scoring the document once misses a French quote inside an English page, which is why
paragraphs are scored one by one. GlotLID v3 is the classifier FineWeb-2 itself was
built with, so its labels match the sources' own.

Writes one column, meta/glotlid/, with a structured dtype (one record per document),
so a single scan gives every number the compilers filter on:

    fra_line        max over the document's lines of P(fra_Latn)
    fra_line_long   the same, over lines of at least LONG_LINE characters only
    zh_line         max over lines that hold a Han character of P(Chinese): the sum
                    over every *_Hani label (cmn, yue, wuu, lzh, hak, nan, ...), since
                    any Chinese is banned
    zh_line_long    the same, long lines only
    fra_doc         P(fra_Latn) of the whole document, newlines as spaces
    zh_doc          P(Chinese) of the whole document, 0 if it holds no Han character

GlotLID puts Hani labels on short lines with no Han character at all ("фото" reads as
lzh_Hani at 0.99, "21.05.2018 17:45" as cmn_Hani at 0.90): in a 19k-document Russian
sample that alone would drop 2.3% of the documents. Text without a Han character can't
be Chinese, so P(Chinese) only counts where there is one. Pages that do hold Han
characters are dropped from L_train anyway, by the V_ban token filter (vban_count.py).

A paragraph is a line: FineWeb and FineWeb-2 keep the page's block structure as
newlines. Short lines (menu items, a lone "Merci") are where LID is least reliable, so
both the all-lines and the long-lines maxima are kept; the compiler picks one after
looking at how many documents each drops (a rescan costs a full pass, a column doesn't).

Probabilities come from the top TOP_K labels; a label outside them counts as 0. The
TOP_K-th probability bounds what that drops, far below the 0.01 document threshold.
"""

import argparse
import functools
import re
from collections.abc import Callable
from typing import Protocol

import numpy

from .harness import Context, TextScorer, run

DTYPE = numpy.dtype(
    [
        ("fra_line", numpy.float32),
        ("fra_line_long", numpy.float32),
        ("zh_line", numpy.float32),
        ("zh_line_long", numpy.float32),
        ("fra_doc", numpy.float32),
        ("zh_doc", numpy.float32),
    ]
)
LONG_LINE = 50
TOP_K = 20
FRENCH = "__label__fra_Latn"
# CJK Unified Ideographs with extension A, compatibility ideographs and extensions B+
HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")


class Model(Protocol):
    """
    The part of fasttext-predict's model this uses (the package ships no types)
    """

    def predict(self, text: str, k: int) -> tuple[list[str], list[float]]: ...


def is_chinese(label: str) -> bool:
    return label.endswith("_Hani")


def probabilities(model: Model, text: str) -> tuple[float, float]:
    """
    (P(French), P(Chinese)) of one line of text
    """
    labels, scores = model.predict(text, k=TOP_K)
    french = chinese = 0.0

    for label, score in zip(labels, scores, strict=True):
        if label == FRENCH:
            french = float(score)
        elif is_chinese(label):
            chinese += float(score)

    # GlotLID's softmax can come out a hair above 1
    return min(french, 1.0), min(chinese, 1.0)


def score_text(model: Model, text: str) -> tuple[float, ...]:
    fra_line = fra_long = zh_line = zh_long = 0.0

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        french, chinese = probabilities(model, line)
        chinese = chinese if HAN.search(line) else 0.0
        fra_line = max(fra_line, french)
        zh_line = max(zh_line, chinese)

        if len(line) >= LONG_LINE:
            fra_long = max(fra_long, french)
            zh_long = max(zh_long, chinese)

    fra_doc, zh_doc = probabilities(model, " ".join(text.split()))
    zh_doc = zh_doc if HAN.search(text) else 0.0
    return fra_line, fra_long, zh_line, zh_long, fra_doc, zh_doc


def build(model_path: str) -> Callable[[list[str]], numpy.ndarray]:
    # Imported here: workers are spawned, and only they need the model
    import fasttext

    model = fasttext.load_model(model_path)

    def score(texts: list[str]) -> numpy.ndarray:
        return numpy.array([score_text(model, text) for text in texts], dtype=DTYPE)

    return score


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", required=True, help="GlotLID fastText model, e.g. model_v3.bin"
    )


def setup(context: Context) -> TextScorer:
    model: str = context.arguments.model
    return TextScorer(
        info={
            "scorer": "glotlid",
            # By file name, not path: the same model may sit elsewhere on another host
            "model": model.rsplit("/", 1)[-1],
            "long_line": LONG_LINE,
            "top_k": TOP_K,
            "chinese_needs_han": True,
            "fields": list(DTYPE.names or ()),
        },
        build=functools.partial(build, model),
        batch_size=256,
    )


if __name__ == "__main__":
    run(setup, add_arguments, kind="glotlid")
