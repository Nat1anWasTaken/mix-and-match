"""
The embedding-ml-transfer training splits (plan §3), one compiler for all of them:

    uv run python -m src.recipe.compilers.embedding_ml_transfer --split multi_train \\
        --store-root s3://embedding-ml-transfer/sources \\
        --total-tokens 20000000000 \\
        --output s3://embedding-ml-transfer/recipes/multi-train-20b

    multi_train     en 40% + es / de / ru 20% each, L_ban filtered out
    mono_train      English only: Multi-Train's English documents plus more English
    oracle_train    Multi-Train (the same documents) + Chinese + Fake-French
    zh_adapt        Chinese not in Oracle-Train (Phase 2 adaptation corpus)
    fake_fra_adapt  Fake-French not in Oracle-Train

Sizes: --total-tokens is what train.py trains on; every split holds it plus --margin.
mono_train and oracle_train also need --multi-tokens, the --total-tokens of the
Multi-Train they contain (default 20B), so they hold exactly its documents: the
plan's "加上去，不是替換", which makes B0-Upper and B0-Lower see the same L_train
tokens, and keeps A1 vs A2 a difference in training languages only. oracle_train
splits what is left after Multi-Train evenly between Chinese and Fake-French.

Same documents without reading another recipe: every component (multi/en,
oracle/zh, ...) samples with its own random generator, seeded by (--seed, its name).
Multi-Train's components are sampled the same way whichever split is compiled, so
oracle_train and mono_train reproduce its selection exactly, given the same stores,
--multi-tokens, --margin and --seed.

Filters, on L_train sources (en, es, de, ru), from the scorers' meta/ columns:
  - meta/glotlid: no line with P(French) > 0.5 or P(Chinese) > 0.5, and the whole
    document under 0.01 for both (--long-lines-only uses the >= 50 character lines)
  - meta/vban_count == 0: no Chinese token at all
Chinese: cmn_Hani, train split, simplified only (meta/fineweb2_chinese). Fake-French
is built from French train documents only, so it needs no filter.
"""

import argparse
import zlib

import numpy

from ..compile import Context, Docs, Source, run

SPLITS = ["multi_train", "mono_train", "oracle_train", "zh_adapt", "fake_fra_adapt"]
L_TRAIN = {
    "en": "fineweb-qwen3",
    "es": "fineweb-2-spa_Latn-qwen3",
    "de": "fineweb-2-deu_Latn-qwen3",
    "ru": "fineweb-2-rus_Cyrl-qwen3",
}
MULTI_WEIGHTS = {"en": 0.4, "es": 0.2, "de": 0.2, "ru": 0.2}
CHINESE = "fineweb-2-cmn_Hani-qwen3"
FAKE_FRENCH = "fake-fra-qwen3"

LINE_THRESHOLD = 0.5
DOCUMENT_THRESHOLD = 0.01
SIMPLIFIED = 1  # fineweb2_chinese.SCRIPTS


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument(
        "--store-root", required=True, help="where the source stores are, e.g. s3://…"
    )
    parser.add_argument(
        "--multi-tokens",
        type=int,
        default=20_000_000_000,
        help="--total-tokens of the Multi-Train that mono/oracle/adapt build on",
    )
    parser.add_argument(
        "--oracle-tokens",
        type=int,
        default=24_000_000_000,
        help="--total-tokens of the Oracle-Train the adaptation corpora avoid",
    )
    parser.add_argument(
        "--long-lines-only",
        action="store_true",
        help="judge paragraphs by the lines of >= 50 characters only",
    )


def sample(context: Context, component: str, documents: Docs, tokens: int) -> Docs:
    """
    documents.sample(tokens) with the component's own generator, see the docstring
    """
    context.random_generator = numpy.random.default_rng(
        [context.seed, zlib.crc32(component.encode())]
    )
    return documents.sample(tokens)


def clean(source: Source, long_lines_only: bool) -> Docs:
    suffix = "_line_long" if long_lines_only else "_line"

    def no_ban_language(scores: numpy.ndarray) -> numpy.ndarray:
        return (
            (scores["fra" + suffix] <= LINE_THRESHOLD)
            & (scores["zh" + suffix] <= LINE_THRESHOLD)
            & (scores["fra_doc"] <= DOCUMENT_THRESHOLD)
            & (scores["zh_doc"] <= DOCUMENT_THRESHOLD)
        )

    lid = source.where("glotlid", no_ban_language)
    tokens = source.where("vban_count", lambda counts: counts == 0)
    kept = lid & tokens
    print(
        f"{source.name}: {source.documents:,} documents; LID drops "
        f"{1 - lid.documents / source.documents:.3%}, V_ban drops "
        f"{1 - tokens.documents / source.documents:.3%}, together "
        f"{1 - kept.documents / source.documents:.3%} "
        f"({1 - kept.tokens / source.tokens:.3%} of tokens)",
        flush=True,
    )
    return kept


def check(name: str, documents: Docs, tokens: int) -> None:
    if documents.tokens < tokens:
        raise ValueError(
            f"{name}: {documents.tokens:,} clean tokens, needs {tokens:,}; prepare "
            "its store with a larger --token-budget"
        )


def multi_train(context: Context, sources: dict[str, Source]) -> dict[str, Docs]:
    """
    Multi-Train's documents per language, for --multi-tokens (plus the margin)
    """
    target = context.arguments.multi_tokens * (1 + context.margin)
    budgets = context.split(MULTI_WEIGHTS, target)
    kept = {
        language: clean(sources[language], context.arguments.long_lines_only)
        for language in MULTI_WEIGHTS
    }

    for language, documents in kept.items():
        check(language, documents, budgets[language])

    return {
        language: sample(
            context, f"multi/{language}", kept[language], budgets[language]
        )
        for language in MULTI_WEIGHTS
    }


def chinese(context: Context) -> Docs:
    source = context.open(
        "zh", f"{context.arguments.store_root}/{CHINESE}", columns=["fineweb2_chinese"]
    )
    kept = source.where(
        "fineweb2_chinese",
        lambda tags: (
            (tags["config"] == 1)
            & (tags["split"] == 0)
            & (tags["script"] == SIMPLIFIED)
        ),
    )
    print(
        f"zh: {kept.documents:,} of {source.documents:,} documents are cmn_Hani "
        f"train simplified ({kept.tokens:,} tokens)",
        flush=True,
    )
    return kept


def fake_french(context: Context) -> Docs:
    return context.open(
        "fake_fra", f"{context.arguments.store_root}/{FAKE_FRENCH}"
    ).all()


def oracle_extra(
    context: Context, oracle_tokens: int, zh: Docs, fake: Docs
) -> tuple[Docs, Docs]:
    """
    Oracle-Train's Chinese and Fake-French, for an Oracle-Train of oracle_tokens: what
    it holds beyond Multi-Train, half each
    """
    extra = (oracle_tokens - context.arguments.multi_tokens) * (1 + context.margin)
    half = int(numpy.ceil(extra / 2))

    if half <= 0:
        raise ValueError("--total-tokens must exceed --multi-tokens for oracle_train")

    check("zh", zh, half)
    check("fake_fra", fake, half)
    return sample(context, "oracle/zh", zh, half), sample(
        context, "oracle/fake_fra", fake, half
    )


def compile(context: Context) -> None:
    split = context.arguments.split
    root = context.arguments.store_root
    context.phase("main")

    if split in ("multi_train", "mono_train", "oracle_train"):
        sources = context.open_many(
            {language: f"{root}/{store}" for language, store in L_TRAIN.items()},
            columns=["glotlid", "vban_count"],
        )
        multi = multi_train(context, sources)

    if split == "multi_train":
        for language in MULTI_WEIGHTS:
            context.cursor(language, multi[language])

    elif split == "mono_train":
        english = clean(sources["en"], context.arguments.long_lines_only)
        more = context.budget() - multi["en"].tokens
        rest = english - multi["en"]
        check("en (beyond Multi-Train)", rest, more)
        context.cursor("en", multi["en"] | sample(context, "mono/en", rest, more))

    elif split == "oracle_train":
        zh, fake = oracle_extra(
            context, context.total_tokens, chinese(context), fake_french(context)
        )

        for language in MULTI_WEIGHTS:
            context.cursor(language, multi[language])

        context.cursor("zh", zh)
        context.cursor("fake_fra", fake)

    else:
        # Adaptation corpora: disjoint from the Chinese and Fake-French that an
        # Oracle-Train of --oracle-tokens holds, recomputed with the same generators
        zh, fake = chinese(context), fake_french(context)
        oracle_zh, oracle_fake = oracle_extra(
            context, context.arguments.oracle_tokens, zh, fake
        )

        if split == "zh_adapt":
            rest = zh - oracle_zh
            check("zh (beyond Oracle-Train)", rest, context.budget())
            context.cursor("zh", sample(context, "adapt/zh", rest, context.budget()))
        else:
            rest = fake - oracle_fake
            check("fake_fra (beyond Oracle-Train)", rest, context.budget())
            context.cursor(
                "fake_fra", sample(context, "adapt/fake_fra", rest, context.budget())
            )


if __name__ == "__main__":
    run(compile, add_arguments)
