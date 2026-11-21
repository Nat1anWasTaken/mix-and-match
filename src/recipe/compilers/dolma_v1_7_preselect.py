"""
Dolma v1.7 in OLMo 7B-v1.7's proportions (the WEIGHTS of compilers/dolma_v1_7.py), each
source drawn from the top --fraction (default 10%) of its clean documents by PreSelect
score (Shum et al., 2025, arXiv:2503.00808).

    uv run python -m src.recipe.compilers.dolma_v1_7_preselect \\
        --store-template 's3://<bucket>/sources/dolma-v1_7-{source}-llama3-150b' \\
        --total-tokens 30000000000 --output s3://<bucket>/recipes/dolma-v1_7-preselect-30b

Run src.recipe.scorers.paloma_decontamination and src.recipe.scorers.preselect on every
store first. Per source, in this order:

  1. Paloma decontamination, as in dolma_v1_7: drop every document with a paragraph
     that matches Paloma (Paloma App. C.1.1)
  2. the pool: the highest-scored clean documents worth --fraction of the clean
     tokens. One ranking over the whole source, ties broken at random. Tokens, not
     documents, as the paper's threshold counts characters (find_threshold.py)
  3. the cursor: the source's share of --total-tokens (plus --margin), split by
     WEIGHTS as in dolma_v1_7
       - the pool holds enough: a uniform random sample of the pool, so what is kept
         is the top --fraction whatever --total-tokens is
       - it does not: a warning, and the top share of the clean documents instead, a
         larger fraction than --fraction (printed). The mixture stays as WEIGHTS says
         and nothing repeats; only the threshold is looser for that source
       - even every clean document is too few: an error, as in dolma_v1_7
"""

import argparse
import math

from ..compile import Context, Docs, run
from .dolma_v1_7 import PLACEHOLDER, WEIGHTS


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store-template",
        required=True,
        help=f"store URI of every source, with {PLACEHOLDER} in it",
    )
    parser.add_argument(
        "--paloma-kind",
        default="paloma_decontamination",
        help="meta/<kind> column from the paloma_decontamination scorer",
    )
    parser.add_argument(
        "--score-kind",
        default="preselect",
        help="meta/<kind> column from the preselect scorer",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="fraction of each source's clean tokens the pool keeps, by score",
    )


def compile(context: Context) -> None:
    template = context.arguments.store_template
    paloma = context.arguments.paloma_kind
    score = context.arguments.score_kind
    fraction = context.arguments.fraction

    if PLACEHOLDER not in template:
        raise ValueError(f"--store-template must contain {PLACEHOLDER}")

    if not 0 < fraction <= 1:
        raise ValueError(f"--fraction must be in (0, 1], not {fraction}")

    budgets = context.split(WEIGHTS)
    sources = context.open_many(
        {name: template.replace(PLACEHOLDER, name) for name in WEIGHTS},
        columns=[paloma, score],
    )
    clean: dict[str, Docs] = {
        name: source.where(paloma, lambda matches: matches == 0)
        for name, source in sources.items()
    }

    # Check every source before selecting any, so one run lists all that fall short
    too_small = []
    loose = []

    for name, documents in clean.items():
        source = sources[name]
        pool_tokens = math.ceil(documents.tokens * fraction)
        needed = budgets[name] / documents.tokens
        print(
            f"{name}: {documents.tokens:,} of {source.tokens:,} tokens clean "
            f"({1 - documents.tokens / source.tokens:.3%} removed), "
            f"top {fraction:.0%} is {pool_tokens:,}, needs {budgets[name]:,} "
            f"({needed:.1%} of clean)",
            flush=True,
        )

        if documents.tokens < budgets[name]:
            too_small.append(
                f"{name} ({source.store.uri}): {documents.tokens:,} clean tokens, "
                f"needs {budgets[name]:,}"
            )
        elif pool_tokens < budgets[name]:
            loose.append(name)

    if too_small:
        raise ValueError(
            "sources too small for their share even with every clean document; "
            "prepare them with a larger --token-budget:\n  " + "\n  ".join(too_small)
        )

    context.phase("main")
    warnings = []

    # In WEIGHTS order: top and sample share one random generator
    for name in WEIGHTS:
        documents = clean[name]

        if name in loose:
            chosen = documents.top(budgets[name], by=score)
            warnings.append(
                f"{name}: top {fraction:.0%} of clean tokens is "
                f"{math.ceil(documents.tokens * fraction):,}, short of its "
                f"{budgets[name]:,}; took the top "
                f"{chosen.tokens / documents.tokens:.1%} instead"
            )
        else:
            pool = documents.top(math.ceil(documents.tokens * fraction), by=score)
            chosen = pool.sample(budgets[name])

        print(
            f"{name}: {chosen.tokens:,} tokens, {chosen.documents:,} documents, "
            f"{chosen.tokens / documents.tokens:.1%} of clean",
            flush=True,
        )
        context.cursor(name, chosen)

    for warning in warnings:
        print(f"warning: {warning}", flush=True)

    if warnings:
        # Budgets scale with --total-tokens, so the tightest source sets the largest
        # one that fits every source in its pool
        scale = min(
            clean[name].tokens * fraction / budget for name, budget in budgets.items()
        )
        print(
            f"warning: {len(warnings)} of {len(WEIGHTS)} sources took more than the "
            f"top {fraction:.0%}; every source fits in its top {fraction:.0%} up to "
            f"about --total-tokens {scale * context.total_tokens:,.0f}",
            flush=True,
        )


if __name__ == "__main__":
    run(compile, add_arguments)
