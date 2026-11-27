"""
Dolma v1.7 with other sources mixed in at fixed token fractions, as DataDecide's
"DCLM-Baseline 25% / Dolma 75%" recipes: each --add source gets its fraction of the
tokens, and Dolma v1.7's 19 sources share the rest in dolma_v1_7.WEIGHTS proportions.

    uv run python -m src.recipe.compilers.dolma_v1_7_mix \\
        --store-template 's3://mix-and-match/sources/dolma-v1_7-{source}-llama3' \\
        --add dclm s3://mix-and-match/sources/dclm-baseline-llama3 0.25 \\
        --total-tokens 30000000000 --output s3://mix-and-match/recipes/dolma-v1_7-dclm0.25-30b

--add NAME STORE FRACTION can be given several times. Fractions are of the recipe's
tokens and must add up to at most 1. At exactly 1 no Dolma source is used, so

    --add dclm s3://mix-and-match/sources/dclm-baseline-llama3 1

is DCLM-Baseline alone (no --store-template needed). With no --add this is
dolma_v1_7.py's recipe: the same budgets, sampled in the same order from the same
generator, so the same stores and --seed give the same documents.

Every source, added ones too, is filtered the way dolma_v1_7.py filters: only documents
with no Paloma paragraph match (meta/<--kind> == 0) are kept, so score each new store
with src.recipe.scorers.paloma_decontamination first. A source with too few clean
tokens for its share is an error, as in dolma_v1_7.py.
"""

import argparse
import math

from ..compile import Context, Docs, run
from .dolma_v1_7 import PLACEHOLDER, WEIGHTS

# Fractions that add up to 1 up to float rounding (0.1 + 0.2 + 0.7) count as 1
TOLERANCE = 1e-9


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store-template",
        help=f"store URI of every Dolma source, with {PLACEHOLDER} in it; "
        "needed unless the --add fractions add up to 1",
    )
    parser.add_argument(
        "--add",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "STORE", "FRACTION"),
        help="a source to mix in: its recipe name, shard store and fraction of tokens",
    )
    parser.add_argument(
        "--kind",
        default="paloma_decontamination",
        help="meta/<kind> column from the paloma_decontamination scorer",
    )


def parse_added(added: list[list[str]]) -> dict[str, tuple[str, float]]:
    """--add's NAME -> (STORE, FRACTION), checked"""
    sources: dict[str, tuple[str, float]] = {}

    for name, store, text in added:
        fraction = float(text)

        if name in WEIGHTS or name in sources:
            raise ValueError(f"--add {name}: the recipe already has a source {name}")

        # Also rejects NaN, which fails every comparison
        if not 0 < fraction <= 1:
            raise ValueError(f"--add {name}: fraction {text} is not in (0, 1]")

        sources[name] = (store, fraction)

    return sources


def compile(context: Context) -> None:
    arguments = context.arguments
    added = parse_added(arguments.add)
    rest = 1 - sum(fraction for _, fraction in added.values())

    if rest < -TOLERANCE:
        raise ValueError(f"--add fractions add up to {1 - rest:g}, more than 1")

    budgets: dict[str, int] = {}
    uris: dict[str, str] = {}

    # Dolma first, in WEIGHTS order, so that with no --add the samples come off the
    # generator exactly as in dolma_v1_7.py
    if rest > TOLERANCE:
        template = arguments.store_template

        if template is None or PLACEHOLDER not in template:
            raise ValueError(
                f"Dolma gets {rest:.3%} of the tokens: pass --store-template with "
                f"{PLACEHOLDER} in it"
            )

        # With no --add, rest is exactly 1 and this is dolma_v1_7.py's context.split
        budgets |= context.split(WEIGHTS, context.target * rest)
        uris |= {name: template.replace(PLACEHOLDER, name) for name in WEIGHTS}

    for name, (store, fraction) in added.items():
        budgets[name] = math.ceil(context.target * fraction)
        uris[name] = store

    sources = context.open_many(uris, columns=[arguments.kind])
    clean: dict[str, Docs] = {
        name: source.where(arguments.kind, lambda matches: matches == 0)
        for name, source in sources.items()
    }

    # Check every source before selecting any, so one run lists all that fall short
    short = []

    for name in budgets:
        source, documents = sources[name], clean[name]
        removed = 1 - documents.tokens / source.tokens
        print(
            f"{name}: {documents.tokens:,} of {source.tokens:,} tokens clean "
            f"({removed:.3%} removed), needs {budgets[name]:,} "
            f"({budgets[name] / context.target:.3%} of the recipe)",
            flush=True,
        )

        if documents.tokens < budgets[name]:
            short.append(
                f"{name} ({source.store.uri}): {documents.tokens:,} clean tokens, "
                f"needs {budgets[name]:,}"
            )

    if short:
        raise ValueError(
            "sources too small for their share; prepare them with a larger "
            "--token-budget:\n  " + "\n  ".join(short)
        )

    context.phase("main")

    # In budgets' order (Dolma, then --add as given): the samples share one generator
    for name in budgets:
        context.cursor(name, clean[name].sample(budgets[name]))


if __name__ == "__main__":
    run(compile, add_arguments)
