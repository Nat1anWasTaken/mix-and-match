"""
A one-phase mixture of several sources in fixed token proportions, each a uniform random
sample of its clean (Paloma-decontaminated) documents.

    uv run python -m src.recipe.compilers.mixture \\
        --mixture src/recipe/mixtures/dolma-v1_7.json \\
        --store-template 's3://<bucket>/sources/dolma-{source}-llama3' \\
        --total-tokens 30000000000 --output s3://<bucket>/recipes/dolma-v1_7-30b

The mixture file names every source and its weight. Weights are normalized, so any unit
works. A source's store is its "store" entry, else --store-template with {source}
replaced by the source name:

    {"sources": {"wiki": {"weight": 7.4},
                 "c4-filtered": {"weight": 69.2, "store": "s3://<bucket>/sources/c4"}}}

Each source gets ceil(weight / total weight * total_tokens * (1 + margin)) tokens.
RecipeTokenStream picks a cursor with probability proportional to its remaining
documents, so every cursor drains at the same relative rate and these shares hold
throughout training, not only on average.

Run src.recipe.scorers.paloma on every store first. As in the decontaminated compiler, a
document is dropped if any of its paragraphs matched. A source with too few clean tokens
for its share is an error, not a quietly different mixture: prepare it again with a
larger --token-budget (and a new --output).
"""

import argparse
import json
import math
from dataclasses import dataclass

import numpy

from ...tokenizer import ShardStore
from ..metadata import read_columns
from ..recipe import (
    Lengths,
    RecipeBuilder,
    Selection,
    selected_tokens,
    take_by_priority,
)

PLACEHOLDER = "{source}"


@dataclass(frozen=True)
class MixtureSource:
    name: str
    store: str
    weight: float


@dataclass(frozen=True)
class CleanSource:
    source: MixtureSource
    store: ShardStore
    lengths: dict[str, Lengths]
    clean: dict[str, Selection]
    all_tokens: int
    clean_tokens: int


def load_mixture(path: str, store_template: str | None) -> list[MixtureSource]:
    with open(path) as file:
        content = json.load(file)

    entries = content.get("sources")

    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"{path} has no sources")

    if store_template is not None and PLACEHOLDER not in store_template:
        raise ValueError(f"--store-template must contain {PLACEHOLDER}")

    sources = []

    for name, entry in entries.items():
        weight = float(entry["weight"])

        # Also rejects NaN, which fails every comparison
        if not weight > 0:
            raise ValueError(f"source {name} needs a positive weight, not {weight}")

        store = entry.get("store")

        if store is None:
            if store_template is None:
                raise ValueError(f"source {name} has no store; pass --store-template")

            store = store_template.replace(PLACEHOLDER, name)

        sources.append(MixtureSource(name, store, weight))

    return sources


def read_clean(builder: RecipeBuilder, source: MixtureSource, kind: str) -> CleanSource:
    """
    The documents of a source's store with no Paloma match
    """
    store = ShardStore(source.store)
    lengths = builder.lengths(store)
    matches = read_columns(store, kind)
    clean: dict[str, Selection] = {}

    for name, shard_matches in matches.items():
        documents = numpy.flatnonzero(shard_matches == 0)

        if documents.size:
            clean[name] = documents.astype(numpy.uint32)

    return CleanSource(
        source,
        store,
        lengths,
        clean,
        all_tokens=sum(int(shard.sum()) for shard in lengths.values()),
        clean_tokens=selected_tokens(clean, lengths),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mixture", required=True, help="mixture JSON file")
    parser.add_argument(
        "--store-template",
        help=f"store URI of sources without a store entry, with {PLACEHOLDER} in it",
    )
    parser.add_argument("--output", required=True, help="recipe directory")
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="extra fraction of tokens, so no rank runs out before --total-tokens",
    )
    parser.add_argument(
        "--kind", default="paloma", help="meta/<kind> column from the paloma scorer"
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    sources = load_mixture(arguments.mixture, arguments.store_template)
    total_weight = sum(source.weight for source in sources)
    target = arguments.total_tokens * (1 + arguments.margin)
    # ceil: the budgets add up to at least the target
    budgets = {
        source.name: math.ceil(target * source.weight / total_weight)
        for source in sources
    }

    builder = RecipeBuilder(seed=arguments.seed)
    cleaned = [read_clean(builder, source, arguments.kind) for source in sources]

    # Check every source before selecting any, so one run lists all that fall short
    short = []

    for entry in cleaned:
        name = entry.source.name
        removed = 1 - entry.clean_tokens / entry.all_tokens
        print(
            f"{name}: {entry.clean_tokens:,} of {entry.all_tokens:,} tokens clean "
            f"({removed:.3%} removed), needs {budgets[name]:,}",
            flush=True,
        )

        if entry.clean_tokens < budgets[name]:
            short.append(
                f"{name} ({entry.store.uri}): {entry.clean_tokens:,} clean tokens, "
                f"needs {budgets[name]:,}"
            )

    if short:
        raise ValueError(
            "sources too small for their share; prepare them with a larger "
            "--token-budget:\n  " + "\n  ".join(short)
        )

    # One generator in mixture-file order, so the recipe is the same on every run
    random_generator = numpy.random.default_rng(arguments.seed)
    builder.add_phase("main")

    for entry in cleaned:
        # A constant priority is a uniform random sample of the clean documents
        selections = take_by_priority(
            {name: numpy.zeros(shard.size) for name, shard in entry.lengths.items()},
            entry.lengths,
            budgets[entry.source.name],
            random_generator,
            eligible=entry.clean,
        )
        builder.add_cursor(
            "main", entry.source.name, entry.source.name, entry.store, selections
        )

    recipe = builder.write(arguments.output)
    print(builder.summary())
    print(f"wrote {recipe.uri}: {recipe.total_tokens:,} tokens")

    weights = {source.name: source.weight / total_weight for source in sources}

    for cursor in recipe.phases[0].cursors:
        print(
            f"{cursor.name}: {cursor.tokens / recipe.total_tokens:.3%} of the recipe, "
            f"weight {weights[cursor.name]:.3%}"
        )


if __name__ == "__main__":
    main()
