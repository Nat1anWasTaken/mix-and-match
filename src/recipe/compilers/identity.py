"""
The baseline recipe: one source, one phase, a uniform random sample of its documents.

    uv run python -m src.recipe.compilers.identity \\
        --store s3://<bucket>/sources/fineweb-edu-llama3 --source fwedu \\
        --total-tokens 30000000000 --output s3://<bucket>/recipes/fwedu-identity-30b

This is what the old TokenStream trained on, as a recipe. Without --total-tokens it
keeps every document.
"""

import argparse

import numpy

from ...tokenizer import ShardStore
from ..recipe import RecipeBuilder, take_by_priority


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", required=True, help="shard store from prepare.py")
    parser.add_argument("--source", required=True, help="source name in the recipe")
    parser.add_argument("--output", required=True, help="recipe directory")
    parser.add_argument(
        "--total-tokens",
        type=int,
        help="train.py --total-tokens; default: every document",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="extra fraction of tokens, so no rank runs out before --total-tokens",
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    store = ShardStore(arguments.store)
    builder = RecipeBuilder(seed=arguments.seed)
    lengths = builder.lengths(store)

    if arguments.total_tokens is None:
        selections = {name: numpy.arange(shard.size) for name, shard in lengths.items()}
    else:
        budget = int(arguments.total_tokens * (1 + arguments.margin))
        # A constant priority is a uniform random sample
        selections = take_by_priority(
            {name: numpy.zeros(shard.size) for name, shard in lengths.items()},
            lengths,
            budget,
            numpy.random.default_rng(arguments.seed),
        )

    builder.add_phase("main")
    builder.add_cursor("main", arguments.source, arguments.source, store, selections)

    recipe = builder.write(arguments.output)
    print(builder.summary())
    print(f"wrote {recipe.uri}: {recipe.total_tokens:,} tokens")


if __name__ == "__main__":
    main()
