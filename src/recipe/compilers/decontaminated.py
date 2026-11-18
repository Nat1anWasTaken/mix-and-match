"""
A one-source, one-phase recipe of documents with no Paloma paragraph: a uniform random
sample of the clean documents.

    uv run python -m src.recipe.compilers.decontaminated \\
        --store s3://<bucket>/sources/fineweb-edu-llama3 --source fwedu \\
        --total-tokens 30000000000 --output s3://<bucket>/recipes/fwedu-decontam-30b

Run src.recipe.scorers.paloma on the store first. A document is dropped if any of its
paragraphs matched (Paloma App. C.1.1 removes whole documents). Without --total-tokens
it keeps every clean document.
"""

import argparse

import numpy

from ...tokenizer import ShardStore
from ..metadata import read_columns
from ..recipe import RecipeBuilder, Selection, selected_tokens, take_by_priority


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", required=True, help="shard store from prepare.py")
    parser.add_argument("--source", required=True, help="source name in the recipe")
    parser.add_argument("--output", required=True, help="recipe directory")
    parser.add_argument(
        "--kind", default="paloma", help="meta/<kind> column from the paloma scorer"
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        help="train.py --total-tokens; default: every clean document",
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
    matches = read_columns(store, arguments.kind)

    clean: dict[str, Selection] = {}

    for name, shard_matches in matches.items():
        documents = numpy.flatnonzero(shard_matches == 0)

        if documents.size:
            clean[name] = documents.astype(numpy.uint32)

    all_tokens = sum(int(shard.sum()) for shard in lengths.values())
    all_documents = sum(shard.size for shard in lengths.values())
    clean_tokens = selected_tokens(clean, lengths)
    clean_documents = sum(documents.size for documents in clean.values())
    print(
        f"removed {all_documents - clean_documents:,} of {all_documents:,} documents "
        f"({1 - clean_documents / all_documents:.3%}), "
        f"{all_tokens - clean_tokens:,} of {all_tokens:,} tokens "
        f"({1 - clean_tokens / all_tokens:.3%})"
    )

    if arguments.total_tokens is None:
        selections = clean
    else:
        budget = int(arguments.total_tokens * (1 + arguments.margin))
        # A constant priority is a uniform random sample of the clean documents
        selections = take_by_priority(
            {name: numpy.zeros(shard.size) for name, shard in lengths.items()},
            lengths,
            budget,
            numpy.random.default_rng(arguments.seed),
            eligible=clean,
        )

    builder.add_phase("main")
    builder.add_cursor("main", arguments.source, arguments.source, store, selections)

    recipe = builder.write(arguments.output)
    print(builder.summary())
    print(f"wrote {recipe.uri}: {recipe.total_tokens:,} tokens")


if __name__ == "__main__":
    main()
