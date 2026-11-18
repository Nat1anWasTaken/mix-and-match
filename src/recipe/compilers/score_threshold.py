"""
A one-source recipe driven by a metadata column: keep the best-scored documents, repeat
the very best for free, and optionally anneal on the best at the end.

    uv run python -m src.recipe.compilers.score_threshold \\
        --store s3://<bucket>/sources/fineweb-edu-llama3 --source fwedu \\
        --kind int_score --min-score 3 --select sample \\
        --total-tokens 30000000000 --repeat-fraction 0.1 --anneal-fraction 0.2 \\
        --output s3://<bucket>/recipes/fwedu-intscore3-30b

Token budgets, with target = total_tokens * (1 + margin):
  - unique = target / (1 + repeat_fraction) documents are kept: the highest scored
    (--select top) or a uniform sample of those with score >= --min-score (--select sample)
  - the highest-scored kept documents worth repeat_fraction * unique tokens get a
    second copy in the main phase, counted as repeated tokens
  - the highest-scored kept documents worth anneal_fraction * target tokens move to a
    final anneal phase. Match it to train.py --decay-fraction, so the anneal phase is the
    part of training where the learning rate decays

Thresholds are always global. Inside each phase RecipeTokenStream mixes uniformly, so
scores never decide when a document is seen, only whether and how often.
"""

import argparse

import numpy

from ...tokenizer import ShardStore
from ..metadata import read_columns
from ..recipe import (
    MAX_REPEAT_FRACTION,
    RecipeBuilder,
    Selection,
    subtract_selections,
    take_by_priority,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", required=True, help="shard store from prepare.py")
    parser.add_argument("--source", required=True, help="source name in the recipe")
    parser.add_argument("--kind", required=True, help="meta/<kind> column to score by")
    parser.add_argument("--output", required=True, help="recipe directory")
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="extra fraction of tokens, so no rank runs out before --total-tokens",
    )
    parser.add_argument(
        "--min-score", type=float, help="drop documents scored below this"
    )
    parser.add_argument(
        "--select",
        choices=["top", "sample"],
        default="top",
        help="keep the highest-scored documents, or a uniform sample of the eligible ones",
    )
    parser.add_argument(
        "--repeat-fraction",
        type=float,
        default=0.0,
        help=f"repeated tokens as a fraction of the counted ones, at most "
        f"{MAX_REPEAT_FRACTION}",
    )
    parser.add_argument(
        "--anneal-fraction",
        type=float,
        default=0.0,
        help="fraction of training in the final anneal phase; 0 means no anneal phase. "
        "Use train.py's --decay-fraction (0.2 by default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    if not 0 <= arguments.repeat_fraction <= MAX_REPEAT_FRACTION:
        raise ValueError(f"--repeat-fraction must be in [0, {MAX_REPEAT_FRACTION}]")

    if not 0 <= arguments.anneal_fraction < 1:
        raise ValueError("--anneal-fraction must be in [0, 1)")

    store = ShardStore(arguments.store)
    builder = RecipeBuilder(seed=arguments.seed)
    lengths = builder.lengths(store)
    scores = {
        name: column.astype(numpy.float64)
        for name, column in read_columns(store, arguments.kind).items()
    }
    random_generator = numpy.random.default_rng(arguments.seed)

    target = int(arguments.total_tokens * (1 + arguments.margin))
    unique_budget = int(target / (1 + arguments.repeat_fraction))
    repeat_budget = int(arguments.repeat_fraction * unique_budget)
    anneal_budget = int(arguments.anneal_fraction * target)

    eligible: dict[str, Selection] | None = None

    if arguments.min_score is not None:
        eligible = {}

        for name, shard_scores in scores.items():
            documents = numpy.flatnonzero(shard_scores >= arguments.min_score)

            if documents.size:
                eligible[name] = documents.astype(numpy.uint32)

    if arguments.select == "top":
        priorities = scores
    else:
        priorities = {name: numpy.zeros(column.size) for name, column in scores.items()}

    kept = take_by_priority(
        priorities, lengths, unique_budget, random_generator, eligible=eligible
    )

    builder.add_phase("main")
    main_selection = kept

    if anneal_budget > 0:
        anneal = take_by_priority(
            scores, lengths, anneal_budget, random_generator, eligible=kept
        )
        main_selection = subtract_selections(kept, anneal)

    builder.add_cursor(
        "main", arguments.source, arguments.source, store, main_selection
    )

    if repeat_budget > 0:
        # Within the budget, not past it: the repeat allowance is a hard limit
        repeat = take_by_priority(
            scores,
            lengths,
            repeat_budget,
            random_generator,
            eligible=kept,
            exceed=False,
        )
        builder.add_cursor(
            "main",
            f"{arguments.source}-repeat",
            arguments.source,
            store,
            repeat,
            repeat=True,
        )

    if anneal_budget > 0:
        builder.add_phase("anneal")
        builder.add_cursor("anneal", arguments.source, arguments.source, store, anneal)

    recipe = builder.write(arguments.output)
    print(builder.summary())
    print(
        f"wrote {recipe.uri}: {recipe.total_tokens:,} tokens, "
        f"{recipe.repeated_tokens:,} repeated"
    )

    for phase in recipe.phases:
        print(
            f"phase {phase.name}: {phase.tokens / recipe.total_tokens:.1%} of training"
        )


if __name__ == "__main__":
    main()
