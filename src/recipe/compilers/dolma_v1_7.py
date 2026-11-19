"""
Dolma v1.7 as OLMo 7B-v1.7 mixed it: one phase of 19 sources in fixed token proportions,
each a uniform random sample of its clean (Paloma-decontaminated) documents.

    uv run python -m src.recipe.compilers.dolma_v1_7 \\
        --store-template 's3://<bucket>/dolma-v1_7-{source}-llama3' \\
        --total-tokens 30000000000 --output s3://<bucket>/recipes/dolma-v1_7-30b

--store-template is every source's store with {source} replaced by its name in WEIGHTS.

Each source gets ceil(weight / total weight * total_tokens * (1 + margin)) tokens.
RecipeTokenStream picks a cursor with probability proportional to its remaining
documents, so every cursor drains at the same relative rate and these shares hold
throughout training, not only on average.

Run src.recipe.scorers.paloma_decontamination on every store first. A document is dropped if any of its
paragraphs matched (Paloma App. C.1.1 removes whole documents). A source with too few
clean tokens for its share is an error, not a quietly different mixture: prepare it
again with a larger --token-budget (and a new --output).
"""

import argparse

from ..compile import Context, Docs, run

PLACEHOLDER = "{source}"

# Billions of tokens OLMo 7B-v1.7 trained on per source: the "OLMo tokens (billions)"
# column of the Dolma v1.7 summary table on huggingface.co/datasets/allenai/dolma, times
# its "Sample Proportion". Only their ratios matter.
#
#   - The card gives one row each for Dolma's CC (1,195.5 x 50% = 597.75) and CC News
#     (14.3 x 100%). They are split over cc_en_head/middle/tail and
#     cc_news_head/middle/tail by the on-disk size of each folder's Lance data on
#     flock-datasets (CC: 955.07, 1219.06, 1108.78 GB; CC News: 23.44, 10.74,
#     4.36 GB), an estimate.
#   - C4 uses 100% (138.4), not the 50% printed in its row: the card's sampled total
#     of 1,715.1 only matches C4 at 100% (at 50% the rows sum to 1,645.15).
#   - Wikipedia's 200% means two epochs in OLMo. Here it only sets the share; the
#     compiler samples unique documents.
WEIGHTS = {
    "cc_en_head": 173.9,
    "cc_en_middle": 221.97,
    "cc_en_tail": 201.89,
    "falcon-refinedweb-filtered": 456.4,
    "starcoder": 263.8,
    "c4-filtered": 138.4,
    "reddit": 79.9,
    "pes2o": 57.2,
    "redpajama-arxiv": 28.0,
    "redpajama-stackexchange": 19.6,
    "tulu_flan": 16.5,
    "cc_news_head": 8.7,
    "cc_news_middle": 3.99,
    "cc_news_tail": 1.62,
    "proof_pile_2-open_web_math": 12.6,
    "proof_pile_2-algebraic_stack": 12.6,
    "books": 5.3,
    "wikiref_megawika": 4.6,
    "wiki": 7.4,
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store-template",
        required=True,
        help=f"store URI of every source, with {PLACEHOLDER} in it",
    )
    parser.add_argument(
        "--kind",
        default="paloma_decontamination",
        help="meta/<kind> column from the paloma_decontamination scorer",
    )


def compile(context: Context) -> None:
    template = context.arguments.store_template
    kind = context.arguments.kind

    if PLACEHOLDER not in template:
        raise ValueError(f"--store-template must contain {PLACEHOLDER}")

    budgets = context.split(WEIGHTS)
    sources = context.open_many(
        {name: template.replace(PLACEHOLDER, name) for name in WEIGHTS},
        columns=[kind],
    )
    clean: dict[str, Docs] = {
        name: source.where(kind, lambda matches: matches == 0)
        for name, source in sources.items()
    }

    # Check every source before selecting any, so one run lists all that fall short
    short = []

    for name, documents in clean.items():
        source = sources[name]
        removed = 1 - documents.tokens / source.tokens
        print(
            f"{name}: {documents.tokens:,} of {source.tokens:,} tokens clean "
            f"({removed:.3%} removed), needs {budgets[name]:,}",
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

    # In WEIGHTS order: the samples share one random generator
    for name in WEIGHTS:
        context.cursor(name, clean[name].sample(budgets[name]))


if __name__ == "__main__":
    run(compile, add_arguments)
