"""
The baseline recipe: one source, one phase, a uniform random sample of its documents.

    uv run python -m src.recipe.compilers.identity \\
        --store s3://<bucket>/fineweb-edu-llama3 --source fwedu \\
        --total-tokens 30000000000 --output s3://<bucket>/recipes/fwedu-identity-30b

This is what the old TokenStream trained on, as a recipe. Without --total-tokens it
keeps every document. Unlike the other compilers it holds no rules of its own, so it
takes the store as an argument and serves any source.
"""

import argparse

from ..compile import Context, run


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", required=True, help="shard store from prepare.py")
    parser.add_argument("--source", required=True, help="source name in the recipe")


def compile(context: Context) -> None:
    name = context.arguments.source
    documents = context.open(name, context.arguments.store).all()

    if context.total_tokens is not None:
        documents = documents.sample(context.budget())

    context.phase("main")
    context.cursor(name, documents)


if __name__ == "__main__":
    run(compile, add_arguments)
