"""
The Pile (uncopyrighted) in its natural proportions: every clean, deduplicated document
of the store, no sampling and no per-subset weights.

    uv run python -m src.recipe.compilers.pile \\
        --store s3://<bucket>/sources/pile-uncopyrighted-llama3 \\
        --output s3://<bucket>/recipes/pile-uncopyrighted

Run src.recipe.scorers.paloma_decontamination and src.recipe.scorers.text_hash on the
store first. prepare.py took a uniform random sample of the dataset's fragments, so the
store already holds the Pile's subsets in their natural proportions; keeping every
remaining document keeps them.

Two rules, in this order:
  1. Paloma decontamination (Paloma App. C.1.1): drop every document with a paragraph
     that matches Paloma, and every copy of it too, i.e. every document with the same
     text_hash. A copy can differ in case or whitespace, which the hash ignores but the
     Paloma match does not, so it could pass the match on its own
  2. exact dedup: of the documents left, keep one per text_hash, the first in the
     store's order. prepare.py shuffles documents uniformly into shards, so the first
     is a random one of its copies, and copies are the same text anyway

The recipe holds every token that is left, so it sets train.py's budget instead of
following one: --total-tokens is only checked, to fail early if the store is too
small for it (plus --margin). Without it the recipe is just everything that is left.
"""

import argparse

import numpy

from ..compile import Context, Docs, Source, run

SOURCE = "pile"
PALOMA = "paloma_decontamination"
HASH = "text_hash"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", required=True, help="shard store from prepare.py")


def clean_unique(source: Source) -> Docs:
    """
    The documents that pass both rules, as one selection over every shard
    """
    names = [name for name, shard in source.lengths.items() if shard.size]
    hashes_by_shard = source.column(HASH)
    paloma_by_shard = source.column(PALOMA)

    # One global index over the shards in order: shard i's documents are
    # starts[i] .. starts[i + 1]
    sizes = numpy.array([source.lengths[name].size for name in names])
    starts = numpy.concatenate([[0], numpy.cumsum(sizes)])
    hashes = numpy.concatenate([hashes_by_shard[name] for name in names])
    matches = numpy.concatenate([paloma_by_shard[name] for name in names])

    if hashes.shape != matches.shape or hashes.size != starts[-1]:
        raise ValueError(
            f"{HASH} ({hashes.size}), {PALOMA} ({matches.size}) and the shards "
            f"({starts[-1]}) disagree on the number of documents"
        )

    # Rule 1: every copy of a contaminated document goes too
    contaminated = numpy.isin(hashes, numpy.unique(hashes[matches > 0]))
    candidates = numpy.flatnonzero(~contaminated)

    # Rule 2: unique's return_index is each hash's first position in candidates,
    # which is in store order
    _, first = numpy.unique(hashes[candidates], return_index=True)
    kept = numpy.sort(candidates[first])

    print(
        f"{source.name}: {hashes.size:,} documents, "
        f"{int(numpy.count_nonzero(matches)):,} match Paloma, "
        f"{int(numpy.count_nonzero(contaminated)):,} dropped with their copies, "
        f"{candidates.size - kept.size:,} duplicates dropped, {kept.size:,} kept",
        flush=True,
    )

    # Back to per-shard indices into each shard's stored order
    bounds = numpy.searchsorted(kept, starts)
    selections = {
        name: (kept[bounds[i] : bounds[i + 1]] - starts[i]).astype(numpy.uint32)
        for i, name in enumerate(names)
        if bounds[i + 1] > bounds[i]
    }
    return Docs(source, selections)


def compile(context: Context) -> None:
    source = context.open(SOURCE, context.arguments.store, columns=[PALOMA, HASH])
    documents = clean_unique(source)
    print(
        f"{SOURCE}: {documents.tokens:,} of {source.tokens:,} tokens kept "
        f"({1 - documents.tokens / source.tokens:.3%} removed)",
        flush=True,
    )

    if context.total_tokens is not None and documents.tokens < context.target:
        raise ValueError(
            f"{source.store.uri} has {documents.tokens:,} clean unique tokens, "
            f"needs {context.target:,.0f} for --total-tokens {context.total_tokens:,} "
            "with its margin; prepare it with a larger --token-budget"
        )

    context.phase("main")
    context.cursor(SOURCE, documents)


if __name__ == "__main__":
    run(compile, add_arguments)
