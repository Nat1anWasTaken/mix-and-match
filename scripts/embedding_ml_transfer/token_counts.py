"""
Per-token counts of a recipe's training set, for the masked softmax (plan Phase 1 design
revision "訓練期 softmax 遮罩"): train.py --softmax-keep takes the ids counted > 0.

    python scripts/embedding_ml_transfer/token_counts.py \\
        --recipe s3://embedding-ml-transfer/recipes/multi-train-20b-v2 \\
        --recipe-cache /work/u9216411/acd115198/recipe-cache \\
        --vocab-size 181632 --output <dir>/multi-train-20b-v2

Reads the local copy of the recipe (src.recipe.cache, the same files train.py
--recipe-cache reads) and counts every token of every selected document of every
cursor: the whole recipe, a superset of the share a run actually consumes. A token the
run never reaches only stays an ordinary candidate in the softmax, so the superset is the
safe side. Writes <output>.npz: counts [vocab] int64, keep (the ids counted > 0) int64,
and <output>.json with the totals.
"""

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.recipe.cache import localize_recipe
from src.recipe.recipe import load_recipe
from src.tokenizer import ShardStore

_vocab = 0


def init(vocab: int) -> None:
    global _vocab
    _vocab = vocab


def count_shard(task: tuple[str, str, str]) -> numpy.ndarray:
    """
    bincount of the tokens of the selected documents of one shard
    """
    store_path, shard, selection_path = task
    store = ShardStore(store_path)
    tokens = store.read_tokens(shard)
    offsets = store.read_offsets(shard)
    selected = numpy.load(selection_path)
    # +1 at each selected document's start, -1 at its stop: the running sum is > 0
    # exactly on the tokens inside a selected document
    edges = numpy.zeros(tokens.size + 1, dtype=numpy.int32)
    numpy.add.at(edges, offsets[selected], 1)
    numpy.add.at(edges, offsets[selected + 1], -1)
    chosen = tokens[numpy.cumsum(edges[:-1]) > 0]
    if chosen.size and int(chosen.max()) >= _vocab:
        raise ValueError(f"{store_path}/{shard}: id {int(chosen.max())} >= {_vocab}")
    return numpy.bincount(chosen, minlength=_vocab).astype(numpy.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recipe", required=True, help="the s3:// recipe URI")
    parser.add_argument("--recipe-cache", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True, help="<dir>/<name>")
    arguments = parser.parse_args()

    recipe = localize_recipe(load_recipe(arguments.recipe), arguments.recipe_cache)
    tasks = [
        (cursor.store, shard.name, f"{recipe.uri}/{shard.selection}")
        for phase in recipe.phases
        for cursor in phase.cursors
        for shard in cursor.shards
    ]
    counts = numpy.zeros(arguments.vocab_size, dtype=numpy.int64)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        arguments.workers, mp_context=context, initializer=init,
        initargs=(arguments.vocab_size,),
    ) as pool:
        for done, part in enumerate(pool.map(count_shard, tasks, chunksize=4), 1):
            counts += part
            if done % 200 == 0:
                print(f"{done}/{len(tasks)} shards", flush=True)

    keep = numpy.flatnonzero(counts > 0).astype(numpy.int64)
    info = {
        "recipe": arguments.recipe,
        "vocab_size": arguments.vocab_size,
        "shards": len(tasks),
        "tokens": int(counts.sum()),
        "kept": int(keep.size),
        "dropped": int(arguments.vocab_size - keep.size),
        "kept_below_151643": int(numpy.count_nonzero(keep < 151643)),
        "kept_from_151643": [int(i) for i in keep[keep >= 151643]],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez(f"{arguments.output}.npz", counts=counts, keep=keep)
    Path(f"{arguments.output}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
