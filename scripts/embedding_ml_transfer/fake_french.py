"""
Fake-French (plan §3): French whose tokens are moved to new ids, so it is the same
language as French but shares no token with any training language except a fixed set
of anchors. Two steps, run on a compute node against the French store:

    # 1. how many new ids each coverage threshold needs, and how much longer the text
    #    gets; pick a threshold from this
    python scripts/embedding_ml_transfer/fake_french.py measure \\
        --store /work1/$USER/stores/embedding-ml-transfer/fineweb-2-fra_Latn-qwen3 \\
        --output /work1/$USER/p0-1/fake-french-measure.json

    # 2. (once a threshold is chosen) the shift table and the fake_fra store
    python scripts/embedding_ml_transfer/fake_french.py build --coverage 0.999 ...

How a French token t is mapped:

    SHARED  stays t. Special tokens (<|endoftext|> separates every document) and every
            token whose bytes are all ASCII digits, whitespace or ASCII punctuation: the
            same anchors real Chinese shares with English, so the two conditions compare
    in S    becomes 151,936 + idx(t). S is the K most frequent non-shared French tokens,
            K the smallest count covering --coverage of non-shared French occurrences,
            plus every non-shared single-byte token
    rare    (non-shared, not in S) is spelled out as its bytes, and each byte token is
            mapped like any other token (shifted, or kept when it is a shared anchor)

measure counts tokens over --sample-shards shards. The store is globally shuffled, so
any shards are a random sample of its documents; the first ones are used and recorded.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.recipe.metadata import read_shard_names
from src.recipe.scorers.vban_count import FIRST_SPECIAL, token_bytes
from src.tokenizer import ShardStore

BASE_VOCAB = 151936
COVERAGES = [0.99, 0.995, 0.999, 0.9995, 0.9999]
# ASCII whitespace, digits and punctuation: the anchors every condition shares
ANCHOR_BYTES = frozenset(
    b"\t\n\x0b\x0c\r 0123456789"
    + bytes(range(0x21, 0x30))
    + bytes(range(0x3A, 0x41))
    + bytes(range(0x5B, 0x61))
    + bytes(range(0x7B, 0x7F))
)


def vocabulary(
    tokenizer: str,
) -> tuple[list[bytes | None], numpy.ndarray, numpy.ndarray]:
    """
    (bytes of each id, shared mask over BASE_VOCAB ids, id of each single byte value)
    """
    spelled = token_bytes(tokenizer)
    shared = numpy.ones(BASE_VOCAB, dtype=bool)
    byte_ids = numpy.full(256, -1, dtype=numpy.int64)

    for index, data in enumerate(spelled):
        if data is None:
            continue

        shared[index] = bool(data) and all(byte in ANCHOR_BYTES for byte in data)

        if len(data) == 1:
            byte_ids[data[0]] = index

    missing = numpy.flatnonzero(byte_ids < 0)

    if missing.size:
        raise ValueError(f"no single-byte token for bytes {missing.tolist()}")

    # Ids from FIRST_SPECIAL up are special tokens or padding: never shifted
    shared[FIRST_SPECIAL:] = True
    return spelled, shared, byte_ids


def count_tokens(store: ShardStore, shards: list[str]) -> numpy.ndarray:
    counts = numpy.zeros(BASE_VOCAB, dtype=numpy.int64)

    for name in shards:
        counts += numpy.bincount(store.read_tokens(name), minlength=BASE_VOCAB)

    return counts


def coverage_point(
    coverage: float,
    counts: numpy.ndarray,
    shared: numpy.ndarray,
    byte_ids: numpy.ndarray,
    spelled: list[bytes | None],
) -> dict[str, float | int]:
    """
    |S| and the cost of the rare tokens at one coverage threshold
    """
    candidates = numpy.flatnonzero(~shared & (counts > 0))
    order = candidates[numpy.argsort(-counts[candidates], kind="stable")]
    cumulative = numpy.cumsum(counts[order])
    total = int(cumulative[-1])
    top = int(numpy.searchsorted(cumulative, coverage * total) + 1)
    top = min(top, order.size)

    in_s = numpy.zeros(BASE_VOCAB, dtype=bool)
    in_s[order[:top]] = True
    single_bytes = byte_ids[~shared[byte_ids]]
    in_s[single_bytes] = True

    rare = order[top:]
    rare = rare[~in_s[rare]]
    lengths = numpy.array([len(spelled[index] or b"") for index in rare])
    extra = int((counts[rare] * (lengths - 1)).sum())
    all_tokens = int(counts.sum())

    return {
        "coverage": coverage,
        "top_tokens": top,
        "byte_tokens": int(single_bytes.size),
        "size_s": int(in_s.sum()),
        "vocab_size": BASE_VOCAB + int(in_s.sum()),
        "vocab_size_128": -(-(BASE_VOCAB + int(in_s.sum())) // 128) * 128,
        "rare_token_types": int(rare.size),
        "rare_occurrence_fraction": float(counts[rare].sum() / all_tokens),
        "length_increase": extra / all_tokens,
        "added_params_d1024_m": int(in_s.sum()) * 1024 / 1e6,
    }


def measure(arguments: argparse.Namespace) -> None:
    store = ShardStore(arguments.store)
    tokenizer = str((store.read_json("metadata.json") or {})["tokenizer"])
    spelled, shared, byte_ids = vocabulary(tokenizer)
    shards = read_shard_names(store)[: arguments.sample_shards]
    counts = count_tokens(store, shards)
    all_tokens = int(counts.sum())

    points = [
        coverage_point(coverage, counts, shared, byte_ids, spelled)
        for coverage in COVERAGES
    ]
    report = {
        "store": arguments.store,
        "tokenizer": tokenizer,
        "shards": shards,
        "tokens": all_tokens,
        "shared_occurrence_fraction": float(counts[shared].sum() / all_tokens),
        "distinct_non_shared": int(((counts > 0) & ~shared).sum()),
        "points": points,
    }
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    Path(arguments.output).write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"{all_tokens:,} tokens from {len(shards)} shards; "
        f"{report['shared_occurrence_fraction']:.1%} shared anchors"
    )
    print("coverage  |S|     vocab(128)  rare occ.  length +  params(d=1024)")

    for point in points:
        print(
            f"{point['coverage']:<8}  {point['size_s']:<6}  "
            f"{point['vocab_size_128']:<10}  {point['rare_occurrence_fraction']:.3%}"
            f"     {point['length_increase']:.3%}    {point['added_params_d1024_m']:.1f}M"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--store", required=True)
    measure_parser.add_argument("--output", required=True)
    measure_parser.add_argument("--sample-shards", type=int, default=40)
    # f1-cpu.sbatch passes --workers right after the script name
    parser.add_argument("--workers", type=int, default=1, help="unused")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.command == "measure":
        measure(arguments)


if __name__ == "__main__":
    main()
