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

from wandb_summary import add_arguments as add_wandb_arguments
from wandb_summary import log_summary

from src.recipe.metadata import read_shard_names
from src.recipe.scorers.vban_count import FIRST_SPECIAL, token_bytes
from src.tokenizer import ShardStore, document_token_ids, load_tokenizer

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


def choose_s(
    coverage: float,
    counts: numpy.ndarray,
    shared: numpy.ndarray,
    byte_ids: numpy.ndarray,
) -> tuple[numpy.ndarray, int]:
    """
    (S as a mask over BASE_VOCAB ids, K): the K most frequent non-shared tokens that
    cover `coverage` of non-shared occurrences, plus every non-shared single byte
    """
    candidates = numpy.flatnonzero(~shared & (counts > 0))
    order = candidates[numpy.argsort(-counts[candidates], kind="stable")]
    cumulative = numpy.cumsum(counts[order])
    top = int(numpy.searchsorted(cumulative, coverage * int(cumulative[-1])) + 1)
    top = min(top, order.size)

    in_s = numpy.zeros(BASE_VOCAB, dtype=bool)
    in_s[order[:top]] = True
    in_s[byte_ids[~shared[byte_ids]]] = True
    return in_s, top


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
    in_s, top = choose_s(coverage, counts, shared, byte_ids)
    rare = numpy.flatnonzero(~shared & ~in_s & (counts > 0))
    lengths = numpy.array([len(spelled[index] or b"") for index in rare])
    extra = int((counts[rare] * (lengths - 1)).sum())
    all_tokens = int(counts.sum())
    size = int(in_s.sum())

    return {
        "coverage": coverage,
        "top_tokens": top,
        "byte_tokens": size - top,
        "size_s": size,
        "vocab_size": BASE_VOCAB + size,
        "vocab_size_128": padded(BASE_VOCAB + size),
        "rare_token_types": int(rare.size),
        "rare_occurrence_fraction": float(counts[rare].sum() / all_tokens),
        "length_increase": extra / all_tokens,
        "added_params_d1024_m": size * 1024 / 1e6,
    }


def padded(vocab_size: int) -> int:
    """
    Rounded up to a multiple of 128, for the tensor cores (plan §3)
    """
    return -(-vocab_size // 128) * 128


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
    log_summary(
        arguments,
        {
            "tokens": all_tokens,
            "shared_occurrence_fraction": report["shared_occurrence_fraction"],
            **{
                f"coverage_{point['coverage']}/{key}": value
                for point in points
                for key, value in point.items()
                if key != "coverage"
            },
        },
    )
    print("coverage  |S|     vocab(128)  rare occ.  length +  params(d=1024)")

    for point in points:
        print(
            f"{point['coverage']:<8}  {point['size_s']:<6}  "
            f"{point['vocab_size_128']:<10}  {point['rare_occurrence_fraction']:.3%}"
            f"     {point['length_increase']:.3%}    {point['added_params_d1024_m']:.1f}M"
        )


def expansion(
    in_s: numpy.ndarray,
    shared: numpy.ndarray,
    byte_ids: numpy.ndarray,
    spelled: list[bytes | None],
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    How every original id is written in Fake-French, as CSR arrays: id t becomes
    flat[start[t] : start[t] + length[t]]. Shared ids map to themselves, ids in S to
    BASE_VOCAB + their rank in S (S sorted by original id), anything else to its bytes'
    single-byte tokens, each mapped the same way
    """
    shifted = numpy.full(BASE_VOCAB, -1, dtype=numpy.int64)
    members = numpy.flatnonzero(in_s)
    shifted[members] = BASE_VOCAB + numpy.arange(members.size)

    def single(index: int) -> int:
        return index if shared[index] else int(shifted[index])

    pieces: list[list[int]] = []

    for index in range(BASE_VOCAB):
        if shared[index] or in_s[index]:
            pieces.append([single(index)])
        else:
            pieces.append(
                [single(int(byte_ids[byte])) for byte in spelled[index] or b""]
            )

    length = numpy.array([len(piece) for piece in pieces], dtype=numpy.int64)
    start = numpy.concatenate([[0], numpy.cumsum(length)[:-1]])
    flat = numpy.array(
        [value for piece in pieces for value in piece], dtype=numpy.int32
    )

    if (flat < 0).any():
        raise ValueError("a non-shared byte token is missing from S")

    return flat, start, length


def translate(
    tokens: numpy.ndarray,
    offsets: numpy.ndarray,
    table: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray],
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    One shard (or eval set) in Fake-French: the new tokens and document offsets
    """
    flat, start, length = table
    lengths = length[tokens]
    ends = numpy.cumsum(lengths)
    total = int(ends[-1]) if ends.size else 0
    # For output position j of input token i: start[t_i] + (j - first output of i)
    firsts = numpy.repeat(ends - lengths, lengths)
    source = numpy.repeat(start[tokens], lengths) + numpy.arange(total) - firsts
    new_tokens = flat[source]
    new_offsets = numpy.concatenate([[0], ends])[offsets]
    return new_tokens, new_offsets


def l_train_counts(stores: list[str], shards: int) -> numpy.ndarray:
    """
    Token counts over the first shards of every L_train store, for the BLI answers
    """
    counts = numpy.zeros(BASE_VOCAB, dtype=numpy.int64)

    for uri in stores:
        store = ShardStore(uri)
        counts += count_tokens(store, read_shard_names(store)[:shards])

    return counts


def build(arguments: argparse.Namespace) -> None:
    source = ShardStore(arguments.store)
    metadata = source.read_json("metadata.json") or {}
    manifest = source.read_json("manifest.json") or {}
    tokenizer = str(metadata["tokenizer"])
    spelled, shared, byte_ids = vocabulary(tokenizer)
    names = read_shard_names(source)
    counts = count_tokens(source, names[: arguments.sample_shards])
    in_s, top = choose_s(arguments.coverage, counts, shared, byte_ids)
    table = expansion(in_s, shared, byte_ids, spelled)
    members = numpy.flatnonzero(in_s)
    vocab_size = BASE_VOCAB + members.size

    folder = Path(arguments.table)
    folder.mkdir(parents=True, exist_ok=True)
    # BLI answers: each shifted id's original token, where L_train uses that token
    seen = l_train_counts(arguments.l_train_store, arguments.sample_shards) > 0
    answers = members[seen[members]]
    numpy.savez(
        folder / "fake_french.npz",
        members=members,
        shifted=BASE_VOCAB + numpy.arange(members.size),
        shared=shared,
        flat=table[0],
        start=table[1],
        length=table[2],
        bli_shifted=BASE_VOCAB + numpy.searchsorted(members, answers),
        bli_original=answers,
    )
    info = {
        "source_store": arguments.store,
        "tokenizer": tokenizer,
        "coverage": arguments.coverage,
        "sample_shards": names[: arguments.sample_shards],
        "top_tokens": top,
        "size_s": int(members.size),
        "vocab_size": vocab_size,
        "vocab_size_padded": padded(vocab_size),
        "base_vocab": BASE_VOCAB,
        "bli_pairs": int(answers.size),
        "l_train_stores": arguments.l_train_store,
    }
    (folder / "fake_french.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)

    # The store: every French document, translated; row_ids still point at the French
    # Lance rows, so each document traces back to its source
    target = ShardStore(arguments.output)
    shards = []

    for name in names:
        tokens, offsets = translate(
            source.read_tokens(name), source.read_offsets(name), table
        )
        target.write_shard(name, tokens, offsets, source.read_row_ids(name))
        shards.append({"name": name, "tokens": int(tokens.size)})

    target.write_json(
        "metadata.json",
        {
            **metadata,
            "fake_french": {k: info[k] for k in ("coverage", "size_s", "vocab_size")},
        },
    )
    target.write_json(
        "manifest.json",
        {
            **manifest,
            "total_tokens": sum(shard["tokens"] for shard in shards),
            "shards": shards,
        },
    )
    print(
        f"wrote {arguments.output}: {len(shards)} shards, "
        f"{sum(shard['tokens'] for shard in shards):,} tokens",
        flush=True,
    )
    log_summary(
        arguments,
        {
            **{
                key: value for key, value in info.items() if not isinstance(value, list)
            },
            "store_tokens": sum(shard["tokens"] for shard in shards),
            "source_tokens": int(manifest.get("total_tokens", 0)),
        },
    )

    for path in arguments.eval_text:
        texts = [
            json.loads(line)["text"] for line in Path(path).read_text().splitlines()
        ]
        encoded = load_tokenizer(tokenizer)(texts, add_special_tokens=False)[
            "input_ids"
        ]
        start_id, end_id = document_token_ids(load_tokenizer(tokenizer))
        documents = [
            numpy.array([start_id, *ids, end_id], dtype=numpy.int32) for ids in encoded
        ]
        offsets = numpy.concatenate([[0], numpy.cumsum([d.size for d in documents])])
        tokens, new_offsets = translate(numpy.concatenate(documents), offsets, table)
        output = folder / (Path(path).stem.replace("fra_Latn", "fake_fra") + ".npz")
        numpy.savez(
            output,
            tokens=tokens,
            offsets=new_offsets,
            # BPB's denominator: the original French text's UTF-8 bytes (plan §3)
            utf8_bytes=numpy.array([len(text.encode()) for text in texts]),
        )
        print(f"wrote {output}: {len(texts)} documents", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    measure_parser = commands.add_parser("measure")
    measure_parser.add_argument("--store", required=True)
    measure_parser.add_argument("--output", required=True)
    measure_parser.add_argument("--sample-shards", type=int, default=40)
    add_wandb_arguments(measure_parser, run_name_required=False)
    build_parser = commands.add_parser("build")
    add_wandb_arguments(build_parser, run_name_required=False)
    build_parser.add_argument("--store", required=True, help="the French store")
    build_parser.add_argument("--coverage", required=True, type=float)
    build_parser.add_argument("--output", required=True, help="the fake_fra store")
    build_parser.add_argument("--table", required=True, help="folder for the table")
    build_parser.add_argument(
        "--l-train-store", action="append", required=True, help="for BLI answers"
    )
    build_parser.add_argument(
        "--eval-text", action="append", default=[], help="French jsonl to translate"
    )
    build_parser.add_argument("--sample-shards", type=int, default=40)
    # f1-cpu.sbatch passes --workers right after the script name
    parser.add_argument("--workers", type=int, default=1, help="unused")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.command == "measure":
        measure(arguments)
    else:
        build(arguments)


if __name__ == "__main__":
    main()
