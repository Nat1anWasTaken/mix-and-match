"""
Count each document's V_ban tokens, the token-level half of the embedding-ml-transfer
leakage filter, reading the tokens prepare.py stored (not the text): what this checks
is exactly what training reads.

    uv run python -m src.recipe.scorers.vban_count \\
        --store /work1/$USER/stores/fineweb-2-spa_Latn-qwen3 \\
        --run-name score-vban_count-fineweb-2-spa_Latn-qwen3

    uv run python -m src.recipe.scorers.vban_count --write-table vban/ \\
        --tokenizer Qwen/Qwen3-0.6B-Base       # just the table, for the audit and evals

V_ban (plan §3) is the set of Chinese tokens, defined on bytes: Qwen3 is byte-level BPE,
so a token can hold half a character, and tokenizer.decode([id]) turns that into U+FFFD.
Every BPE token (ids below the first special token, 151,643) gets one category from its
bytes, which byte_categories() computes and --write-table saves:

    BAN           holds a UTF-8 lead byte (or lead + second byte) of a CJK range:
                  E3-E9 (U+3000-U+9FFF: CJK punctuation, kana, Han, extension A),
                  E2 BA-BF (U+2E80-U+2FFF: radicals, Kangxi), EF A4-AB (U+F900-U+FAFF:
                  compatibility ideographs), EF BC-BD (U+FF00-U+FF7F: fullwidth forms,
                  halfwidth katakana) and F0 A0-B3 (Han extensions B-H)
    CONTINUATION  only continuation bytes (80-BF): the tail of some multi-byte
                  character, shared by every script that uses that byte. Not banned; the
                  audit counts them on their own (plan: "列為共用 token 另外統計")
    SHARED        everything else

A complete Han character starts with its lead byte, so any token holding one is BAN
whatever comes after it. Ids from 151,643 up (special tokens and padding rows) are
SPECIAL and never banned: <|endoftext|> separates every document.

The process stage writes meta/vban_count/ (uint32, the BAN tokens per document).
Training splits keep only documents with vban_count == 0. Takes --stage like the other
scorers; fetch does nothing (it reads the store only), upload copies the store.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy

from ...stages import add_stage_arguments, check_stage, upload_store
from ...tokenizer import ShardStore, load_tokenizer
from ..compile import compiler_record
from ..metadata import (
    META,
    ensure_info,
    has_column,
    is_complete,
    read_shard_names,
    write_column,
)

KIND = "vban_count"
# The first special token of Qwen3; everything from here up is never banned
FIRST_SPECIAL = 151643

SHARED, BAN, CONTINUATION, SPECIAL = 0, 1, 2, 3
CATEGORY_NAMES = {SHARED: "shared", BAN: "ban", CONTINUATION: "continuation",
                  SPECIAL: "special"}  # fmt: skip


def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2's byte <-> printable character map, which byte-level BPE vocabularies use to
    spell tokens (e.g. "Ġ" is the space byte 0x20)
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    characters = list(printable)
    extra = 0

    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            characters.append(256 + extra)
            extra += 1

    return dict(zip(printable, map(chr, characters), strict=True))


def token_bytes(tokenizer_name: str) -> list[bytes | None]:
    """
    The bytes of every token id below FIRST_SPECIAL; None for an id with no BPE token
    """
    tokenizer = load_tokenizer(tokenizer_name)
    decoder = {character: byte for byte, character in bytes_to_unicode().items()}
    spelled: list[bytes | None] = [None] * FIRST_SPECIAL

    for token, index in tokenizer.get_vocab().items():
        if index < FIRST_SPECIAL:
            spelled[index] = bytes(decoder[character] for character in token)

    return spelled


def is_ban(data: bytes) -> bool:
    for position, byte in enumerate(data):
        following = data[position + 1] if position + 1 < len(data) else None

        if 0xE3 <= byte <= 0xE9:
            return True

        if following is None:
            continue

        if byte == 0xE2 and 0xBA <= following <= 0xBF:
            return True

        if byte == 0xEF and (0xA4 <= following <= 0xAB or 0xBC <= following <= 0xBD):
            return True

        if byte == 0xF0 and 0xA0 <= following <= 0xB3:
            return True

    return False


def categorize(data: bytes | None) -> int:
    if data is None:
        return SPECIAL

    if is_ban(data):
        return BAN

    if data and all(0x80 <= byte <= 0xBF for byte in data):
        return CONTINUATION

    return SHARED


def byte_categories(tokenizer_name: str, vocab_size: int) -> numpy.ndarray:
    """
    uint8 category per token id in [0, vocab_size), ids from FIRST_SPECIAL up SPECIAL
    """
    categories = numpy.full(vocab_size, SPECIAL, dtype=numpy.uint8)

    for index, data in enumerate(token_bytes(tokenizer_name)):
        categories[index] = categorize(data)

    return categories


def count_shard(store: ShardStore, name: str, banned: numpy.ndarray) -> numpy.ndarray:
    """
    The BAN tokens of each document of one shard
    """
    tokens = store.read_tokens(name)
    offsets = store.read_offsets(name)

    if tokens.size and int(tokens.max()) >= banned.size:
        raise ValueError(
            f"shard {name} holds token {int(tokens.max())}, past the table's "
            f"{banned.size} ids; pass a larger --vocab-size"
        )

    # Every document holds at least its two separators, so no segment is empty and
    # reduceat's "empty segment gives the next value" quirk never applies
    per_token = banned[tokens].astype(numpy.uint32)
    return numpy.add.reduceat(per_token, offsets[:-1]).astype(numpy.uint32)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--store", help="shard store from prepare.py")
    parser.add_argument("--write-table", help="write vban.npy + categories.npy here")
    parser.add_argument("--tokenizer", help="default: the store's")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=151936,
        help="ids the table covers; track 1's extended vocab is larger",
    )
    parser.add_argument("--run-name", help="score-vban_count-<store name>")
    parser.add_argument("--workers", type=int, default=1, help="unused; for sbatch")
    parser.add_argument("--no-wandb", action="store_true", help="unused; console only")
    add_stage_arguments(parser)
    arguments = parser.parse_args()

    if (arguments.store is None) == (arguments.write_table is None):
        parser.error("pass exactly one of --store and --write-table")

    return arguments


def write_table(arguments: argparse.Namespace) -> None:
    if arguments.tokenizer is None:
        raise SystemExit("--write-table needs --tokenizer")

    categories = byte_categories(arguments.tokenizer, arguments.vocab_size)
    folder = Path(arguments.write_table)
    folder.mkdir(parents=True, exist_ok=True)
    numpy.save(folder / "categories.npy", categories)
    numpy.save(folder / "vban.npy", numpy.flatnonzero(categories == BAN))

    counts = {
        name: int((categories == value).sum()) for value, name in CATEGORY_NAMES.items()
    }
    print(f"{folder}: {counts}")


def main() -> None:
    arguments = parse_arguments()

    if arguments.write_table is not None:
        write_table(arguments)
        return

    check_stage(arguments, "store")
    store = ShardStore(arguments.store)

    if arguments.stage == "fetch":
        return

    if arguments.stage == "upload":
        if not is_complete(store, KIND):
            raise FileNotFoundError(f"{store.uri} lacks shards of meta/{KIND}")

        upload_store(store.uri, arguments.remote)
        return

    metadata = store.read_json("metadata.json") or {}
    tokenizer = arguments.tokenizer or str(metadata["tokenizer"])
    info = {
        "scorer": KIND,
        "tokenizer": tokenizer,
        "vocab_size": arguments.vocab_size,
        "first_special": FIRST_SPECIAL,
    }
    ensure_info(store, KIND, info)

    if is_complete(store, KIND):
        print(f"{store.uri} already has every shard of meta/{KIND}", flush=True)
        return

    start = time.monotonic()
    banned = byte_categories(tokenizer, arguments.vocab_size) == BAN
    total_documents = flagged = total_banned = 0

    for name in read_shard_names(store):
        if has_column(store, KIND, name):
            continue

        counts = count_shard(store, name, banned)
        write_column(store, KIND, name, counts)
        total_documents += counts.size
        flagged += int(numpy.count_nonzero(counts))
        total_banned += int(counts.sum())

    path = Path(__file__).resolve()
    store.write_json(
        f"{META}/{KIND}/record.json", compiler_record(main, path, arguments)
    )
    store.write_text(f"{META}/{KIND}/scorer.py", path.read_text())
    print(
        f"meta/{KIND}: documents={total_documents} with_ban={flagged} "
        f"ban_tokens={total_banned} seconds={time.monotonic() - start:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
