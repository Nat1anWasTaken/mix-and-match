"""
Pre-tokenize FineWeb-Edu into globally shuffled shards, until a token budget.

    eval "$(flock s3 credentials --ttl 24h)"
    uv run prepare.py --dataset s3://flock-datasets/fineweb-edu \\
        --output s3://<bucket>/sources/fineweb-edu-llama3 \\
        --token-budget 30000000000 \\
        --fragments /work/<user>/fragments/fineweb-edu-llama3 \\
        --run-name prepare-fineweb-edu-llama3

Each output shard is data/<bucket>/ with tokens.npy, offsets.npy and row_ids.npy. Row
ids are the documents' Lance _rowaddr, so scorers under src/recipe/ can join any
column of the source dataset back onto the shards and write it under meta/.

A two-pass shuffle:
  1. Tokenize fragments in a fixed shuffled order until the budget. Every document draws
     a bucket uniformly at random, and each fragment is stored sorted by bucket in the
     --fragments directory (default <output>/fragments). Phase 2 reads small slices of
     every fragment, so a shared filesystem is much faster for it than S3.
  2. Gather each bucket's documents from every fragment into one output shard.
TokenStream then permutes the documents inside each shard. Uniform buckets plus a
uniform permutation inside each bucket make the whole document order uniformly random,
so any prefix of the shards is a uniform sample of the data.

Rerunning with the same --output skips finished fragments and shards, so an interrupted
run (or an expired S3 key) only needs a rerun with the same --fragments. The bucket
count comes from the budget, so a different budget needs a different --output and
--fragments. The --fragments directory can be deleted once manifest.json is written.

On hosts whose compute nodes have no internet (Forerunner1), split it into stages with
the same arguments (scripts/submit-f1.sh runs all three):

    --stage fetch     login node: caches the tokenizer and development ids, and copies
                      the first fragments of the shuffled order into --mirror, until
                      their estimated tokens reach --mirror-margin x the budget. The
                      estimate is the source's token_count column if it has one (FineWeb-
                      Edu), else each fragment's rows x its sampled characters per row x
                      Llama 3 tokens per character, calibrated by tokenizing a few MB
                      (Dolma; see sample_texts)
    --stage process   compute node: the two phases above, reading --mirror and writing a
                      local --output, offline
    --stage upload    login node: copies the local --output to --remote

metadata.json names --dataset, not the mirror, so the uploaded store is the same as one
prepared from S3 directly. If the mirror runs out before the budget, process stops
before phase 2; rerun fetch with a larger --mirror-margin, then process again.
"""

import argparse
import itertools
import math
import multiprocessing
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy
from lance.fragment import LanceFragment
from transformers import PreTrainedTokenizerFast

from src.loggers import ConsoleLogger, WandbLogger
from src.mirror import fetch_fragments, open_mirror, read_mirror
from src.stages import add_stage_arguments, check_stage, upload_store
from src.tokenizer import (
    BucketGatherer,
    FragmentTokenizer,
    Offsets,
    ShardStore,
    fragment_shard_name,
    load_development_ids,
    load_tokenizer,
    open_dataset,
    shuffled_fragment_ids,
)

TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
DEVELOPMENT_DATASET = "cerulean-works/fineweb-edu-dev"
DEVELOPMENT_SIZE = 5000
SHARD_FORMAT = (
    "data/<shard>/: [BOS, text, EOS] int32 tokens.npy + int64 offsets.npy + "
    "int64 row_ids.npy (Lance _rowaddr), globally shuffled into buckets"
)
LAYOUT_VERSION = 2
WANDB_PROJECT = "lab7-data-curation"
# FineWeb-Edu's per-document GPT-2 token count, which --stage fetch sizes the mirror by.
# Sources without it (Dolma) are sized from a sample of their text instead
TOKEN_COUNT_COLUMN = "token_count"
# Each fragment's sample: SAMPLE_SLICES runs of SAMPLE_SLICE_ROWS consecutive rows at
# random offsets. Reading text is cheap, so every fragment gets its own sample for its
# mean length; tokenizing is not, so only CALIBRATION_CHARACTERS characters, drawn from
# the samples of the first CALIBRATION_FRAGMENTS fragments, are tokenized, for the
# source's tokens per character
SAMPLE_SLICES = 16
SAMPLE_SLICE_ROWS = 16
CALIBRATION_FRAGMENTS = 8
CALIBRATION_CHARACTERS = 8_000_000

# One per worker process, built by initialize_tokenizer or initialize_gatherer
tokenizer_worker: FragmentTokenizer | None = None
gatherer_worker: BucketGatherer | None = None


def initialize_tokenizer(
    dataset_uri: str,
    dataset_version: int,
    tokenizer_name: str,
    exclude_ids: frozenset[str],
    fragments_uri: str,
    seed: int,
    num_buckets: int,
    batch_size: int,
) -> None:
    global tokenizer_worker

    # Each process already uses one core; tokenizer threads on top would oversubscribe
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer_worker = FragmentTokenizer(
        open_dataset(dataset_uri, dataset_version),
        load_tokenizer(tokenizer_name),
        exclude_ids,
        ShardStore(fragments_uri),
        seed,
        num_buckets,
        batch_size,
    )


def tokenize_fragment(fragment_id: int) -> tuple[int, int, Offsets]:
    if tokenizer_worker is None:
        raise RuntimeError("worker is not initialized")

    return fragment_id, *tokenizer_worker.tokenize_fragment(fragment_id)


def initialize_gatherer(
    fragments_uri: str,
    output_uri: str,
    fragment_names: list[str],
    bucket_offsets: Offsets,
) -> None:
    global gatherer_worker

    gatherer_worker = BucketGatherer(
        ShardStore(fragments_uri), ShardStore(output_uri), fragment_names, bucket_offsets
    )


def gather_bucket(bucket: int) -> tuple[int, int]:
    if gatherer_worker is None:
        raise RuntimeError("worker is not initialized")

    return bucket, gatherer_worker.gather_bucket(bucket)


def check_metadata(store: ShardStore, metadata: dict[str, object]) -> None:
    """
    Shards are skipped by name only, so refuse to mix shards made with different settings
    """
    existing = store.read_json("metadata.json")

    if existing is None:
        store.write_json("metadata.json", metadata)
    elif existing != metadata:
        raise ValueError(
            f"{store.uri} holds shards made with {existing}, "
            f"not {metadata}; use a different --output"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", required=True, help="local path or s3://bucket/prefix"
    )
    parser.add_argument(
        "--fragments",
        help="local path or s3://bucket/prefix for the phase 1 fragments; "
        "default <output>/fragments",
    )
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=50_000_000,
        help="target tokens per output shard; sets the bucket count from the budget",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument(
        "--dataset", required=True, help="Lance dataset URI, local path or s3://"
    )
    parser.add_argument("--development-dataset", default=DEVELOPMENT_DATASET)
    parser.add_argument("--run-name", required=True, help="names the wandb run")
    parser.add_argument("--no-wandb", action="store_true", help="log to console only")
    add_stage_arguments(parser)
    parser.add_argument(
        "--mirror-margin",
        type=float,
        default=1.25,
        help="--stage fetch mirrors estimated tokens up to this x --token-budget. "
        "token_count is GPT-2's, which runs above Llama 3's; the calibrated estimate "
        "has a sampling error, printed by fetch",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    check_stage(arguments, "output", "fragments")

    if arguments.stage == "fetch":
        fetch(arguments)
    elif arguments.stage == "upload":
        upload(arguments)
    else:
        process(arguments)


type FragmentCounter = Callable[[int], tuple[int, float]]


def sample_texts(
    fragment: LanceFragment, exclude_ids: frozenset[str], seed: int
) -> list[str]:
    """
    A sample of the fragment's rows, as --stage process keeps them: the text, or "" for
    the empty and development rows it drops. The offsets depend only on (seed, fragment)
    """
    rows = fragment.count_rows()
    limit = min(SAMPLE_SLICE_ROWS, rows)
    random_generator = numpy.random.default_rng([seed, 1, fragment.fragment_id])
    texts: list[str] = []

    for offset in random_generator.integers(rows - limit + 1, size=SAMPLE_SLICES):
        table = fragment.to_table(
            columns=["id", "text"], offset=int(offset), limit=limit
        )
        texts.extend(
            text if text and document_id not in exclude_ids else ""
            for document_id, text in zip(
                table.column("id").to_pylist(),
                table.column("text").to_pylist(),
                strict=True,
            )
        )

    return texts


def calibrate_tokens_per_character(
    samples: list[list[str]], tokenizer: PreTrainedTokenizerFast
) -> tuple[float, int]:
    """
    Llama 3 tokens per character of text, and the characters it was measured on.
    Documents are taken round-robin over the fragments' samples, so every fragment
    contributes, until CALIBRATION_CHARACTERS
    """
    batch: list[str] = []
    characters = 0

    for round_ in itertools.zip_longest(*samples):
        for text in round_:
            if text:
                batch.append(text)
                characters += len(text)

        if characters >= CALIBRATION_CHARACTERS:
            break

    if not batch:
        raise ValueError("every sampled row is empty")

    encoded = tokenizer(batch, add_special_tokens=False, return_attention_mask=False)
    tokens = sum(len(token_ids) for token_ids in encoded["input_ids"])
    return tokens / characters, characters


def estimate_tokens(
    rows: int, texts: list[str], tokens_per_character: float
) -> tuple[int, float]:
    """
    A fragment's tokens as process stores them ([BOS, text, EOS] per kept row), from its
    row count and a sample of its rows, and the variance of that estimate
    """
    characters = numpy.array([len(text) for text in texts], dtype=numpy.float64)
    row_tokens = characters * tokens_per_character + 2 * (characters > 0)
    variance = (
        rows**2 * float(row_tokens.var(ddof=1)) / row_tokens.size
        if row_tokens.size > 1
        else 0.0
    )
    return int(rows * float(row_tokens.mean())), variance


def select_fragments(
    fragment_ids: list[int], count: FragmentCounter, target_tokens: float, workers: int
) -> tuple[list[int], int, float]:
    """
    The shortest prefix of fragment_ids whose estimated tokens reach target_tokens, that
    estimate and its variance, counting a chunk of fragments at a time
    """
    selected: list[int] = []
    total = 0
    variance = 0.0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(fragment_ids), workers):
            chunk = fragment_ids[start : start + workers]

            for fragment_id, (tokens, fragment_variance) in zip(
                chunk, pool.map(count, chunk), strict=True
            ):
                selected.append(fragment_id)
                total += tokens
                variance += fragment_variance

                if total >= target_tokens:
                    return selected, total, variance

    return selected, total, variance


def fetch(arguments: argparse.Namespace) -> None:
    """
    Everything --stage process reads from the internet, cached or mirrored locally
    """
    # Keep the mirror's version once there is one, so a newer source version can't
    # quietly mix into it
    existing = read_mirror(arguments.mirror)
    dataset = open_dataset(
        arguments.dataset, existing.version if existing is not None else None
    )

    tokenizer = load_tokenizer(arguments.tokenizer)
    exclude_ids = load_development_ids(arguments.development_dataset, DEVELOPMENT_SIZE)

    workers = min(arguments.workers, 32)
    fragment_ids = shuffled_fragment_ids(dataset, arguments.seed)

    def get_fragment(fragment_id: int) -> LanceFragment:
        fragment = dataset.get_fragment(fragment_id)

        if fragment is None:
            raise ValueError(f"fragment {fragment_id} does not exist")

        return fragment

    if TOKEN_COUNT_COLUMN in dataset.schema.names:
        estimate = TOKEN_COUNT_COLUMN

        def count(fragment_id: int) -> tuple[int, float]:
            column = (
                get_fragment(fragment_id)
                .to_table(columns=[TOKEN_COUNT_COLUMN])
                .column(0)
            )
            # nansum: a column with nulls comes out as float with NaN for them
            return int(numpy.nansum(column.to_numpy(zero_copy_only=False))), 0.0

    else:
        start_time = time.monotonic()
        calibration_ids = fragment_ids[:CALIBRATION_FRAGMENTS]

        def sample(fragment_id: int) -> list[str]:
            return sample_texts(get_fragment(fragment_id), exclude_ids, arguments.seed)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            samples = dict(zip(calibration_ids, pool.map(sample, calibration_ids)))

        tokens_per_character, characters = calibrate_tokens_per_character(
            list(samples.values()), tokenizer
        )
        estimate = (
            f"rows x sampled characters per row x {tokens_per_character:.4f} tokens "
            "per character"
        )
        print(
            f"calibrated {tokens_per_character:.4f} tokens per character on "
            f"{characters:,} characters of {len(samples)} fragments, in "
            f"{time.monotonic() - start_time:.1f}s",
            flush=True,
        )

        def count(fragment_id: int) -> tuple[int, float]:
            fragment = get_fragment(fragment_id)
            # The calibration fragments are sampled already
            texts = samples.get(fragment_id)

            if texts is None:
                texts = sample_texts(fragment, exclude_ids, arguments.seed)

            return estimate_tokens(fragment.count_rows(), texts, tokens_per_character)

    start_time = time.monotonic()
    selected, source_tokens, variance = select_fragments(
        fragment_ids,
        count,
        arguments.token_budget * arguments.mirror_margin,
        workers,
    )
    relative_error = math.sqrt(variance) / source_tokens if source_tokens else 0.0
    print(
        f"{len(selected)} of {len(fragment_ids)} fragments hold {source_tokens:,} "
        f"estimated tokens ({estimate}; ±{relative_error:.1%} standard error), "
        f"{arguments.mirror_margin}x the budget, in "
        f"{time.monotonic() - start_time:.1f}s",
        flush=True,
    )

    # Three standard errors under the margin, or the mirror may well run short
    if 3 * relative_error > arguments.mirror_margin - 1:
        print(
            f"warning: the estimate's standard error ({relative_error:.1%}) is large "
            f"for --mirror-margin {arguments.mirror_margin}; if process runs out of "
            "mirror, rerun fetch with a larger --mirror-margin",
            flush=True,
        )

    fetch_fragments(
        arguments.dataset, dataset.version, arguments.mirror, selected, workers
    )


def upload(arguments: argparse.Namespace) -> None:
    store = ShardStore(arguments.output)

    if store.read_json("manifest.json") is None:
        raise FileNotFoundError(
            f"no manifest.json in {store.uri}; --stage process has not finished"
        )

    upload_store(store.uri, arguments.remote)


def process(arguments: argparse.Namespace) -> None:
    store = ShardStore(arguments.output)
    fragment_store = ShardStore(
        arguments.fragments or arguments.output.rstrip("/") + "/fragments"
    )

    if arguments.stage == "process":
        mirror = open_mirror(arguments.mirror, arguments.dataset)
        dataset_uri = os.path.abspath(arguments.mirror)
        dataset = open_dataset(dataset_uri, mirror.version)
    else:
        mirror = None
        dataset_uri = arguments.dataset
        dataset = open_dataset(dataset_uri)

    num_buckets = math.ceil(arguments.token_budget / arguments.shard_tokens)

    # The fragments' bucket assignment depends on seed and num_buckets, so both must
    # match for a rerun to reuse them
    metadata = {
        "tokenizer": arguments.tokenizer,
        "dataset": arguments.dataset,
        "dataset_version": dataset.version,
        "development_dataset": arguments.development_dataset,
        "format": SHARD_FORMAT,
        "layout": LAYOUT_VERSION,
        "seed": arguments.seed,
        "num_buckets": num_buckets,
    }
    check_metadata(store, metadata)
    # Fragments are also skipped by name, and --fragments can point anywhere
    check_metadata(fragment_store, metadata)

    if arguments.no_wandb:
        logger: ConsoleLogger | WandbLogger = ConsoleLogger()
    else:
        logger = WandbLogger(
            WANDB_PROJECT, arguments.run_name, {**vars(arguments), **metadata}
        )

    # Download into the HF cache once here, so workers don't all race to download it,
    load_tokenizer(arguments.tokenizer)

    exclude_ids = load_development_ids(arguments.development_dataset, DEVELOPMENT_SIZE)
    fragment_ids = shuffled_fragment_ids(dataset, arguments.seed)
    # The mirror lists every fragment but holds the data of a prefix of this order only
    mirror_ran_out = False

    if mirror is not None:
        mirrored = list(
            itertools.takewhile(lambda i: i in mirror.fragment_ids, fragment_ids)
        )
        mirror_ran_out = len(mirrored) < len(fragment_ids)
        fragment_ids = mirrored

    fragments: list[dict[str, int]] = []
    bucket_offsets: list[Offsets] = []
    total_tokens = 0
    start_time = time.monotonic()

    # spawn, not fork: this process already runs Lance's background threads, and
    # forking a process with live threads can deadlock the child
    context = multiprocessing.get_context("spawn")

    # Phase 1: tokenize and scatter
    with context.Pool(
        arguments.workers,
        initializer=initialize_tokenizer,
        initargs=(
            dataset_uri,
            dataset.version,
            arguments.tokenizer,
            exclude_ids,
            fragment_store.uri,
            arguments.seed,
            num_buckets,
            arguments.batch_size,
        ),
    ) as pool:
        # imap yields in fragment_ids order however the workers finish, so the
        # selected fragments are always the same prefix of the shuffled order
        for fragment_id, tokens, offsets in pool.imap(tokenize_fragment, fragment_ids):
            fragments.append({"fragment_id": fragment_id, "tokens": tokens})
            bucket_offsets.append(offsets)
            total_tokens += tokens

            elapsed = time.monotonic() - start_time
            logger.log(
                {
                    "prepare/fragment_id": fragment_id,
                    "prepare/fragment_tokens": tokens,
                    "prepare/total_tokens": total_tokens,
                    "prepare/budget_fraction": total_tokens / arguments.token_budget,
                    "prepare/tokens_per_second": total_tokens / elapsed,
                },
                step=len(fragments),
            )

            if total_tokens >= arguments.token_budget:
                break

        # Leaving the with block terminates workers still busy on later fragments.
        # Their shards never got offsets.npy, so a rerun redoes them

    if total_tokens < arguments.token_budget:
        if mirror_ran_out:
            # Phase 2 needs the final fragment list, so stop before it. The tokenized
            # fragments are kept, and the rerun only tokenizes the new ones
            raise RuntimeError(
                f"the mirror ran out at {total_tokens:,} of {arguments.token_budget:,} "
                f"tokens after {len(fragments)} fragments; rerun --stage fetch with a "
                "larger --mirror-margin, then --stage process again"
            )

        print(f"warning: dataset ran out at {total_tokens} tokens", flush=True)

    shards: list[dict[str, object]] = []
    gathered_tokens = 0
    start_time = time.monotonic()

    # Phase 2: gather each bucket into an output shard
    with context.Pool(
        arguments.workers,
        initializer=initialize_gatherer,
        initargs=(
            fragment_store.uri,
            store.uri,
            [fragment_shard_name(fragment["fragment_id"]) for fragment in fragments],
            numpy.stack(bucket_offsets),
        ),
    ) as pool:
        for bucket, tokens in pool.imap(gather_bucket, range(num_buckets)):
            shards.append({"name": f"{bucket:06d}", "tokens": tokens})
            gathered_tokens += tokens

            elapsed = time.monotonic() - start_time
            logger.log(
                {
                    "prepare/shard_tokens": tokens,
                    "prepare/gathered_tokens": gathered_tokens,
                    "prepare/gathered_fraction": gathered_tokens / total_tokens,
                    "prepare/gather_tokens_per_second": gathered_tokens / elapsed,
                },
                step=len(fragments) + len(shards),
            )

    # Every document lands in exactly one bucket
    if gathered_tokens != total_tokens:
        raise RuntimeError(
            f"gathered {gathered_tokens} tokens but tokenized {total_tokens}"
        )

    store.write_json(
        "manifest.json",
        {
            "seed": arguments.seed,
            "token_budget": arguments.token_budget,
            "num_buckets": num_buckets,
            "total_tokens": total_tokens,
            "fragments": fragments,
            "shards": shards,
        },
    )

    if isinstance(logger, WandbLogger):
        logger.run.summary["total_tokens"] = total_tokens
        logger.run.summary["num_fragments"] = len(fragments)
        logger.run.summary["num_shards"] = len(shards)
        logger.run.summary["ran_out"] = total_tokens < arguments.token_budget
        logger.finish()


# Required with spawn: each worker re-imports this file, and must not rerun main()
if __name__ == "__main__":
    main()
