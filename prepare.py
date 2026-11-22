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

Phase 1 works in chunks, not whole fragments: Dolma's fragments hold up to 2^20 rows
and hundreds of millions of tokens, and a source may span fewer fragments than there
are workers. Chunks are contiguous row ranges, sized so the candidate fragments split
into about TASKS_PER_WORKER chunks per worker (see plan_chunks). The budget is still
checked only once a fragment's last chunk is in, so the selected fragments are a
prefix of the shuffled order, as before. Bucket draws are per row of the fragment, so
the output shards do not depend on the chunking, i.e. on --workers. Workers report the
tokens they encode as they go, and the run logs tokenizing progress every
PROGRESS_SECONDS, not only when a fragment finishes.

Rerunning with the same --output skips finished chunks and shards, so an interrupted
run (or an expired S3 key) only needs a rerun with the same --fragments. Chunks are
reused only with the same --workers (it sets their ranges; others are retokenized). The
bucket count comes from the budget, so a different budget needs a different --output
and --fragments. The --fragments directory can be deleted once manifest.json is
written.

On hosts whose compute nodes have no internet (Forerunner1), split it into stages with
the same arguments (scripts/submit-f1.sh runs all three):

    --stage fetch     login node: caches the tokenizer, and copies
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
import queue
import time
from collections.abc import Callable, Sequence
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
    chunk_ranges,
    chunk_shard_name,
    load_tokenizer,
    open_dataset,
    shuffled_fragment_ids,
)

TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
SHARD_FORMAT = (
    "data/<shard>/: [BOS, text, EOS] int32 tokens.npy + int64 offsets.npy + "
    "int64 row_ids.npy (Lance _rowaddr), globally shuffled into buckets"
)
# The same layout for a tokenizer without a BOS (Qwen3): documents start with the EOS.
# A separate string, so stores made with a BOS keep their metadata.json and can resume
SHARD_FORMAT_NO_BOS = SHARD_FORMAT.replace("[BOS, text, EOS]", "[EOS, text, EOS]")
# 3: phase 1 in chunks, bucket draws per row of the fragment (2 drew them per kept
# document), so a layout 2 fragment store can't be continued
LAYOUT_VERSION = 3
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
# Phase 1 chunks per worker: more balance the load better (a fragment waits for its
# last chunk), fewer cost less per-task overhead and fewer files for phase 2 to read
TASKS_PER_WORKER = 4
# How often phase 1 logs the tokens workers have encoded, between finished fragments
PROGRESS_SECONDS = 30

# One per worker process, built by initialize_tokenizer or initialize_gatherer
tokenizer_worker: FragmentTokenizer | None = None
gatherer_worker: BucketGatherer | None = None
# Where tokenizer workers put the tokens of each piece they encode, for progress
progress_queue: multiprocessing.Queue[int] | None = None

# A phase 1 task: (fragment_id, offset, limit, rows of the fragment)
type Chunk = tuple[int, int, int, int]


def initialize_tokenizer(
    dataset_uri: str,
    dataset_version: int,
    tokenizer_name: str,
    fragments_uri: str,
    seed: int,
    num_buckets: int,
    batch_size: int,
    progress: multiprocessing.Queue[int],
) -> None:
    global tokenizer_worker, progress_queue

    # Each process already uses one core; tokenizer threads on top would oversubscribe
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer_worker = FragmentTokenizer(
        open_dataset(dataset_uri, dataset_version),
        load_tokenizer(tokenizer_name),
        ShardStore(fragments_uri),
        seed,
        num_buckets,
        batch_size,
    )
    progress_queue = progress


def tokenize_chunk(chunk: Chunk) -> tuple[int, Offsets]:
    if tokenizer_worker is None or progress_queue is None:
        raise RuntimeError("worker is not initialized")

    fragment_id, offset, limit, rows = chunk
    return tokenizer_worker.tokenize_chunk(
        fragment_id, offset, limit, rows, progress=progress_queue.put
    )


def initialize_gatherer(
    fragments_uri: str,
    output_uri: str,
    fragment_names: list[str],
    bucket_offsets: Offsets,
) -> None:
    global gatherer_worker

    gatherer_worker = BucketGatherer(
        ShardStore(fragments_uri),
        ShardStore(output_uri),
        fragment_names,
        bucket_offsets,
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
    parser.add_argument("--run-name", required=True, help="names the wandb run")
    parser.add_argument("--no-wandb", action="store_true", help="log to console only")
    parser.add_argument(
        "--wandb-project",
        default=WANDB_PROJECT,
        help=f"wandb project to log to (default {WANDB_PROJECT}, the lab's)",
    )
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
    fragment: LanceFragment, seed: int
) -> list[str]:
    """
    A sample of the fragment's rows, as --stage process keeps them: the text, or "" for
    the empty rows it drops. The offsets depend only on (seed, fragment)
    """
    rows = fragment.count_rows()
    limit = min(SAMPLE_SLICE_ROWS, rows)
    random_generator = numpy.random.default_rng([seed, 1, fragment.fragment_id])
    texts: list[str] = []

    for offset in random_generator.integers(rows - limit + 1, size=SAMPLE_SLICES):
        table = fragment.to_table(columns=["text"], offset=int(offset), limit=limit)
        texts.extend(text or "" for text in table.column("text").to_pylist())

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
    A fragment's tokens as process stores them ([start, text, EOS] per kept row), from its
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


def plan_chunks(
    fragment_rows: Sequence[tuple[int, int]], workers: int
) -> list[tuple[int, list[Chunk]]]:
    """
    Every fragment's chunks, as (fragment_id, [(fragment_id, offset, limit, rows), ...])
    in fragment order. The chunk size comes from all the candidate fragments' rows, so
    they split into about TASKS_PER_WORKER chunks per worker. Process's candidates are
    the mirrored fragments, about --mirror-margin x the budget; --stage all has every
    fragment of the source as a candidate, so its chunks can be whole fragments
    """
    total = sum(rows for _, rows in fragment_rows)
    chunk_rows = max(math.ceil(total / (workers * TASKS_PER_WORKER)), 1)
    return [
        (
            fragment_id,
            [
                (fragment_id, offset, limit, rows)
                for offset, limit in chunk_ranges(rows, chunk_rows)
            ],
        )
        for fragment_id, rows in fragment_rows
    ]


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
            return sample_texts(get_fragment(fragment_id), arguments.seed)

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
                texts = sample_texts(fragment, arguments.seed)

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
    # Download into the HF cache once here, so workers don't all race to download it,
    # and see which document layout it gets
    has_bos = load_tokenizer(arguments.tokenizer).bos_token_id is not None

    # The fragments' bucket assignment depends on seed and num_buckets, so both must
    # match for a rerun to reuse them
    metadata = {
        "tokenizer": arguments.tokenizer,
        "dataset": arguments.dataset,
        "dataset_version": dataset.version,
        "format": SHARD_FORMAT if has_bos else SHARD_FORMAT_NO_BOS,
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
            arguments.wandb_project,
            arguments.run_name,
            {**vars(arguments), **metadata},
        )

    fragment_ids = shuffled_fragment_ids(dataset, arguments.seed)
    # The mirror lists every fragment but holds the data of a prefix of this order only
    mirror_ran_out = False

    if mirror is not None:
        mirrored = list(
            itertools.takewhile(lambda i: i in mirror.fragment_ids, fragment_ids)
        )
        mirror_ran_out = len(mirrored) < len(fragment_ids)
        fragment_ids = mirrored

    fragment_rows = []

    for fragment_id in fragment_ids:
        fragment = dataset.get_fragment(fragment_id)

        if fragment is None:
            raise ValueError(f"fragment {fragment_id} does not exist")

        # From the fragment's metadata, no data read. Rows after deletions, which
        # to_batches' offset and limit index into
        fragment_rows.append((fragment_id, fragment.count_rows()))

    plan = plan_chunks(fragment_rows, arguments.workers)
    print(
        f"phase 1: {len(plan)} candidate fragments in "
        f"{sum(len(chunks) for _, chunks in plan)} chunks",
        flush=True,
    )

    fragments: list[dict[str, int]] = []
    chunk_names: list[str] = []
    bucket_offsets: list[Offsets] = []
    total_tokens = 0
    # Tokens the workers have encoded so far, from progress messages: also counts
    # chunks still in flight, and chunks past the budget that end up unused
    encoded_tokens = 0
    step = 0
    start_time = time.monotonic()
    last_log_time = start_time

    # spawn, not fork: this process already runs Lance's background threads, and
    # forking a process with live threads can deadlock the child
    context = multiprocessing.get_context("spawn")
    progress: multiprocessing.Queue[int] = context.Queue()

    def log_phase_1(metrics: dict[str, float]) -> None:
        nonlocal encoded_tokens, step, last_log_time

        while True:
            try:
                encoded_tokens += progress.get_nowait()
            except queue.Empty:
                break

        now = time.monotonic()
        elapsed = now - start_time
        step += 1
        last_log_time = now
        logger.log(
            {
                **metrics,
                "prepare/encoded_tokens": encoded_tokens,
                "prepare/encoded_tokens_per_second": encoded_tokens / elapsed,
                "prepare/total_tokens": total_tokens,
                "prepare/budget_fraction": total_tokens / arguments.token_budget,
                "prepare/tokens_per_second": total_tokens / elapsed,
            },
            step=step,
        )

    # Phase 1: tokenize and scatter
    with context.Pool(
        arguments.workers,
        initializer=initialize_tokenizer,
        initargs=(
            dataset_uri,
            dataset.version,
            arguments.tokenizer,
            fragment_store.uri,
            arguments.seed,
            num_buckets,
            arguments.batch_size,
            progress,
        ),
    ) as pool:
        # imap yields in chunk order, fragment by fragment, however the workers
        # finish. The budget is checked at fragment ends only, so the selected
        # fragments are always the same prefix of the shuffled order
        results = pool.imap(
            tokenize_chunk, [chunk for _, chunks in plan for chunk in chunks]
        )

        for fragment_id, chunks in plan:
            fragment_tokens = 0

            for chunk in chunks:
                # Waits for the chunk, logging progress every PROGRESS_SECONDS
                while True:
                    timeout = last_log_time + PROGRESS_SECONDS - time.monotonic()

                    try:
                        tokens, offsets = results.next(timeout=max(timeout, 0))
                        break
                    except multiprocessing.TimeoutError:
                        log_phase_1({})

                fragment_tokens += tokens
                chunk_names.append(chunk_shard_name(*chunk[:3]))
                bucket_offsets.append(offsets)

            fragments.append({"fragment_id": fragment_id, "tokens": fragment_tokens})
            total_tokens += fragment_tokens
            log_phase_1(
                {
                    "prepare/fragment_id": fragment_id,
                    "prepare/fragment_tokens": fragment_tokens,
                }
            )

            if total_tokens >= arguments.token_budget:
                break

        # Leaving the with block terminates workers still busy on later chunks.
        # Their shards never got offsets.npy, so a rerun redoes them

    if total_tokens < arguments.token_budget:
        if mirror_ran_out:
            # Phase 2 needs the final fragment list, so stop before it. A larger
            # mirror has more candidate rows, hence other chunk ranges (plan_chunks),
            # so the rerun tokenizes everything again
            raise RuntimeError(
                f"the mirror ran out at {total_tokens:,} of {arguments.token_budget:,} "
                f"tokens after {len(fragments)} fragments; rerun --stage fetch with a "
                "larger --mirror-margin, then --stage process again"
            )

        print(f"warning: dataset ran out at {total_tokens} tokens", flush=True)

    shards: list[dict[str, object]] = []
    gathered_tokens = 0
    start_time = time.monotonic()

    # Phase 2: gather each bucket into an output shard, from the selected fragments'
    # chunks in phase 1 order, so a bucket's documents keep (fragment, row) order
    with context.Pool(
        arguments.workers,
        initializer=initialize_gatherer,
        initargs=(
            fragment_store.uri,
            store.uri,
            chunk_names,
            numpy.stack(bucket_offsets),
        ),
    ) as pool:
        for bucket, tokens in pool.imap(gather_bucket, range(num_buckets)):
            shards.append({"name": f"{bucket:06d}", "tokens": tokens})
            gathered_tokens += tokens

            elapsed = time.monotonic() - start_time
            # Continues phase 1's steps
            step += 1
            logger.log(
                {
                    "prepare/shard_tokens": tokens,
                    "prepare/gathered_tokens": gathered_tokens,
                    "prepare/gathered_fraction": gathered_tokens / total_tokens,
                    "prepare/gather_tokens_per_second": gathered_tokens / elapsed,
                },
                step=step,
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
        logger.run.summary["num_chunks"] = len(chunk_names)
        logger.run.summary["num_shards"] = len(shards)
        logger.run.summary["ran_out"] = total_tokens < arguments.token_budget
        logger.finish()


# Required with spawn: each worker re-imports this file, and must not rerun main()
if __name__ == "__main__":
    main()
