"""
P0-1's Token Frequency Audit and leakage spot check (plan §3, Phase 0 data gate), run
on a compute node over local copies of the recipes and stores:

    python scripts/embedding_ml_transfer/audit.py \\
        --recipe multi_train=/work1/$USER/p0-1/recipes/multi-train-20b \\
        --recipe mono_train=/work1/$USER/p0-1/recipes/mono-train-20b \\
        --recipe oracle_train=/work1/$USER/p0-1/recipes/oracle-train-24b \\
        --store-map s3://embedding-ml-transfer/sources=/work1/$USER/stores/embedding-ml-transfer \\
        --lid-model /work1/$USER/models/lid/lid.176.bin \\
        --output /work1/$USER/p0-1/audit

Recipes name their stores by S3 URI, and compute nodes have no S3, so --store-map
rewrites a URI prefix to the local path the same store sits at.

1. Token counts. For every cursor of every recipe, over every selected token: tokens
   per V_ban category (vban_count.py's BAN / CONTINUATION / SHARED / SPECIAL), tokens
   with ids >= 151,936 (Fake-French's shifted ids) and the largest id. The gate:
   BAN and shifted ids are 0 in every cursor of Multi-Train and Mono-Train.
2. Leakage spot check with a second, independent LID (fastText lid.176, not the GlotLID
   the filter used): --sample documents per recipe drawn uniformly from its selected
   documents, decoded, and scored line by line. A document counts as leaked if a line
   reads as French (P > 0.5), or a line holding a Han character reads as Chinese (the
   same Han gate as the filter). The rate comes with a Wilson 95% interval; the gate is
   its upper bound below 0.01. Flagged lines go to flagged.jsonl, for reading by hand.
3. --eval-text FILE (jsonl with "text"): how the evaluation text tokenizes, by V_ban
   category: the plan's "中文評估文字中有多少比例的 token 落在 V_ban、多少是共用 token".

Writes audit.json and audit.md to --output.
"""

import argparse
import json
import math
import multiprocessing
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wandb_summary import add_arguments as add_wandb_arguments
from wandb_summary import log_summary

from src.recipe.recipe import Cursor, Recipe, load_recipe
from src.recipe.scorers.vban_count import CATEGORY_NAMES, byte_categories
from src.tokenizer import ShardStore, load_tokenizer

BASE_VOCAB = 151936
TABLE_SIZE = 1 << 18  # covers the extended vocab of track 1
HAN = re.compile("[㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f]")
GATED = {"multi_train", "mono_train"}

_categories: numpy.ndarray | None = None
_lid = None


def local(uri: str, mapping: list[tuple[str, str]]) -> str:
    for prefix, path in mapping:
        if uri.startswith(prefix):
            return path + uri[len(prefix) :]

    return uri


def init_counter(tokenizer: str) -> None:
    global _categories
    _categories = byte_categories(tokenizer, TABLE_SIZE)


def count_shard(task: tuple[str, str, str]) -> dict[str, int]:
    """
    Token counts of the selected documents of one shard
    """
    store_path, shard, selection_path = task
    store = ShardStore(store_path)
    tokens = store.read_tokens(shard)
    offsets = store.read_offsets(shard)
    selected = numpy.load(selection_path)
    # One boolean per token: inside a selected document
    starts, stops = offsets[selected], offsets[selected + 1]
    mask = numpy.zeros(tokens.size + 1, dtype=numpy.int32)
    numpy.add.at(mask, starts, 1)
    numpy.add.at(mask, stops, -1)
    chosen = tokens[numpy.cumsum(mask[:-1]) > 0]

    assert _categories is not None
    per_category = numpy.bincount(_categories[chosen], minlength=4)
    result = {name: int(per_category[value]) for value, name in CATEGORY_NAMES.items()}
    result["tokens"] = int(chosen.size)
    result["documents"] = int(selected.size)
    result["shifted"] = int(numpy.count_nonzero(chosen >= BASE_VOCAB))
    result["max_id"] = int(chosen.max()) if chosen.size else -1
    return result


def cursor_tasks(
    recipe_path: str, cursor: Cursor, mapping: list[tuple[str, str]]
) -> list[tuple[str, str, str]]:
    store = local(cursor.store, mapping)
    return [
        (store, shard.name, f"{recipe_path}/{shard.selection}")
        for shard in cursor.shards
    ]


def init_lid(model: str, tokenizer: str) -> None:
    global _lid
    import fasttext

    _lid = (fasttext.load_model(model), load_tokenizer(tokenizer))


def spot_check(task: tuple[str, str, list[int]]) -> list[dict[str, object]]:
    """
    lid.176 over the given documents of one shard: one record per flagged document
    """
    store_path, shard, documents = task
    assert _lid is not None
    model, tokenizer = _lid
    store = ShardStore(store_path)
    tokens = store.read_tokens(shard)
    offsets = store.read_offsets(shard)
    flagged = []

    for document in documents:
        text = tokenizer.decode(
            tokens[offsets[document] : offsets[document + 1]], skip_special_tokens=True
        )

        for line in text.split("\n"):
            line = line.strip()

            if not line:
                continue

            labels, scores = model.predict(line, k=3)
            label, score = labels[0], float(scores[0])

            if score > 0.5 and (
                label == "__label__fr"
                or (
                    label in ("__label__zh", "__label__wuu", "__label__yue")
                    and HAN.search(line)
                )
            ):
                flagged.append(
                    {
                        "shard": shard,
                        "document": document,
                        "label": label,
                        "p": round(score, 3),
                        "line": line[:300],
                    }
                )
                break

    return flagged


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0

    z = 1.959964
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def sample_documents(
    recipe_path: str,
    recipe: Recipe,
    mapping: list[tuple[str, str]],
    size: int,
    rng: numpy.random.Generator,
) -> list[tuple[str, str, list[int]]]:
    """
    size documents drawn uniformly from every selected document of the recipe
    """
    entries = []

    for phase in recipe.phases:
        for cursor in phase.cursors:
            for shard in cursor.shards:
                entries.append(
                    (
                        local(cursor.store, mapping),
                        shard.name,
                        f"{recipe_path}/{shard.selection}",
                        shard.documents,
                    )
                )

    totals = numpy.array([entry[3] for entry in entries])
    picks = numpy.sort(
        rng.choice(int(totals.sum()), size=min(size, int(totals.sum())), replace=False)
    )
    bounds = numpy.concatenate([[0], numpy.cumsum(totals)])
    tasks = []

    for index, (store, shard, selection, _) in enumerate(entries):
        inside = picks[(picks >= bounds[index]) & (picks < bounds[index + 1])]

        if inside.size:
            documents = numpy.load(selection)[inside - bounds[index]]
            tasks.append((store, shard, documents.astype(int).tolist()))

    return tasks


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recipe", action="append", required=True, help="name=path")
    parser.add_argument("--store-map", action="append", default=[], help="uri=path")
    parser.add_argument("--lid-model", required=True)
    parser.add_argument("--sample", type=int, default=100_000)
    parser.add_argument("--eval-text", action="append", default=[], help="name=jsonl")
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen3-0.6B-Base", help="for --eval-text"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    add_wandb_arguments(parser, run_name_required=False)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    mapping = [tuple(item.split("=", 1)) for item in arguments.store_map]
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    rng = numpy.random.default_rng(arguments.seed)
    report: dict[str, object] = {"recipes": {}, "eval_text": {}}
    flagged_all = []

    for item in arguments.recipe:
        name, path = item.split("=", 1)
        recipe = load_recipe(path)
        cursors = {}

        with ProcessPoolExecutor(
            arguments.workers,
            mp_context=context,
            initializer=init_counter,
            initargs=(recipe.tokenizer,),
        ) as pool:
            for phase in recipe.phases:
                for cursor in phase.cursors:
                    parts = list(
                        pool.map(count_shard, cursor_tasks(path, cursor, mapping))
                    )
                    total = {
                        key: sum(part[key] for part in parts)
                        for key in parts[0]
                        if key != "max_id"
                    }
                    total["max_id"] = max(part["max_id"] for part in parts)
                    cursors[f"{phase.name}/{cursor.name}"] = total
                    print(f"{name} {phase.name}/{cursor.name}: {total}", flush=True)

        tasks = sample_documents(path, recipe, mapping, arguments.sample, rng)
        sampled = sum(len(task[2]) for task in tasks)

        with ProcessPoolExecutor(
            arguments.workers,
            mp_context=context,
            initializer=init_lid,
            initargs=(arguments.lid_model, recipe.tokenizer),
        ) as pool:
            flagged = [
                record for part in pool.map(spot_check, tasks) for record in part
            ]

        low, high = wilson(len(flagged), sampled)
        french = sum(record["label"] == "__label__fr" for record in flagged)
        report["recipes"][name] = {
            "path": path,
            "total_tokens": recipe.total_tokens,
            "cursors": cursors,
            "spot_check": {
                "sampled": sampled,
                "flagged": len(flagged),
                "flagged_french": french,
                "flagged_chinese": len(flagged) - french,
                "rate": len(flagged) / max(sampled, 1),
                "ci95": [low, high],
            },
        }
        flagged_all += [{"recipe": name, **record} for record in flagged]
        print(
            f"{name}: {len(flagged)}/{sampled} flagged, 95% CI [{low:.5f}, {high:.5f}]",
            flush=True,
        )

    for item in arguments.eval_text:
        name, path = item.split("=", 1)
        tokenizer = load_tokenizer(arguments.tokenizer)
        categories = byte_categories(arguments.tokenizer, TABLE_SIZE)
        texts = [
            json.loads(line)["text"] for line in Path(path).read_text().splitlines()
        ]
        ids = numpy.concatenate(
            [
                numpy.array(row, dtype=numpy.int64)
                for row in tokenizer(texts, add_special_tokens=False)["input_ids"]
            ]
        )
        counts = numpy.bincount(categories[ids], minlength=4)
        report["eval_text"][name] = {
            CATEGORY_NAMES[value]: int(counts[value]) for value in CATEGORY_NAMES
        } | {"tokens": int(ids.size)}

    gate = all(
        cursor["ban"] == 0 and cursor["shifted"] == 0
        for name, entry in report["recipes"].items()
        if name in GATED
        for cursor in entry["cursors"].values()
    ) and all(
        entry["spot_check"]["ci95"][1] < 0.01
        for name, entry in report["recipes"].items()
        if name in GATED
    )
    report["gate_pass"] = gate
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    summary: dict[str, object] = {"gate_pass": gate}

    for name, entry in report["recipes"].items():
        for cursor, total in entry["cursors"].items():
            for key in ("tokens", "ban", "continuation", "shifted", "max_id"):
                summary[f"{name}/{cursor}/{key}"] = total[key]

        spot = entry["spot_check"]
        summary[f"{name}/spot_check/rate"] = spot["rate"]
        summary[f"{name}/spot_check/ci95_high"] = spot["ci95"][1]
        summary[f"{name}/spot_check/sampled"] = spot["sampled"]

    for name, counts in report["eval_text"].items():
        for key, value in counts.items():
            summary[f"eval_text/{name}/{key}"] = value

    log_summary(arguments, summary)

    with (output / "flagged.jsonl").open("w") as stream:
        for record in flagged_all:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# P0-1 Token Frequency Audit",
        "",
        "| recipe / cursor | tokens | BAN | continuation | shifted (>= 151,936) | max id |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, entry in report["recipes"].items():
        for cursor, total in entry["cursors"].items():
            lines.append(
                f"| {name} / {cursor} | {total['tokens']:,} | {total['ban']:,} | "
                f"{total['continuation']:,} | {total['shifted']:,} | {total['max_id']} |"
            )

    lines += [
        "",
        "| recipe | sampled | flagged (fr / zh) | rate | 95% CI |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for name, entry in report["recipes"].items():
        spot = entry["spot_check"]
        lines.append(
            f"| {name} | {spot['sampled']:,} | {spot['flagged']} "
            f"({spot['flagged_french']} / {spot['flagged_chinese']}) | "
            f"{spot['rate']:.5f} | [{spot['ci95'][0]:.5f}, {spot['ci95'][1]:.5f}] |"
        )

    if report["eval_text"]:
        lines += [
            "",
            "| eval text | tokens | BAN | continuation | shared |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]

        for name, counts in report["eval_text"].items():
            lines.append(
                f"| {name} | {counts['tokens']:,} | {counts['ban'] / counts['tokens']:.1%} | "
                f"{counts['continuation'] / counts['tokens']:.1%} | "
                f"{counts['shared'] / counts['tokens']:.1%} |"
            )

    lines += [
        "",
        (
            "Gate (Multi/Mono: BAN = 0, shifted = 0, spot-check CI upper < 0.01): "
            f"**{'PASS' if gate else 'FAIL'}**"
        ),
    ]
    (output / "audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
