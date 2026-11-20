"""
A local copy of the files a compiled recipe reads while training, so the ranks read
/work instead of S3. An s3://<bucket>/<key> URI maps to <cache>/<bucket>/<key>:

    <cache>/<bucket>/<recipe key>/recipe.json                 the recipe, as on S3
    <cache>/<bucket>/<recipe key>/selections/...              its selection files
    <cache>/<bucket>/<store key>/data/<shard>/tokens.npy      each selected shard
    <cache>/<bucket>/<store key>/data/<shard>/offsets.npy

So one cache holds any number of recipes, and recipes over the same stores share their
shards. Only what RecipeTokenStream reads is copied: no row_ids.npy, no meta/, no
shards the recipe selects nothing from.

Why: every rank reads every shard (document g goes to rank g % world_size, so the data
order doesn't depend on the rank count), so S3 traffic grows with the ranks, and Flock
S3 gives about 0.85 GB/s in total however many nodes read. On /work, a node's 8 ranks
read the same files through the node's page cache, so each node reads a shard about
once.

    python -m src.recipe.cache --recipe s3://mix-and-match/recipes/dolma-v1_7-30b \\
        --cache /work/u9216411/acd115198/recipe-cache

Run it on a compute node (scripts/cache-recipe.sbatch). Every file is written to a
temporary file of its own and renamed into place once whole, and recipe.json is written
last, so its presence means the copy is complete, even with several copies running at
once. A rerun skips shard files already there with the source's size, so an interrupted
copy picks up where it stopped. train.py --recipe-cache then reads the copy, and
refuses one that is incomplete or whose recipe.json differs from the source rather than
quietly falling back to S3.

Assumes shard stores are immutable once written (prepare.py writes each shard once).
A recipe directory may be recompiled in place, so when the cached recipe.json differs
from the source, the recipe's own files are copied again whatever their size.
"""

import argparse
import dataclasses
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import pyarrow.fs

from ..tokenizer import ShardStore
from .recipe import RECIPE_FILE, Recipe, load_recipe

SHARD_FILES = ("tokens.npy", "offsets.npy")


def cache_path(uri: str, cache: Path) -> str:
    """
    Where an s3:// URI lives in the cache. Local paths are returned as they are, since
    reading them needs no copy
    """
    if not uri.startswith("s3://"):
        return uri

    return str(Path(cache).absolute() / uri.removeprefix("s3://").strip("/"))


@dataclasses.dataclass(frozen=True)
class CachedFile:
    store: str
    # Relative to the store's root
    relative: str
    # Files of the recipe directory itself, recopied when recipe.json changes
    in_recipe: bool


def recipe_files(recipe: Recipe) -> list[CachedFile]:
    """
    Every file RecipeTokenStream reads for this recipe, except recipe.json itself, once
    per cache destination: store URIs that differ only by a trailing slash map to the
    same files, and copying one file twice at once would race
    """
    files: dict[tuple[str, str], CachedFile] = {}

    def add(file: CachedFile) -> None:
        files.setdefault((file.store.rstrip("/"), file.relative), file)

    for phase in recipe.phases:
        for cursor in phase.cursors:
            for shard in cursor.shards:
                add(CachedFile(recipe.uri, shard.selection, in_recipe=True))

                for name in SHARD_FILES:
                    relative = str(PurePosixPath("data", shard.name, name))
                    add(CachedFile(cursor.store, relative, in_recipe=False))

    return list(files.values())


def copy_recipe(uri: str, cache: Path, workers: int = 16) -> Recipe:
    """
    Copies a recipe and the shard files it reads into the cache, recipe.json last.
    Returns the recipe as loaded from the source
    """
    # One read of recipe.json for both the file list and the commit point, so the
    # recipe.json written to the cache names exactly the files copied for it, even if
    # the source is recompiled during the copy
    source_store = ShardStore(uri)
    source_recipe_bytes = _read_bytes(source_store, RECIPE_FILE)
    recipe = Recipe.from_dict(json.loads(source_recipe_bytes), uri=source_store.uri)
    files = [file for file in recipe_files(recipe) if file.store.startswith("s3://")]
    mode = _file_mode()

    cached_recipe = Path(cache_path(recipe.uri, cache)) / RECIPE_FILE
    recipe_changed = (
        not cached_recipe.is_file() or cached_recipe.read_bytes() != source_recipe_bytes
    )
    if recipe_changed and cached_recipe.is_file():
        # Without a complete recipe.json the cache can't be used until this finishes
        cached_recipe.unlink()

    filesystems = {store: ShardStore(store) for store in {file.store for file in files}}
    start = time.perf_counter()
    copied_bytes = 0
    skipped = 0

    def copy(file: CachedFile) -> int:
        store = filesystems[file.store]
        source = str(store.root / file.relative)
        destination = Path(cache_path(file.store, cache)) / file.relative

        info = store.filesystem.get_file_info(source)
        if info.type != pyarrow.fs.FileType.File:
            raise FileNotFoundError(f"{file.store}/{file.relative}")

        reuse = not (file.in_recipe and recipe_changed)
        if reuse and destination.is_file() and destination.stat().st_size == info.size:
            return -1

        # One read() of the whole file, one GET. pyarrow.fs.copy_files reads in 1 MB
        # chunks, each its own ranged GET, so latency capped it at ~12 MB/s per file.
        # A file is at most a few hundred MB, so holding it in memory is fine
        with store.filesystem.open_input_file(source) as stream:
            data = stream.read()
        if len(data) != info.size:
            raise OSError(f"read {len(data)} bytes of {source}, expected {info.size}")
        _write_atomically(destination, data, mode)
        return info.size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, size in enumerate(pool.map(copy, files), start=1):
            if size < 0:
                skipped += 1
            else:
                copied_bytes += size

            if index % 200 == 0 or index == len(files):
                seconds = time.perf_counter() - start
                print(
                    f"cache: {index}/{len(files)} files, {skipped} already cached, "
                    f"{copied_bytes / 1e9:.1f} GB copied in {seconds:.0f} s "
                    f"({copied_bytes / 1e6 / max(seconds, 1e-9):.0f} MB/s)",
                    flush=True,
                )

    # The commit point: written only once every file it names is in place
    _write_atomically(cached_recipe, source_recipe_bytes, mode)
    print(f"cache: {recipe.uri} -> {cached_recipe.parent}", flush=True)

    return recipe


def localize_recipe(recipe: Recipe, cache: Path) -> Recipe:
    """
    The recipe with its URI and every cursor's store pointing into the cache. Raises if
    the cache doesn't hold a complete copy of this exact recipe
    """
    if not recipe.uri.startswith("s3://"):
        raise ValueError(f"--recipe-cache needs an s3:// recipe, got {recipe.uri}")

    local_uri = cache_path(recipe.uri, cache)
    if not (Path(local_uri) / RECIPE_FILE).is_file():
        raise FileNotFoundError(
            f"{local_uri} holds no complete copy of {recipe.uri}; run "
            f"python -m src.recipe.cache --recipe {recipe.uri} --cache {cache}"
        )

    cached = load_recipe(local_uri)
    if cached.to_dict() != recipe.to_dict():
        raise ValueError(
            f"{local_uri}/{RECIPE_FILE} differs from {recipe.uri}/{RECIPE_FILE}, the "
            "recipe was recompiled; copy it again with python -m src.recipe.cache"
        )

    missing = [
        f"{file.store}/{file.relative}"
        for file in recipe_files(recipe)
        if not (Path(cache_path(file.store, cache)) / file.relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files of {recipe.uri} are missing from {cache}, "
            f"e.g. {missing[0]}; copy it again with python -m src.recipe.cache"
        )

    return dataclasses.replace(
        recipe,
        uri=local_uri,
        phases=[
            dataclasses.replace(
                phase,
                cursors=[
                    dataclasses.replace(cursor, store=cache_path(cursor.store, cache))
                    for cursor in phase.cursors
                ],
            )
            for phase in recipe.phases
        ],
    )


def _read_bytes(store: ShardStore, name: str) -> bytes:
    if not store._is_file(store.root / name):
        raise FileNotFoundError(f"no {name} in {store.uri}")

    with store.filesystem.open_input_stream(str(store.root / name)) as stream:
        return stream.read()


def _file_mode() -> int:
    """
    The mode open() would give a new file under the current umask (0o640 for umask
    0o027). mkstemp creates its files 0o600, which would also cut the ACL mask and with
    it the group's read access to the shared cache, so the copies get this mode back.
    Read once before any copy thread starts, since os.umask is process-wide
    """
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def _write_atomically(destination: Path, data: bytes, mode: int) -> None:
    """
    Writes data to destination through a temporary file of its own beside it, renamed
    into place once whole. The temporary name is unique, so two jobs copying the same
    file never write into each other's file, and a file in the cache is always one
    job's complete copy. On failure only this call's temporary file is removed
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f"{destination.name}.", suffix=".tmp"
    )
    temporary = Path(name)

    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
        os.chmod(temporary, mode)

        if temporary.stat().st_size != len(data):
            raise OSError(
                f"wrote {temporary.stat().st_size} bytes to {temporary}, "
                f"expected {len(data)}"
            )

        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--recipe", required=True, help="s3:// compiled recipe")
    parser.add_argument("--cache", type=Path, required=True, help="cache root")
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()

    recipe = copy_recipe(arguments.recipe, arguments.cache, arguments.workers)
    # Check the copy the same way train.py will
    localize_recipe(recipe, arguments.cache)
    print("cache: verified", flush=True)


if __name__ == "__main__":
    main()
