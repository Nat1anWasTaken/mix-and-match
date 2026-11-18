"""
Stages for hosts whose compute nodes have no internet (Forerunner1). Each data script
takes the same arguments in every stage, and only --stage changes:

    login node:    <script> --stage fetch   ...   download what process will read
    compute node:  <script> --stage process ...   offline, local inputs and outputs
    login node:    <script> --stage upload  ...   copy the local store to --remote

--stage all (the default) is the online path: read and write S3 directly in one go.
scripts/submit-f1.sh runs the three stages in order with the same arguments.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import huggingface_hub.constants
import pyarrow.fs

from .tokenizer import open_filesystem

STAGES = ["all", "fetch", "process", "upload"]

# Readers treat these files as commit points (a store is finished once manifest.json
# exists, a recipe once recipe.json does), so they go last, in this order
COMMIT_ORDER = ["info.json", "metadata.json", "manifest.json", "recipe.json"]
# Phase 1 scratch of prepare.py; nothing reads it once manifest.json exists
SKIP_DIRECTORIES = {"fragments"}


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="fetch on a login node, process on a compute node (offline), upload on a "
        "login node; all does everything online",
    )
    parser.add_argument(
        "--mirror",
        help="local copy of the source Lance dataset, written by --stage fetch and "
        "read by --stage process",
    )
    parser.add_argument(
        "--remote", help="s3://bucket/prefix that --stage upload copies the store to"
    )


def check_stage(arguments: argparse.Namespace, *local: str) -> None:
    """
    local names the arguments that must be local paths for fetch, process and upload
    (the store; on a compute node there is no S3)
    """
    stage = arguments.stage

    if stage == "all":
        if arguments.mirror is not None or arguments.remote is not None:
            raise ValueError("--mirror and --remote only apply to the split stages")

        return

    for name in local:
        value = getattr(arguments, name)

        if value is not None and "://" in value:
            raise ValueError(
                f"--stage {stage} needs a local --{name.replace('_', '-')}, "
                f"not {value}; --remote is where it goes"
            )

    if stage in ("fetch", "process") and arguments.mirror is None:
        raise ValueError(f"--stage {stage} needs --mirror")

    if stage == "fetch" and huggingface_hub.constants.HF_HUB_OFFLINE:
        raise RuntimeError("--stage fetch downloads; unset HF_HUB_OFFLINE")

    if stage == "process" and not huggingface_hub.constants.HF_HUB_OFFLINE:
        raise RuntimeError(
            "--stage process runs on compute nodes without internet; "
            "set HF_HUB_OFFLINE=1 (scripts/f1-cpu.sbatch does)"
        )

    if stage == "upload" and arguments.remote is None:
        raise ValueError("--stage upload needs --remote")


def _commit_rank(relative: str) -> int:
    # 0 for plain files, then one rank per commit point, in COMMIT_ORDER
    name = relative.rsplit("/", 1)[-1]

    if name in COMMIT_ORDER:
        return COMMIT_ORDER.index(name) + 1

    return 0


def _list_files(filesystem: pyarrow.fs.FileSystem, root: str) -> dict[str, int]:
    """
    Every file under root by path relative to it, with its size
    """
    selector = pyarrow.fs.FileSelector(root, recursive=True, allow_not_found=True)
    return {
        info.path.removeprefix(root + "/"): info.size
        for info in filesystem.get_file_info(selector)
        if info.type == pyarrow.fs.FileType.File
    }


def _read_json(filesystem: pyarrow.fs.FileSystem, path: str) -> object:
    with filesystem.open_input_stream(path) as stream:
        return json.loads(stream.read())


def upload_store(local: str, remote: str, workers: int = 16) -> None:
    """
    Copies a local shard store (or recipe) to remote. Files already there with the same
    size are skipped, so rerun it after an interruption or after adding meta/ columns.

    Plain files go first and commit points last, each rank only once the one before is
    fully uploaded, so a reader never sees manifest.json before the shards it lists.
    A remote whose commit points differ from the local ones holds a different store,
    and nothing is uploaded
    """
    local_filesystem, local_root = open_filesystem(local)
    remote_filesystem, remote_root = open_filesystem(remote)
    local_root = local_root.rstrip("/")
    remote_root = remote_root.rstrip("/")

    local_files = {
        relative: size
        for relative, size in _list_files(local_filesystem, local_root).items()
        if not relative.endswith(".tmp")
        and relative.split("/", 1)[0] not in SKIP_DIRECTORIES
    }
    remote_files = _list_files(remote_filesystem, remote_root)

    for relative in local_files:
        if _commit_rank(relative) and relative in remote_files:
            local_content = _read_json(local_filesystem, f"{local_root}/{relative}")
            remote_content = _read_json(remote_filesystem, f"{remote_root}/{relative}")

            if local_content != remote_content:
                raise ValueError(
                    f"{remote}/{relative} differs from {local}/{relative}; "
                    "the remote holds a different store, use another --remote"
                )

    def upload(relative: str) -> int:
        size = local_files[relative]

        if remote_files.get(relative) == size:
            return 0

        # S3 has no directories, and create_dir there would leave marker objects
        if isinstance(remote_filesystem, pyarrow.fs.LocalFileSystem):
            parent = f"{remote_root}/{relative}".rsplit("/", 1)[0]
            remote_filesystem.create_dir(parent, recursive=True)

        pyarrow.fs.copy_files(
            f"{local_root}/{relative}",
            f"{remote_root}/{relative}",
            source_filesystem=local_filesystem,
            destination_filesystem=remote_filesystem,
        )
        return size

    for rank in range(len(COMMIT_ORDER) + 1):
        group = sorted(
            relative for relative in local_files if _commit_rank(relative) == rank
        )

        if not group:
            continue

        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = list(pool.map(upload, group))

        uploaded = sum(1 for size in sizes if size)
        print(
            f"uploaded {uploaded} of {len(group)} "
            f"{'files' if rank == 0 else COMMIT_ORDER[rank - 1]} "
            f"({sum(sizes) / 1e9:.2f} GB) to {remote}",
            flush=True,
        )
