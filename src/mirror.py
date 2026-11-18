"""
A local copy of some fragments of a Lance dataset, for compute nodes without internet.

    <mirror>/_versions/, _transactions/, ...   every file outside data/, copied as is
    <mirror>/data/<file>.lance                  the data files of the mirrored fragments
    <mirror>/mirror.json                        source URI, version and mirrored fragments

The manifests are the source's, so the mirror opens at the same version with the same
fragment ids and row addresses: row_ids.npy made from the mirror are valid on the
source too. Fragments that were not copied are still listed, but reading them fails, so
callers check mirror.json first.
"""

import json
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow.fs
from lance import LanceDataset

from .tokenizer import open_dataset, open_filesystem

MIRROR_FILE = "mirror.json"
DATA_DIRECTORY = "data"


@dataclass(frozen=True)
class MirrorInfo:
    source: str
    version: int
    fragment_ids: frozenset[int]


def read_mirror(mirror: str) -> MirrorInfo | None:
    path = Path(mirror) / MIRROR_FILE

    if not path.is_file():
        return None

    content = json.loads(path.read_text())
    return MirrorInfo(
        source=content["source"],
        version=int(content["version"]),
        fragment_ids=frozenset(int(i) for i in content["fragment_ids"]),
    )


def fetch_fragments(
    source: str,
    version: int,
    mirror: str,
    fragment_ids: Iterable[int],
    workers: int = 16,
) -> MirrorInfo:
    """
    Copies the source's metadata and the given fragments' data files into the mirror.
    Files already there with the right size are skipped, so a rerun only adds what is
    missing. Run on a host with internet
    """
    existing = read_mirror(mirror)

    if existing is not None and (existing.source, existing.version) != (
        source,
        version,
    ):
        raise ValueError(
            f"{mirror} mirrors {existing.source} version {existing.version}, "
            f"not {source} version {version}; use a different --mirror"
        )

    dataset = open_dataset(source, version)
    filesystem, root = open_filesystem(source)
    root = root.rstrip("/")
    wanted = sorted(set(fragment_ids))

    # (path relative to the dataset root, size or None if unknown)
    files: list[tuple[str, int | None]] = []

    for info in filesystem.get_file_info(pyarrow.fs.FileSelector(root, recursive=True)):
        relative = info.path.removeprefix(root + "/")

        if info.type == pyarrow.fs.FileType.File and not relative.startswith(
            DATA_DIRECTORY + "/"
        ):
            files.append((relative, info.size))

    for fragment_id in wanted:
        fragment = dataset.get_fragment(fragment_id)

        if fragment is None:
            raise ValueError(f"{source} has no fragment {fragment_id}")

        for data_file in fragment.metadata.files:
            files.append(
                (f"{DATA_DIRECTORY}/{data_file.path}", data_file.file_size_bytes)
            )

    def copy(entry: tuple[str, int | None]) -> int:
        relative, size = entry
        source_path = f"{root}/{relative}"
        local_path = Path(mirror) / relative

        if size is None:
            size = filesystem.get_file_info(source_path).size

        if local_path.is_file() and local_path.stat().st_size == size:
            return 0

        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy next to the target, then rename: a killed copy never looks finished
        temporary_path = local_path.with_name(local_path.name + ".tmp")
        pyarrow.fs.copy_files(
            source_path,
            str(temporary_path),
            source_filesystem=filesystem,
            destination_filesystem=pyarrow.fs.LocalFileSystem(),
        )
        os.replace(temporary_path, local_path)
        return local_path.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        copied = sum(pool.map(copy, files))

    mirrored = frozenset(wanted) | (existing.fragment_ids if existing else frozenset())
    info = MirrorInfo(source=source, version=version, fragment_ids=mirrored)

    # mirror.json last: it only lists fragments whose files are all there
    temporary_path = Path(mirror) / (MIRROR_FILE + ".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "source": source,
                "version": version,
                "fragment_ids": sorted(mirrored),
            },
            indent=2,
        )
    )
    os.replace(temporary_path, Path(mirror) / MIRROR_FILE)

    print(
        f"mirrored {len(wanted)} fragments of {source} into {mirror} "
        f"({copied / 1e9:.1f} GB copied, {len(mirrored)} fragments in total)",
        flush=True,
    )
    return info


def open_mirror(mirror: str, source: str, version: int | None = None) -> MirrorInfo:
    """
    The mirror's info, after checking it copies the given source (and version, if
    given). Open the dataset with open_dataset(mirror, info.version)
    """
    info = read_mirror(mirror)

    if info is None:
        raise FileNotFoundError(
            f"no {MIRROR_FILE} in {mirror}; run --stage fetch on a login node first"
        )

    if info.source != source or (version is not None and info.version != version):
        raise ValueError(
            f"{mirror} mirrors {info.source} version {info.version}, "
            f"not {source} version {version}"
        )

    return info


def check_mirrored(info: MirrorInfo, mirror: str, fragment_ids: Iterable[int]) -> None:
    missing = sorted(set(fragment_ids) - info.fragment_ids)

    if missing:
        raise FileNotFoundError(
            f"{mirror} lacks {len(missing)} fragments (e.g. {missing[:5]}); "
            "rerun --stage fetch on a login node"
        )


def open_mirrored_dataset(mirror: str, info: MirrorInfo) -> LanceDataset:
    return open_dataset(str(Path(mirror).absolute()), info.version)
