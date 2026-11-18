"""
The compiled recipe: what RecipeTokenStream trains on.

A recipe directory holds recipe.json plus one selection per (phase, cursor, shard):

    <recipe>/recipe.json
    <recipe>/selections/<phase>/<cursor>/<shard>.npy    sorted unique uint32 doc indices

A recipe is a list of phases, trained in order. A phase is a set of cursors mixed
uniformly at random; a cursor is a selection of documents from one source store. A
document index is its position in the shard's stored order, the same order meta/
columns and row_ids.npy use. recipe.json only names things, every per-document array
lives in a selection file, so the JSON stays small at any scale.

A cursor marked repeat holds extra copies of documents some non-repeat cursor already
selects. Their tokens are the ones LAB.md lets us repeat for free, up to 10% of the rest.
"""

import dataclasses
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy
from numpy.typing import ArrayLike

from ..tokenizer import ShardStore

RECIPE_VERSION = 1
RECIPE_FILE = "recipe.json"
MAX_REPEAT_FRACTION = 0.1

# Names become path components of the selection files
NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

type Selection = numpy.ndarray[tuple[int], numpy.dtype[numpy.uint32]]
type Lengths = numpy.ndarray[tuple[int], numpy.dtype[numpy.int64]]


@dataclass(frozen=True)
class ShardSelection:
    name: str
    # Path of the selection file, relative to the recipe directory
    selection: str
    tokens: int
    documents: int


@dataclass(frozen=True)
class Cursor:
    name: str
    source: str
    store: str
    repeat: bool
    tokens: int
    documents: int
    shards: list[ShardSelection]


@dataclass(frozen=True)
class Phase:
    name: str
    tokens: int
    documents: int
    cursors: list[Cursor]


@dataclass(frozen=True)
class Recipe:
    version: int
    seed: int
    tokenizer: str
    total_tokens: int
    repeated_tokens: int
    phases: list[Phase]
    # Where the recipe was loaded from; not part of recipe.json
    uri: str = ""

    def sources(self) -> list[str]:
        return sorted(
            {cursor.source for phase in self.phases for cursor in phase.cursors}
        )

    def to_dict(self) -> dict[str, Any]:
        content = dataclasses.asdict(self)
        del content["uri"]
        return content

    @staticmethod
    def from_dict(content: Mapping[str, Any], uri: str = "") -> "Recipe":
        if content["version"] != RECIPE_VERSION:
            raise ValueError(
                f"recipe version {content['version']}, expected {RECIPE_VERSION}"
            )

        return Recipe(
            version=content["version"],
            seed=content["seed"],
            tokenizer=content["tokenizer"],
            total_tokens=content["total_tokens"],
            repeated_tokens=content["repeated_tokens"],
            phases=[
                Phase(
                    name=phase["name"],
                    tokens=phase["tokens"],
                    documents=phase["documents"],
                    cursors=[
                        Cursor(
                            name=cursor["name"],
                            source=cursor["source"],
                            store=cursor["store"],
                            repeat=cursor["repeat"],
                            tokens=cursor["tokens"],
                            documents=cursor["documents"],
                            shards=[
                                ShardSelection(**shard) for shard in cursor["shards"]
                            ],
                        )
                        for cursor in phase["cursors"]
                    ],
                )
                for phase in content["phases"]
            ],
            uri=uri,
        )


def load_recipe(uri: str) -> Recipe:
    store = ShardStore(uri)
    content = store.read_json(RECIPE_FILE)

    if content is None:
        raise FileNotFoundError(f"no {RECIPE_FILE} in {store.uri}")

    return Recipe.from_dict(content, uri=store.uri)


def read_shard_names(store: ShardStore) -> list[str]:
    manifest = store.read_json("manifest.json")

    if manifest is None:
        raise FileNotFoundError(f"no manifest.json in {store.uri}, run prepare.py")

    shards = manifest["shards"]

    if not isinstance(shards, list) or not shards:
        raise ValueError(f"manifest.json in {store.uri} lists no shards")

    return [shard["name"] for shard in shards]


def read_document_lengths(store: ShardStore, workers: int = 16) -> dict[str, Lengths]:
    """
    Token count of every document, per shard, from offsets.npy alone. A 30B-token store
    has ~600 shards of a few hundred KB of offsets each, so threads hide the S3 latency
    """
    names = read_shard_names(store)

    def lengths(name: str) -> Lengths:
        return numpy.diff(store.read_offsets(name))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(zip(names, pool.map(lengths, names), strict=True))


def take_by_priority(
    priorities: Mapping[str, numpy.ndarray],
    lengths: Mapping[str, Lengths],
    token_budget: int,
    random_generator: numpy.random.Generator,
    eligible: Mapping[str, Selection] | None = None,
    exceed: bool = True,
) -> dict[str, Selection]:
    """
    Documents in descending priority until token_budget, as per-shard selections.
    Ties break at random, so a coarse score like int_score still takes a uniform sample
    of its boundary class instead of whichever shards come first. A constant priority
    is a uniform random sample of token_budget tokens.

    eligible limits the candidates to a selection (default: every document). exceed
    takes the smallest prefix that reaches the budget; without it, the largest that
    stays within it, for budgets that are a hard limit like the repeat allowance.

    Always one global ranking: a top-k per shard would tie the threshold to a shard's
    contents, which is only harmless while every shard is a uniform sample
    """
    names = list(priorities)

    if eligible is None:
        candidates = [numpy.arange(priorities[name].size) for name in names]
    else:
        candidates = [
            eligible.get(name, numpy.empty(0, dtype=numpy.uint32)).astype(numpy.int64)
            for name in names
        ]

    shard_of = numpy.concatenate(
        [
            numpy.full(documents.size, index)
            for index, documents in enumerate(candidates)
        ]
    )
    document_of = numpy.concatenate(candidates)
    all_priorities = numpy.concatenate(
        [
            priorities[name][documents]
            for name, documents in zip(names, candidates, strict=True)
        ]
    )
    all_lengths = numpy.concatenate(
        [
            lengths[name][documents]
            for name, documents in zip(names, candidates, strict=True)
        ]
    )

    tiebreak = random_generator.random(all_priorities.size)
    # lexsort sorts by the last key first: priority descending, then the random tiebreak
    order = numpy.lexsort((tiebreak, -all_priorities))
    cumulative = numpy.cumsum(all_lengths[order])

    if exceed:
        if cumulative.size == 0 or cumulative[-1] < token_budget:
            available = int(cumulative[-1]) if cumulative.size else 0
            raise ValueError(
                f"need {token_budget} tokens but only {available} are available"
            )

        # Smallest prefix that reaches the budget
        keep = order[: int(numpy.searchsorted(cumulative, token_budget)) + 1]
    else:
        # Largest prefix that stays within it
        keep = order[: int(numpy.searchsorted(cumulative, token_budget, side="right"))]

    # Group the kept documents by shard, each shard's in ascending order
    kept_shards = shard_of[keep]
    kept_documents = document_of[keep]
    grouped = numpy.lexsort((kept_documents, kept_shards))
    counts = numpy.bincount(kept_shards, minlength=len(names))
    parts = numpy.split(kept_documents[grouped], numpy.cumsum(counts)[:-1])

    return {
        name: documents.astype(numpy.uint32)
        for name, documents in zip(names, parts, strict=True)
        if documents.size
    }


def subtract_selections(
    selections: Mapping[str, Selection], removed: Mapping[str, Selection]
) -> dict[str, Selection]:
    result: dict[str, Selection] = {}

    for name, documents in selections.items():
        if name in removed:
            documents = numpy.setdiff1d(documents, removed[name]).astype(numpy.uint32)

        if documents.size:
            result[name] = documents

    return result


def selected_tokens(
    selections: Mapping[str, Selection], lengths: Mapping[str, Lengths]
) -> int:
    return sum(
        int(lengths[name][documents].sum()) for name, documents in selections.items()
    )


@dataclass
class _PendingCursor:
    phase: str
    name: str
    source: str
    store: ShardStore
    repeat: bool
    selections: dict[str, Selection]


class RecipeBuilder:
    """
    Collects cursors, validates them against the stores, and writes the recipe. Every
    count in recipe.json comes from offsets.npy, so train.py knows the token totals
    before reading a single token
    """

    def __init__(
        self, seed: int = 42, max_repeat_fraction: float = MAX_REPEAT_FRACTION
    ) -> None:
        self.seed = seed
        self.max_repeat_fraction = max_repeat_fraction
        self.phase_names: list[str] = []
        self.cursors: list[_PendingCursor] = []
        self._lengths: dict[str, dict[str, Lengths]] = {}
        self._tokenizers: dict[str, str] = {}

    def lengths(self, store: ShardStore) -> dict[str, Lengths]:
        if store.uri not in self._lengths:
            self._lengths[store.uri] = read_document_lengths(store)

        return self._lengths[store.uri]

    def add_phase(self, name: str) -> None:
        """
        Phases train in the order they are added
        """
        _check_name(name)

        if name in self.phase_names:
            raise ValueError(f"phase {name} already exists")

        self.phase_names.append(name)

    def add_cursor(
        self,
        phase: str,
        name: str,
        source: str,
        store: ShardStore,
        selections: Mapping[str, ArrayLike],
        repeat: bool = False,
    ) -> None:
        _check_name(name)

        if phase not in self.phase_names:
            raise ValueError(f"add phase {phase} before its cursors")

        if any(c.phase == phase and c.name == name for c in self.cursors):
            raise ValueError(f"phase {phase} already has a cursor {name}")

        self._check_tokenizer(store)
        lengths = self.lengths(store)
        checked: dict[str, Selection] = {}

        for shard, documents in selections.items():
            if shard not in lengths:
                raise ValueError(f"{store.uri} has no shard {shard}")

            documents = numpy.asarray(documents)

            if documents.size == 0:
                continue

            if not numpy.issubdtype(documents.dtype, numpy.integer):
                raise ValueError(
                    f"selection for shard {shard} has non-integer indices "
                    f"({documents.dtype})"
                )

            # Pairwise, not numpy.diff: diff wraps around on unsigned indices
            if documents.ndim != 1 or numpy.any(documents[1:] <= documents[:-1]):
                raise ValueError(
                    f"selection for shard {shard} is not sorted and unique"
                )

            if documents[0] < 0 or documents[-1] >= lengths[shard].size:
                raise ValueError(f"selection for shard {shard} is out of range")

            checked[shard] = documents.astype(numpy.uint32)

        if not checked:
            raise ValueError(f"cursor {phase}/{name} selects no documents")

        self.cursors.append(
            _PendingCursor(
                phase, name, source, store, repeat, dict(sorted(checked.items()))
            )
        )

    def build(self) -> Recipe:
        self._check_repeats()

        phases: list[Phase] = []

        for phase_name in self.phase_names:
            cursors = [
                self._build_cursor(cursor)
                for cursor in self.cursors
                if cursor.phase == phase_name
            ]

            if not cursors:
                raise ValueError(f"phase {phase_name} has no cursors")

            phases.append(
                Phase(
                    name=phase_name,
                    tokens=sum(cursor.tokens for cursor in cursors),
                    documents=sum(cursor.documents for cursor in cursors),
                    cursors=cursors,
                )
            )

        if not phases:
            raise ValueError("recipe has no phases")

        total_tokens = sum(phase.tokens for phase in phases)
        repeated_tokens = sum(
            cursor.tokens
            for phase in phases
            for cursor in phase.cursors
            if cursor.repeat
        )

        if repeated_tokens > self.max_repeat_fraction * (
            total_tokens - repeated_tokens
        ):
            raise ValueError(
                f"{repeated_tokens} repeated tokens exceed {self.max_repeat_fraction} of "
                f"the {total_tokens - repeated_tokens} counted ones"
            )

        return Recipe(
            version=RECIPE_VERSION,
            seed=self.seed,
            tokenizer=next(iter(self._tokenizers.values())),
            total_tokens=total_tokens,
            repeated_tokens=repeated_tokens,
            phases=phases,
        )

    def write(self, uri: str) -> Recipe:
        """
        Selections first, recipe.json last as the commit point. A recipe names a run's
        data, so an existing one is never overwritten
        """
        recipe = self.build()
        store = ShardStore(uri)

        if store.read_json(RECIPE_FILE) is not None:
            raise FileExistsError(f"{store.uri} already holds a recipe; pick a new one")

        for cursor in self.cursors:
            for shard, documents in cursor.selections.items():
                store.write_array(
                    _selection_path(cursor.phase, cursor.name, shard), documents
                )

        store.write_json(RECIPE_FILE, recipe.to_dict())
        return dataclasses.replace(recipe, uri=store.uri)

    def summary(self) -> str:
        """
        One line per cursor, with how much of its source it keeps
        """
        lines = []

        for cursor in self.cursors:
            lengths = self.lengths(cursor.store)
            available = sum(int(shard.sum()) for shard in lengths.values())
            tokens = selected_tokens(cursor.selections, lengths)
            kind = "repeat" if cursor.repeat else "unique"
            lines.append(
                f"{cursor.phase}/{cursor.name} ({kind}, {cursor.source}): "
                f"{tokens:,} tokens, {tokens / available:.1%} of {cursor.store.uri}"
            )

        return "\n".join(lines)

    def _build_cursor(self, cursor: _PendingCursor) -> Cursor:
        lengths = self.lengths(cursor.store)
        shards = [
            ShardSelection(
                name=shard,
                selection=_selection_path(cursor.phase, cursor.name, shard),
                tokens=int(lengths[shard][documents].sum()),
                documents=int(documents.size),
            )
            for shard, documents in cursor.selections.items()
        ]

        return Cursor(
            name=cursor.name,
            source=cursor.source,
            store=cursor.store.uri,
            repeat=cursor.repeat,
            tokens=sum(shard.tokens for shard in shards),
            documents=sum(shard.documents for shard in shards),
            shards=shards,
        )

    def _check_tokenizer(self, store: ShardStore) -> None:
        metadata = store.read_json("metadata.json")

        if metadata is None:
            raise FileNotFoundError(f"no metadata.json in {store.uri}")

        self._tokenizers[store.uri] = str(metadata["tokenizer"])

        if len(set(self._tokenizers.values())) > 1:
            raise ValueError(f"sources use different tokenizers: {self._tokenizers}")

    def _check_repeats(self) -> None:
        """
        Unique cursors never share a document, across all phases, so every document is
        counted once. Repeat cursors may only copy documents a unique cursor holds
        """
        unique: dict[tuple[str, str], list[Selection]] = {}

        for cursor in self.cursors:
            if not cursor.repeat:
                for shard, documents in cursor.selections.items():
                    unique.setdefault((cursor.store.uri, shard), []).append(documents)

        merged: dict[tuple[str, str], Selection] = {}

        for key, parts in unique.items():
            documents = numpy.concatenate(parts)
            merged[key] = numpy.unique(documents)

            if merged[key].size != documents.size:
                raise ValueError(
                    f"unique cursors share documents of shard {key[1]} in {key[0]}"
                )

        for cursor in self.cursors:
            if cursor.repeat:
                for shard, documents in cursor.selections.items():
                    held = merged.get((cursor.store.uri, shard))

                    if held is None or not numpy.isin(documents, held).all():
                        raise ValueError(
                            f"repeat cursor {cursor.phase}/{cursor.name} copies documents "
                            f"of shard {shard} that no unique cursor selects"
                        )


def _selection_path(phase: str, cursor: str, shard: str) -> str:
    return f"selections/{phase}/{cursor}/{shard}.npy"


def _check_name(name: str) -> None:
    # "." and ".." pass the pattern but would walk the selection paths
    if not NAME_PATTERN.match(name) or name in (".", ".."):
        raise ValueError(
            f"name {name!r} may only use letters, digits, '.', '_' and '-'"
        )
