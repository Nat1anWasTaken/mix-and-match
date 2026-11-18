from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, NamedTuple

import numpy

from .data import TokenArray
from .recipe.recipe import Cursor, Recipe, Selection
from .tokenizer import Offsets, ShardStore

# Background shard reads shared by a phase's cursors. Each cursor keeps its current
# shard and the next one in memory, compacted to the selected documents
PREFETCH_WORKERS = 4

# Second entry of the RNG seeds, so the cursor and mixing streams never collide
CURSOR_STREAM = 0
MIX_STREAM = 1


class DocumentOrigin(NamedTuple):
    phase: int
    cursor: int
    source: str
    shard: str
    # Index in the shard's stored order, the order of row_ids.npy and meta/ columns
    document: int
    repeat: bool


class LoadedShard(NamedTuple):
    tokens: TokenArray
    offsets: Offsets
    # Which stored document each compacted document is
    selection: Selection


class CursorStream:
    """
    One cursor of a phase: its shards in a random order, each shard's selected documents
    in a random order. The shard order and the in-shard permutations come from
    (seed, phase, cursor), so a repeat cursor's copies land at times independent of the
    originals
    """

    def __init__(
        self,
        recipe_store: ShardStore,
        cursor: Cursor,
        seed: int,
        phase_index: int,
        cursor_index: int,
    ) -> None:
        self.recipe_store = recipe_store
        self.store = ShardStore(cursor.store)
        self.cursor = cursor

        random_generator = numpy.random.default_rng(
            [seed, phase_index, CURSOR_STREAM, cursor_index]
        )
        # Redrawn identically on resume, so it needs no saved state
        self.shard_order = random_generator.permutation(len(cursor.shards))

        # The next document: which shard (in shard_order), and where in its permutation
        self.shard_index = 0
        self.position = 0
        self.remaining_documents = cursor.documents
        # The RNG as it was before drawing shard_index's permutation
        self.rng_state = random_generator.bit_generator.state

        self._loaded: LoadedShard | None = None
        self._order: numpy.ndarray | None = None
        self._random_generator: numpy.random.Generator | None = None
        self._pending: Future[LoadedShard] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "position": self.position,
            "remaining_documents": self.remaining_documents,
            "rng_state": self.rng_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.shard_index = state["shard_index"]
        self.position = state["position"]
        self.remaining_documents = state["remaining_documents"]
        self.rng_state = state["rng_state"]
        self._loaded = None
        self._order = None
        self._pending = None

    def next_document(self, pool: ThreadPoolExecutor) -> tuple[TokenArray, str, int]:
        """
        The next document's tokens, its shard and its index in the shard's stored order
        """
        if self.remaining_documents <= 0:
            raise RuntimeError(f"cursor {self.cursor.name} has no documents left")

        if self._loaded is None:
            self._start_shard(pool)

        assert self._loaded is not None and self._order is not None
        shard = self.cursor.shards[self.shard_order[self.shard_index]]
        compacted = int(self._order[self.position])
        tokens = self._loaded.tokens[
            self._loaded.offsets[compacted] : self._loaded.offsets[compacted + 1]
        ]
        document = int(self._loaded.selection[compacted])

        # Advance before returning: the state means "after this document"
        self.position += 1
        self.remaining_documents -= 1

        if self.position == self._order.size:
            assert self._random_generator is not None
            self.shard_index += 1
            self.position = 0
            self.rng_state = self._random_generator.bit_generator.state
            self._loaded = None
            self._order = None

        return tokens, shard.name, document

    def _start_shard(self, pool: ThreadPoolExecutor) -> None:
        if self._pending is None:
            self._pending = pool.submit(self._read_shard, self.shard_index)

        self._loaded = self._pending.result()
        self._pending = None

        random_generator = numpy.random.default_rng()
        random_generator.bit_generator.state = self.rng_state
        # The shards store their documents in fragment order, so this permutation is
        # still the last step of prepare.py's global shuffle and must not be removed
        self._order = random_generator.permutation(self._loaded.selection.size)
        self._random_generator = random_generator

        if self.shard_index + 1 < len(self.cursor.shards):
            self._pending = pool.submit(self._read_shard, self.shard_index + 1)

    def _read_shard(self, shard_index: int) -> LoadedShard:
        """
        Reads a whole shard and keeps only the selected documents. pyarrow releases the
        GIL during the read, so it overlaps training
        """
        shard = self.cursor.shards[self.shard_order[shard_index]]
        selection: Selection = self.recipe_store.read_array(shard.selection)
        tokens = self.store.read_tokens(shard.name)
        offsets = self.store.read_offsets(shard.name)

        if selection.size != shard.documents:
            raise ValueError(
                f"{shard.selection} selects {selection.size} documents, "
                f"recipe.json says {shard.documents}"
            )

        if selection.size == offsets.size - 1:
            return LoadedShard(tokens, offsets, selection)

        starts = offsets[selection]
        stops = offsets[selection.astype(numpy.int64) + 1]
        compacted_offsets = numpy.zeros(selection.size + 1, dtype=numpy.int64)
        numpy.cumsum(stops - starts, out=compacted_offsets[1:])
        compacted_tokens = numpy.concatenate(
            [tokens[start:stop] for start, stop in zip(starts, stops, strict=True)]
        )
        return LoadedShard(compacted_tokens, compacted_offsets, selection)


class RecipeTokenStream:
    """
    Documents of a compiled recipe, phase by phase. Inside a phase, the next document is
    drawn uniformly from all the documents the phase has left: a cursor is picked with
    probability proportional to its remaining documents. So every cursor drains at the
    same relative rate and all of them end together, and the token mix stays the same
    throughout the phase. Weighting by remaining tokens instead would drift: a cursor of
    long documents loses more tokens per draw and its share would shrink over the phase.

    Every rank draws the same sequence, and document g belongs to rank g % world_size
    """

    def __init__(self, recipe: Recipe, rank: int = 0, world_size: int = 1) -> None:
        self.recipe = recipe
        self.recipe_store = ShardStore(recipe.uri)
        self.rank = rank
        self.world_size = world_size

        # Global document counter across phases
        self.documents_seen = 0
        # This rank's yielded tokens, for logging total_tokens and oj_total_tokens
        self.tokens_by_source = dict.fromkeys(recipe.sources(), 0)
        self.repeated_tokens = 0

        self._start_phase(0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "phase_index": self.phase_index,
            "mix_rng_state": self.mix_rng.bit_generator.state,
            "cursors": [cursor.state_dict() for cursor in self.cursors],
            "documents_seen": self.documents_seen,
            "tokens_by_source": dict(self.tokens_by_source),
            "repeated_tokens": self.repeated_tokens,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._start_phase(state["phase_index"])
        self.mix_rng.bit_generator.state = state["mix_rng_state"]

        for cursor, cursor_state in zip(self.cursors, state["cursors"], strict=True):
            cursor.load_state_dict(cursor_state)

        self.documents_seen = state["documents_seen"]
        self.tokens_by_source = dict(state["tokens_by_source"])
        self.repeated_tokens = state["repeated_tokens"]

    def stats(self) -> dict[str, int]:
        """
        Tokens this rank has handed to the packer, by source. They run ahead of the
        trained tokens by the packer's buffer and the loader's prefetch, a few rows
        """
        return {
            **{
                f"tokens/{source}": tokens
                for source, tokens in self.tokens_by_source.items()
            },
            "repeated_tokens": self.repeated_tokens,
        }

    def __iter__(self) -> Iterator[TokenArray]:
        for tokens, _ in self.iter_with_origin():
            yield tokens

    def iter_with_origin(self) -> Iterator[tuple[TokenArray, DocumentOrigin]]:
        with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
            while self.phase_index < len(self.recipe.phases):
                remaining = numpy.array(
                    [cursor.remaining_documents for cursor in self.cursors],
                    dtype=numpy.int64,
                )
                total = int(remaining.sum())

                while total > 0:
                    draw = self.mix_rng.random() * total
                    index = int(
                        numpy.searchsorted(numpy.cumsum(remaining), draw, side="right")
                    )
                    cursor = self.cursors[index]
                    tokens, shard, document = cursor.next_document(pool)
                    remaining[index] -= 1
                    total -= 1

                    mine = self.documents_seen % self.world_size == self.rank
                    self.documents_seen += 1

                    if mine:
                        self.tokens_by_source[cursor.cursor.source] += tokens.size

                        if cursor.cursor.repeat:
                            self.repeated_tokens += tokens.size

                        yield (
                            tokens,
                            DocumentOrigin(
                                phase=self.phase_index,
                                cursor=index,
                                source=cursor.cursor.source,
                                shard=shard,
                                document=document,
                                repeat=cursor.cursor.repeat,
                            ),
                        )

                self._start_phase(self.phase_index + 1)

    def _start_phase(self, phase_index: int) -> None:
        self.phase_index = phase_index
        self.mix_rng = numpy.random.default_rng(
            [self.recipe.seed, phase_index, MIX_STREAM]
        )
        self.cursors: list[CursorStream] = []

        if phase_index < len(self.recipe.phases):
            self.cursors = [
                CursorStream(
                    self.recipe_store,
                    cursor,
                    self.recipe.seed,
                    phase_index,
                    cursor_index,
                )
                for cursor_index, cursor in enumerate(
                    self.recipe.phases[phase_index].cursors
                )
            ]
