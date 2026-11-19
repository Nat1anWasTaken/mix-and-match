"""
The harness recipe compilers run in. A compiler is a Python file in
src/recipe/compilers/ that holds one recipe's rules as code; the recipe is what it
writes, for train.py --recipe. Rules are code, so a compiler can do anything numpy can;
this module handles the rest:

    from ..compile import Context, run

    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--store", required=True)

    def compile(context: Context) -> None:
        source = context.open("fwedu", context.arguments.store, columns=["paloma"])
        clean = source.where("paloma", lambda matches: matches == 0)
        context.phase("main")
        context.cursor("fwedu", clean.sample(context.budget()))

    if __name__ == "__main__":
        run(compile, add_arguments)

run() parses the common arguments (--output, --total-tokens, --margin, --seed,
--workers) plus the compiler's own from add_arguments, calls compile, then writes the
recipe with a copy of the compiler's code and the git commit it ran at.

Speed: a compiler reads per-document arrays of many shards, often on S3, where latency
dominates. Context.open_many reads every store at once, its shards in parallel too, and
fetches the columns asked for in the same pass; Context.map runs other per-source work
on the same kind of thread pool. Selections (Docs) are sorted index arrays per shard,
so every set operation and sample is a few numpy calls, however many documents.

Determinism: one random generator, seeded by --seed, consumed in call order. Keep
Docs.sample and Docs.top on the main thread in a fixed order (not inside Context.map),
and the same code and arguments give the same recipe.
"""

import argparse
import dataclasses
import functools
import inspect
import math
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy

from ..tokenizer import ShardStore
from .metadata import read_columns
from .recipe import (
    Lengths,
    RecipeBuilder,
    Selection,
    selected_tokens,
    subtract_selections,
    take_by_priority,
)

DEFAULT_WORKERS = 16


@dataclass(eq=False)
class Source:
    """
    One shard store, opened under the name the recipe calls it. Lengths are read on
    open; columns on first use (or on open, if asked for), then kept
    """

    name: str
    store: ShardStore
    lengths: dict[str, Lengths]
    context: "Context"
    columns: dict[str, dict[str, numpy.ndarray]] = dataclasses.field(
        default_factory=dict
    )

    @functools.cached_property
    def tokens(self) -> int:
        return sum(int(shard.sum()) for shard in self.lengths.values())

    @functools.cached_property
    def documents(self) -> int:
        return sum(shard.size for shard in self.lengths.values())

    def column(self, kind: str) -> dict[str, numpy.ndarray]:
        """
        meta/<kind>/ of every shard, e.g. a quality score or the Paloma match count
        """
        if kind not in self.columns:
            self.columns[kind] = read_columns(self.store, kind, self.context.workers)

        return self.columns[kind]

    def all(self) -> "Docs":
        return Docs(
            self,
            {
                name: numpy.arange(shard.size, dtype=numpy.uint32)
                for name, shard in self.lengths.items()
                if shard.size
            },
        )

    def where(
        self, kind: str, predicate: Callable[[numpy.ndarray], numpy.ndarray]
    ) -> "Docs":
        """
        The documents whose meta/<kind> value passes predicate, a vectorized test that
        maps a shard's column to a boolean mask, e.g. lambda score: score >= 3
        """
        selections: dict[str, Selection] = {}

        for name, values in self.column(kind).items():
            mask = numpy.asarray(predicate(values))

            if mask.dtype != numpy.bool_ or mask.shape != values.shape:
                raise ValueError(
                    f"predicate on {kind} must return a boolean mask of shape "
                    f"{values.shape}, not {mask.dtype} {mask.shape}"
                )

            documents = numpy.flatnonzero(mask)

            if documents.size:
                selections[name] = documents.astype(numpy.uint32)

        return Docs(self, selections)


@dataclass(frozen=True, eq=False)
class Docs:
    """
    A selection of one source's documents: per shard, sorted unique indices into the
    shard's stored order, the form a recipe cursor takes. Immutable, so a rule can
    keep intermediate selections around and combine them with &, | and -
    """

    source: Source
    selections: Mapping[str, Selection]

    @functools.cached_property
    def tokens(self) -> int:
        return selected_tokens(self.selections, self.source.lengths)

    @functools.cached_property
    def documents(self) -> int:
        return sum(documents.size for documents in self.selections.values())

    def __and__(self, other: "Docs") -> "Docs":
        self._check_source(other)
        selections = {}

        for name, documents in self.selections.items():
            if name in other.selections:
                both = numpy.intersect1d(
                    documents, other.selections[name], assume_unique=True
                )

                if both.size:
                    selections[name] = both.astype(numpy.uint32)

        return Docs(self.source, selections)

    def __or__(self, other: "Docs") -> "Docs":
        self._check_source(other)
        selections = dict(self.selections)

        for name, documents in other.selections.items():
            if name in selections:
                documents = numpy.union1d(selections[name], documents)

            selections[name] = documents.astype(numpy.uint32)

        return Docs(self.source, dict(sorted(selections.items())))

    def __sub__(self, other: "Docs") -> "Docs":
        self._check_source(other)
        return Docs(self.source, subtract_selections(self.selections, other.selections))

    def sample(self, tokens: int, exceed: bool = True) -> "Docs":
        """
        A uniform random sample worth tokens tokens. exceed takes the smallest sample
        that reaches it; without it, the largest that stays within it (for hard limits
        like the repeat allowance)
        """
        priorities = {
            name: numpy.zeros(shard.size) for name, shard in self.source.lengths.items()
        }
        return self._take(priorities, tokens, exceed)

    def top(self, tokens: int, by: str, exceed: bool = True) -> "Docs":
        """
        The highest-scored documents by meta/<by> worth tokens tokens, ties broken at
        random. One global ranking across shards, never a top-k per shard
        """
        priorities = {
            name: column.astype(numpy.float64)
            for name, column in self.source.column(by).items()
        }
        return self._take(priorities, tokens, exceed)

    def _take(
        self, priorities: dict[str, numpy.ndarray], tokens: int, exceed: bool
    ) -> "Docs":
        try:
            selections = take_by_priority(
                priorities,
                self.source.lengths,
                tokens,
                self.source.context.random_generator,
                eligible=self.selections,
                exceed=exceed,
            )
        except ValueError as error:
            raise ValueError(f"source {self.source.name}: {error}") from error

        return Docs(self.source, selections)

    def _check_source(self, other: "Docs") -> None:
        if other.source is not self.source:
            raise ValueError(
                f"can't combine documents of {self.source.name} and {other.source.name}"
            )


class Context:
    """
    What a compiler builds the recipe through: opened sources, token budgets, phases
    and cursors. The arguments are on context.arguments, the compiler's own included
    """

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.seed: int = arguments.seed
        self.total_tokens: int | None = arguments.total_tokens
        self.margin: float = arguments.margin
        self.workers: int = arguments.workers
        self.random_generator = numpy.random.default_rng(self.seed)
        self.builder = RecipeBuilder(seed=self.seed)
        self.sources: dict[str, Source] = {}
        self._phase: str | None = None

    @property
    def target(self) -> float:
        """
        total_tokens plus the margin: what a recipe for --total-tokens should hold, so
        no rank runs out a little before the end
        """
        if self.total_tokens is None:
            raise ValueError("this compiler needs --total-tokens")

        return self.total_tokens * (1 + self.margin)

    def budget(self, fraction: float = 1.0) -> int:
        """
        A fraction of the target, rounded up
        """
        return math.ceil(self.target * fraction)

    def split(
        self, weights: Mapping[str, float], tokens: float | None = None
    ) -> dict[str, int]:
        """
        tokens (default: the target) in proportion to weights, each share rounded up so
        the shares add up to at least tokens. Weights are normalized, so any unit works
        """
        for name, weight in weights.items():
            # Also rejects NaN, which fails every comparison
            if not weight > 0:
                raise ValueError(f"{name} needs a positive weight, not {weight}")

        tokens = self.target if tokens is None else tokens
        total_weight = sum(weights.values())
        return {
            name: math.ceil(tokens * weight / total_weight)
            for name, weight in weights.items()
        }

    def open(self, name: str, uri: str, columns: Iterable[str] = ()) -> Source:
        return self.open_many({name: uri}, columns)[name]

    def open_many(
        self, uris: Mapping[str, str], columns: Iterable[str] = ()
    ) -> dict[str, Source]:
        """
        Opens stores by recipe source name: every store at once, each one's shards in
        parallel, columns read in the same pass. Returns them in the order given
        """
        columns = list(columns)

        for name in uris:
            if name in self.sources:
                raise ValueError(f"source {name} is already open")

        def load(item: tuple[str, str]) -> Source:
            name, uri = item
            store = ShardStore(uri)
            # Through the builder, which keeps the lengths for add_cursor's checks.
            # Threads only ever load different stores, so its cache needs no lock
            source = Source(name, store, self.builder.lengths(store), self)

            for kind in columns:
                source.column(kind)

            return source

        opened = self.map(load, list(uris.items()))
        sources = {source.name: source for source in opened}
        self.sources.update(sources)
        return sources

    def map[T, R](self, function: Callable[[T], R], items: Sequence[T]) -> list[R]:
        """
        function over items on a thread pool, results in order. For reads (S3 releases
        the GIL) and large numpy work; not for sampling, see the module docstring
        """
        if not items:
            return []

        with ThreadPoolExecutor(max_workers=min(self.workers, len(items))) as pool:
            return list(pool.map(function, items))

    def phase(self, name: str) -> None:
        """
        Starts a phase; phases train in the order they start, and cursor adds to the
        latest one
        """
        self.builder.add_phase(name)
        self._phase = name

    def cursor(self, name: str, documents: Docs, repeat: bool = False) -> None:
        if self._phase is None:
            raise ValueError("start a phase before adding cursors")

        self.builder.add_cursor(
            self._phase,
            name,
            documents.source.name,
            documents.source.store,
            documents.selections,
            repeat=repeat,
        )


def _git(directory: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def compiler_record(
    compile: Callable[[Context], None], path: Path, arguments: argparse.Namespace
) -> dict[str, Any]:
    """
    Where the recipe's rules came from. dirty means the working tree had uncommitted
    changes, so the commit alone may not reproduce the recipe; compiler.py still has
    the compiler's own code, though not the harness's
    """
    module = compile.__module__

    # Under python -m the module is __main__; its spec has the real name
    if module == "__main__":
        spec = getattr(sys.modules["__main__"], "__spec__", None)
        module = spec.name if spec is not None else module

    root = _git(path.parent, "rev-parse", "--show-toplevel")
    status = _git(path.parent, "status", "--porcelain")

    return {
        "module": module,
        "path": str(path.relative_to(root)) if root else str(path),
        "git_commit": _git(path.parent, "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "argv": sys.argv[1:],
        "arguments": vars(arguments),
    }


def run(
    compile: Callable[[Context], None],
    add_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """
    The entry point of a compiler module: parse arguments, compile, write, summarize.
    The module's docstring is the --help text
    """
    module = sys.modules[compile.__module__]
    parser = argparse.ArgumentParser(
        description=module.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", required=True, help="recipe directory; never overwritten"
    )
    parser.add_argument(
        "--total-tokens", type=int, help="train.py --total-tokens the recipe is for"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="extra fraction of tokens, so no rank runs out before --total-tokens",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="threads for parallel reads, per level (stores, then shards)",
    )

    if add_arguments is not None:
        add_arguments(parser)

    arguments = parser.parse_args()

    # Read before compiling, so an edit during a long compile can't change the copy
    source_file = inspect.getsourcefile(compile)

    if source_file is None:
        raise ValueError(f"can't find the source of {compile}")

    path = Path(source_file).resolve()
    compiler_source = path.read_text()
    record = compiler_record(compile, path, arguments)

    context = Context(arguments)
    compile(context)

    recipe = context.builder.write(
        arguments.output, compiler=record, compiler_source=compiler_source
    )
    print(context.builder.summary())

    for phase in recipe.phases:
        print(
            f"phase {phase.name}: {phase.tokens / recipe.total_tokens:.1%} of training"
        )

        for cursor in phase.cursors:
            print(f"  {cursor.name}: {cursor.tokens / recipe.total_tokens:.3%}")

    print(
        f"wrote {recipe.uri}: {recipe.total_tokens:,} tokens, "
        f"{recipe.repeated_tokens:,} repeated"
    )

    if record["git_dirty"]:
        print(
            "warning: uncommitted changes in the working tree; "
            f"git commit {record['git_commit']} may not reproduce this recipe"
        )
