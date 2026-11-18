import copy
import io
import itertools
from collections import Counter
from pathlib import Path

import numpy
import pytest
import torch

from src.recipe.recipe import Recipe, RecipeBuilder
from src.token_stream import RecipeTokenStream

from .conftest import document_code, make_store


def documents_of(stream: RecipeTokenStream, limit: int | None = None) -> list[int]:
    """
    The document codes the stream yields, in order
    """
    return [int(tokens[0]) for tokens in itertools.islice(stream, limit)]


@pytest.fixture
def recipe(tmp_path: Path) -> Recipe:
    """
    Two sources with very different document lengths, a repeat cursor and an anneal
    phase, spread over several shards with partial selections
    """
    rng = numpy.random.default_rng(0)
    long_docs = make_store(
        tmp_path / "long", [rng.integers(30, 60, size=40).tolist() for _ in range(3)]
    )
    short_docs = make_store(
        tmp_path / "short", [rng.integers(2, 6, size=80).tolist() for _ in range(4)]
    )

    builder = RecipeBuilder()
    builder.add_phase("main")
    builder.add_cursor(
        "main",
        "long",
        "long",
        long_docs,
        {f"{s:06d}": range(0, 40, 2) for s in range(3)},
    )
    builder.add_cursor(
        "main", "short", "short", short_docs, {f"{s:06d}": range(70) for s in range(4)}
    )
    builder.add_cursor(
        "main", "short-repeat", "short", short_docs, {"000000": range(10)}, repeat=True
    )
    builder.add_phase("anneal")
    builder.add_cursor(
        "anneal",
        "long",
        "long",
        long_docs,
        {f"{s:06d}": range(1, 40, 2) for s in range(3)},
    )
    return builder.write(str(tmp_path / "recipe"))


def expected_counts(recipe: Recipe) -> Counter[tuple[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    for phase in recipe.phases:
        for cursor in phase.cursors:
            for shard in cursor.shards:
                selection = numpy.load(Path(recipe.uri) / shard.selection)
                for document in selection:
                    counts[
                        (cursor.source, document_code(int(shard.name), int(document)))
                    ] += 1
    return counts


def test_every_selected_document_appears_as_often_as_selected(recipe: Recipe) -> None:
    stream = RecipeTokenStream(recipe)
    seen: Counter[tuple[str, int]] = Counter()
    total_tokens = 0

    for tokens, origin in stream.iter_with_origin():
        assert numpy.all(tokens == document_code(int(origin.shard), origin.document))
        seen[(origin.source, int(tokens[0]))] += 1
        total_tokens += tokens.size

    assert seen == expected_counts(recipe)
    assert total_tokens == recipe.total_tokens
    assert stream.repeated_tokens == recipe.repeated_tokens
    assert sum(stream.tokens_by_source.values()) == recipe.total_tokens


def test_phases_run_in_order(recipe: Recipe) -> None:
    phases = [
        origin.phase for _, origin in RecipeTokenStream(recipe).iter_with_origin()
    ]
    assert phases == sorted(phases)
    assert phases.count(0) == recipe.phases[0].documents


def test_ranks_split_the_single_rank_sequence(recipe: Recipe) -> None:
    sequence = documents_of(RecipeTokenStream(recipe))
    for world_size in (2, 3):
        for rank in range(world_size):
            stream = RecipeTokenStream(recipe, rank=rank, world_size=world_size)
            assert documents_of(stream) == sequence[rank::world_size]


@pytest.mark.parametrize("world_size", [1, 2])
def test_resume_continues_the_same_sequence(recipe: Recipe, world_size: int) -> None:
    rank = world_size - 1
    sequence = documents_of(RecipeTokenStream(recipe, rank, world_size))
    phase_end = recipe.phases[0].documents // world_size
    # Every early stop, including the phase boundary and the very end
    stops = sorted(
        {*range(0, len(sequence) + 1, 17), phase_end, phase_end + 1, len(sequence)}
    )

    for stop in stops:
        stream = RecipeTokenStream(recipe, rank, world_size)
        head = documents_of(stream, stop)

        # Through torch.save, as the trainer's checkpoint does with weights_only=True
        buffer = io.BytesIO()
        torch.save(copy.deepcopy(stream.state_dict()), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True)

        resumed = RecipeTokenStream(recipe, rank, world_size)
        resumed.load_state_dict(state)
        assert head + documents_of(resumed) == sequence, f"stop {stop}"


def test_cursors_drain_together(recipe: Recipe) -> None:
    """
    Halfway through the main phase, each cursor has used about half of its tokens,
    though one cursor's documents are ten times longer than the other's
    """
    main = recipe.phases[0]
    halfway = main.documents // 2
    used: Counter[int] = Counter()

    for tokens, origin in itertools.islice(
        RecipeTokenStream(recipe).iter_with_origin(), halfway
    ):
        used[origin.cursor] += tokens.size

    for index, cursor in enumerate(main.cursors):
        assert 0.35 < used[index] / cursor.tokens < 0.65, cursor.name


def test_draws_are_spread_over_the_phase(recipe: Recipe) -> None:
    """
    The repeat copies are not bunched next to their originals in time
    """
    main = recipe.phases[0]
    positions: dict[int, list[int]] = {}
    for position, (tokens, origin) in enumerate(
        RecipeTokenStream(recipe).iter_with_origin()
    ):
        if origin.phase == 0 and origin.source == "short" and origin.shard == "000000":
            positions.setdefault(int(tokens[0]), []).append(position)

    gaps = [abs(a - b) for a, b in (p for p in positions.values() if len(p) == 2)]
    assert len(gaps) == 10
    # Uniform placement puts two copies about a third of the phase apart on average
    assert numpy.mean(gaps) > main.documents / 6


def test_same_seed_same_order(recipe: Recipe) -> None:
    assert documents_of(RecipeTokenStream(recipe)) == documents_of(
        RecipeTokenStream(recipe)
    )


def test_documents_are_shuffled_inside_each_shard(recipe: Recipe) -> None:
    """
    Shards store their documents in fragment order, so the in-shard permutation is the
    last step of the global shuffle: the order a shard's documents come out in must not
    follow their stored order
    """
    order: dict[tuple[int, str, str], list[int]] = {}
    for _, origin in RecipeTokenStream(recipe).iter_with_origin():
        order.setdefault((origin.cursor, origin.source, origin.shard), []).append(
            origin.document
        )

    for key, documents in order.items():
        if len(documents) < 20:
            continue
        ranks = numpy.argsort(numpy.argsort(documents))
        correlation = numpy.corrcoef(numpy.arange(len(documents)), ranks)[0, 1]
        assert abs(correlation) < 0.5, key
