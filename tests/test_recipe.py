from pathlib import Path

import numpy
import pytest

from src.recipe.recipe import (
    RecipeBuilder,
    load_recipe,
    subtract_selections,
    take_by_priority,
)

from .conftest import make_store


def test_builder_counts_tokens_from_offsets(tmp_path: Path) -> None:
    store = make_store(tmp_path / "store", [[3, 1, 2], [5, 4]])
    builder = RecipeBuilder()
    builder.add_phase("main")
    builder.add_cursor("main", "a", "src", store, {"000000": [0, 2], "000001": [1]})

    recipe = builder.write(str(tmp_path / "recipe"))

    assert recipe.total_tokens == 3 + 2 + 4
    assert recipe.phases[0].documents == 3
    assert load_recipe(str(tmp_path / "recipe")) == recipe
    assert store.read_offsets("000000").size == 4, (
        "the builder must not touch the source store"
    )


def test_builder_refuses_to_overwrite_a_recipe(tmp_path: Path) -> None:
    store = make_store(tmp_path / "store", [[3, 1]])
    for attempt in range(2):
        builder = RecipeBuilder()
        builder.add_phase("main")
        builder.add_cursor("main", "a", "src", store, {"000000": [0]})
        if attempt == 0:
            builder.write(str(tmp_path / "recipe"))
        else:
            with pytest.raises(FileExistsError):
                builder.write(str(tmp_path / "recipe"))


@pytest.mark.parametrize(
    "selection",
    [[1, 0], [0, 0], [3], [-1]],
    ids=["unsorted", "duplicate", "past-end", "negative"],
)
def test_builder_rejects_bad_selections(tmp_path: Path, selection: list[int]) -> None:
    store = make_store(tmp_path, [[3, 1, 2]])
    builder = RecipeBuilder()
    builder.add_phase("main")
    with pytest.raises(ValueError):
        builder.add_cursor("main", "a", "src", store, {"000000": selection})


def test_unique_cursors_may_not_share_documents_across_phases(tmp_path: Path) -> None:
    store = make_store(tmp_path, [[3, 1, 2]])
    builder = RecipeBuilder()
    builder.add_phase("main")
    builder.add_phase("anneal")
    builder.add_cursor("main", "a", "src", store, {"000000": [0, 1]})
    builder.add_cursor("anneal", "a", "src", store, {"000000": [1, 2]})
    with pytest.raises(ValueError, match="share"):
        builder.build()


def test_repeats_must_copy_selected_documents(tmp_path: Path) -> None:
    store = make_store(tmp_path, [[10] * 40])
    builder = RecipeBuilder()
    builder.add_phase("main")
    builder.add_cursor("main", "a", "src", store, {"000000": list(range(30))})
    builder.add_cursor("main", "r", "src", store, {"000000": [35]}, repeat=True)
    with pytest.raises(ValueError, match="no unique cursor"):
        builder.build()


def test_repeat_allowance_is_enforced(tmp_path: Path) -> None:
    store = make_store(tmp_path, [[10] * 40])

    def build(repeated: int) -> int:
        builder = RecipeBuilder()
        builder.add_phase("main")
        builder.add_cursor("main", "a", "src", store, {"000000": list(range(30))})
        builder.add_cursor(
            "main", "r", "src", store, {"000000": list(range(repeated))}, repeat=True
        )
        return builder.build().repeated_tokens

    assert build(3) == 30  # exactly 10% of the 300 counted tokens
    with pytest.raises(ValueError, match="exceed"):
        build(4)


def test_sources_must_share_a_tokenizer(tmp_path: Path) -> None:
    a = make_store(tmp_path / "a", [[3]])
    b = make_store(tmp_path / "b", [[3]], tokenizer="other")
    builder = RecipeBuilder()
    builder.add_phase("main")
    builder.add_cursor("main", "a", "a", a, {"000000": [0]})
    with pytest.raises(ValueError, match="tokenizer"):
        builder.add_cursor("main", "b", "b", b, {"000000": [0]})


def test_take_by_priority_ranks_globally(
    random_generator: numpy.random.Generator,
) -> None:
    priorities = {"x": numpy.array([5.0, 1.0, 3.0]), "y": numpy.array([4.0, 2.0])}
    lengths = {"x": numpy.array([10, 10, 10]), "y": numpy.array([10, 10])}

    top = take_by_priority(priorities, lengths, 25, random_generator)
    assert {k: v.tolist() for k, v in top.items()} == {"x": [0, 2], "y": [0]}

    within = take_by_priority(priorities, lengths, 25, random_generator, exceed=False)
    assert {k: v.tolist() for k, v in within.items()} == {"x": [0], "y": [0]}

    eligible = {"x": numpy.array([1, 2], dtype=numpy.uint32)}
    limited = take_by_priority(
        priorities, lengths, 10, random_generator, eligible=eligible
    )
    assert {k: v.tolist() for k, v in limited.items()} == {"x": [2]}

    with pytest.raises(ValueError, match="available"):
        take_by_priority(priorities, lengths, 51, random_generator)


def test_take_by_priority_breaks_ties_at_random() -> None:
    # One class of equal scores across many shards: the kept half must not favour
    # the first shards
    priorities = {f"{s:06d}": numpy.ones(100) for s in range(10)}
    lengths = {name: numpy.ones(100, dtype=numpy.int64) for name in priorities}
    kept = take_by_priority(priorities, lengths, 500, numpy.random.default_rng(42))
    counts = [kept[name].size for name in priorities]
    assert sum(counts) == 500
    assert min(counts) > 30 and max(counts) < 70


def test_subtract_selections() -> None:
    a = {
        "x": numpy.array([0, 1, 2], dtype=numpy.uint32),
        "y": numpy.array([4], dtype=numpy.uint32),
    }
    b = {
        "x": numpy.array([1], dtype=numpy.uint32),
        "y": numpy.array([4], dtype=numpy.uint32),
    }
    result = subtract_selections(a, b)
    assert {k: v.tolist() for k, v in result.items()} == {"x": [0, 2]}
