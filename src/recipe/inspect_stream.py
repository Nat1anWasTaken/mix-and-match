"""
Check that a recipe's stream is stationary, without a GPU: the property whose absence
made the loss oscillate with shard boundaries before the global shuffle (72fcd70).

    uv run python -m src.recipe.inspect_stream \\
        --recipe s3://<bucket>/recipes/fwedu-intscore3-30b --kind int_score \\
        --global-batch-tokens 4194304 --max-steps 2000

Streams the recipe as one rank would see it with world_size 1, cuts it into steps of
--global-batch-tokens, and per step records each source's token share and, with --kind,
the token-weighted mean of that metadata column. Then prints the autocorrelation of each
series. A stationary stream is white noise: every lag stays within about 2 / sqrt(steps).
A peak near shard_tokens / step_tokens means documents of one shard cluster in time.

Phases change the data on purpose, so the series is checked phase by phase.
"""

import argparse

import numpy

from ..token_stream import RecipeTokenStream
from ..tokenizer import ShardStore
from .metadata import read_column
from .recipe import load_recipe


def autocorrelation(series: numpy.ndarray, max_lag: int) -> numpy.ndarray:
    centered = series - series.mean()
    variance = float(centered @ centered)

    if variance == 0:
        return numpy.zeros(max_lag)

    return numpy.array(
        [
            float(centered[:-lag] @ centered[lag:]) / variance
            for lag in range(1, max_lag + 1)
        ]
    )


def report(name: str, series: numpy.ndarray, max_lag: int) -> bool:
    """
    Prints the lags outside the white-noise band; True when there are none
    """
    if series.size < 3:
        print(f"  {name}: too few steps")
        return True

    lags = min(max_lag, series.size - 1)
    values = autocorrelation(series, lags)
    band = 2 / numpy.sqrt(series.size)
    outside = numpy.flatnonzero(numpy.abs(values) > band) + 1
    # About 5% of the lags leave a 2-sigma band by chance alone
    expected = 0.05 * lags
    stationary = outside.size <= max(2 * expected, 2)
    verdict = "ok" if stationary else "NOT STATIONARY"

    print(
        f"  {name}: mean {series.mean():.4g}, std {series.std():.3g}, "
        f"{outside.size}/{lags} lags outside ±{band:.3f} ({verdict})"
    )

    if outside.size and not stationary:
        worst = outside[numpy.argsort(-numpy.abs(values[outside - 1]))][:5]
        print(
            "    largest at lags "
            + ", ".join(f"{lag} ({values[lag - 1]:+.3f})" for lag in worst)
        )

    return stationary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--kind", help="meta/<kind> column to track per step")
    parser.add_argument("--global-batch-tokens", type=int, default=4096 * 8 * 16 * 8)
    parser.add_argument("--max-steps", type=int, help="default: the whole recipe")
    parser.add_argument("--max-lag", type=int, default=100)
    arguments = parser.parse_args()

    recipe = load_recipe(arguments.recipe)
    stream = RecipeTokenStream(recipe)
    sources = recipe.sources()
    columns: dict[tuple[str, str], numpy.ndarray] = {}
    stores: dict[str, ShardStore] = {}
    source_store = {
        cursor.source: cursor.store
        for phase in recipe.phases
        for cursor in phase.cursors
    }

    def score(source: str, shard: str, document: int) -> float:
        key = (source, shard)

        if key not in columns:
            uri = source_store[source]
            stores.setdefault(uri, ShardStore(uri))
            columns[key] = read_column(stores[uri], arguments.kind, shard)

        return float(columns[key][document])

    # Per phase, a list of per-step [tokens per source..., score sum]
    steps: dict[int, list[numpy.ndarray]] = {}
    current = numpy.zeros(len(sources) + 1)
    current_phase = 0
    source_index = {source: index for index, source in enumerate(sources)}

    for tokens, origin in stream.iter_with_origin():
        if origin.phase != current_phase:
            # Drop the partial step at a phase boundary
            current = numpy.zeros(len(sources) + 1)
            current_phase = origin.phase

        current[source_index[origin.source]] += tokens.size

        if arguments.kind is not None:
            current[-1] += tokens.size * score(
                origin.source, origin.shard, origin.document
            )

        if current[:-1].sum() >= arguments.global_batch_tokens:
            steps.setdefault(origin.phase, []).append(current)
            current = numpy.zeros(len(sources) + 1)

            if (
                arguments.max_steps
                and sum(map(len, steps.values())) >= arguments.max_steps
            ):
                break

    all_stationary = True

    for phase_index, phase_steps in steps.items():
        table = numpy.stack(phase_steps)
        step_tokens = table[:, :-1].sum(axis=1)
        print(f"phase {recipe.phases[phase_index].name}: {table.shape[0]} steps")

        for source in sources:
            share = table[:, source_index[source]] / step_tokens
            if len(sources) > 1:
                all_stationary &= report(f"share of {source}", share, arguments.max_lag)

        if arguments.kind is not None:
            mean_score = table[:, -1] / step_tokens
            all_stationary &= report(
                f"mean {arguments.kind}", mean_score, arguments.max_lag
            )

    print(
        "stationary"
        if all_stationary
        else "NOT STATIONARY: check the recipe and shards"
    )


if __name__ == "__main__":
    main()
