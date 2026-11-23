"""
P0-2a point selection (§1 判定標準): the L_train BPB mean over 7 items, FLORES+ devtest
en/es/de/ru and FineWeb-2 test es/de/ru, for each evaluate.py output; then, against the
best point, the paired bootstrap 95% CI of every other point's difference (same items,
resampled together), so "within CI" can be read off directly.

    uv run python scripts/embedding_ml_transfer/select_lr.py evals/p0-2a/<run> ... \\
        --output evals/p0-2a/select.json
"""

import argparse
import json
from pathlib import Path

import numpy

LN2 = numpy.log(2)
FLORES = ("en", "es", "de", "ru")
FINEWEB2 = ("spa", "deu", "rus")


def train_keys(arrays) -> list[str]:  # noqa: ANN001
    names = {k.rsplit("/", 1)[0] for k in arrays.files if k.startswith("bpb/")}
    keys = [f"bpb/flores/{lang}" for lang in FLORES]
    for prefix in FINEWEB2:
        match = [n for n in names if n.startswith(f"bpb/fineweb2/{prefix}")]
        assert len(match) == 1, (prefix, sorted(names))
        keys.append(match[0])
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="+", help="evaluate.py --output prefixes")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    runs = {Path(p).name: numpy.load(f"{p}.npz") for p in arguments.outputs}
    keys = train_keys(next(iter(runs.values())))
    rng = numpy.random.default_rng(arguments.seed)
    # One shared resample per item set, so every run sees the same items
    picks = {}
    for key in keys:
        n = next(iter(runs.values()))[f"{key}/bytes"].size
        picks[key] = rng.integers(0, n, size=(arguments.bootstrap, n))

    means, samples = {}, {}
    for name, arrays in runs.items():
        per_key, per_key_samples = [], []
        for key in keys:
            nll, utf8 = arrays[f"{key}/nll"], arrays[f"{key}/bytes"]
            per_key.append(nll.sum() / (LN2 * utf8.sum()))
            p = picks[key]
            per_key_samples.append(nll[p].sum(1) / (LN2 * utf8[p].sum(1)))
        means[name] = float(numpy.mean(per_key))
        samples[name] = numpy.mean(per_key_samples, axis=0)

    best = min(means, key=means.get)
    report = {"keys": keys, "best": best, "runs": {}}
    for name in sorted(means, key=means.get):
        diff = samples[name] - samples[best]
        low, high = numpy.percentile(diff, [2.5, 97.5])
        report["runs"][name] = {
            "l_train_bpb_mean7": means[name],
            "minus_best": means[name] - means[best],
            "minus_best_ci95": [float(low), float(high)],
        }
        print(f"{name}: {means[name]:.5f}  vs best {means[name] - means[best]:+.5f} "
              f"[{low:+.5f}, {high:+.5f}]")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
