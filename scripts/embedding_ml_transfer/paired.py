"""
Paired comparison of two Phase 1 evaluations (evaluate.py .npz files), unit by unit
(FLORES+ sentence, FineWeb-2 document), with a 95% paired bootstrap:

    uv run python scripts/embedding_ml_transfer/paired.py \\
        --a evals/phase1/<run-a>.npz --b evals/phase1/<run-b>.npz [<more b>...] \\
        --output <json>

Every value is B - A. Several --b files are averaged unit by unit first (the shuffled
control's 5 seeds: B = their mean). Reported, when both sides have them:

    bpb/<set>/<lang>, split/<set>/<lang>/<term>   BPB difference (bits per byte)
    delta/<pair>, delta_masked/<pair>              translation-gain difference
    l_train7                                       the mean over select_lr.py's 7 L_train
                                                   BPB items (each item resampled on its own)

A difference is significant when its CI excludes 0 ("significant": true).
"""

import argparse
import json
import math
from pathlib import Path

import numpy

LN2 = math.log(2)
L_TRAIN7 = [
    "bpb/flores/en", "bpb/flores/es", "bpb/flores/de", "bpb/flores/ru",
    "bpb/fineweb2/spa_Latn", "bpb/fineweb2/deu_Latn", "bpb/fineweb2/rus_Cyrl",
]


def load_mean(paths: list[Path]) -> dict[str, numpy.ndarray]:
    files = [numpy.load(p) for p in paths]
    keys = set(files[0].files)
    for f in files[1:]:
        keys &= set(f.files)
    return {k: numpy.mean([f[k] for f in files], axis=0) for k in keys}


def bpb_metrics(arrays: dict[str, numpy.ndarray]) -> set[str]:
    return {k[: -len("/nll")] for k in arrays if k.endswith("/nll")}


def delta_metrics(arrays: dict[str, numpy.ndarray]) -> set[str]:
    return {k[: -len("/nll_real")] for k in arrays if k.endswith("/nll_real")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, nargs="+", required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    a, b = load_mean([arguments.a]), load_mean(arguments.b)
    rng = numpy.random.default_rng(arguments.seed)
    report: dict = {"a": str(arguments.a), "b": [str(p) for p in arguments.b], "metrics": {}}
    draws_by_key: dict[str, numpy.ndarray] = {}

    def summarize(key: str, value: float, samples: numpy.ndarray) -> None:
        low, high = numpy.percentile(samples, [2.5, 97.5])
        report["metrics"][key] = {
            "b_minus_a": float(value),
            "ci95": [float(low), float(high)],
            "significant": bool(low > 0 or high < 0),
        }

    for key in sorted(bpb_metrics(a) & bpb_metrics(b)):
        nll_a, nll_b, utf8 = a[f"{key}/nll"], b[f"{key}/nll"], a[f"{key}/bytes"]
        if nll_a.shape != nll_b.shape or not numpy.array_equal(utf8, b[f"{key}/bytes"]):
            print(f"{key}: different units, skipped", flush=True)
            continue
        diff = nll_b - nll_a
        picks = rng.integers(0, diff.size, size=(arguments.draws, diff.size))
        samples = diff[picks].sum(1) / (LN2 * utf8[picks].sum(1))
        draws_by_key[key] = samples
        summarize(key, diff.sum() / (LN2 * utf8.sum()), samples)

    for key in sorted(delta_metrics(a) & delta_metrics(b)):
        # delta_masked/<pair> shares delta/<pair>'s target bytes
        utf8 = a.get(f"{key}/bytes", a.get(key.replace("delta_masked/", "delta/") + "/bytes"))
        gain_a = a[f"{key}/nll_unrelated"] - a[f"{key}/nll_real"]
        gain_b = b[f"{key}/nll_unrelated"] - b[f"{key}/nll_real"]
        if gain_a.shape != gain_b.shape:
            print(f"{key}: different units, skipped", flush=True)
            continue
        diff = gain_b - gain_a
        picks = rng.integers(0, diff.size, size=(arguments.draws, diff.size))
        samples = diff[picks].sum(1) / (LN2 * utf8[picks].sum(1))
        summarize(key, diff.sum() / (LN2 * utf8.sum()), samples)

    if all(k in draws_by_key for k in L_TRAIN7):
        value = numpy.mean([report["metrics"][k]["b_minus_a"] for k in L_TRAIN7])
        summarize("l_train7", value, numpy.mean([draws_by_key[k] for k in L_TRAIN7], axis=0))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    for key, m in report["metrics"].items():
        if key.startswith(("bpb/flores", "split/flores", "delta", "l_train7")):
            low, high = m["ci95"]
            star = " *" if m["significant"] else ""
            print(f"{key}: {m['b_minus_a']:+.5f} [{low:+.5f}, {high:+.5f}]{star}", flush=True)


if __name__ == "__main__":
    main()
