"""
Pair evaluate.py outputs sentence by sentence (plan §6 statistics):

    # the ruler gate (Phase 0 gate 3): upper bound vs. Floor on Delta(L_ban -> En)
    uv run python scripts/embedding_ml_transfer/compare.py ruler \\
        --upper b0-upper.npz --floor b0-lower-floor.npz --output ruler.json

    # tau for any checkpoint x
    uv run python scripts/embedding_ml_transfer/compare.py tau \\
        --upper b0-upper.npz --floor b0-lower-floor.npz --model x.npz --output tau.json

Every interval is a 95% paired bootstrap over the same sentences (documents) in all
inputs, --draws resamples.

ruler   for each L_ban -> En pair: gap = Delta_upper - Delta_floor, its CI and CI width;
        the gate passes for a pair when gap >= 5 x width. The plan's decision pairs are
        zh-en and fake-en (fr-en is reported too)
tau     (BPB_floor - BPB_x) / (BPB_floor - BPB_upper) for every BPB set, and
        Delta_x - Delta_floor for every Delta pair
"""

import argparse
import json
import math
from pathlib import Path

import numpy

LN2 = math.log(2)
RULER_PAIRS = ["zh-en", "fake-en", "fr-en"]


def load(path: Path) -> dict[str, numpy.ndarray]:
    with numpy.load(path) as data:
        return {key: data[key] for key in data.files}


def bpb_sets(arrays: dict[str, numpy.ndarray]) -> list[str]:
    return sorted(key[: -len("/nll")] for key in arrays if key.endswith("/nll"))


def delta_gain(arrays: dict[str, numpy.ndarray], pair: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    key = f"delta/{pair}"
    gain = (arrays[f"{key}/nll_unrelated"] - arrays[f"{key}/nll_real"]) / LN2
    return gain, arrays[f"{key}/bytes"]


def interval(samples: numpy.ndarray) -> list[float]:
    low, high = numpy.percentile(samples, [2.5, 97.5])
    return [float(low), float(high)]


def ruler(arguments: argparse.Namespace) -> dict:
    upper, floor = load(arguments.upper), load(arguments.floor)
    rng = numpy.random.default_rng(arguments.seed)
    report: dict = {"upper": str(arguments.upper), "floor": str(arguments.floor), "pairs": {}}

    for pair in RULER_PAIRS:
        if f"delta/{pair}/bytes" not in upper or f"delta/{pair}/bytes" not in floor:
            continue
        gain_u, utf8 = delta_gain(upper, pair)
        gain_f, utf8_f = delta_gain(floor, pair)
        assert numpy.array_equal(utf8, utf8_f), f"{pair}: different sentences"
        gap = (gain_u.sum() - gain_f.sum()) / utf8.sum()
        picks = rng.integers(0, utf8.size, size=(arguments.draws, utf8.size))
        samples = (gain_u[picks].sum(1) - gain_f[picks].sum(1)) / utf8[picks].sum(1)
        ci = interval(samples)
        width = ci[1] - ci[0]
        report["pairs"][pair] = {
            "delta_upper": float(gain_u.sum() / utf8.sum()),
            "delta_floor": float(gain_f.sum() / utf8.sum()),
            "gap": float(gap),
            "gap_ci95": ci,
            "ci_width": float(width),
            "gap_over_width": float(gap / width) if width > 0 else float("inf"),
            "pass": bool(gap >= 5 * width),
        }

    decision = [report["pairs"][p]["pass"] for p in ("zh-en", "fake-en") if p in report["pairs"]]
    report["pass"] = bool(decision) and all(decision)
    return report


def tau(arguments: argparse.Namespace) -> dict:
    upper, floor, model = load(arguments.upper), load(arguments.floor), load(arguments.model)
    rng = numpy.random.default_rng(arguments.seed)
    report: dict = {"model": str(arguments.model), "tau": {}, "delta_minus_floor": {}}

    for key in bpb_sets(model):
        if f"{key}/nll" not in upper or f"{key}/nll" not in floor:
            continue
        utf8 = model[f"{key}/bytes"]
        nll = [arrays[f"{key}/nll"] for arrays in (floor, model, upper)]
        picks = rng.integers(0, utf8.size, size=(arguments.draws, utf8.size))

        def ratio(index: numpy.ndarray | slice) -> numpy.ndarray:
            total = utf8[index].sum(-1) * LN2
            f, x, u = (values[index].sum(-1) / total for values in nll)
            return (f - x) / (f - u)

        report["tau"][key] = {
            "value": float(ratio(slice(None))),
            "ci95": interval(ratio(picks)),
        }

    for key in model:
        if not key.startswith("delta/") or not key.endswith("/bytes"):
            continue
        pair = key[len("delta/") : -len("/bytes")]
        if f"delta/{pair}/bytes" not in floor:
            continue
        gain_x, utf8 = delta_gain(model, pair)
        gain_f, _ = delta_gain(floor, pair)
        picks = rng.integers(0, utf8.size, size=(arguments.draws, utf8.size))
        samples = (gain_x[picks].sum(1) - gain_f[picks].sum(1)) / utf8[picks].sum(1)
        report["delta_minus_floor"][pair] = {
            "value": float((gain_x.sum() - gain_f.sum()) / utf8.sum()),
            "ci95": interval(samples),
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["ruler", "tau"])
    parser.add_argument("--upper", type=Path, required=True)
    parser.add_argument("--floor", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = ruler(arguments) if arguments.mode == "ruler" else tau(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
