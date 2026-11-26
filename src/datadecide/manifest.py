"""
List the DataDecide checkpoints we score on Paloma, as a CSV with one row per checkpoint.

    uv run python -m src.datadecide.manifest --out src/datadecide/manifest.csv

Asks the Hugging Face Hub for every repo's branches (step<N>-seed-<seed>), so each row names
a revision that exists. Tiers, scored in this order:

  1. 25 corpora x {150M, 530M, 1B}, last step, default seed: rankings, and whether 150M's
     ranking predicts 1B's
  2. 25 x {20M, 60M, 300M, 750M}, last step: the rest of the ladder
  3. 25 x 1B x seeds large-aux-2 and large-aux-3, last step: the seed noise floor. Only
     1B's large-aux seeds are used: smaller sizes' aux seeds run to other step counts
     (150M small-aux-2 to step 351260, against 38157 for default), so they aren't the
     same runs; 1B's small-aux seeds stop at step 17339
  4. 25 x 1B at steps 2500, 5000, 10000 (3.6B, 7.2B, 14.4B tokens at 1.44M per step) and
     25 x 150M at step 5000 (2.0B tokens at 393K per step): early in training

"Last step" is each repo's own largest step, which differs by a few steps between corpora.
"""

import argparse
import collections
import concurrent.futures
import csv
import re
from pathlib import Path

from huggingface_hub import HfApi

PREFIX = "allenai/DataDecide-"
SIZES = ["20M", "60M", "150M", "300M", "530M", "750M", "1B"]
FIELDS = ["tier", "corpus", "size", "seed", "step", "repo", "revision", "name"]


def branches(api: HfApi, repo: str) -> dict[str, list[int]]:
    """Each seed's steps, from the repo's step<N>-seed-<seed> branches."""
    steps = collections.defaultdict(list)
    for branch in api.list_repo_refs(repo).branches:
        match = re.fullmatch(r"step(\d+)-seed-(.+)", branch.name)
        if match:
            steps[match.group(2)].append(int(match.group(1)))
    return {seed: sorted(s) for seed, s in steps.items()}


def manifest(api: HfApi) -> list[dict]:
    repos = [m.id for m in api.list_models(author="allenai", search="DataDecide-")]
    repos = [r for r in repos if r.rsplit("-", 1)[1] in SIZES]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        refs = dict(zip(repos, pool.map(lambda r: branches(api, r), repos)))
    corpora = sorted({r.removeprefix(PREFIX).rsplit("-", 1)[0] for r in repos})

    rows = []

    def add(tier: int, corpus: str, size: str, seed: str, step: int | None = None) -> None:
        repo = f"{PREFIX}{corpus}-{size}"
        steps = refs[repo][seed]
        step = steps[-1] if step is None else step
        if step not in steps:
            raise ValueError(f"{repo} has no step {step} for seed {seed}")
        revision = f"step{step}-seed-{seed}"
        rows.append(dict(tier=tier, corpus=corpus, size=size, seed=seed, step=step, repo=repo,
                         revision=revision, name=f"{corpus}-{size}-{revision}"))

    for corpus in corpora:
        for size in ["150M", "530M", "1B"]:
            add(1, corpus, size, "default")
        for size in ["20M", "60M", "300M", "750M"]:
            add(2, corpus, size, "default")
        for seed in ["large-aux-2", "large-aux-3"]:
            add(3, corpus, "1B", seed)
        for step in [2500, 5000, 10000]:
            add(4, corpus, "1B", "default", step)
        add(4, corpus, "150M", "default", 5000)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    rows = manifest(HfApi())
    with arguments.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(collections.Counter(row["tier"] for row in rows), f"{len(rows)} rows -> {arguments.out}")


if __name__ == "__main__":
    main()
