"""
Embedding drift of a Phase 1 run whose embedding trains from Oracle_Emb (A4a, A4b, A4c):
for every step-N checkpoint of the run, read the tied embedding from resume.pt (the
unfolded weights, as Oracle_Emb itself was taken) and measure how far it moved.

    uv run python scripts/embedding_ml_transfer/drift.py \\
        --run checkpoints/<run> --reference donors/oracle_emb.pt \\
        --fake-french p0-1/fake-french --muse eval-data/muse/en-zh.txt --output <json>

Per checkpoint, for three row groups (L_train: below 151,643 and not V_ban; zh: V_ban;
fake: the Fake-French rows):
    rel_distance   ||E - O|| / ||O||, O = Oracle_Emb, on the group's rows
    row_rms        the group's row RMS
    mean_cosine    mean cosine between each row and its Oracle_Emb row (ignores the norm
                   growth that wd 0 allows, which rel_distance counts as drift)
and for the whole matrix:
    procrustes     ||E R - O|| / ||O|| on the L_train rows, R the orthogonal map that
                   minimizes it (R = U V^T, U S V^T = svd(E^T O)): the drift a rotation
                   cannot explain, i.e. what changes the geometry rather than the frame
    bli            bli.py's en-zh and fake P@1 (CSLS k=10) on E, as on Oracle_Emb in
                   G-Emb, with 1 shuffle (the baseline is only a sanity check here)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bli import BASE_VOCAB, evaluate, muse_pairs  # noqa: E402
from src.recipe.scorers.vban_count import BAN, FIRST_SPECIAL, byte_categories  # noqa: E402
from src.tokenizer import load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="checkpoints/<run>")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fake-french", type=Path, required=True)
    parser.add_argument("--muse", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = torch.load(arguments.reference, map_location="cpu", weights_only=True)
    reference = (reference["weight"] if isinstance(reference, dict) else reference).float()
    reference = reference.to(device)
    vocab = reference.shape[0]
    categories = byte_categories(arguments.tokenizer, max(vocab, BASE_VOCAB))[:vocab]
    ids = numpy.arange(vocab)
    table = numpy.load(arguments.fake_french / "fake_french.npz")
    fake_rows = table["members"].size
    groups = {
        "l_train": (ids < FIRST_SPECIAL) & (categories != BAN),
        "zh": (ids < FIRST_SPECIAL) & (categories == BAN),
        "fake": (ids >= BASE_VOCAB) & (ids < BASE_VOCAB + fake_rows),
    }
    groups = {k: torch.from_numpy(v).to(device) for k, v in groups.items()}
    tokenizer = load_tokenizer(arguments.tokenizer)
    zh_queries, zh_answers = muse_pairs(arguments.muse, tokenizer, categories)
    zh_candidates = numpy.flatnonzero((categories == BAN) & (ids < FIRST_SPECIAL))
    fake_queries = table["bli_shifted"].astype(numpy.int64)
    fake_answers = [{int(o)} for o in table["bli_original"].astype(numpy.int64)]
    fake_candidates = numpy.flatnonzero((categories != BAN) & (ids < FIRST_SPECIAL))

    def measure(embedding: torch.Tensor) -> dict:
        out: dict = {}
        for name, rows in groups.items():
            e, o = embedding[rows], reference[rows]
            out[f"{name}/rel_distance"] = float((e - o).norm() / o.norm())
            out[f"{name}/row_rms"] = float(e.pow(2).mean().sqrt())
            # Scale-free: wd 0 lets row norms grow, which rel_distance counts as drift
            out[f"{name}/mean_cosine"] = float(
                torch.nn.functional.cosine_similarity(e, o, dim=1).mean()
            )
        e, o = embedding[groups["l_train"]].double(), reference[groups["l_train"]].double()
        u, _, vt = torch.linalg.svd(e.T @ o)
        out["procrustes"] = float((e @ (u @ vt) - o).norm() / o.norm())
        out["bli/en-zh"] = evaluate(embedding, zh_queries, zh_answers, zh_candidates, 1, 42)
        out["bli/fake"] = evaluate(
            embedding, fake_queries, fake_answers, fake_candidates, 1, 42
        )
        return out

    steps = sorted(
        (int(m.group(1)), p)
        for p in arguments.run.iterdir()
        if (m := re.fullmatch(r"step-(\d+)", p.name)) and (p / "resume.pt").exists()
    )
    report: dict = {"run": str(arguments.run), "reference": str(arguments.reference), "steps": {}}
    report["steps"]["0"] = measure(reference)
    for step, path in steps:
        state = torch.load(path / "resume.pt", map_location="cpu", weights_only=False)
        embedding = state["model"]["model.embed_tokens.weight"].float().to(device)
        del state
        report["steps"][str(step)] = measure(embedding)
        row = report["steps"][str(step)]
        print(
            f"step {step}: procrustes {row['procrustes']:.4f} "
            f"l_train {row['l_train/rel_distance']:.4f} zh {row['zh/rel_distance']:.4f} "
            f"fake {row['fake/rel_distance']:.4f} cos l_train {row['l_train/mean_cosine']:.3f} "
            f"zh {row['zh/mean_cosine']:.3f} fake {row['fake/mean_cosine']:.3f} bli zh {row['bli/en-zh']['p_at_1']:.4f} "
            f"fake {row['bli/fake']['p_at_1']:.4f}",
            flush=True,
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
