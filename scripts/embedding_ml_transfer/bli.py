"""
The Embedding gate (plan Phase 0, gate 2): bilingual lexicon induction on an embedding
matrix, with no GPU training.

    uv run python scripts/embedding_ml_transfer/bli.py \\
        --embedding oracle_emb.pt --name oracle --fake-french <dir> \\
        --muse en-zh.txt --output <dir>/bli-oracle.json
    uv run python scripts/embedding_ml_transfer/bli.py --hf Qwen/Qwen3-0.6B-Base ...

Two dictionaries, each scored with CSLS (k=10) retrieval, P@1:

    en-zh  MUSE en-zh pairs where both words are a single Qwen3 token (tried with and
           without a leading space, the form running text uses). Queries: the English
           rows. Candidates: every V_ban row (the Chinese tokens)
    fake   each Fake-French row's answer is the original token it was shifted from
           (P0-1's bli_shifted / bli_original, only originals L_train uses).
           Queries: the shifted rows. Candidates: every base row below 151,643 that is
           not V_ban (the rows Fake-French could have been aligned to). Only for
           embeddings that have the shifted rows (track 1)

The random baseline permutes the candidate rows (which row is which token) and scores
the same queries again, --shuffles times; the gate needs P@1 well above it, judged by a
95% Wilson interval on P@1 against the baseline's mean.
Rows from 151,643 up are never candidates (special and padding rows).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.recipe.scorers.vban_count import BAN, FIRST_SPECIAL, byte_categories
from src.tokenizer import load_tokenizer

BASE_VOCAB = 151936


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959964
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (
        1 + z * z / total
    )
    return (centre - half, centre + half)


def csls_top1(
    queries: torch.Tensor, candidates: torch.Tensor, k: int = 10, chunk: int = 4096
) -> torch.Tensor:
    """
    For each query row, the index of the candidate with the highest CSLS score,
    2 cos(q, c) - r_C(q) - r_Q(c), r the mean cosine to the k nearest of the other side
    """
    q = torch.nn.functional.normalize(queries.float(), dim=1)
    c = torch.nn.functional.normalize(candidates.float(), dim=1)
    # r_Q(c): each candidate's mean cosine to its k nearest queries
    r_candidates = torch.zeros(c.size(0), device=c.device)
    for start in range(0, c.size(0), chunk):
        sims = c[start : start + chunk] @ q.T
        r_candidates[start : start + chunk] = sims.topk(min(k, q.size(0)), dim=1).values.mean(1)
    best = torch.empty(q.size(0), dtype=torch.long, device=q.device)
    for start in range(0, q.size(0), chunk):
        sims = q[start : start + chunk] @ c.T
        r_queries = sims.topk(min(k, c.size(0)), dim=1).values.mean(1)
        scores = 2 * sims - r_queries[:, None] - r_candidates[None, :]
        best[start : start + chunk] = scores.argmax(1)
    return best


def evaluate(
    embedding: torch.Tensor,
    query_ids: numpy.ndarray,
    answer_ids: list[set[int]],
    candidate_ids: numpy.ndarray,
    shuffles: int,
    seed: int,
) -> dict:
    device = embedding.device
    queries = embedding[torch.from_numpy(query_ids).to(device)]
    candidates = embedding[torch.from_numpy(candidate_ids).to(device)]
    best = csls_top1(queries, candidates).cpu().numpy()
    hits = sum(int(candidate_ids[b]) in answers for b, answers in zip(best, answer_ids))
    rng = numpy.random.default_rng(seed)
    baseline = []
    for _ in range(shuffles):
        permutation = rng.permutation(candidate_ids.size)
        shuffled = candidates[torch.from_numpy(permutation).to(device)]
        best = csls_top1(queries, shuffled).cpu().numpy()
        baseline.append(
            sum(int(candidate_ids[b]) in answers for b, answers in zip(best, answer_ids))
            / len(answer_ids)
        )
    low, high = wilson(hits, len(answer_ids))
    return {
        "pairs": len(answer_ids),
        "candidates": int(candidate_ids.size),
        "p_at_1": hits / len(answer_ids),
        "p_at_1_ci95": [low, high],
        "random_p_at_1_mean": float(numpy.mean(baseline)),
        "random_p_at_1_max": float(numpy.max(baseline)),
        "chance": 1 / candidate_ids.size,
        "pass": bool(low > numpy.max(baseline)),
    }


def single_token(tokenizer, word: str) -> int | None:  # noqa: ANN001
    for form in (" " + word, word):
        ids = tokenizer(form, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    return None


def muse_pairs(
    path: Path, tokenizer, categories: numpy.ndarray  # noqa: ANN001
) -> tuple[numpy.ndarray, list[set[int]]]:
    answers: dict[int, set[int]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        english, chinese = parts
        source = single_token(tokenizer, english)
        target = single_token(tokenizer, chinese)
        if source is None or target is None:
            continue
        if categories[source] == BAN or categories[target] != BAN:
            continue
        answers.setdefault(source, set()).add(target)
    query_ids = numpy.array(sorted(answers), dtype=numpy.int64)
    return query_ids, [answers[int(q)] for q in query_ids]


def load_embedding(arguments: argparse.Namespace) -> torch.Tensor:
    if arguments.embedding is not None:
        loaded = torch.load(arguments.embedding, map_location="cpu", weights_only=True)
        return loaded["weight"] if isinstance(loaded, dict) else loaded
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(arguments.hf, torch_dtype=torch.float32)
    return model.get_input_embeddings().weight.detach()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--embedding", type=Path, help=".pt tensor or {'weight': ...}")
    source.add_argument("--hf", help="a HF model whose input embedding to score")
    parser.add_argument("--name", required=True)
    parser.add_argument("--fake-french", type=Path, help="P0-1 table dir; omit to skip fake")
    parser.add_argument("--muse", type=Path, required=True, help="MUSE en-zh.txt")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--shuffles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = load_embedding(arguments).float().to(device)
    vocab = embedding.shape[0]
    categories = byte_categories(arguments.tokenizer, max(vocab, BASE_VOCAB))[:vocab]
    tokenizer = load_tokenizer(arguments.tokenizer)
    ids = numpy.arange(vocab)
    report: dict = {
        "name": arguments.name,
        "source": str(arguments.embedding or arguments.hf),
        "shape": list(embedding.shape),
        "row_rms": float(embedding[:FIRST_SPECIAL].pow(2).mean().sqrt()),
    }

    query_ids, answers = muse_pairs(arguments.muse, tokenizer, categories)
    candidates = numpy.flatnonzero((categories == BAN) & (ids < FIRST_SPECIAL))
    report["en-zh"] = evaluate(
        embedding, query_ids, answers, candidates, arguments.shuffles, arguments.seed
    )
    print("en-zh", json.dumps(report["en-zh"]), flush=True)

    table = (
        numpy.load(arguments.fake_french / "fake_french.npz")
        if arguments.fake_french is not None
        else None
    )
    if table is not None and vocab >= BASE_VOCAB + table["members"].size:
        shifted = table["bli_shifted"].astype(numpy.int64)
        original = table["bli_original"].astype(numpy.int64)
        candidates = numpy.flatnonzero((categories != BAN) & (ids < FIRST_SPECIAL))
        report["fake"] = evaluate(
            embedding,
            shifted,
            [{int(o)} for o in original],
            candidates,
            arguments.shuffles,
            arguments.seed,
        )
        print("fake", json.dumps(report["fake"]), flush=True)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
