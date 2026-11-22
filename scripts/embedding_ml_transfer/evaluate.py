"""
Tier 0-1 of the embedding-ml-transfer evaluation (plan §6), for one checkpoint:

    uv run python scripts/embedding_ml_transfer/evaluate.py \\
        --checkpoint checkpoints/<run>/step-N --fake-french <dir with fake_french.npz> \\
        --flores <dir> --fineweb2 <dir> --output <dir>/<name> [--floor]

Writes <output>.json (every metric with a 95% bootstrap CI) and <output>.npz (the
per-sentence / per-document sums, so compare.py can pair two checkpoints sentence by
sentence for tau and the ruler gate).

What it scores, all in bits per UTF-8 byte of the original text (never token PPL):

    bpb/<set>/<lang>   FLORES+ devtest (1,012 sentences, the same in every language)
                       and a seeded sample of FineWeb-2's test split. Each item is
                       [151643, text] and only the text tokens are scored (the EOS
                       start token is context, the closing EOS is not a target). Long
                       documents are scored in independent windows of --max-length.
                       fake_fra is French translated with the P0-1 table; its
                       denominator is the original French bytes, so it compares
                       directly with fra_Latn
    delta/<src>-<tgt>  translation gain Delta = BPB(y | x') - BPB(y | x) on FLORES+
                       devtest, 3-shot (dev sentences 0-2) with the prompt
                       "<src>\\n<tgt>\\n\\n" x 3 + "<x>\\n", y scored. x' is an unrelated
                       devtest sentence of about the same length (the neighbour in
                       byte-length order). Positive: the source's meaning reached y

--floor builds B0-Lower's Floor first: every L_ban row (Chinese V_ban and the
Fake-French ids, plus the never-used padding rows past them) is set to the mean of the
trained rows (ids below 151,643 that are not V_ban), on the tied embedding, so the
output side moves with it. Rows from 151,643 up never enter a mean (AGENTS.md).

The checkpoint's HF directory is used: it is a plain Llama computing the same function
as training (multipliers folded in), which is all scoring needs. Runs on one GPU.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
from transformers import LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.recipe.scorers.vban_count import BAN, FIRST_SPECIAL, byte_categories
from src.tokenizer import load_tokenizer

BASE_VOCAB = 151936
EOS = 151643
LN2 = math.log(2)
FLORES_LANGS = {
    "eng_Latn": "en",
    "spa_Latn": "es",
    "deu_Latn": "de",
    "rus_Cyrl": "ru",
    "fra_Latn": "fr",
    "cmn_Hans": "zh",
}
FINEWEB2_LANGS = ["spa_Latn", "deu_Latn", "rus_Cyrl", "fra_Latn", "cmn_Hani"]
# (source, target): the L_ban <-> En pairs the plan decides on, plus Spanish, a
# training language, as a reference for what a seen language gives
DELTA_PAIRS = [
    ("zh", "en"),
    ("fake", "en"),
    ("fr", "en"),
    ("es", "en"),
    ("en", "zh"),
    ("en", "fake"),
    ("en", "fr"),
    ("en", "es"),
]
SHOTS = 3


@dataclass
class Item:
    """One scored sequence: tokens, and the [start, end) positions that are targets"""

    tokens: numpy.ndarray
    start: int
    end: int


class FakeFrench:
    """The P0-1 shift table: original Qwen3 ids -> Fake-French ids"""

    def __init__(self, path: Path) -> None:
        table = numpy.load(path)
        self.flat = table["flat"]
        self.start = table["start"]
        self.length = table["length"]
        self.members = table["members"]

    def translate(self, tokens: numpy.ndarray) -> numpy.ndarray:
        lengths = self.length[tokens]
        ends = numpy.cumsum(lengths)
        total = int(ends[-1]) if ends.size else 0
        firsts = numpy.repeat(ends - lengths, lengths)
        source = numpy.repeat(self.start[tokens], lengths) + numpy.arange(total) - firsts
        return self.flat[source].astype(numpy.int64)


class Encoder:
    def __init__(self, tokenizer_name: str, fake: FakeFrench | None) -> None:
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.fake = fake

    def encode(self, text: str, language: str) -> numpy.ndarray:
        """language 'fake' is French, tokenized and then shifted"""
        ids = numpy.array(
            self.tokenizer(text, add_special_tokens=False)["input_ids"], dtype=numpy.int64
        )
        if language == "fake":
            assert self.fake is not None
            return self.fake.translate(ids)
        return ids


def read_jsonl(path: Path) -> list[str]:
    return [json.loads(line)["text"] for line in path.read_text().splitlines()]


def windows(tokens: numpy.ndarray, max_length: int) -> list[Item]:
    """
    [EOS, text]: the text tokens as targets, in windows of at most max_length positions,
    each window starting with its own context-free first token
    """
    sequence = numpy.concatenate([[EOS], tokens])
    items = []
    # Window w covers targets sequence[1 + w*(L-1) : 1 + (w+1)*(L-1)], inputs from one before
    step = max_length - 1

    for first in range(1, sequence.size, step):
        last = min(first + step, sequence.size)
        chunk = sequence[first - 1 : last]
        items.append(Item(chunk, 1, chunk.size))

    return items


@torch.no_grad()
def score(
    model: LlamaForCausalLM, items: list[Item], batch_tokens: int, device: torch.device
) -> numpy.ndarray:
    """Summed NLL in nats over each item's target positions"""
    order = sorted(range(len(items)), key=lambda index: -items[index].tokens.size)
    result = numpy.zeros(len(items), dtype=numpy.float64)
    position = 0

    while position < len(order):
        width = items[order[position]].tokens.size
        count = max(1, batch_tokens // width)
        batch = order[position : position + count]
        position += count

        input_ids = torch.full((len(batch), width), EOS, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, index in enumerate(batch):
            tokens = items[index].tokens
            input_ids[row, : tokens.size] = torch.from_numpy(tokens)
            mask[row, : tokens.size] = 1
        input_ids, mask = input_ids.to(device), mask.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=input_ids, attention_mask=mask).last_hidden_state

        weight = model.lm_head.weight
        for row, index in enumerate(batch):
            item = items[index]
            # Position t predicts token t + 1: targets [start, end) come from [start-1, end-1)
            states = hidden[row, item.start - 1 : item.end - 1]
            logits = (states @ weight.T.to(states.dtype)).float()
            targets = input_ids[row, item.start : item.end]
            nll = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
            result[index] = nll.item()

    return result


def bootstrap(
    nll: numpy.ndarray, utf8: numpy.ndarray, rng: numpy.random.Generator, draws: int
) -> tuple[float, float, float]:
    """BPB = sum nll / (ln2 sum bytes), with a 95% CI resampling the units"""
    value = nll.sum() / (LN2 * utf8.sum())
    picks = rng.integers(0, nll.size, size=(draws, nll.size))
    samples = nll[picks].sum(1) / (LN2 * utf8[picks].sum(1))
    low, high = numpy.percentile(samples, [2.5, 97.5])
    return float(value), float(low), float(high)


def reset_floor_rows(model: LlamaForCausalLM, categories: numpy.ndarray) -> dict:
    embedding = model.model.embed_tokens.weight
    assert model.lm_head.weight is embedding, "the export lost its tie"
    vocab = embedding.shape[0]
    ids = numpy.arange(vocab)
    trained = (ids < FIRST_SPECIAL) & (categories[:vocab] != BAN)
    # Chinese V_ban, and every row past the base vocab: Fake-French and the padding
    reset = (categories[:vocab] == BAN) | (ids >= BASE_VOCAB)

    with torch.no_grad():
        mean = embedding[torch.from_numpy(trained).to(embedding.device)].float().mean(0)
        embedding[torch.from_numpy(reset).to(embedding.device)] = mean.to(embedding.dtype)

    return {"floor_rows_reset": int(reset.sum()), "floor_mean_over": int(trained.sum())}


def flores_items(
    encoder: Encoder, flores: Path, max_length: int
) -> dict[str, tuple[list[list[Item]], numpy.ndarray]]:
    """Per language: each sentence's windows and its UTF-8 bytes"""
    sets = {}

    extra = [("fra_Latn", "fake")] if encoder.fake is not None else []
    for code, short in [*FLORES_LANGS.items(), *extra]:
        texts = read_jsonl(flores / f"devtest-{code}.jsonl")
        sets[short] = (
            [windows(encoder.encode(text, short), max_length) for text in texts],
            numpy.array([len(text.encode()) for text in texts], dtype=numpy.float64),
        )

    return sets


def fineweb2_items(
    encoder: Encoder, fineweb2: Path, documents: int, seed: int, max_length: int
) -> dict[str, tuple[list[list[Item]], numpy.ndarray]]:
    sets = {}

    for code in FINEWEB2_LANGS:
        texts = read_jsonl(fineweb2 / f"fineweb-2-{code}-test.jsonl")
        picks = numpy.random.default_rng(seed).choice(
            len(texts), size=min(documents, len(texts)), replace=False
        )
        texts = [texts[index] for index in sorted(picks)]
        languages = [code] + (
            ["fake_fra"] if code == "fra_Latn" and encoder.fake is not None else []
        )

        for language in languages:
            kind = "fake" if language == "fake_fra" else language
            sets[language] = (
                [windows(encoder.encode(text, kind), max_length) for text in texts],
                numpy.array([len(text.encode()) for text in texts], dtype=numpy.float64),
            )

    return sets


def delta_items(
    encoder: Encoder, flores: Path, source: str, target: str
) -> tuple[list[Item], list[Item], numpy.ndarray]:
    """
    For source -> target: the prompt with the real source, with the unrelated one, and
    the target bytes. The pieces are tokenized one by one, so the separators are the
    same anchor tokens (newlines) in every language and condition
    """
    code = {short: code for code, short in FLORES_LANGS.items()} | {"fake": "fra_Latn"}

    def load(language: str, split: str) -> list[str]:
        return read_jsonl(flores / f"{split}-{code[language]}.jsonl")

    newline = encoder.encode("\n", "en")
    blank = encoder.encode("\n\n", "en")
    dev_source, dev_target = load(source, "dev"), load(target, "dev")
    prefix = [numpy.array([EOS])]
    for index in range(SHOTS):
        prefix += [
            encoder.encode(dev_source[index], source),
            newline,
            encoder.encode(dev_target[index], target),
            blank,
        ]
    prefix_tokens = numpy.concatenate(prefix)

    sources, targets = load(source, "devtest"), load(target, "devtest")
    # x': the neighbour in byte-length order (each sentence swaps with its pair)
    lengths = numpy.array([len(text.encode()) for text in sources])
    by_length = numpy.argsort(lengths, kind="stable")
    unrelated = numpy.empty(len(sources), dtype=numpy.int64)
    for rank, index in enumerate(by_length):
        partner = rank + 1 if rank % 2 == 0 else rank - 1
        partner = partner if partner < len(by_length) else rank - 1
        unrelated[index] = by_length[partner]

    real, fake = [], []
    for index, text in enumerate(targets):
        y = numpy.concatenate([encoder.encode(text, target), newline])
        for collection, source_index in [(real, index), (fake, int(unrelated[index]))]:
            context = numpy.concatenate(
                [prefix_tokens, encoder.encode(sources[source_index], source), newline]
            )
            tokens = numpy.concatenate([context, y])
            # y's text tokens are the targets; its closing newline is not
            collection.append(Item(tokens, context.size, tokens.size - newline.size))

    utf8 = numpy.array([len(text.encode()) for text in targets], dtype=numpy.float64)
    return real, fake, utf8


def flatten(groups: list[list[Item]]) -> tuple[list[Item], numpy.ndarray]:
    owner = numpy.repeat(numpy.arange(len(groups)), [len(group) for group in groups])
    return [item for group in groups for item in group], owner


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="a step-N dir")
    parser.add_argument(
        "--fake-french",
        type=Path,
        help="dir with P0-1's fake_french.npz; without it the fake conditions are skipped",
    )
    parser.add_argument("--flores", type=Path, required=True)
    parser.add_argument("--fineweb2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="<dir>/<name>")
    parser.add_argument("--floor", action="store_true", help="B0-Lower's Floor")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--fineweb2-documents", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-tokens", type=int, default=65536)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-delta", action="store_true")
    arguments = parser.parse_args()

    started = time.time()
    device = torch.device("cuda")
    model = LlamaForCausalLM.from_pretrained(
        arguments.checkpoint, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    vocab = model.config.vocab_size
    categories = byte_categories(arguments.tokenizer, max(vocab, BASE_VOCAB))
    report: dict = {
        "checkpoint": str(arguments.checkpoint),
        "floor": arguments.floor,
        "vocab_size": vocab,
        "metrics": {},
    }
    if arguments.floor:
        report |= reset_floor_rows(model, categories)

    fake = (
        FakeFrench(arguments.fake_french / "fake_french.npz")
        if arguments.fake_french is not None
        else None
    )
    has_fake = fake is not None and vocab >= BASE_VOCAB + fake.members.size
    if not has_fake:
        print(f"vocab {vocab}: no Fake-French table or rows, skipping fake", flush=True)
    encoder = Encoder(arguments.tokenizer, fake)
    rng = numpy.random.default_rng(arguments.seed)
    arrays: dict[str, numpy.ndarray] = {}

    for set_name, sets in [
        ("flores", flores_items(encoder, arguments.flores, arguments.max_length)),
        (
            "fineweb2",
            fineweb2_items(
                encoder,
                arguments.fineweb2,
                arguments.fineweb2_documents,
                arguments.seed,
                arguments.max_length,
            ),
        ),
    ]:
        for language, (groups, utf8) in sets.items():
            if "fake" in language and not has_fake:
                continue
            items, owner = flatten(groups)
            nll = numpy.bincount(
                owner,
                weights=score(model, items, arguments.batch_tokens, device),
                minlength=len(groups),
            )
            key = f"bpb/{set_name}/{language}"
            arrays[f"{key}/nll"], arrays[f"{key}/bytes"] = nll, utf8
            value, low, high = bootstrap(nll, utf8, rng, arguments.bootstrap)
            report["metrics"][key] = {"value": value, "ci95": [low, high], "n": len(groups)}
            print(f"{key}: {value:.4f} [{low:.4f}, {high:.4f}]", flush=True)

    if not arguments.no_delta:
        for source, target in DELTA_PAIRS:
            if "fake" in (source, target) and not has_fake:
                continue
            real, unrelated, utf8 = delta_items(encoder, arguments.flores, source, target)
            nll_real = score(model, real, arguments.batch_tokens, device)
            nll_unrelated = score(model, unrelated, arguments.batch_tokens, device)
            key = f"delta/{source}-{target}"
            arrays[f"{key}/nll_real"] = nll_real
            arrays[f"{key}/nll_unrelated"] = nll_unrelated
            arrays[f"{key}/bytes"] = utf8
            gain = (nll_unrelated - nll_real) / LN2
            value = gain.sum() / utf8.sum()
            picks = rng.integers(0, utf8.size, size=(arguments.bootstrap, utf8.size))
            samples = gain[picks].sum(1) / utf8[picks].sum(1)
            low, high = numpy.percentile(samples, [2.5, 97.5])
            report["metrics"][key] = {
                "value": float(value),
                "ci95": [float(low), float(high)],
                "bpb_real": float(nll_real.sum() / (LN2 * utf8.sum())),
                "bpb_unrelated": float(nll_unrelated.sum() / (LN2 * utf8.sum())),
                "n": int(utf8.size),
            }
            print(f"{key}: {value:.4f} [{low:.4f}, {high:.4f}]", flush=True)

    train_keys = [f"bpb/flores/{short}" for short in ("en", "es", "de", "ru")]
    report["metrics"]["bpb/flores/l_train_mean"] = {
        "value": float(numpy.mean([report["metrics"][k]["value"] for k in train_keys]))
    }
    report["seconds"] = time.time() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    numpy.savez(arguments.output.with_suffix(".npz"), **arrays)
    print(f"wrote {arguments.output}.json/.npz in {report['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
