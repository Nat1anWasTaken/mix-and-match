"""
Check a converted DataDecide model against the original hf_olmo one, in float32.

Two steps in two environments, since hf_olmo needs ai2-olmo (numpy<2, so Python <= 3.12)
while flock scores with this project's transformers. Comparing against that stack tests
exactly what flock will load:

    # 1. hf_olmo's numbers, in a throwaway environment with ai2-olmo
    uv run --isolated --no-project --python 3.12 --exclude-newer 2025-04-01 \\
        --with ai2-olmo==0.6.0 --with datasets --with transformers==4.50.3 \\
        python src/datadecide/parity.py reference \\
        --repo allenai/DataDecide-dolma1_7-20M --revision step14594-seed-default \\
        --out $WORK/datadecide/parity/dolma1_7-20M.pt

    # 2. the converted model's numbers on the same inputs, in the project environment
    uv run python -m src.datadecide.parity compare \\
        --model $WORK/datadecide/models/dolma1_7-20M-step14594-seed-default \\
        --reference $WORK/datadecide/parity/dolma1_7-20M.pt

Inputs are two rows of 2048 tokens (Paloma's window): uniformly random ids (seed 42) and
the start of the repo's README.md, tokenized, as real text. Compared:

  - every position's log-probability of the next token, as Paloma's bits per byte sums
  - whole log-softmax rows at a few positions, so errors on other tokens show too
  - the mean next-token loss of each row

compare exits 1 if any log-probability differs by more than --tolerance.
"""

import argparse
import sys
from pathlib import Path

import torch

LENGTH = 2048
POSITIONS = [0, 1, 511, 1024, 2046]


def scores(logits: torch.Tensor, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    target = logprobs[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
    return {
        "target": target.cpu(),
        "rows": logprobs[:, POSITIONS].cpu(),
        "loss": (-target).mean(dim=-1).cpu(),
    }


def reference(arguments: argparse.Namespace) -> None:
    from hf_olmo import OLMoForCausalLM  # noqa: F401, registers hf_olmo with transformers
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sha = HfApi().model_info(arguments.repo, revision=arguments.revision).sha
    tokenizer = AutoTokenizer.from_pretrained(arguments.repo, revision=sha)
    readme = Path(hf_hub_download(arguments.repo, "README.md", revision=sha)).read_text()
    text = tokenizer(readme, add_special_tokens=False)["input_ids"][:LENGTH]
    if len(text) < LENGTH:
        raise ValueError(f"README.md has only {len(text)} tokens")
    generator = torch.Generator().manual_seed(42)
    noise = torch.randint(0, len(tokenizer), (LENGTH,), generator=generator)
    ids = torch.stack([noise, torch.tensor(text)])

    model = OLMoForCausalLM.from_pretrained(
        arguments.repo, revision=sha, torch_dtype=torch.float32, flash_attention=False
    ).to("cuda").eval()
    with torch.no_grad():
        logits = model(input_ids=ids.cuda()).logits
    result = {"repo": arguments.repo, "revision": arguments.revision, "commit": sha, "ids": ids}
    result |= scores(logits, ids.cuda())
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, arguments.out)
    print(f"{arguments.repo}@{arguments.revision} ({sha}): loss {result['loss'].tolist()}")


def compare(arguments: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = torch.load(arguments.reference)
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model, dtype=torch.float32, device_map="cuda"
    ).eval()
    ids = ref["ids"].cuda()
    with torch.no_grad():
        ours = scores(model(input_ids=ids).logits, ids)

    target = (ours["target"] - ref["target"]).abs()
    rows = (ours["rows"] - ref["rows"]).abs()
    loss = (ours["loss"] - ref["loss"]).abs()
    print(f"reference {ref['repo']}@{ref['revision']} ({ref['commit']})")
    for i, name in enumerate(["random ids", "README.md"]):
        print(
            f"{name:>10}: loss {ref['loss'][i]:.6f} vs {ours['loss'][i]:.6f} (|d| {loss[i]:.2e}), "
            f"next-token logprob max |d| {target[i].max():.2e} mean {target[i].mean():.2e}, "
            f"full rows max |d| {rows[i].max():.2e}"
        )
    worst = max(target.max().item(), rows.max().item())
    if worst > arguments.tolerance:
        print(f"FAIL: {worst:.2e} > tolerance {arguments.tolerance:.0e}")
        sys.exit(1)
    print(f"PASS: max |d| {worst:.2e} <= tolerance {arguments.tolerance:.0e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference", help="hf_olmo's scores (needs ai2-olmo)")
    ref.add_argument("--repo", required=True)
    ref.add_argument("--revision", required=True)
    ref.add_argument("--out", type=Path, required=True)
    cmp = commands.add_parser("compare", help="the converted model's scores against them")
    cmp.add_argument("--model", type=Path, required=True)
    cmp.add_argument("--reference", type=Path, required=True)
    cmp.add_argument("--tolerance", type=float, default=1e-3)
    arguments = parser.parse_args()
    {"reference": reference, "compare": compare}[arguments.command](arguments)


if __name__ == "__main__":
    main()
