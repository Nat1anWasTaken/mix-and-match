"""
Donor embeddings for --donor-embedding, as fp32 {'weight': [V, d], ...} .pt files:

    # Oracle_Emb, from B0-Upper's resume.pt (NOT the HF dir: that one has the embedding
    # multiplier folded into the weights, AGENTS.md)
    uv run python scripts/embedding_ml_transfer/extract_embedding.py resume \\
        --checkpoint checkpoints/<b0-upper run>/step-N --output oracle_emb.pt

    # an open Qwen3 donor (track 2), bf16 on the Hub, saved as fp32
    uv run python scripts/embedding_ml_transfer/extract_embedding.py hf \\
        --model Qwen/Qwen3-0.6B-Base --output qwen3-0.6b-base-emb.pt

Each file also records where it came from, its shape and the RMS over the rows below
151,643 (special and padding rows from there up are never counted, AGENTS.md).
"""

import argparse
import json
from pathlib import Path

import torch

FIRST_SPECIAL = 151643


def rms(weight: torch.Tensor) -> float:
    return float(weight[:FIRST_SPECIAL].float().pow(2).mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", choices=["resume", "hf"])
    parser.add_argument("--checkpoint", type=Path, help="a step-N dir (resume)")
    parser.add_argument("--model", help="HF model id (hf)")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    info: dict = {"source": arguments.source}

    if arguments.source == "resume":
        state = torch.load(
            arguments.checkpoint / "resume.pt", map_location="cpu", weights_only=False
        )
        weight = state["model"]["model.embed_tokens.weight"].float().clone()
        trainer_state = json.loads((arguments.checkpoint / "trainer_state.json").read_text())
        config = json.loads((arguments.checkpoint / "config.json").read_text())
        info |= {
            "checkpoint": str(arguments.checkpoint),
            "step": trainer_state["step"],
            "tokens_seen": trainer_state["tokens_seen"],
            # The training-time multiplier this embedding was learned under (unfolded)
            "hf_config_vocab_size": config.get("vocab_size"),
        }
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(arguments.model, torch_dtype=torch.bfloat16)
        assert model.config.tie_word_embeddings, f"{arguments.model} is not tied"
        weight = model.get_input_embeddings().weight.detach().float().clone()
        info |= {"model": arguments.model}

    info |= {"shape": list(weight.shape), "rms_below_151643": rms(weight)}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight, "info": info}, arguments.output)
    arguments.output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
