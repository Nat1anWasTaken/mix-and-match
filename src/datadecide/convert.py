"""
Convert one DataDecide checkpoint from OLMo's `hf_olmo` format to HF `LlamaForCausalLM`.

    uv run python -m src.datadecide.convert \\
        --repo allenai/DataDecide-dolma1_7-1B --revision step69369-seed-default \\
        --out $WORK/datadecide/models/dolma1_7-1B-step69369-seed-default

DataDecide's checkpoints need `hf_olmo` (pip install ai2-olmo), which transformers can't
load, so flock can't score them as they are. Their architecture is Llama's under other
names, which `check_config` makes sure of:

  - pre-norm blocks, RMSNorm with a weight, no biases anywhere
  - SwiGLU MLP, RoPE (rotate-half, as HF Llama), no QK norm, no clipping, no ALiBi
  - untied input and output embeddings

So conversion only splits two fused matrices and renames:

  OLMo (model.transformer.)        Llama
  wte                              model.embed_tokens
  blocks.i.attn_norm               model.layers.i.input_layernorm
  blocks.i.att_proj  [q; k; v]     model.layers.i.self_attn.{q,k,v}_proj
  blocks.i.attn_out                model.layers.i.self_attn.o_proj
  blocks.i.ff_norm                 model.layers.i.post_attention_layernorm
  blocks.i.ff_proj   [up; gate]    model.layers.i.mlp.{up,gate}_proj
  blocks.i.ff_out                  model.layers.i.mlp.down_proj
  ln_f                             model.norm
  ff_out                           lm_head

ff_proj's order: OLMo's SwiGLU is `x, gate = ff_proj(h).chunk(2); silu(gate) * x`, so the
first half is Llama's up_proj and the second its gate_proj (Llama: down(silu(gate) * up)).

The vocabulary stays at embedding_size (50304), not the tokenizer's 50280: OLMo's logits
have embedding_size columns, so its softmax normalizes over the padding rows too, and
keeping them gives the same log-probabilities.

The tokenizer files are copied as they are, except tokenizer_config.json's
`tokenizer_class`, OLMoTokenizer, which only hf_olmo defines: PreTrainedTokenizerFast
reads the same tokenizer.json. It has no BOS token, so flock starts each document with
EOS, as OLMES does for OLMo.

datadecide.json in --out records the repo, revision and the commit it resolved to; score
the folder as `<out>@<commit>` so results.json names the exact upstream weights.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig

PREFIX = "model.transformer."
TOKENIZER_FILES = ["tokenizer.json", "special_tokens_map.json"]

# Every hf_olmo config field the mapping above depends on, and the value it assumes.
EXPECTED = {
    "activation_type": "swiglu",
    "alibi": False,
    "attention_layer_norm": False,
    "bias_for_layer_norm": False,
    "block_type": "sequential",
    "clip_qkv": None,
    "include_bias": False,
    "layer_norm_type": "rms",
    "layer_norm_with_affine": True,
    "norm_after": False,
    "rope": True,
    "scale_logits": False,
    "weight_tying": False,
}


def check_config(config: dict) -> None:
    """Raise if the checkpoint isn't the Llama-shaped OLMo the mapping assumes."""
    wrong = {k: config.get(k) for k, v in EXPECTED.items() if config.get(k) != v}
    if config.get("multi_query_attention"):
        wrong["multi_query_attention"] = config["multi_query_attention"]
    if wrong:
        raise ValueError(f"not a Llama-shaped OLMo checkpoint: {wrong}")


def llama_config(config: dict, intermediate_size: int) -> LlamaConfig:
    n_heads = config["n_heads"]
    return LlamaConfig(
        vocab_size=config["embedding_size"] or config["vocab_size"],
        hidden_size=config["d_model"],
        intermediate_size=intermediate_size,
        num_hidden_layers=config["n_layers"],
        num_attention_heads=n_heads,
        num_key_value_heads=config["n_kv_heads"] or n_heads,
        hidden_act="silu",
        max_position_embeddings=config["max_sequence_length"],
        rms_norm_eps=config["layer_norm_eps"],
        rope_parameters={"rope_type": "default", "rope_theta": float(config["rope_theta"])},
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=None,
        eos_token_id=config["eos_token_id"],
        pad_token_id=config["pad_token_id"],
        dtype="float32",
    )


def convert_state(olmo: dict[str, torch.Tensor], config: dict) -> tuple[dict, int]:
    """Llama's state dict from OLMo's, and the MLP's intermediate size."""
    d = config["d_model"]
    n_heads = config["n_heads"]
    kv_dim = (config["n_kv_heads"] or n_heads) * (d // n_heads)
    state = {
        "model.embed_tokens.weight": olmo.pop(PREFIX + "wte.weight"),
        "model.norm.weight": olmo.pop(PREFIX + "ln_f.weight"),
        "lm_head.weight": olmo.pop(PREFIX + "ff_out.weight"),
    }
    intermediate = None
    for i in range(config["n_layers"]):
        src, dst = f"{PREFIX}blocks.{i}.", f"model.layers.{i}."
        q, k, v = olmo.pop(src + "att_proj.weight").split([d, kv_dim, kv_dim], dim=0)
        up, gate = olmo.pop(src + "ff_proj.weight").chunk(2, dim=0)
        intermediate = up.shape[0]
        state |= {
            dst + "input_layernorm.weight": olmo.pop(src + "attn_norm.weight"),
            dst + "self_attn.q_proj.weight": q,
            dst + "self_attn.k_proj.weight": k,
            dst + "self_attn.v_proj.weight": v,
            dst + "self_attn.o_proj.weight": olmo.pop(src + "attn_out.weight"),
            dst + "post_attention_layernorm.weight": olmo.pop(src + "ff_norm.weight"),
            dst + "mlp.up_proj.weight": up,
            dst + "mlp.gate_proj.weight": gate,
            dst + "mlp.down_proj.weight": olmo.pop(src + "ff_out.weight"),
        }
    if olmo:
        raise ValueError(f"weights left unmapped: {sorted(olmo)}")
    # split/chunk return views; safetensors wants each tensor to own its storage
    return {k: t.contiguous().clone() for k, t in state.items()}, intermediate


def convert(repo: str, revision: str, out: Path, cache_dir: Path | None) -> str:
    """Convert repo@revision into out and return the commit it resolved to."""
    sha = HfApi().model_info(repo, revision=revision).sha
    snapshot = Path(snapshot_download(repo, revision=sha, cache_dir=cache_dir))
    config = json.loads((snapshot / "config.json").read_text())
    check_config(config)

    olmo = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        olmo |= load_file(shard)
    state, intermediate = convert_state(olmo, config)

    out.mkdir(parents=True, exist_ok=False)
    llama_config(config, intermediate).save_pretrained(out)
    save_file(state, out / "model.safetensors", metadata={"format": "pt"})
    for name in TOKENIZER_FILES:
        shutil.copy(snapshot / name, out / name)
    tokenizer_config = json.loads((snapshot / "tokenizer_config.json").read_text())
    tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))
    (out / "datadecide.json").write_text(
        json.dumps({"repo": repo, "revision": revision, "commit": sha}, indent=2)
    )
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--repo", required=True, help="e.g. allenai/DataDecide-dolma1_7-1B")
    parser.add_argument("--revision", required=True, help="e.g. step69369-seed-default")
    parser.add_argument("--out", type=Path, required=True, help="new folder for the Llama model")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    arguments = parser.parse_args()
    sha = convert(arguments.repo, arguments.revision, arguments.out, arguments.cache_dir)
    print(f"{arguments.repo}@{arguments.revision} ({sha}) -> {arguments.out}")


if __name__ == "__main__":
    main()
