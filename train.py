import argparse
import dataclasses
import json
import logging
import math
import os
from collections import namedtuple
from pathlib import Path

import numpy
import torch

from src.checkpoint import PretrainedCheckpointer
from src.data import PackedDataLoader, SequencePacker
from src.evaluation import FlockEvaluator, parse_eval_spec
from src.hooks import parse_stop_condition
from src.loggers import ConsoleLogger, WandbLogger
from src.loss import LinearCrossEntropySum
from src.model import LlamaCausalLM, ScalingLlamaConfig
from src.optimizer import build_muon_adamw, build_warmup_stable_decay
from src.recipe.cache import localize_recipe
from src.recipe.recipe import load_recipe
from src.token_stream import RecipeTokenStream
from src.tokenizer import document_token_ids, load_tokenizer
from src.trainer import Trainer, TrainingConfig

TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
WANDB_PROJECT = "lab7-data-curation"
# Llama 3's 128,256. Other tokenizers pass --vocab-size: embedding-ml-transfer's track 1
# is Qwen3's 151,936 plus the Fake-French rows (181,632), track 2 Qwen3's own 151,936
VOCAB_SIZE = 128256
# Lab 7: repeats of up to this fraction of the training data don't count toward the score
FREE_REPEAT_FRACTION = 0.1


DistributedData = namedtuple("DistributedData", ["rank", "world_size", "device"])


def setup_distributed(requested: str | None) -> DistributedData:
    if "WORLD_SIZE" not in os.environ:
        return DistributedData(rank=0, world_size=1, device=pick_device(requested))

    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available() and requested != "cpu":
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        torch.distributed.init_process_group("gloo")
        device = torch.device("cpu")

    return DistributedData(
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
        device=device,
    )


@dataclasses.dataclass(frozen=True)
class ModelSize:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_query_heads: int
    num_key_value_heads: int


# The reference configurations from LAB.md, plus a tiny one for local smoke tests
MODEL_SIZES = {
    "debug": ModelSize(2, 128, 512, 4, 2),
    "150M": ModelSize(6, 768, 3072, 12, 3),
    "300M": ModelSize(9, 1024, 5184, 16, 4),
    "500M": ModelSize(11, 1536, 4672, 24, 6),
    "750M": ModelSize(13, 1792, 5888, 28, 7),
    "1B": ModelSize(15, 1792, 8000, 28, 7),
    # embedding-ml-transfer: d=1024 (Qwen3-0.6B's width), head_dim 64, only the depth
    # changes, so CompleteP transfers from t1-l12 with m_d = 1 and m_L = L / 12
    "t1-l12": ModelSize(12, 1024, 3072, 16, 4),
    "t1-l24": ModelSize(24, 1024, 3072, 16, 4),
    "t1-l40": ModelSize(40, 1024, 3072, 16, 4),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--recipe", required=True, help="compiled recipe directory from src/recipe/"
    )
    parser.add_argument(
        "--recipe-cache",
        type=Path,
        help="read the recipe and its shards from this local copy, made by "
        "python -m src.recipe.cache, instead of S3. Same data in the same order; "
        "the run still logs the S3 --recipe URI",
    )
    parser.add_argument("--model-size", choices=MODEL_SIZES, required=True)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    # Replaces --grad-accum-steps: the global batch is part of the recipe, while the
    # micro batch and GPU count only decide how it is computed
    parser.add_argument(
        "--global-batch-tokens",
        type=int,
        default=4096 * 8 * 16 * 8,
        help="tokens per optimizer step summed over all ranks (default: 4,194,304)",
    )

    # muP: hyperparameters are tuned on --mup-base-size and transferred. Muon's lr is in
    # spectral units (update spectral norm per step), not comparable to the AdamW lr
    parser.add_argument(
        "--mup-base-size",
        choices=MODEL_SIZES,
        help="model size whose hyperparameters this run transfers from (default: none, "
        "i.e. plain SP with every multiplier at 1)",
    )
    parser.add_argument("--embedding-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--output-multiplier",
        type=float,
        default=1.0,
        help="alpha_out on the logits (tied output side), for frozen donor embeddings",
    )
    parser.add_argument(
        "--donor-embedding",
        type=Path,
        help="a .pt holding a [vocab_size, hidden_size] tensor (or a dict with "
        "'weight') copied into the tied embedding after init, e.g. Oracle_Emb from "
        "B0-Upper's resume.pt or Qwen3-0.6B-Base's embed_tokens",
    )
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="don't train the tied embedding (requires_grad False before the optimizer "
        "is built, so it gets no optimizer state, no weight decay and no all-reduce)",
    )
    parser.add_argument(
        "--embedding-lr-multiplier",
        type=float,
        default=1.0,
        help="the tied embedding's AdamW lr is --adamw-lr times this; the RMSNorm gains "
        "keep --adamw-lr (embedding-ml-transfer A4a-lr0.1)",
    )
    parser.add_argument(
        "--untie-output",
        action="store_true",
        help="embedding-ml-transfer A6a: an untied lm_head, initialized like the embedding "
        "and trained by AdamW with --adamw-weight-decay; the embedding is input-only",
    )
    parser.add_argument(
        "--output-delta",
        action="store_true",
        help="embedding-ml-transfer A6b: logits = E (I + Delta) h through the tied lm_head, "
        "Delta a d x d matrix starting at 0 and trained by Muon, whose weight decay pulls "
        "it back to 0, i.e. the map back to the identity",
    )
    parser.add_argument(
        "--softmax-keep",
        type=Path,
        help="masked softmax: a .npz whose 'keep' holds the vocab ids the logits cover "
        "(scripts/embedding_ml_transfer/token_counts.py, the ids counted > 0 in the "
        "training set). Every other embedding row is never a negative and is held fixed "
        "against weight decay",
    )
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adamw-lr", type=float, default=3e-3)
    parser.add_argument(
        "--muon-weight-decay",
        type=float,
        default=0.015,
        help="per-step hidden decay is muon_lr * this, scaled by 1 / m_in per matrix",
    )
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.95)
    parser.add_argument("--adamw-eps", type=float, default=1e-12)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--decay-fraction", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument(
        "--eval-every-steps", type=int, help="default: every 10%% of training"
    )
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        metavar="NAME[@SPLIT]",
        help="required, repeatable: a flock suite or task to score at every eval, by its "
        "flock.eval.tasks registry name (as `flock eval run --suite/--task` takes it), "
        "on its own split or on SPLIT, e.g. --eval paloma@val or --eval paloma (test)",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        help="score only each task's first N docs, for debug runs (default: all)",
    )
    parser.add_argument(
        "--stop-when",
        action="append",
        default=[],
        metavar="METRIC<=VALUE",
        help="repeatable: end training early, with a checkpoint, once a logged metric "
        "crosses VALUE (<=, <, >=, >), e.g. "
        "--stop-when 'eval/paloma@val/macro/bits_per_byte<=1.15'. eval/... metrics are "
        "checked after each eval, so --eval-every-steps sets how soon a stop is seen",
    )
    parser.add_argument(
        "--ckpt-every-steps", type=int, help="default: about every 5%% of training"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest checkpoint in the run's checkpoint directory",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        help="start this new run from another run's checkpoint (a step-N directory): "
        "model, optimizer, schedule step, data stream and tokens_seen all carry over, "
        "under this run's own name. For WSD decay branches off a stable-phase checkpoint. "
        "Once the run holds checkpoints of its own, --resume continues those instead",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-name",
        required=True,
        help="names the wandb run and the checkpoint directory, so runs never overwrite",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="model init; the data order comes from the recipe's own seed",
    )
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help=f"embedding rows (default {VOCAB_SIZE:,}, Llama 3's); at least the "
        "tokenizer's length, and above it when the recipe holds ids the tokenizer "
        "doesn't (Fake-French's shifted ids)",
    )
    parser.add_argument("--device", help="default: cuda, else mps, else cpu")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the model, on by default (--no-compile turns it off)",
    )
    parser.add_argument(
        "--ce-chunk-tokens",
        type=int,
        # Fastest on an H200 at 150M (2048-16384 tried): ~8 GB of bf16 [C, V] transients
        default=16384,
        help="positions per chunk of the fused lm_head + cross entropy; the largest "
        "transient is chunk x vocab logits (bf16) plus their gradient",
    )
    parser.add_argument(
        "--attn-implementation",
        help="default: flash_attention_3 on CUDA, else sdpa",
    )
    parser.add_argument("--no-wandb", action="store_true", help="log to console only")
    parser.add_argument(
        "--wandb-project",
        default=WANDB_PROJECT,
        help=f"wandb project to log to (default {WANDB_PROJECT}, the lab's)",
    )

    arguments = parser.parse_args()

    # No default: which eval a run is judged by is a choice each run makes
    if not arguments.eval:
        parser.error(
            "--eval is required: name the flock suite or task to score during training, "
            "e.g. --eval paloma@val (Paloma's val split) or --eval paloma (its test "
            "split, what `flock eval run --suite paloma` scores)"
        )
    for text in arguments.eval:
        try:
            parse_eval_spec(text)
        except ValueError as error:
            parser.error(str(error))
    if arguments.eval_limit is not None and arguments.eval_limit < 1:
        parser.error("--eval-limit must be at least 1")
    for text in arguments.stop_when:
        try:
            parse_stop_condition(text)
        except ValueError as error:
            parser.error(str(error))

    return arguments


def oj_counted_tokens(total_tokens: int, repeated_share: float) -> int:
    """
    oj_total_tokens: the trained tokens minus the repeats that are free, which are at
    most FREE_REPEAT_FRACTION of the unrepeated data. LAB.md's example: 220B trained, 20B
    of them repeats, counts 200B; 230B with 30B repeats counts 210B, since only 20B of
    the 30B are free
    """
    repeated = total_tokens * repeated_share
    free = min(repeated, FREE_REPEAT_FRACTION * (total_tokens - repeated))
    return round(total_tokens - free)


def default_ckpt_every_steps(total_steps: int, log_every_steps: int) -> int:
    """
    About every 5% of training, rounded down to a multiple of the log interval
    """
    every = total_steps // 20 // log_every_steps * log_every_steps
    return max(every, log_every_steps)


def pick_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scheduler_decay_start(total_steps: int, decay_fraction: float) -> int:
    """The first step of the WSD decay, as build_warmup_stable_decay computes it"""
    return total_steps - max(1, math.ceil(total_steps * decay_fraction))


def check_branch_point(
    checkpoint: Path, total_steps: int, tokens_per_step: int, decay_start: int
) -> None:
    """
    Refuse an --init-from checkpoint this run's schedule can't continue smoothly.

    The LR multiplier depends only on the step, so a checkpoint taken while the source
    run's LR was constant (after warmup, before its decay) continues exactly as if this
    run had been that long from the start, provided this run's decay starts at or after
    the checkpoint's step: the multiplier is 1.0 up to decay_start either way
    """
    state_path = checkpoint / "trainer_state.json"
    if not (checkpoint / "resume.pt").is_file() or not state_path.is_file():
        raise FileNotFoundError(f"--init-from {checkpoint} is not a step-N checkpoint")

    state = json.loads(state_path.read_text())
    step, tokens_seen = state["step"], state["tokens_seen"]
    if tokens_seen != step * tokens_per_step:
        raise ValueError(
            f"--init-from {checkpoint}: {tokens_seen:,} tokens in {step} steps is not "
            f"{tokens_per_step:,} per step; use the source run's --global-batch-tokens"
        )
    if step >= total_steps:
        raise ValueError(
            f"--init-from {checkpoint} is at step {step}, not before this run's last "
            f"step {total_steps}"
        )
    if step > decay_start:
        raise ValueError(
            f"--init-from {checkpoint} is at step {step}, after this run's decay starts "
            f"(step {decay_start}): the LR would jump down at the branch. Lower "
            "--decay-fraction or raise --total-tokens"
        )


def build_model_config(
    size: ModelSize,
    base_size: ModelSize | None,
    embedding_multiplier: float,
    seq_len: int,
    bos_token_id: int,
    eos_token_id: int,
    vocab_size: int = VOCAB_SIZE,
    output_multiplier: float = 1.0,
    untie_output: bool = False,
    output_delta: bool = False,
) -> ScalingLlamaConfig:
    base_size = base_size or size
    return ScalingLlamaConfig(
        base_hidden_size=base_size.hidden_size,
        base_intermediate_size=base_size.intermediate_size,
        base_num_hidden_layers=base_size.num_layers,
        embedding_multiplier=embedding_multiplier,
        output_multiplier=output_multiplier,
        output_delta=output_delta,
        vocab_size=vocab_size,
        hidden_size=size.hidden_size,
        intermediate_size=size.intermediate_size,
        num_hidden_layers=size.num_layers,
        num_attention_heads=size.num_query_heads,
        num_key_value_heads=size.num_key_value_heads,
        max_position_embeddings=seq_len,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=not untie_output,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )


def load_donor_embedding(
    model: LlamaCausalLM, path: Path | None, freeze: bool, rank: int
) -> None:
    """
    Copy a donor into the tied embedding and/or freeze it. Runs after init and before
    the optimizer, DDP and compile, so a frozen embedding is never an optimizer param
    """
    embedding = model.model.embed_tokens.weight

    if path is not None:
        donor = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(donor, dict):
            donor = donor["weight"]
        if tuple(donor.shape) != tuple(embedding.shape):
            raise ValueError(
                f"--donor-embedding {path} is {tuple(donor.shape)}, the model's "
                f"embedding {tuple(embedding.shape)}: check --vocab-size and the model size"
            )
        with torch.no_grad():
            embedding.copy_(donor.to(embedding.dtype))
        if rank == 0:
            rms = donor.float().pow(2).mean().sqrt().item()
            print(f"donor embedding: {path} {tuple(donor.shape)}, rms {rms:.4g}")

    if freeze:
        embedding.requires_grad_(False)

    # The tie must survive both: lm_head is the same Parameter as the embedding
    if model.config.tie_word_embeddings:
        assert model.lm_head.weight is model.model.embed_tokens.weight, "tie broken"
    else:
        assert model.lm_head.weight is not model.model.embed_tokens.weight


def hold_dropped_rows(
    adamw: torch.optim.Optimizer, embedding: torch.nn.Parameter, dropped: torch.Tensor
) -> None:
    """
    Keep the masked-out embedding rows exactly where they are. They get no gradient
    (never an input in the training set, never in the masked softmax), so Adam moves them
    by 0, but AdamW's decoupled weight decay would still shrink them every step. Save them
    before each AdamW step and write them back after it. The first step also checks that
    their gradient really is zero, so the write-back can't hide a leak
    """
    saved: dict[str, torch.Tensor] = {}

    def before(_optimizer: torch.optim.Optimizer, _args: object, _kwargs: object) -> None:
        if "checked" not in saved and embedding.grad is not None:
            leak = embedding.grad.index_select(0, dropped).abs().max().item()
            if leak != 0:
                raise RuntimeError(f"masked rows have a nonzero gradient ({leak:.3g})")
            saved["checked"] = torch.ones(())
        saved["rows"] = embedding.detach().index_select(0, dropped)

    def after(_optimizer: torch.optim.Optimizer, _args: object, _kwargs: object) -> None:
        with torch.no_grad():
            embedding.index_copy_(0, dropped, saved.pop("rows"))

    adamw.register_step_pre_hook(before)
    adamw.register_step_post_hook(after)


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)

    rank, world_size, device = setup_distributed(arguments.device)

    tokens_per_micro_step = arguments.seq_len * arguments.micro_batch_size * world_size
    if arguments.global_batch_tokens % tokens_per_micro_step != 0:
        raise ValueError(
            f"global_batch_tokens ({arguments.global_batch_tokens}) must be divisible by "
            f"seq_len * micro_batch_size * world_size ({tokens_per_micro_step})"
        )

    grad_accum_steps = arguments.global_batch_tokens // tokens_per_micro_step
    tokens_per_step = arguments.global_batch_tokens

    if arguments.total_tokens % tokens_per_step != 0:
        raise ValueError(
            f"total_tokens ({arguments.total_tokens}) must be divisible by "
            f"tokens_per_step ({tokens_per_step})"
        )

    total_steps = math.ceil(arguments.total_tokens / tokens_per_step)

    training_config = TrainingConfig(
        total_tokens=arguments.total_tokens,
        seq_len=arguments.seq_len,
        micro_batch_size=arguments.micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=arguments.max_grad_norm,
        log_every_steps=arguments.log_every_steps,
        eval_every_steps=arguments.eval_every_steps or max(1, total_steps // 10),
        ckpt_every_steps=arguments.ckpt_every_steps
        or default_ckpt_every_steps(total_steps, arguments.log_every_steps),
    )

    tokenizer = load_tokenizer(arguments.tokenizer)
    # Without a BOS (Qwen3), documents start with the EOS, as prepare.py writes them
    bos_token_id, eos_token_id = document_token_ids(tokenizer)
    # A row per token id: fewer rows than the tokenizer's ids is an out-of-range lookup,
    # which on CUDA is a device-side assert with no useful message
    if arguments.vocab_size < len(tokenizer):
        raise ValueError(
            f"--vocab-size {arguments.vocab_size:,} is smaller than {arguments.tokenizer}'s "
            f"{len(tokenizer):,} tokens"
        )

    model_config = build_model_config(
        MODEL_SIZES[arguments.model_size],
        MODEL_SIZES[arguments.mup_base_size] if arguments.mup_base_size else None,
        arguments.embedding_multiplier,
        arguments.seq_len,
        bos_token_id,
        eos_token_id,
        arguments.vocab_size,
        arguments.output_multiplier,
        arguments.untie_output,
        arguments.output_delta,
    )
    model = LlamaCausalLM(model_config).to(device)
    load_donor_embedding(
        model, arguments.donor_embedding, arguments.freeze_embeddings, rank
    )

    # FA3 needs Hopper-class CUDA GPUs, so CPU and MPS debug runs fall back to SDPA
    attn_implementation = arguments.attn_implementation or (
        "flash_attention_3" if device.type == "cuda" else "sdpa"
    )
    model.set_attn_implementation(attn_implementation)

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    if rank == 0:
        print(
            f"model: {arguments.model_size}, {num_parameters:,} parameters on {device}"
        )

    optimizer = build_muon_adamw(
        model,
        muon_lr=arguments.muon_lr,
        adamw_lr=arguments.adamw_lr,
        muon_weight_decay=arguments.muon_weight_decay,
        adamw_weight_decay=arguments.adamw_weight_decay,
        adamw_betas=(arguments.adamw_beta1, arguments.adamw_beta2),
        adamw_eps=arguments.adamw_eps,
        embedding_lr_multiplier=arguments.embedding_lr_multiplier,
    )
    scheduler = build_warmup_stable_decay(
        optimizer,
        total_steps=total_steps,
        warmup_steps=arguments.warmup_steps,
        decay_fraction=arguments.decay_fraction,
    )

    training_model = model

    if world_size > 1:
        device_ids = [device.index] if device.type == "cuda" else None
        training_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=device_ids, output_device=device
        )

    # Compile after wrapping, so dynamo sees DDP and keeps all-reduce overlapped with backward
    if arguments.compile:
        # transformers' flash-attention path logs a warning_once (fp32 q/k cast back to bf16)
        # inside the compiled region, which dynamo can't trace: a graph break in every layer
        torch._dynamo.config.ignore_logging_functions.add(logging.Logger.warning_once)
        # The hub FA3 op has no fake impl, so attention still splits into frames that guard
        # on module.layer_idx: one compiled entry per layer, two under DDP (train + eval)
        torch._dynamo.config.recompile_limit = max(
            8, 2 * model_config.num_hidden_layers + 4
        )
        training_model.compile()
        # The unwrapped model, which the DDP compile above doesn't cover, was compiled for
        # eval. Eval no longer goes through it (TrainingScorer calls model.forward,
        # uncompiled), but DDP's forward calls the module, so it is left as it was rather
        # than risk changing how training traces
        if training_model is not model:
            model.compile()

    keep_ids = None
    if arguments.softmax_keep is not None:
        keep_ids = torch.from_numpy(
            numpy.load(arguments.softmax_keep)["keep"].astype("int64")
        ).to(device)
        # The output weight is the one the masked softmax leaves out rows of: the tied
        # embedding, or with --untie-output the separate head (the input embedding then
        # gets gradients only on rows that occur as inputs, which are all kept)
        embedding = model.lm_head.weight
        if embedding.requires_grad:
            dropped = torch.ones(model_config.vocab_size, dtype=torch.bool, device=device)
            dropped[keep_ids] = False
            adamw = optimizer.optimizers[1]
            assert any(p is embedding for g in adamw.param_groups for p in g["params"])
            hold_dropped_rows(adamw, embedding, dropped.nonzero().squeeze(1))
        if rank == 0:
            print(
                f"masked softmax: {keep_ids.numel():,} of {model_config.vocab_size:,} "
                f"rows kept ({arguments.softmax_keep})"
            )
    loss_fn = LinearCrossEntropySum(
        chunk_tokens=arguments.ce_chunk_tokens,
        compile=arguments.compile,
        keep_ids=keep_ids,
        vocab_size=model_config.vocab_size,
    )
    recipe = load_recipe(arguments.recipe)

    # Compilers add a small margin on top, since the ranks' shares vary a little and
    # the packer drops each rank's last partial row
    if recipe.total_tokens < arguments.total_tokens:
        raise ValueError(
            f"recipe {recipe.uri} holds {recipe.total_tokens:,} tokens, fewer than "
            f"--total-tokens {arguments.total_tokens:,}"
        )
    if recipe.tokenizer != arguments.tokenizer:
        raise ValueError(
            f"recipe {recipe.uri} was tokenized with {recipe.tokenizer}, "
            f"not {arguments.tokenizer}"
        )

    # Where the recipe came from, for the run's config, even when read from the cache
    recipe_uri = recipe.uri
    if arguments.recipe_cache is not None:
        recipe = localize_recipe(recipe, arguments.recipe_cache)
        if rank == 0:
            print(f"recipe: reading {recipe_uri} from {recipe.uri}")

    # The stream shards documents across ranks before packing, so each rank only
    # packs its own 1/world_size
    documents = RecipeTokenStream(recipe, rank=rank, world_size=world_size)
    rows = SequencePacker(documents, arguments.seq_len)
    train_data = PackedDataLoader(rows, arguments.micro_batch_size)

    # Every rank scores its share of the eval tasks, so every rank needs an evaluator:
    # a rank without one would skip the eval gather and hang the others
    evaluator = FlockEvaluator(
        [parse_eval_spec(text) for text in arguments.eval],
        model,
        tokenizer,
        max_length=arguments.seq_len,
        limit=arguments.eval_limit,
        rank=rank,
        world_size=world_size,
    )

    run_config: dict[str, object] = {
        **vars(arguments),
        "recipe_uri": recipe_uri,
        "recipe_total_tokens": recipe.total_tokens,
        "recipe_repeated_tokens": recipe.repeated_tokens,
        "recipe_phases": [
            {"name": phase.name, "tokens": phase.tokens} for phase in recipe.phases
        ],
        "output_dir": str(arguments.output_dir),
        "model": model_config.to_dict(),
        "num_parameters": num_parameters,
        "tokens_per_step": tokens_per_step,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
        "total_steps": total_steps,
        "training": dataclasses.asdict(training_config),
        "initialization": (
            f"muP: hidden normal(0, {model_config.initializer_range} * sqrt(base_fan_in "
            "/ fan_in)), embedding constant std; logits * base_d / d, residual branches "
            "* base_L / L (CompleteP alpha=1), embedding * embedding_multiplier"
        ),
        "optimizer": (
            "Muon(spectral_unclamped) for decoder 2D weights with wd / m_in, "
            "AdamW for the tied embedding and RMSNorm gains"
        ),
        "schedule": "linear warmup, constant, linear decay to 0 (WSD)",
        "attention": attn_implementation,
    }

    run_dir = arguments.output_dir / arguments.run_name
    checkpointer = PretrainedCheckpointer(run_dir, tokenizer)
    resume_from = checkpointer.latest()

    # Checked before wandb starts, so a refused launch leaves no empty run behind
    if arguments.resume and resume_from is None:
        raise FileNotFoundError(f"--resume, but {run_dir} holds no checkpoint")
    if not arguments.resume and resume_from is not None:
        raise FileExistsError(
            f"{run_dir} already holds {resume_from.name}; pass --resume to continue "
            "it, or pick a new --run-name"
        )
    # A branch run starts from the other run's checkpoint only until it has its own
    if resume_from is None and arguments.init_from is not None:
        resume_from = arguments.init_from
        check_branch_point(
            resume_from,
            total_steps,
            tokens_per_step,
            scheduler_decay_start(total_steps, arguments.decay_fraction),
        )
        run_config["init_from"] = str(resume_from)

    if arguments.no_wandb or rank != 0:
        logger: ConsoleLogger | WandbLogger = ConsoleLogger()
    else:
        logger = WandbLogger(
            arguments.wandb_project, arguments.run_name, run_config
        )

    trainer = Trainer(
        model=training_model,
        train_data=train_data,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        ckpt=checkpointer,
        logger=logger,
        cfg=training_config,
        data_stats=documents.stats,
        hooks=[parse_stop_condition(text) for text in arguments.stop_when],
    )

    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
        if rank == 0:
            print(f"resumed from {resume_from}")

    if rank == 0:
        print(f"training {total_steps} steps of {tokens_per_step:,} tokens")
    trainer.train()

    if isinstance(logger, WandbLogger):
        # The stream's counters run a few rows ahead of training, so the repeated share
        # of the streamed tokens is applied to the trained ones
        streamed = sum(
            value
            for key, value in trainer.data_metrics.items()
            if key.startswith("tokens/")
        )
        repeated_share = (
            trainer.data_metrics.get("repeated_tokens", 0.0) / streamed if streamed else 0.0
        )
        total_tokens = trainer.state.tokens_seen
        logger.run.summary["total_tokens"] = total_tokens
        if trainer.stop_reason is not None:
            logger.run.summary["stop_reason"] = trainer.stop_reason
        # The lab's OJ scores the lab project's oj_total_tokens: only its runs write it
        if arguments.wandb_project == WANDB_PROJECT:
            logger.run.summary["oj_total_tokens"] = oj_counted_tokens(
                total_tokens, repeated_share
            )
        logger.finish()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
