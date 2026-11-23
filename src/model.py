import math
from typing import Any, cast

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

MUP_CONFIG_KEYS = (
    "base_hidden_size",
    "base_intermediate_size",
    "base_num_hidden_layers",
    "embedding_multiplier",
    "output_multiplier",
    "output_delta",
)


class ScalingLlamaConfig(LlamaConfig):
    def __init__(
        self,
        base_hidden_size: int | None = None,
        base_intermediate_size: int | None = None,
        base_num_hidden_layers: int | None = None,
        embedding_multiplier: float = 1.0,
        output_multiplier: float = 1.0,
        output_delta: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        # None means "this model is its own base": every multiplier is 1 and we get plain SP
        self.base_hidden_size = base_hidden_size or self.hidden_size
        self.base_intermediate_size = base_intermediate_size or self.intermediate_size
        self.base_num_hidden_layers = base_num_hidden_layers or self.num_hidden_layers
        self.embedding_multiplier = embedding_multiplier
        # alpha_out: a constant on the logits, for a frozen donor embedding whose row
        # norms differ from our init's (embedding-ml-transfer P0-2b); 1 leaves it out
        self.output_multiplier = output_multiplier
        # embedding-ml-transfer A6b: a trainable d x d map on the readout side, logits =
        # E (I + Delta) h with Delta starting at 0, so the run starts as the tied model
        self.output_delta = output_delta


class LlamaCausalLM(LlamaForCausalLM):
    config_class = ScalingLlamaConfig

    def __init__(self, config: ScalingLlamaConfig) -> None:
        # muP width multiplier m_d and CompleteP depth multiplier m_L, both 1 at the base
        self.width_multiplier = config.hidden_size / config.base_hidden_size
        self.depth_multiplier = config.num_hidden_layers / config.base_num_hidden_layers

        super().__init__(config)

        for layer in self.model.layers:
            assert isinstance(layer, LlamaDecoderLayer)
            layer.self_attn.o_proj.register_forward_hook(self._scale_branch)
            layer.mlp.down_proj.register_forward_hook(self._scale_branch)

        # Created after super().__init__, whose init pass has already drawn every other
        # weight, so adding it leaves the rest of the init (and the RNG) as without it
        self.output_delta: nn.Linear | None = None
        if config.output_delta:
            self.output_delta = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            nn.init.zeros_(self.output_delta.weight)

    def _scale_branch(
        self, module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> torch.Tensor:
        return output / self.depth_multiplier

    def initialize_weights(self) -> None:
        # HF's version dispatches every submodel (here self.model, a LlamaModel) to its own
        # generic _init_weights, so ours would only ever see lm_head. Apply ours everywhere
        self.apply(self._init_weights)

        # Every residual branch starts at exactly 0 (as in the CompleteP + Muon reference
        # code), so h_L = embedding at init and the stream carries no depth-dependent noise
        for layer in self.model.layers:
            assert isinstance(layer, LlamaDecoderLayer)
            nn.init.zeros_(layer.self_attn.o_proj.weight)
            nn.init.zeros_(layer.mlp.down_proj.weight)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range

        if module is self.lm_head:
            if self.config.tie_word_embeddings:
                # tied to the embedding right after init, so its own init never matters
                return
            # embedding-ml-transfer A6a: an untied head, drawn like the embedding
            nn.init.normal_(module.weight, std=std)
            return
        if isinstance(module, nn.Linear):
            if module.in_features == self.config.hidden_size:
                base_fan_in = self.config.base_hidden_size
            elif module.in_features == self.config.intermediate_size:
                base_fan_in = self.config.base_intermediate_size
            else:
                raise ValueError(f"unexpected fan_in {module.in_features}")
            nn.init.normal_(
                module.weight, std=std * math.sqrt(base_fan_in / module.in_features)
            )
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, LlamaRMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        return_logits: bool = True,
        **kwargs: Any,  # noqa: ANN401
    ) -> CausalLMOutputWithPast | BaseModelOutputWithPast:
        # alpha_emb: a tunable constant on the input side (Cerebras 10, MiniCPM 12). Passing
        # inputs_embeds makes HF skip its own embedding lookup
        inputs_embeds = (
            self.model.embed_tokens(input_ids) * self.config.embedding_multiplier
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        # muP readout: logits scaled by 1/m_d, applied in the forward pass rather than the
        # weight, since the weight is the tied embedding. lm_head is linear with no bias, so
        # we scale its [B,T,d] input instead of the [B,T,V] logits; the division then fuses
        # into the final RMSNorm kernel instead of costing an extra pass over the logits
        readout_hidden = outputs.last_hidden_state / self.width_multiplier
        if self.config.output_multiplier != 1.0:
            readout_hidden = readout_hidden * self.config.output_multiplier
        if self.output_delta is not None:
            # (I + Delta) h: logits = E (I + Delta) h through the tied lm_head below
            readout_hidden = readout_hidden + self.output_delta(readout_hidden)
        if not return_logits:
            # Training and eval stop here: LinearCrossEntropySum applies lm_head.weight
            # chunk by chunk, so the full [B,T,V] logits are never materialized
            return BaseModelOutputWithPast(last_hidden_state=readout_hidden)
        return CausalLMOutputWithPast(logits=self.lm_head(readout_hidden))

    @torch.no_grad()
    def export_plain_llama(self) -> LlamaForCausalLM:
        """
        Fold the muP/CompleteP multipliers into the weights, giving a plain Llama that
        computes the same function, for the HF upload / flock eval
        """
        plain_config = LlamaConfig(
            **{
                k: v
                for k, v in self.config.to_dict().items()
                if k not in MUP_CONFIG_KEYS
            }
        )
        # The readout map has no place in a plain Llama: it folds into an untied lm_head
        if self.output_delta is not None:
            plain_config.tie_word_embeddings = False
        plain = LlamaForCausalLM(plain_config)
        state = {k: v for k, v in self.state_dict().items() if not k.startswith("output_delta.")}
        if self.output_delta is not None:
            state["lm_head.weight"] = self.lm_head.weight.clone()
        plain.load_state_dict(state)

        # The branches are linear with no bias, so 1/m_L folds into their last weight
        for layer in plain.model.layers:
            layer = cast(LlamaDecoderLayer, layer)
            layer.self_attn.o_proj.weight.div_(self.depth_multiplier)
            layer.mlp.down_proj.weight.div_(self.depth_multiplier)

        # The embedding is tied, so alpha_emb scaled into it also scales the logits by
        # alpha_emb. That, the 1/m_d readout and alpha_out fold into the final RMSNorm gain
        alpha = self.config.embedding_multiplier
        plain.model.embed_tokens.weight.mul_(alpha)
        plain.model.norm.weight.div_(self.width_multiplier * alpha)
        plain.model.norm.weight.mul_(self.config.output_multiplier)
        if not plain_config.tie_word_embeddings:
            # Untied: the gain above divided the logits by alpha, which a tied embedding
            # multiplied back in. Here the head carries it itself, after (I + Delta)
            head = plain.lm_head.weight
            if self.output_delta is not None:
                delta = self.output_delta.weight.to(head.dtype)
                head.copy_(head + head @ delta)
            head.mul_(alpha)
        else:
            assert plain.lm_head.weight is plain.model.embed_tokens.weight
        return plain
