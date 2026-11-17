from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import torch
from torch import Tensor

IGNORE_INDEX = -100


class LossOutput(NamedTuple):
    # Summed over the scored tokens, so the trainer can normalize by the global count
    loss_sum: Tensor
    num_tokens: Tensor
    # Per position logsumexp of the logits [N], and the max logit, both detached. Only for
    # the logit scale metrics: the fused loss never materializes the logits themselves
    logsumexp: Tensor
    logit_max: Tensor


class CausalLMLossSum(Protocol):
    def __call__(self, hidden: Tensor, weight: Tensor, labels: Tensor) -> LossOutput:
        raise NotImplementedError


def _chunk_step(
    hidden: Tensor, weight: Tensor, labels: Tensor, compute_grad: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    """
    Linear + cross entropy for one chunk of C positions, never wider than [C, V].

    hidden [C, d] and weight [V, d] are already in the compute dtype (bf16 on CUDA), so the
    logits come out of the matmul in bf16 exactly as autocast made them before. The
    softmax math runs in fp32: inside torch.compile the .float() fuses into the reduction
    kernels, so no fp32 [C, V] copy is ever written
    """
    logits = (hidden @ weight.T).float()  # [C, V]
    mask = labels != IGNORE_INDEX
    safe_labels = torch.where(mask, labels, 0)

    logsumexp = torch.logsumexp(logits, dim=-1)  # [C]
    target_logits = logits.gather(1, safe_labels[:, None]).squeeze(1)  # [C]
    loss_sum = torch.where(mask, logsumexp - target_logits, 0.0).sum()
    logit_max = logits.max()

    if not compute_grad:
        return loss_sum, logsumexp, logit_max, None, None

    # d(lse - z_y)/dz = softmax(z) - onehot(y), zero on ignored positions. The one-hot is
    # an elementwise compare, so softmax, subtract and mask fuse into one pass over [C, V]
    vocab_ids = torch.arange(logits.size(1), device=logits.device)
    grad_logits = torch.exp(logits - logsumexp[:, None])
    grad_logits = grad_logits - (vocab_ids[None, :] == safe_labels[:, None]).float()
    grad_logits = (grad_logits * mask[:, None]).to(hidden.dtype)  # [C, V] bf16

    grad_hidden = grad_logits @ weight  # [C, d]
    grad_weight = grad_logits.T @ hidden  # [V, d]
    return loss_sum, logsumexp, logit_max, grad_hidden, grad_weight


ChunkStep = Callable[..., tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]]


def _chunked(
    chunk_step: ChunkStep,
    hidden: Tensor,
    weight: Tensor,
    labels: Tensor,
    chunk_tokens: int,
    compute_grad: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    """
    Run chunk_step over the positions chunk by chunk. hidden [N, d], weight [V, d]
    and labels [N] (already shifted). Returns loss_sum, logsumexp [N], logit_max and,
    with compute_grad, the gradients of loss_sum w.r.t. hidden and weight
    """
    # Under autocast the old lm_head ran in bf16; cast once here, not once per chunk
    compute_dtype = torch.bfloat16 if hidden.is_cuda else hidden.dtype
    hidden_c = hidden.to(compute_dtype)
    weight_c = weight.to(compute_dtype)

    loss_sum = hidden.new_zeros((), dtype=torch.float32)
    logit_max = hidden.new_full((), float("-inf"), dtype=torch.float32)
    logsumexps: list[Tensor] = []
    grad_hidden = torch.empty_like(hidden) if compute_grad else None
    # fp32 accumulator: each chunk's bf16 grad_weight is rounded once, then summed in fp32
    grad_weight = (
        torch.zeros_like(weight, dtype=torch.float32) if compute_grad else None
    )

    for start in range(0, hidden.size(0), chunk_tokens):
        end = start + chunk_tokens
        chunk_loss, chunk_lse, chunk_max, chunk_grad_h, chunk_grad_w = chunk_step(
            hidden_c[start:end], weight_c, labels[start:end], compute_grad
        )
        loss_sum += chunk_loss
        logit_max = torch.maximum(logit_max, chunk_max)
        logsumexps.append(chunk_lse)
        if grad_hidden is not None and grad_weight is not None:
            grad_hidden[start:end] = chunk_grad_h
            grad_weight += chunk_grad_w

    return loss_sum, torch.cat(logsumexps), logit_max, grad_hidden, grad_weight


class _FusedLinearCrossEntropy(torch.autograd.Function):
    """
    loss_sum = sum_t CE(hidden_t @ weight.T, label_t), with the gradients computed in the
    forward pass chunk by chunk (as Liger does): the [C, V] logits of a chunk are needed
    for both the loss and its gradient, so using them right away means the full [N, V]
    logits are never stored for backward. Same three matmuls as autograd, no recompute
    """

    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401
        hidden: Tensor,
        weight: Tensor,
        labels: Tensor,
        chunk_tokens: int,
        chunk_step: ChunkStep,
    ) -> tuple[Tensor, Tensor, Tensor]:
        loss_sum, logsumexp, logit_max, grad_hidden, grad_weight = _chunked(
            chunk_step, hidden, weight, labels, chunk_tokens, compute_grad=True
        )
        assert grad_hidden is not None and grad_weight is not None
        ctx.save_for_backward(grad_hidden, grad_weight.to(weight.dtype))
        ctx.mark_non_differentiable(logsumexp, logit_max)
        return loss_sum, logsumexp, logit_max

    @staticmethod
    def backward(
        ctx: Any,  # noqa: ANN401
        grad_loss: Tensor,
        _grad_logsumexp: Tensor | None,
        _grad_logit_max: Tensor | None,
    ) -> tuple[Tensor, Tensor, None, None, None]:
        grad_hidden, grad_weight = ctx.saved_tensors
        # loss_sum is a scalar, so the chain rule is just a scale (1 for loss_sum.backward())
        return grad_hidden * grad_loss, grad_weight * grad_loss, None, None, None


class LinearCrossEntropySum:
    """
    Fused lm_head + cross entropy: takes the final hidden states (already scaled by the
    muP readout 1/m_d) and the tied output weight instead of logits
    """

    def __init__(self, chunk_tokens: int = 16384, compile: bool = True) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be greater than 0")
        self.chunk_tokens = chunk_tokens
        # dynamic=False: every chunk has the same shape (except a ragged last one), so
        # static shapes give the best kernels at the cost of at most one extra compile
        self.chunk_step: ChunkStep = (
            torch.compile(_chunk_step, dynamic=False) if compile else _chunk_step
        )

    def __call__(self, hidden: Tensor, weight: Tensor, labels: Tensor) -> LossOutput:
        # Position t predicts token t + 1. Shift the labels left and give the last
        # position no target, instead of slicing the hidden states
        shifted_labels = torch.nn.functional.pad(
            labels[:, 1:], (0, 1), value=IGNORE_INDEX
        ).reshape(-1)
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        num_tokens = (shifted_labels != IGNORE_INDEX).sum()

        needs_grad = torch.is_grad_enabled() and (
            flat_hidden.requires_grad or weight.requires_grad
        )
        if needs_grad:
            loss_sum, logsumexp, logit_max = _FusedLinearCrossEntropy.apply(
                flat_hidden, weight, shifted_labels, self.chunk_tokens, self.chunk_step
            )
        else:
            # Eval: same chunked loss, skipping the gradient matmuls
            loss_sum, logsumexp, logit_max, _, _ = _chunked(
                self.chunk_step,
                flat_hidden,
                weight,
                shifted_labels,
                self.chunk_tokens,
                compute_grad=False,
            )

        return LossOutput(loss_sum, num_tokens, logsumexp, logit_max)
