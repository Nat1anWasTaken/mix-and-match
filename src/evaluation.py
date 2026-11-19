"""
In-training evaluation with flock.eval: any flock suite or task, scored between steps.

An eval is named the way `flock eval run` names it, a key of flock's SUITES or TASKS
registry, with an optional split: `paloma` scores the registered suite as it is (Paloma's
test split), `paloma@val` the same tasks on the val split. The scoring is flock's own
(prepare_task once, score_prepared at every eval), so a test-split number here is the
number `flock eval run` gives for the same weights, up to the dtype and attention kernel
(see TrainingScorer).
"""

import dataclasses
import difflib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import torch
from flock.eval.model import DEFAULT_BATCH_TOKENS, FlockCausalLM, ScoringModel
from flock.eval.run import PreparedTask, prepare_task, score_prepared
from flock.eval.tasks import SUITES, TASKS, PerplexityTask, Suite, Task
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPast

from .model import LlamaCausalLM
from .trainer import EvaluationResult, autocast

# The key an eval's tasks are prepared and scored under: (eval label, task name). Two
# evals can hold tasks of one name, e.g. paloma and paloma@val
type TaskKey = tuple[str, str]


@dataclass(frozen=True)
class EvalSpec:
    """
    One --eval: a registered flock suite or task, on its own split or on `split`
    """

    name: str
    split: str | None
    item: Task | Suite

    @property
    def label(self) -> str:
        """The name its metrics are logged under, eval/<label>/..."""
        return f"{self.name}@{self.split}" if self.split else self.name

    @property
    def tasks(self) -> tuple[Task, ...]:
        return self.item.tasks if isinstance(self.item, Suite) else (self.item,)


def parse_eval_spec(text: str) -> EvalSpec:
    """
    NAME or NAME@SPLIT, NAME a key of flock's SUITES or TASKS. Raises ValueError, naming
    the closest registered names for an unknown one
    """
    name, at, split = text.partition("@")
    registry: dict[str, Task | Suite] = {**TASKS, **SUITES}

    if name not in registry:
        close = difflib.get_close_matches(name, registry, n=5)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise ValueError(
            f"--eval {text}: {name!r} is not a flock suite or task{hint} "
            "(flock.eval.tasks.SUITES and TASKS list them)"
        )
    if at and (not split or "@" in split):
        raise ValueError(f"--eval {text}: expected NAME or NAME@SPLIT")

    item = registry[name]
    return EvalSpec(name, split or None, with_split(item, split) if split else item)


def with_split(item: Task | Suite, split: str) -> Task | Suite:
    """
    The same suite or task, with every task scoring `split` of its dataset instead. The
    names, domains, windows and suite tree stay, so the scores average the same way. A
    multiple-choice task's few-shot examples are not moved
    """
    if isinstance(item, Suite):
        return dataclasses.replace(
            item, members=tuple(with_split(member, split) for member in item.members)
        )

    return dataclasses.replace(
        item, dataset=dataclasses.replace(item.dataset, split=split)
    )


def suite_metric(suite: Suite) -> str:
    """The metric a scored suite's score is, e.g. bits_per_byte for a Paloma source"""
    return suite.metric or suite.scale


class TrainingScorer(FlockCausalLM):
    """
    Our LlamaCausalLM, called the way flock's ScoringModel calls a transformers causal LM.
    It is a FlockCausalLM, flock's wrapper for models that aren't transformers ones, and
    keeps its device and dtype; only the config and the forward are ours.

    It runs our own forward, with the muP/CompleteP multipliers, so what is scored is the
    function the run trains, not an export of it. The forward stops at the readout hidden
    states (return_logits=False, as in training), and only the last logits_to_keep
    positions go through lm_head, which is all ScoringModel reads: [B, T, d] ->
    [B, keep, d] -> [B, keep, V]. It runs under bf16 autocast with the run's attention
    kernel, as training does, so its numbers differ slightly from flock's float32 + SDPA
    official run.

    It calls model.forward, not model(...): the in-place torch.compile only wraps
    __call__, and eval batches change shape with every padded width, so a compiled eval
    would recompile. The submodules' forward hooks (the 1/m_L branch scaling) still run
    """

    def __init__(self, model: LlamaCausalLM, max_length: int) -> None:
        super().__init__(model)
        self.llama = model
        # ScoringModel reads max_position_embeddings and the attention implementation
        # from here. With flash_attention_3 it pins no SDPA kernel, since none runs
        self.config = SimpleNamespace(
            max_position_embeddings=max_length,
            _attn_implementation=model.config._attn_implementation,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool = False,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        with autocast(input_ids.device):
            output = self.llama.forward(
                input_ids=cast(torch.LongTensor, input_ids),
                attention_mask=attention_mask,
                position_ids=cast(torch.LongTensor, position_ids),
                use_cache=False,
                return_logits=False,
            )
            hidden = cast(BaseModelOutputWithPast, output).last_hidden_state
            assert hidden is not None
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:]
            # lm_head is the tied embedding, with no bias
            logits = torch.nn.functional.linear(hidden, self.llama.lm_head.weight)

        return SimpleNamespace(logits=logits)


def task_cost(task: Task, limit: int | None) -> int:
    """
    Roughly how long a task takes to score: a perplexity task's UTF-8 bytes (its token
    count is about proportional to them), a multiple-choice task's doc count
    """
    if isinstance(task, PerplexityTask):
        return sum(len(doc.text.encode()) for doc in task.load(limit))

    docs, _ = task.load(limit)
    return len(docs)


def assign_tasks(costs: Sequence[int], world_size: int) -> list[int]:
    """
    Each task's rank: the costliest first, each to the rank with the least cost so far
    (LPT), ties to the earlier task and the lower rank, so the answer is deterministic
    """
    loads = [0] * world_size
    ranks = [0] * len(costs)

    for index in sorted(range(len(costs)), key=lambda index: (-costs[index], index)):
        rank = min(range(world_size), key=lambda rank: (loads[rank], rank))
        ranks[index] = rank
        loads[rank] += costs[index]

    return ranks


def spec_metrics(
    spec: EvalSpec, by_task: dict[str, dict[str, float]]
) -> dict[str, float]:
    """
    A task eval logs each of its metrics, eval/<label>/<metric>. A suite eval logs every
    scored suite in it, eval/<label>/<suite>/<metric>. An unscored suite of scored suites
    on one metric (Paloma: a suite per source, no score over all of them) also logs the
    unweighted mean of those, eval/<label>/macro/<metric>, and how many it averages
    """
    prefix = f"eval/{spec.label}"
    item = spec.item

    if not isinstance(item, Suite):
        return {f"{prefix}/{key}": value for key, value in by_task[item.name].items()}

    metrics = {
        f"{prefix}/{suite.name}/{suite_metric(suite)}": suite.score(by_task)
        for suite in item.suites
        if suite.scored
    }

    members = [member for member in item.members if isinstance(member, Suite)]
    if (
        not item.scored
        and len(members) == len(item.members)
        and all(member.scored for member in members)
        and len({suite_metric(member) for member in members}) == 1
    ):
        scores = [member.score(by_task) for member in members]
        macro = sum(scores) / len(scores)
        metrics[f"{prefix}/macro/{suite_metric(members[0])}"] = macro
        metrics[f"{prefix}/macro/members"] = len(scores)

    return metrics


class FlockEvaluator:
    """
    Scores flock suites and tasks with the training model, for eval/... during training.

    Each task is scored whole on one rank, and the ranks split the tasks: rank 0 costs
    them, assign_tasks balances them, and each rank prepares (loads and tokenizes) only
    its own, once, here. At each eval every rank scores its tasks and rank 0 gathers the
    per-task metrics, so every rank must call the evaluator, or the gather hangs. No task
    is split across ranks, so no number depends on the world size
    """

    def __init__(
        self,
        specs: Iterable[EvalSpec],
        model: LlamaCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int,
        limit: int | None = None,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.specs = tuple(specs)
        labels = [spec.label for spec in self.specs]
        if not labels:
            raise ValueError("no eval given")
        if len(set(labels)) != len(labels):
            raise ValueError(f"--eval repeats an eval: {labels}")

        tasks: dict[TaskKey, Task] = {
            (spec.label, task.name): task for spec in self.specs for task in spec.tasks
        }
        for (label, name), task in tasks.items():
            # Checked here, so the run fails at startup rather than at its first eval
            if isinstance(task, PerplexityTask) and task.window > max_length:
                raise ValueError(
                    f"--eval {label}: {name} scores windows of {task.window} tokens, "
                    f"above the model's max_length {max_length} (--seq-len)"
                )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_tokens = batch_tokens
        self.rank = rank
        self.world_size = world_size

        start = time.perf_counter()
        # Rank 0 loads every split first, so the other ranks then read them from the cache
        assignment: list[list[int] | None] = [None]
        if rank == 0:
            costs = [task_cost(task, limit) for task in tasks.values()]
            assignment[0] = assign_tasks(costs, world_size)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(assignment, src=0)
        ranks = assignment[0]
        assert ranks is not None

        scorer = self.scoring_model(model)
        # In suite order, so the tasks of one split load it once (flock caches one split)
        self.prepared: dict[TaskKey, PreparedTask] = {
            key: prepare_task(scorer, task, limit)
            for (key, task), task_rank in zip(tasks.items(), ranks, strict=True)
            if task_rank == rank
        }

        if rank == 0:
            print(
                f"eval: {', '.join(labels)}: {len(tasks)} tasks on {world_size} "
                f"rank(s), {len(self.prepared)} on rank 0, prepared in "
                f"{time.perf_counter() - start:.1f} s"
            )

    def scoring_model(self, model: LlamaCausalLM) -> ScoringModel:
        # Always this tokenizer object: prepared tasks are tied to it by identity
        return ScoringModel(
            TrainingScorer(model, self.max_length),
            self.tokenizer,
            name="training model",
            max_length=self.max_length,
            batch_tokens=self.batch_tokens,
        )

    def __call__(self, model: PreTrainedModel) -> EvaluationResult:
        # TrainingScorer runs our forward (return_logits=False); a stock HF model has none
        if not isinstance(model, LlamaCausalLM):
            raise TypeError(f"FlockEvaluator scores a LlamaCausalLM, got {type(model)}")
        start = time.perf_counter()
        scorer = self.scoring_model(model)
        local = {
            key: score_prepared(scorer, prepared).metrics
            for key, prepared in self.prepared.items()
        }
        local_seconds = time.perf_counter() - start

        parts: list[tuple[dict[TaskKey, dict[str, float]], float]] = [
            (local, local_seconds)
        ]
        if torch.distributed.is_initialized():
            gathered: list[object] | None = (
                [None] * self.world_size if self.rank == 0 else None
            )
            torch.distributed.gather_object((local, local_seconds), gathered, dst=0)
            parts = gathered or []  # ty: ignore[invalid-assignment]

        if self.rank != 0:
            return EvaluationResult(metrics={})

        results = {key: value for part, _ in parts for key, value in part.items()}
        metrics: dict[str, float] = {}
        for spec in self.specs:
            by_task = {
                name: task_metrics
                for (label, name), task_metrics in results.items()
                if label == spec.label
            }
            metrics.update(spec_metrics(spec, by_task))

        # The slowest rank sets the eval's cost; the fastest shows how even the split is
        rank_seconds = [seconds for _, seconds in parts]
        metrics["eval/seconds"] = time.perf_counter() - start
        metrics["eval/rank_seconds_min"] = min(rank_seconds)
        metrics["eval/rank_seconds_max"] = max(rank_seconds)
        return EvaluationResult(metrics=metrics)
