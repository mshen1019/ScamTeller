"""Full generation of one record: dialogue -> prefix labelling -> programmatic validation.

This layer only assembles; it adds no rules, and the boundaries all live in the modules it calls. It exists to write
"what one record is made of" in one place: transcript, labels and **provenance** (C4).

Provenance is not optional. Without ``derived_seed`` and each agent's provider/model, two weeks later nobody can say
how a batch was made; without ``seed_honoured_by_provider``, the Anthropic path would be
mistaken for bit-reproducible (N5).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .agents import CallerAgent, ReceiverAgent
from .dialogue import DialogueFailure, run_dialogue
from .evaluator import DEFAULT_DETECTION_THRESHOLD, EvaluatorAgent, evaluate_conversation
from .providers import LLMProvider, ProviderRoles
from .schema import Conversation, GenerationTask
from .validators import annotate_repetition, validate_conversation


@dataclass
class GenerationSettings:
    min_rounds: int = 5
    max_rounds: int = 20
    temperature: float = 0.9              # R4.10, dialogue wants diversity
    eval_stride: int = 1
    prefix_eval: bool = True
    global_eval: bool = True
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD
    #: True = the old detect/v1 rule (also requires sufficient_evidence). Defaults to False;
    #: see the comment on evaluator.DEFAULT_DETECTION_THRESHOLD.
    detection_require_sufficient: bool = False
    repetition_threshold: float = 0.5
    empty_retries: int = 1
    max_consecutive_fallbacks: int = 2
    #: Labelling requests in flight at once within one conversation. None = unlimited (this is where N1's peak request count comes from).
    eval_concurrency: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def agent_provenance(provider: LLMProvider, *, temperature: float, seed: int | None) -> dict[str, Any]:
    """Who generated this record, and with what parameters.

    The two ``*_honoured_by_provider`` fields aren't decoration: Claude's Messages API has no seed and no
    longer accepts temperature. If metadata only recorded the values we **requested**, two weeks later this batch would be
    read as "temperature 0.9, fixed seed" data (N5).
    """
    honoured = bool(provider.supports_temperature)
    return {
        "provider": provider.config.name,
        "model": provider.model,
        "temperature": temperature if honoured else None,
        "temperature_requested": temperature,
        "temperature_honoured_by_provider": honoured,
        "seed": seed,
        "seed_honoured_by_provider": bool(provider.supports_seed),
    }


async def generate_conversation(
    task: GenerationTask,
    roles: ProviderRoles,
    *,
    settings: GenerationSettings | None = None,
    run_id: str = "run",
    p4_warnings: list[str] | None = None,
) -> Conversation:
    """Run one record end to end. Dialogue-level failures raise ``DialogueFailure``, which the batch layer records in failures."""
    settings = settings or GenerationSettings()
    seed = task.derived_seed

    caller = CallerAgent(roles.caller, task, seed=seed, temperature=settings.temperature)
    receiver = ReceiverAgent(roles.receiver, task, seed=seed, temperature=settings.temperature)

    started = time.monotonic()
    dialogue = await run_dialogue(
        task, caller, receiver,
        min_rounds=settings.min_rounds,
        max_rounds=settings.max_rounds,
        empty_retries=settings.empty_retries,
        max_consecutive_fallbacks=settings.max_consecutive_fallbacks,
    )

    evaluator = EvaluatorAgent(roles.evaluator, seed=seed)
    evaluation = await evaluate_conversation(
        dialogue.utterances,
        evaluator,
        stride=settings.eval_stride,
        prefix_eval=settings.prefix_eval,
        global_eval=settings.global_eval,
        threshold=settings.detection_threshold,
        require_sufficient=settings.detection_require_sufficient,
        concurrency=settings.eval_concurrency,
    )

    conv = Conversation(
        conversation_id=f"{run_id}-{task.index:05d}",
        utterances=dialogue.utterances,
        task=task,
        generation={
            "run_id": run_id,
            "sample_index": task.index,
            "created_at": time.time(),
            "elapsed_s": round(time.monotonic() - started, 2),
            "bank_version": task.bank_version,
            "seed": task.seed,
            "derived_seed": task.derived_seed,
            "agents": {
                "caller": agent_provenance(
                    roles.caller, temperature=settings.temperature, seed=caller.seed
                ),
                "receiver": agent_provenance(
                    roles.receiver, temperature=settings.temperature, seed=receiver.seed
                ),
                "evaluator": evaluator.provenance(),
            },
            # When P4 is explicitly waived, leave a trace here; otherwise there's no telling afterwards which batches graded themselves.
            "p4_warnings": list(p4_warnings or []),
            "settings": settings.to_dict(),
            "dialogue": dialogue.to_dict(),
            "evaluation": {
                "stride": evaluation.stride,
                "threshold": evaluation.threshold,
                "require_persistence": evaluation.require_persistence,
                "require_sufficient": evaluation.require_sufficient,
                # How the detection point was derived from the prefix labels. The labelling itself is identified by rubric_version;
                # this identifies the **derivation rule**. The two evolve separately, and mixing them gives incomparable labels.
                "detection_rule": (
                    "detect/v1" if evaluation.require_sufficient else "detect/v2"
                ),
                "prefixes_labelled": len(evaluation.prefix_labels),
                "failures": [f.to_dict() for f in evaluation.failures],
            },
            # Filled in by the gatekeeper step (C4); the slot is reserved so the metadata shape stays stable across steps.
            "rubric_version": None,
        },
        prefix_labels=evaluation.prefix_labels,
        utterance_labels=evaluation.utterance_labels,
        blind_assessment=evaluation.blind_assessment,
        earliest_detection_index=evaluation.earliest_detection_index,
    )

    annotate_repetition(conv, settings.repetition_threshold)
    report = validate_conversation(
        conv,
        min_rounds=settings.min_rounds,
        max_rounds=settings.max_rounds,
        repetition_threshold=settings.repetition_threshold,
    )
    conv.validation = report.to_list()
    return conv


__all__ = [
    "GenerationSettings",
    "generate_conversation",
    "agent_provenance",
    "DialogueFailure",
]
