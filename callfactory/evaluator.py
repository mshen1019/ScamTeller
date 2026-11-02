"""Evaluator agent (component 5): prefix-level labelling.

This is where label quality comes from for the whole project, so the rules are stricter than elsewhere:

* **One independent, stateless call per prefix** (R5.1). Not fed turn by turn into one session: that way the 3rd
  judgement gets contaminated by the context of the 5th, and the label stops being "the conclusion reachable having heard only 3 turns".
* **Temperature 0** (R5.3): labelling must be reproducible.
* **Blind labelling** (R5.5): the prompt has no place for the scenario; the evaluator doesn't know the ground truth.
* **Only the caller is scored** (R5.9): the receiver's suspicion is a reaction, not a risk signal. Otherwise a wary persona would
  push scores up, and the model would learn "listen for who is suspicious" instead of "listen for who is applying pressure".
* **signals are monotonic, risk_score is not** (R5.8). Once evidence appears it doesn't go away, so signals accumulate;
  but the score may fall. When a challenged caller readily accepts verification the risk really does drop, and erasing that drop
  deletes the hard negative's discriminating information.

Direction of ``label_from_index`` (decided at implementation time): the label for utterance i may **only** come from a prefix with index ≤ i.
When ``--eval-stride`` skips intermediate prefixes, a missing one carries over the **previous** evaluated prefix, never a later one:
borrowing a later one writes future information into the label at turn i, and the model learns to alarm earlier than the evidence, which is exactly
the metric this project measures. The cost is that labels "lag" under stride, which is the cost on the correct side.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence

from .prompts import (
    GLOBAL_EVAL_SYSTEM,
    PREFIX_EVAL_RUBRIC_VERSION,
    PREFIX_EVAL_SYSTEM,
    global_eval_user_message,
    prefix_eval_user_message,
    rubric_fingerprint,
)
from .providers import LLMProvider, ProviderError, extract_json
from .schema import (
    BlindAssessment,
    PrefixLabel,
    UtteranceLabel,
    Utterance,
    clean_signals,
    coerce_tier,
)

#: Default score line for "an alarm is warranted now". It is ANDed with sufficient_evidence (R5.10).
#: Score line for the detection point.
#:
#: 0.5 -> 0.7, and ``sufficient_evidence`` is no longer required: in testing this boolean was the step in the pipeline that lost
#: the most information. Of 169 real misses, 136 (81%) peaked at exactly 0.7 with the boolean false, while
#: the score channel itself discriminates much better: 180/366 hard-negative peaks sit at 0.3, and only 5
#: reach 0.7. Setting the detection point at score >= 0.7: scam detection 0.616 -> 0.927, at the cost of 5 false alarms among 366
#: hard negatives and 1 among 177 ordinary legitimate calls.
#:
#: This **moves the operating point**; it doesn't make the labeller smarter. It just uses what the labeller already knows.
#: The persistence rule for ``earliest_detection_index`` already looked only at the score, not at this boolean (see that function's
#: docstring); this step applies the same conclusion to the threshold itself. Whether the 0.7 level really counts as "sufficient evidence"
#: is still awaiting a human ruling (EVAL_SYSTEM §5.9).
DEFAULT_DETECTION_THRESHOLD = 0.7


@dataclass
class EvalFailure:
    """A labelling call returned nothing usable. Recorded in failures.jsonl without stopping the batch (R5.11)."""

    kind: str                          # unparseable | provider_error
    stage: str                         # prefix | global
    through_index: int | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "stage": self.stage,
            "through_index": self.through_index,
            "detail": self.detail[:400],
        }


@dataclass
class EvaluationResult:
    prefix_labels: list[PrefixLabel]
    utterance_labels: list[UtteranceLabel]
    earliest_detection_index: int | None = None
    blind_assessment: BlindAssessment | None = None
    failures: list[EvalFailure] = field(default_factory=list)
    stride: int = 1
    threshold: float = DEFAULT_DETECTION_THRESHOLD
    require_persistence: bool = True
    require_sufficient: bool = False


def prefix_indices(num_utterances: int, stride: int = 1) -> list[int]:
    """Which prefixes actually get sent for labelling.

    The first (T0) and the last are always labelled: T0 is the R7.5 criterion "every model alarms at T0 => non-discriminative",
    and the last carries the overall conclusion. The middle is sampled by stride.
    """
    if num_utterances <= 0:
        return []
    if stride < 1:
        raise ValueError("eval_stride must be >= 1")
    keep = {0, num_utterances - 1}
    keep.update(i for i in range(num_utterances) if (i + 1) % stride == 0)
    return sorted(keep)


class EvaluatorAgent:
    """Prefix labelling plus one full quality assessment."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        seed: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        self.provider = provider
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens

    def provenance(self) -> dict[str, Any]:
        """Provenance recorded in metadata (C4).

        When ``temperature_honoured_by_provider`` is false, R5.3 (temperature 0 for reproducible labels) didn't actually
        take effect. Put the evaluator on Claude, for instance: it takes no temperature and has no seed,
        so relabelling the same conversation twice can give different labels. This has to be written in the data, not just in the docs.
        """
        honoured = bool(self.provider.supports_temperature)
        return {
            "provider": self.provider.config.name,
            "model": self.provider.model,
            # Which rubric version produced the labels. Without these two fields, mixed batches produce contradictory labels and nobody is warned.
            "rubric_version": PREFIX_EVAL_RUBRIC_VERSION,
            "rubric_fingerprint": rubric_fingerprint(),
            "temperature": self.temperature if honoured else None,
            "temperature_requested": self.temperature,
            "temperature_honoured_by_provider": honoured,
            "seed": self.seed,
            "seed_honoured_by_provider": bool(self.provider.supports_seed),
        }

    async def label_prefix(
        self, utterances: Sequence[Utterance]
    ) -> tuple[PrefixLabel, EvalFailure | None]:
        """Label ``utterances``, which must already be a prefix slice; later content is never passed in."""
        if not utterances:
            raise ValueError("prefix must not be empty")
        through = utterances[-1].index
        try:
            raw = await self.provider.complete(
                system=PREFIX_EVAL_SYSTEM,
                messages=[
                    {"role": "user", "content": prefix_eval_user_message(list(utterances))}
                ],
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
                tag="evaluator_prefix",
            )
        except ProviderError as exc:
            return _empty_label(through), EvalFailure(
                "provider_error", "prefix", through, str(exc)
            )
        try:
            data = extract_json(raw)
        except (ValueError, TypeError) as exc:
            # Degrade to empty labels rather than guess: a guessed label is harder to spot than a missing one (R5.11).
            return _empty_label(through), EvalFailure(
                "unparseable", "prefix", through, f"{exc}: {raw[:200]!r}"
            )

        score = _clamp(data.get("risk_score"), 0.0, 1.0)
        return PrefixLabel(
            through_index=through,
            risk_score=score,
            risk_tier=coerce_tier(data.get("risk_tier"), score),
            signals=clean_signals(data.get("signals")),
            new_signals=clean_signals(data.get("new_signals_in_last_turn")),
            sufficient_evidence=bool(data.get("sufficient_evidence", False)),
            rationale=str(data.get("rationale", ""))[:240],
        ), None

    async def assess_global(
        self, utterances: Sequence[Utterance]
    ) -> tuple[BlindAssessment | None, EvalFailure | None]:
        """Full quality assessment. ``blind_is_suspicious`` is never used as an early-warning label (R5.13)."""
        try:
            raw = await self.provider.complete(
                system=GLOBAL_EVAL_SYSTEM,
                messages=[
                    {"role": "user", "content": global_eval_user_message(list(utterances))}
                ],
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
                tag="evaluator_global",
            )
        except ProviderError as exc:
            return None, EvalFailure("provider_error", "global", None, str(exc))
        try:
            data = extract_json(raw)
        except (ValueError, TypeError) as exc:
            return None, EvalFailure("unparseable", "global", None, f"{exc}: {raw[:200]!r}")

        blind = data.get("blind_is_suspicious")
        return BlindAssessment(
            coherence=_clamp(data.get("coherence"), 0.0, 5.0),
            naturalness=_clamp(data.get("naturalness"), 0.0, 5.0),
            repetition=_clamp(data.get("repetition"), 0.0, 1.0),
            role_consistency=bool(data.get("role_consistency", True)),
            blind_is_suspicious=None if blind is None else bool(blind),
            notes=str(data.get("notes", ""))[:240],
        ), None


async def evaluate_conversation(
    utterances: Sequence[Utterance],
    evaluator: EvaluatorAgent,
    *,
    stride: int = 1,
    prefix_eval: bool = True,
    global_eval: bool = True,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    require_persistence: bool = True,
    require_sufficient: bool = False,
    concurrency: int | None = None,
) -> EvaluationResult:
    """Label one conversation. Prefixes don't depend on each other, so they're sent concurrently (R5.12)."""
    utterances = list(utterances)
    failures: list[EvalFailure] = []
    labels: list[PrefixLabel] = []

    if prefix_eval and utterances:
        wanted = prefix_indices(len(utterances), stride)
        sem = asyncio.Semaphore(concurrency) if concurrency else None

        async def one(k: int) -> tuple[PrefixLabel, EvalFailure | None]:
            # The slicing happens here, so utterances[k+1:] structurally can't reach the request body (R5.2).
            if sem is None:
                return await evaluator.label_prefix(utterances[: k + 1])
            async with sem:
                return await evaluator.label_prefix(utterances[: k + 1])

        results = await asyncio.gather(*(one(k) for k in wanted))
        for label, failure in results:
            labels.append(label)
            if failure:
                failures.append(failure)
        labels.sort(key=lambda p: p.through_index)
        _enforce_monotonic(labels)

    blind: BlindAssessment | None = None
    if global_eval and utterances:
        blind, failure = await evaluator.assess_global(utterances)
        if failure:
            failures.append(failure)

    earliest = earliest_detection_index(
        labels, threshold,
        require_persistence=require_persistence,
        require_sufficient=require_sufficient,
    )
    mark_evidence_established(labels, earliest)
    return EvaluationResult(
        prefix_labels=labels,
        utterance_labels=build_utterance_labels(utterances, labels, earliest),
        earliest_detection_index=earliest,
        blind_assessment=blind,
        failures=failures,
        stride=stride,
        threshold=threshold,
        require_persistence=require_persistence,
        require_sufficient=require_sufficient,
    )


def _enforce_monotonic(labels: list[PrefixLabel]) -> None:
    """signals accumulate, and new_signals is derived back from the accumulation. risk_score is never touched (R5.8).

    ``new_signals`` doesn't trust the model's own report: models often read "newly appeared this turn" as "also shown this turn",
    so the same signal gets reported as new on every prefix and the detection point can't be computed.
    """
    seen: list[str] = []
    for label in labels:
        union = list(seen)
        for sig in label.signals:
            if sig not in union:
                union.append(sig)
        label.new_signals = [s for s in union if s not in seen]
        label.signals = sorted(union)
        seen = union


def earliest_detection_index(
    labels: Sequence[PrefixLabel],
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    *,
    require_persistence: bool = True,
    require_sufficient: bool = False,
) -> int | None:
    """The first prefix with ``risk_score >= threshold`` (R5.10).

    With ``require_sufficient=True``, ``sufficient_evidence`` must also be true. That is
    the detect/v1 rule, kept only so old labels can be reproduced; it isn't the default. For why the default dropped it,
    see the comment on ``DEFAULT_DETECTION_THRESHOLD``.

    ``require_persistence``: the prefix must also **not be overturned by the next labelled prefix**.

    Motivation from testing: a hard negative (genuine IT support) poked above 0.5 on one prefix and set
    sufficient_evidence, then the evaluator itself brought the score back to 0.20. But "take the first prefix that meets the condition"
    made that self-overturned misjudgement a permanent "decidable at T7" in the ground truth, so a model that rightly didn't alarm
    would be marked wrong on it.

    Only the next prefix's **score** is checked, not its sufficient_evidence: the score is the well-calibrated channel
    (in testing hard negatives 0.00-0.20 vs scams 0.60-0.90), and the boolean is the noisy one.
    Real scam scores rise all the way, so this rule costs them almost nothing.
    """
    for i, label in enumerate(labels):
        if label.risk_score < threshold:
            continue
        if require_sufficient and not label.sufficient_evidence:
            continue
        if require_persistence and i + 1 < len(labels):
            if labels[i + 1].risk_score < threshold:
                continue
        return label.through_index
    return None


def mark_evidence_established(
    labels: Sequence[PrefixLabel], earliest: int | None
) -> None:
    """Mark the monotonic "evidence is sufficient" in place: prefixes from ``earliest`` on are always True.

    Must be called after ``earliest_detection_index`` is computed: it's derived from that moment,
    not given by the labeller. For its meaning and cost see ``PrefixLabel.evidence_established``.
    """
    for label in labels:
        label.evidence_established = earliest is not None and label.through_index >= earliest


def build_utterance_labels(
    utterances: Sequence[Utterance],
    labels: Sequence[PrefixLabel],
    earliest: int | None,
) -> list[UtteranceLabel]:
    """Spread prefix labels onto each utterance, looking only backwards (see the module docstring)."""
    by_index = {label.through_index: label for label in labels}
    out: list[UtteranceLabel] = []
    current: PrefixLabel | None = None
    for utt in utterances:
        current = by_index.get(utt.index, current)
        if current is None:
            # Only happens when the stride doesn't include T0; prefix_indices guarantees T0, so this is just defensive.
            continue
        out.append(UtteranceLabel(
            index=utt.index,
            speaker=utt.speaker,
            risk_score=current.risk_score,
            risk_tier=current.risk_tier,
            new_signals=list(current.new_signals) if current.through_index == utt.index else [],
            cumulative_signals=list(current.signals),
            is_detection_point=earliest is not None and utt.index == earliest,
            evidence_established=earliest is not None and utt.index >= earliest,
            label_from_index=current.through_index,
        ))
    return out


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return low


def _empty_label(through_index: int) -> PrefixLabel:
    return PrefixLabel(
        through_index=through_index,
        risk_score=0.0,
        risk_tier="none",
        signals=[],
        new_signals=[],
        sufficient_evidence=False,
        rationale="",
    )
