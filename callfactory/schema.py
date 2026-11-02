"""Data contracts.

Three record kinds, matching docs/REQUIREMENTS.md §3:

* Scenario bank   -- ``Scenario``, human-editable, versioned
* Candidate data  -- ``Conversation``, written as two outputs: transcript (no labels) and metadata (labels)
* Verdicts        -- ``AdmissionDecision``, kept in a separate decisions.jsonl for auditing

P2 (labels physically separate from inputs) is enforced by ``transcript_record()`` and ``LABEL_KEYS`` together:
the first emits only three fields, the second lets the validator find leakage at any depth.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import asdict, dataclass, field
from typing import Any

SPEAKERS = ("caller", "receiver")
CATEGORIES = ("scam", "legitimate")
RISK_TIERS = ("none", "low", "medium", "high")
DECISIONS = ("admit", "reject", "review")

#: Closed vocabulary of risk signals. The scenario author and the evaluator may only use these words;
#: anything outside it is dropped, never guess-mapped (R1.3 / R5.7). Free-text signals can't be trained on.
RISK_SIGNALS = (
    "urgency_pressure",
    "authority_claim",
    "unverified_identity",
    "secrecy_request",
    "unusual_payment_channel",
    "threat_of_consequence",
    "personal_info_request",
    "remote_access_request",
    "inconsistent_details",
    "refusal_to_verify",
    "too_good_to_be_true",
    "script_like_repetition",
    "callback_evasion",
    "emotional_manipulation",
)

#: Signals that come only from the caller persona's speech style and have nothing to do with the call's purpose.
#: Scenarios must not declare them in expected_signs: a caller of any family may read from a script,
#: so their coverage is checked against the persona bank, not the scenario bank.
STYLE_ONLY_SIGNALS = ("script_like_repetition",)

#: Signals set by the call's purpose, which Scenario.expected_signs should declare.
SCENARIO_SIGNALS = tuple(s for s in RISK_SIGNALS if s not in STYLE_ONLY_SIGNALS)

#: Signals the evaluator's anchors treat as "enough on their own" (see prompts.PREFIX_EVAL_SYSTEM).
#: The two must agree. A scenario that declares a strong signal can't be "no observable evidence"; this catches self-contradicting scenarios.
STRONG_SIGNALS = (
    "refusal_to_verify",
    "callback_evasion",
    "secrecy_request",
    "threat_of_consequence",
    "remote_access_request",
    "unusual_payment_channel",
    "inconsistent_details",
    "too_good_to_be_true",
)

PLACEHOLDER_TOKENS = (
    # [PERSON_3] was added after testing: a third person in a call (the receiver's daughter, a colleague) is common,
    # and with the vocabulary stopping at _2 the model invents [PERSON_3] itself and trips the privacy warning.
    "[PERSON_1]", "[PERSON_2]", "[PERSON_3]", "[COMPANY_A]", "[COMPANY_B]",
    "[PHONE_1]", "[ACCOUNT_REF]", "[AMOUNT_1]", "[AMOUNT_2]", "[CITY_1]",
    "[URL_1]", "[CODE_1]",
)


# --------------------------------------------------------------------------- #
# Scenario bank
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioProvenance:
    """Where a scenario came from. Human-edited scenarios set author_model to "human"."""

    author_model: str
    bank_version: int
    context: str = ""
    created_at: float = field(default_factory=time.time)
    edited_by_human: bool = False
    #: Which version of the author prompt wrote it (``prompts.SCENARIO_AUTHOR_VERSION``). Empty for hand-written scenarios.
    #: Once the author prompt changes, the bank holds scenarios written under several rule versions; this is the only
    #: way to tell afterwards that "this one was written under the old rules".
    rubric_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Scenario:
    scenario_id: str
    category: str                      # scam | legitimate
    family: str
    pretext: str
    org_kind: str
    expected_signs: list[str]
    plausible_outcomes: list[str]
    hard_negative: bool = False
    #: Hard negatives only: how it behaves like a real organisation when challenged (R1.4).
    #: A hard negative without this degrades into an ordinary negative, so empty means rejected.
    disambiguator: str = ""
    #: The scenario **suits** a slow-burn performance (a fully ordinary opening, pressure later). Whether a given run is slow-burn
    #: is decided by GenerationTask.late_onset; the sampler allocates it among suitable scenarios per --late-onset-ratio.
    late_onset: bool = False
    #: Subset of ``plausible_outcomes``: endings where the receiver is **won over**.
    #:
    #: They exist because of a shortcut found in testing: 5 of the 6 scam outcomes in the seed bank were "receiver resists",
    #: and none of the 5 legitimate ones were. "Receiver resists" separated the two classes almost perfectly, so a model could
    #: score by reading the **receiver** and ignoring the caller entirely. Compliant outcomes break that shortcut.
    #:
    #: The safety boundary is unchanged (P8): the call ends the moment the receiver agrees and never acts it out.
    compliant_outcomes: list[str] = field(default_factory=list)
    #: A scam by intent, but no usable evidence surfaces at any point. Tests the detector's restraint, not its sensitivity.
    #: Note this is not "the scam isn't completed": no scam in this bank is completed (P8).
    no_observable_evidence: bool = False
    #: Kept out of training by default, eval only. Used by the split command.
    eval_only: bool = False
    notes: str = ""
    provenance: ScenarioProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.provenance is None:
            out.pop("provenance")
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        prov = data.get("provenance")
        return cls(
            **{k: v for k, v in data.items() if k != "provenance"},
            provenance=ScenarioProvenance(**prov) if prov else None,
        )


# --------------------------------------------------------------------------- #
# Personas (one set each for caller and receiver; the fields differ on purpose -- R3 / R4)
# --------------------------------------------------------------------------- #

@dataclass
class CallerPersona:
    persona_id: str
    role: str
    speech_style: str
    pressure_profile: str
    #: The **tone** of the reaction when challenged (turns bureaucratic / more forceful / over-explains / softens).
    #: Note: this is style only. Whether verification is **actually** accepted is decided by the scenario's category and
    #: disambiguator. The two are orthogonal, so any persona can pair with any scenario.
    challenge_response: str
    competence: str = "practised"
    #: Style-level signals this persona inherently shows (see STYLE_ONLY_SIGNALS).
    #: Scenarios can't declare these signals, so the persona bank is responsible for their coverage.
    style_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReceiverPersona:
    persona_id: str
    skepticism: str                    # very_low | low | medium | high
    tech_familiarity: str
    emotional_state: str
    situation: str
    speech_style: str
    prior_exposure: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _persona_from_dict(cls: Any, data: dict[str, Any]) -> Any:
    """Keep only fields the dataclass knows; older batches' metadata may lack newly added fields."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class GenerationTask:
    """All deterministic inputs to one generation. (bank_version, seed, index) determines it uniquely (P5)."""

    task_id: str
    index: int
    seed: int
    derived_seed: int
    bank_version: int
    scenario: Scenario
    caller_persona: CallerPersona
    receiver_persona: ReceiverPersona
    outcome: str
    target_rounds: int
    #: Whether this run is played slow-burn. Only set true for scenarios with scenario.late_onset true.
    late_onset: bool = False
    turn_directives: list[str | None] = field(default_factory=list)

    @property
    def fingerprint(self) -> tuple[str, str, str, str]:
        """Diversity level-1 fingerprint (R6.1①)."""
        return (
            self.scenario.family,
            self.caller_persona.persona_id,
            self.receiver_persona.persona_id,
            self.outcome,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationTask":
        """Rebuild from metadata. ``relabel`` uses this to reuse the validator without rerunning the dialogue."""
        return cls(
            task_id=data["task_id"],
            index=data["index"],
            seed=data["seed"],
            derived_seed=data["derived_seed"],
            bank_version=data["bank_version"],
            scenario=Scenario.from_dict(data["scenario"]),
            caller_persona=_persona_from_dict(CallerPersona, data["caller_persona"]),
            receiver_persona=_persona_from_dict(ReceiverPersona, data["receiver_persona"]),
            outcome=data["outcome"],
            target_rounds=data["target_rounds"],
            late_onset=data.get("late_onset", False),
            turn_directives=list(data.get("turn_directives") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "index": self.index,
            "seed": self.seed,
            "derived_seed": self.derived_seed,
            "bank_version": self.bank_version,
            "scenario": self.scenario.to_dict(),
            "caller_persona": self.caller_persona.to_dict(),
            "receiver_persona": self.receiver_persona.to_dict(),
            "outcome": self.outcome,
            "target_rounds": self.target_rounds,
            "late_onset": self.late_onset,
            "turn_directives": self.turn_directives,
        }


# --------------------------------------------------------------------------- #
# Dialogue and labels
# --------------------------------------------------------------------------- #

@dataclass
class Utterance:
    index: int
    speaker: str
    text: str

    def to_transcript(self) -> dict[str, Any]:
        return {"index": self.index, "speaker": self.speaker, "text": self.text}


@dataclass
class PrefixLabel:
    """Verdict on turns 1..through_index; sees nothing after it (P3)."""

    through_index: int
    risk_score: float
    risk_tier: str
    signals: list[str] = field(default_factory=list)
    new_signals: list[str] = field(default_factory=list)
    sufficient_evidence: bool = False
    #: **Monotonic** version: always True from ``earliest_detection_index`` on.
    #:
    #: The difference from ``sufficient_evidence`` is not wording: that one is the verdict on this prefix **in place** and may
    #: fall back (the labeller grows suspicious and is then reassured; seen in 8.3% of conversations), whereas a real
    #: detector can't take back an alarm it has raised. So **this is the training target**.
    #:
    #: The cost needs stating: it is **derived with hindsight**. The persistence rule for ``earliest_detection_index``
    #: looks at the next prefix, so its value at k == earliest uses information from k+1. That is allowed for a
    #: label (only inputs may not see the future), but it is no longer "a quantity derivable from the first k turns".
    #: Don't put it in the same class as ``risk_score`` / ``sufficient_evidence``.
    evidence_established: bool = False
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UtteranceLabel:
    index: int
    speaker: str
    risk_score: float
    risk_tier: str
    new_signals: list[str] = field(default_factory=list)
    cumulative_signals: list[str] = field(default_factory=list)
    is_detection_point: bool = False
    #: Monotonic "evidence is sufficient": see ``PrefixLabel.evidence_established``.
    #: ``is_detection_point`` is true only on **that one** utterance; this stays true from there on.
    evidence_established: bool = False
    #: Which prefix the label actually comes from, **always <= index**. With eval_stride > 1 a missing one carries over the
    #: last evaluated prefix, never a later one: borrowing a later one writes future information into the label at turn i.
    label_from_index: int | None = None
    repetition_of: int | None = None
    repetition_similarity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlindAssessment:
    """The evaluator's full quality score. Never used as an early-warning label (R5.13)."""

    coherence: float = 0.0
    naturalness: float = 0.0
    repetition: float = 0.0
    role_consistency: bool = True
    blind_is_suspicious: bool | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiversityScore:
    """Computed by the program, not asked of the LLM (P6)."""

    fingerprint_seen: int = 0          # times the same fingerprint appeared in the admitted pool
    max_text_similarity: float = 0.0
    nearest_conversation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Conversation:
    conversation_id: str
    utterances: list[Utterance]
    task: GenerationTask
    generation: dict[str, Any]
    prefix_labels: list[PrefixLabel] = field(default_factory=list)
    utterance_labels: list[UtteranceLabel] = field(default_factory=list)
    blind_assessment: BlindAssessment | None = None
    diversity: DiversityScore | None = None
    earliest_detection_index: int | None = None
    validation: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rounds(self) -> int:
        return sum(1 for u in self.utterances if u.speaker == "receiver")

    @property
    def earliest_detection_round(self) -> int | None:
        if self.earliest_detection_index is None:
            return None
        return self.earliest_detection_index // 2 + 1

    def transcript_record(self) -> dict[str, Any]:
        """The only view the model under test and downstream training may see (P2 / R7.2)."""
        return {
            "conversation_id": self.conversation_id,
            "language": "en",
            "channel": "phone",
            "num_utterances": len(self.utterances),
            "utterances": [u.to_transcript() for u in self.utterances],
        }

    def metadata_record(self) -> dict[str, Any]:
        scn = self.task.scenario
        return {
            "conversation_id": self.conversation_id,
            "task": self.task.to_dict(),
            "generation": self.generation,
            "ground_truth": {
                "is_scam": scn.category == "scam",
                "scam_family": scn.family,
                "hard_negative": scn.hard_negative,
                "late_onset": self.task.late_onset,
                "expected_signs": list(scn.expected_signs),
                # This run **intends** the receiver to be won over. bench can slice on it to answer "does the
                # model's alarm rate drop when the receiver is credulous". What actually happened is in the transcript.
                "compliance_intended": self.task.outcome in scn.compliant_outcomes,
                "no_observable_evidence": scn.no_observable_evidence,
                "eval_only": scn.eval_only,
            },
            "earliest_detection_index": self.earliest_detection_index,
            "earliest_detection_round": self.earliest_detection_round,
            "prefix_labels": [p.to_dict() for p in self.prefix_labels],
            "utterance_labels": [u.to_dict() for u in self.utterance_labels],
            "blind_assessment": (
                self.blind_assessment.to_dict() if self.blind_assessment else None
            ),
            "diversity": self.diversity.to_dict() if self.diversity else None,
            "validation": self.validation,
        }


# --------------------------------------------------------------------------- #
# Gatekeeper verdict
# --------------------------------------------------------------------------- #

@dataclass
class GatekeeperScores:
    """Four-axis score. Each axis is 1-5 and the rubric must have anchors, otherwise scores aren't comparable across models or batches (component 6)."""

    realism: int = 0
    consistency: int = 0
    detection_value: int = 0
    #: Specific contradictions, e.g. "turn 4 and turn 9 disagree on which department it is". A score without listed points fails.
    inconsistencies: list[str] = field(default_factory=list)
    realism_notes: str = ""
    detection_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdmissionDecision:
    conversation_id: str
    decision: str                      # admit | reject | review
    scores: GatekeeperScores
    diversity: DiversityScore
    reasons: list[str] = field(default_factory=list)
    judge_model: str = ""
    rubric_version: str = ""
    decided_by: str = "gatekeeper"     # gatekeeper | human
    decided_at: float = field(default_factory=time.time)
    #: When human review overrides the gatekeeper, the original verdict stays here for comparison (R6.10).
    supersedes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["scores"] = self.scores.to_dict()
        out["diversity"] = self.diversity.to_dict()
        return out


# --------------------------------------------------------------------------- #
# Leakage protection
# --------------------------------------------------------------------------- #

#: Any of these keys at any depth in a transcript record counts as leakage (P2).
LABEL_KEYS = frozenset({
    "task", "scenario", "generation", "ground_truth", "caller_persona",
    "receiver_persona", "expected_signs", "disambiguator", "is_scam",
    "scam_family", "hard_negative", "late_onset", "prefix_labels",
    "utterance_labels", "blind_assessment", "blind_is_suspicious", "diversity",
    "risk_score", "risk_tier", "signals", "new_signals", "cumulative_signals",
    "sufficient_evidence", "evidence_established",
    "earliest_detection_index", "earliest_detection_round",
    "is_detection_point", "label_from_index", "decision", "scores", "realism",
    "consistency", "detection_value", "turn_directives", "outcome",
})


def clean_signals(values: Any) -> list[str]:
    """Normalise model-supplied signals onto the closed vocabulary; drop anything outside it (R1.3 / R5.7)."""
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        key = v.strip().lower().replace(" ", "_").replace("-", "_")
        if key in RISK_SIGNALS and key not in out:
            out.append(key)
    return out


def coerce_tier(tier: Any, score: float) -> str:
    if isinstance(tier, str) and tier.lower() in RISK_TIERS:
        return tier.lower()
    if score < 0.1:
        return "none"
    if score < 0.35:
        return "low"
    if score < 0.6:
        return "medium"
    return "high"
