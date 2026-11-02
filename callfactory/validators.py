"""Programmatic hard rules. Run **before** the gatekeeper, so failures don't waste LLM calls (R6.8).

Six groups, one per way this data goes bad: schema, role alternation, length, repetition, anonymisation, label structure.
Two of them catch things nothing else does:

* **Label leakage**: a label key at any depth in ``conversations.jsonl`` is an ERROR (P2 / C1).
  This is the one most likely to break silently: adding a field is easy, and the downstream model will happily use it.
* **Label causality**: ``label_from_index <= index``. If that fails, the label at turn i carries future information
  and the detection latency measured on the batch is fake.

ERROR blocks data, WARNING only records it. Making "this turn is too short" a hard error would throw away lots of usable data,
but not recording it leaves no way to review batch quality later, so both are needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .schema import (
    LABEL_KEYS,
    PLACEHOLDER_TOKENS,
    RISK_TIERS,
    SPEAKERS,
    Conversation,
    Utterance,
)

ERROR = "error"
WARNING = "warning"

_WORD_RE = re.compile(r"[a-z0-9']+")
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
#: Digit runs that should have been covered by a placeholder (years and small counts don't count).
_LONG_DIGITS_RE = re.compile(r"(?<!\[)\b\d[\d\s\-]{5,}\b")


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "index": self.index,
        }


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == ERROR for i in self.issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    def add(self, severity: str, code: str, message: str, index: int | None = None) -> None:
        self.issues.append(Issue(severity, code, message, index))

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)

    def to_list(self) -> list[dict[str, Any]]:
        return [i.to_dict() for i in self.issues]


# --------------------------------------------------------------------------- #
# Repetition
# --------------------------------------------------------------------------- #

def shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i: i + n]) for i in range(len(words) - n + 1)}


def similarity(a: str, b: str, n: int = 3) -> float:
    """Word-trigram Jaccard: a lexical proxy for semantic repetition.

    No embedding model: the failure that actually happens here is one side restating the previous turn in near-identical wording,
    which lexical similarity catches, and it is cheap, deterministic and explainable (Diversity maintains its index incrementally with it).
    """
    sa, sb = shingles(a, n), shingles(b, n)
    if not sa or not sb:
        return 0.0
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def find_repetitions(
    utterances: Iterable[Utterance], threshold: float = 0.5
) -> dict[int, tuple[int, float]]:
    """index -> (index of the utterance it repeats, similarity). Same speaker only."""
    found: dict[int, tuple[int, float]] = {}
    items = list(utterances)
    for i, later in enumerate(items):
        best: tuple[int, float] | None = None
        for earlier in items[:i]:
            if earlier.speaker != later.speaker:
                continue
            score = similarity(earlier.text, later.text)
            if score >= threshold and (best is None or score > best[1]):
                best = (earlier.index, score)
        if best:
            found[later.index] = best
    return found


def annotate_repetition(conv: Conversation, threshold: float = 0.5) -> None:
    """Write repetition findings back into the per-utterance labels (in place)."""
    reps = find_repetitions(conv.utterances, threshold)
    for label in conv.utterance_labels:
        hit = reps.get(label.index)
        if hit:
            label.repetition_of, label.repetition_similarity = hit[0], round(hit[1], 3)


def placeholder_count(text: str) -> int:
    return len(_PLACEHOLDER_RE.findall(text))


# --------------------------------------------------------------------------- #
# transcript view (C1)
# --------------------------------------------------------------------------- #

def leaked_keys(node: Any) -> set[str]:
    """Search for label keys at any depth. Recurse into both dict values and list items."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in LABEL_KEYS:
                found.add(key)
            found |= leaked_keys(value)
    elif isinstance(node, list):
        for item in node:
            found |= leaked_keys(item)
    return found


def validate_transcript_record(record: dict[str, Any]) -> ValidationReport:
    report = ValidationReport()
    for key in ("conversation_id", "utterances"):
        if key not in record:
            report.add(ERROR, "schema.missing_field", f"missing field {key!r}")
    utterances = record.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        report.add(ERROR, "schema.no_utterances", "utterances must be a non-empty list")
        return report
    for position, utt in enumerate(utterances):
        if not isinstance(utt, dict):
            report.add(ERROR, "schema.bad_utterance", "utterance is not an object", position)
            continue
        if set(utt) - {"index", "speaker", "text"}:
            report.add(
                ERROR, "schema.extra_utterance_field",
                f"utterance has extra fields {sorted(set(utt) - {'index', 'speaker', 'text'})}",
                position,
            )
        if utt.get("speaker") not in SPEAKERS:
            report.add(ERROR, "schema.bad_speaker", f"speaker {utt.get('speaker')!r}", position)
        if not isinstance(utt.get("text"), str) or not utt.get("text", "").strip():
            report.add(ERROR, "schema.empty_text", "text missing or empty", position)
        if utt.get("index") != position:
            report.add(
                ERROR, "schema.bad_index",
                f"index {utt.get('index')!r} != {position}", position,
            )
    leaked = sorted(leaked_keys(record))
    if leaked:
        report.add(ERROR, "leakage.label_in_transcript", f"transcript contains label keys: {leaked}")
    return report


def validate_record_pair(
    transcript: dict[str, Any], metadata: dict[str, Any]
) -> ValidationReport:
    """Join the two files on conversation_id (C2) and recheck the transcript side for leakage."""
    report = validate_transcript_record(transcript)
    tid, mid = transcript.get("conversation_id"), metadata.get("conversation_id")
    if tid != mid:
        report.add(ERROR, "pair.id_mismatch", f"conversation_id mismatch: {tid!r} vs {mid!r}")
    n_utt = len(transcript.get("utterances") or [])
    labels = metadata.get("utterance_labels") or []
    if labels and len(labels) != n_utt:
        report.add(
            WARNING, "pair.label_count",
            f"{len(labels)} utterance labels vs {n_utt} utterances",
        )
    return report


# --------------------------------------------------------------------------- #
# Whole conversation
# --------------------------------------------------------------------------- #

def validate_conversation(
    conv: Conversation,
    *,
    min_rounds: int = 5,
    max_rounds: int = 20,
    min_words_per_utterance: int = 2,
    repetition_threshold: float = 0.5,
    max_repetition_rate: float = 0.2,
) -> ValidationReport:
    report = validate_transcript_record(conv.transcript_record())
    utterances = conv.utterances
    if not utterances:
        return report

    # --- Role alternation: caller first, strictly alternating (R4.7) ---
    for utt in utterances:
        expected = "caller" if utt.index % 2 == 0 else "receiver"
        if utt.speaker != expected:
            report.add(
                ERROR, "role.alternation",
                f"index {utt.index} should be {expected}, got {utt.speaker}", utt.index,
            )

    # --- Length ---
    if conv.rounds < min_rounds:
        report.add(ERROR, "length.too_short", f"{conv.rounds} rounds < min_rounds={min_rounds}")
    if conv.rounds > max_rounds:
        report.add(WARNING, "length.too_long", f"{conv.rounds} rounds > max_rounds={max_rounds}")
    for utt in utterances:
        if len(_WORD_RE.findall(utt.text.lower())) < min_words_per_utterance:
            report.add(WARNING, "length.thin_utterance", f"{utt.text!r}", utt.index)

    # --- Repetition ---
    reps = find_repetitions(utterances, repetition_threshold)
    for index, (source, score) in reps.items():
        report.add(
            WARNING, "repetition.near_duplicate",
            f"index {index} repeats index {source} (jaccard={score:.2f})", index,
        )
    if len(reps) / len(utterances) > max_repetition_rate:
        report.add(
            ERROR, "repetition.excessive",
            f"{len(reps)}/{len(utterances)} utterances are near-duplicates",
        )

    # --- Anonymisation ---
    for utt in utterances:
        if _LONG_DIGITS_RE.search(utt.text):
            report.add(WARNING, "privacy.raw_digits", "long digit string should be replaced with a placeholder", utt.index)
        unknown = [
            t for t in _PLACEHOLDER_RE.findall(utt.text)
            if t not in PLACEHOLDER_TOKENS and t != "[END_CALL]"
        ]
        if unknown:
            report.add(
                WARNING, "privacy.unknown_placeholder",
                f"placeholder not in the agreed vocabulary: {unknown}", utt.index,
            )

    report.extend(validate_labels(conv))
    return report


def validate_labels(conv: Conversation) -> ValidationReport:
    """Label structure. Causality is the crux of this project, so it is an ERROR, not a WARNING."""
    report = ValidationReport()
    utterances = conv.utterances
    if not utterances:
        return report
    last_index = utterances[-1].index

    if conv.prefix_labels:
        indices = [p.through_index for p in conv.prefix_labels]
        if indices != sorted(indices):
            report.add(ERROR, "labels.unordered", "prefix labels not sorted by turn")
        if len(set(indices)) != len(indices):
            report.add(ERROR, "labels.duplicate_prefix", "same prefix labelled more than once")
        if any(p.risk_tier not in RISK_TIERS for p in conv.prefix_labels):
            report.add(ERROR, "labels.bad_tier", "risk_tier outside the vocabulary")
        if any(not 0.0 <= p.risk_score <= 1.0 for p in conv.prefix_labels):
            report.add(ERROR, "labels.score_out_of_range", "risk_score not in [0,1]")
        if max(indices) != last_index:
            report.add(
                WARNING, "labels.incomplete",
                f"last prefix label at index {max(indices)}, call ends at {last_index}",
            )
        # signals are monotonic (R5.8)
        seen: set[str] = set()
        for label in conv.prefix_labels:
            current = set(label.signals)
            lost = seen - current
            if lost:
                report.add(
                    ERROR, "labels.signal_regression",
                f"index {label.through_index} dropped already-evidenced signals {sorted(lost)}",
                    label.through_index,
                )
            seen |= current

    for label in conv.utterance_labels:
        if label.label_from_index is None:
            continue
        if label.label_from_index > label.index:
            report.add(
                ERROR, "labels.future_leak",
                f"label at index {label.index} comes from index {label.label_from_index}: future information leak",
                label.index,
            )

    if conv.earliest_detection_index is not None:
        if not 0 <= conv.earliest_detection_index <= last_index:
            report.add(
                ERROR, "labels.detection_out_of_range",
                f"earliest_detection_index={conv.earliest_detection_index}",
            )
        marked = [u.index for u in conv.utterance_labels if u.is_detection_point]
        if marked and marked != [conv.earliest_detection_index]:
            report.add(
                ERROR, "labels.detection_mismatch",
                f"detection point marked at {marked}, earliest_detection_index="
                f"{conv.earliest_detection_index}",
            )
    return report
