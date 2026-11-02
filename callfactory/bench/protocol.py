"""Interface for the system under test (REQUIREMENTS §6).

**Streaming is the reference**, because early warning is inherently a streaming task: a real detector is stateful, and on hearing utterance k
knows only the first k. So the interface is ``push`` one utterance and get the verdict at that moment, not "score me the whole call".

Stateless models (prompt-driven LLMs) are wrapped as streaming by ``ReplayAdapter``: each ``push`` replays
the full prefix. The cost is stated openly: token cost is **O(N²)**, and the report must say so (R7.1).

Jev goes through the same adapter: its SDK only has ``system_one(state, questions)``, with **no
session and no streaming interface**, so "one score per round" is likewise implemented as one call per round carrying the accumulated state.
The earlier claim in REQUIREMENTS §6 that "Jev is natively streaming with no O(N²) cost" has been corrected accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

from ..schema import RISK_SIGNALS, Utterance


@dataclass
class Verdict:
    """The verdict at this moment. ``risk_score`` is continuous, so prefix-level AUC/PR is available (R7.7)."""

    risk_score: float
    is_alarm: bool
    signals: list[str] = field(default_factory=list)
    rationale: str = ""
    latency_ms: float | None = None
    #: Whether this push actually sent a request or reused the previous verdict (granularity alignment for R7.6)
    fresh: bool = True
    usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.risk_score = max(0.0, min(1.0, float(self.risk_score)))
        # Signals outside the closed set are dropped, no guesswork repairs; the same rule as on the labelling side
        self.signals = [s for s in self.signals if s in RISK_SIGNALS]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 4),
            "is_alarm": self.is_alarm,
            "signals": list(self.signals),
            "rationale": self.rationale,
            "latency_ms": self.latency_ms,
            "fresh": self.fresh,
        }


@runtime_checkable
class DetectorSession(Protocol):
    async def push(self, utterance: Utterance) -> Verdict: ...
    async def close(self) -> None: ...


@runtime_checkable
class EarlyWarningDetector(Protocol):
    name: str

    def open_call(self, call_id: str) -> DetectorSession: ...


class StatelessJudge(Protocol):
    """A single-shot model: given a prefix, return a verdict. The adapter turns it into streaming."""

    name: str

    async def judge(
        self, call_id: str, prefix: Sequence[Utterance]
    ) -> Verdict: ...

    async def aclose(self) -> None: ...


class ReplaySession:
    """Wrap a stateless judge as a streaming session (R7.1).

    ``ask_at`` gives the utterance indices at which a request is really sent; elsewhere it **returns the same verdict
    as last time** (R7.6: granularity alignment; a model that updates once per round repeats its last result on the push in between).
    By default it asks on every utterance; the cheap way is to ask only on the labelled prefixes, which the runner passes in.
    """

    def __init__(self, judge: StatelessJudge, call_id: str, ask_at: set[int] | None = None):
        self.judge = judge
        self.call_id = call_id
        self.ask_at = ask_at
        self.prefix: list[Utterance] = []
        self.last = Verdict(risk_score=0.0, is_alarm=False, fresh=False)
        self.calls = 0

    async def push(self, utterance: Utterance) -> Verdict:
        self.prefix.append(utterance)
        if self.ask_at is not None and utterance.index not in self.ask_at:
            # Reuse the previous one: copy it and mark fresh=False so callers don't mistake it for a new judgement
            carried = Verdict(
                risk_score=self.last.risk_score,
                is_alarm=self.last.is_alarm,
                signals=list(self.last.signals),
                rationale=self.last.rationale,
                fresh=False,
            )
            return carried
        self.last = await self.judge.judge(self.call_id, self.prefix)
        self.calls += 1
        return self.last

    async def close(self) -> None:
        return None


class ReplayAdapter:
    """``StatelessJudge`` → ``EarlyWarningDetector``."""

    def __init__(self, judge: StatelessJudge, ask_at: set[int] | None = None):
        self.judge = judge
        self.name = judge.name
        self.ask_at = ask_at

    def open_call(self, call_id: str) -> ReplaySession:
        return ReplaySession(self.judge, call_id, self.ask_at)

    async def aclose(self) -> None:
        await self.judge.aclose()


async def replay_call(
    detector: EarlyWarningDetector,
    call_id: str,
    utterances: Sequence[Utterance],
    threshold: float,
) -> dict[str, Any]:
    """Run one call to the end and return per-utterance verdicts.

    ``alarm_index`` is computed from the **score** (``risk_score >= threshold``) and counts only once:
    once raised it stays raised. A real detector can't withdraw an alarm that has gone off; this is the same reasoning as
    the monotonicity of ``evidence_established`` on the label side.

    **The decision looks only at the score, never at the model's own ``alarm`` boolean.** We were bitten by this: it used to read
    ``is_alarm or score >= threshold``, and the model's boolean short-circuited the threshold entirely:
    raising the threshold from 0.5 to 0.95 barely moved the false-alarm rate (0.92 -> 0.82), because whatever it said "alarm" on
    bypassed the threshold. It is the same class of bug as the label-side "0.5 threshold short-circuited into a no-op by
    sufficient_evidence" in §5.9.

    The model's own boolean is still stored per utterance (``is_alarm``); it is the vendor's chosen operating point and worth
    its own row in the report, but it must not be mixed into the decision at our operating point.
    """
    session = detector.open_call(call_id)
    steps: list[dict[str, Any]] = []
    alarm_at: int | None = None
    try:
        for utt in utterances:
            verdict = await session.push(utt)
            fired = verdict.risk_score >= threshold
            if fired and alarm_at is None:
                alarm_at = utt.index
            steps.append({"index": utt.index, **verdict.to_dict(), "fired": fired})
    finally:
        await session.close()
    return {
        "conversation_id": call_id,
        "alarm_index": alarm_at,
        "alarm_round": None if alarm_at is None else alarm_at // 2 + 1,
        "steps": steps,
    }
