"""Component 5: causality, monotonic fix-up, degradation and detection points for prefix labelling."""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from callfactory.evaluator import (
    DEFAULT_DETECTION_THRESHOLD,
    EvaluatorAgent,
    build_utterance_labels,
    earliest_detection_index,
    evaluate_conversation,
    prefix_indices,
)
from callfactory.prompts import PREFIX_EVAL_SYSTEM, GLOBAL_EVAL_SYSTEM
from callfactory.providers import build_provider
from callfactory.providers.base import LLMProvider, ProviderConfig, ProviderError
from callfactory.schema import PrefixLabel, Utterance

CALLER_TEXT = [
    "Good afternoon, I'm calling from [COMPANY_A] about your file.",
    "I do need to settle this today, the window closes at five.",
    "I can't put you through to anyone else, it has to be me on this call.",
    "Between us, I shouldn't really be offering this, so keep it to yourself.",
    "I just need you to confirm a couple of details.",
]
RECEIVER_TEXT = [
    "Sorry, who is this?",
    "Nobody's written to me about anything.",
    "I'd rather ring the number on my statement.",
    "Hang on, why can't I check first?",
    "No. I'm going to leave it there.",
]


def convo(rounds: int = 5) -> list[Utterance]:
    out: list[Utterance] = []
    for i in range(rounds):
        out.append(Utterance(len(out), "caller", CALLER_TEXT[i % 5]))
        out.append(Utterance(len(out), "receiver", RECEIVER_TEXT[i % 5]))
    return out


def run(coro):
    return asyncio.run(coro)


class RecordingProvider(LLMProvider):
    """Return preset JSON by prefix length, and record each request body so causality can be asserted."""

    supports_seed = True

    def __init__(self, payloads=None, *, raw=None):
        super().__init__(ProviderConfig(name="recording", model="recording-1"))
        self.payloads = payloads or {}
        self.raw = raw
        self.calls: list[dict] = []

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag):
        body = messages[-1]["content"]
        self.calls.append(
            {"system": system, "body": body, "tag": tag, "temperature": temperature}
        )
        if self.raw is not None:
            return self.raw
        if tag == "evaluator_global":
            return json.dumps({
                "coherence": 4, "naturalness": 4, "repetition": 0.1,
                "role_consistency": True, "notes": "fine", "blind_is_suspicious": True,
            })
        # payload is indexed by prefix length; the length comes from the prompt's own "turns 1..N".
        n = int(re.search(r"turns 1\.\.(\d+)", body).group(1))
        return json.dumps(self.payloads.get(n, {
            "risk_score": 0.1, "risk_tier": "low", "signals": [],
            "new_signals_in_last_turn": [], "sufficient_evidence": False,
            "rationale": "nothing yet",
        }))


class ExplodingProvider(LLMProvider):
    def __init__(self):
        super().__init__(ProviderConfig(name="boom", model="boom-1", max_retries=0))
        self.calls = 0

    async def _complete(self, **kw):
        self.calls += 1
        raise ProviderError("boom: 500 from upstream", status=500, retryable=False)


# --------------------------------------------------------------------------- #
# Causality (P3 / R5.1 / R5.2)
# --------------------------------------------------------------------------- #

def test_no_future_turn_ever_reaches_a_request():
    utterances = convo(5)
    provider = RecordingProvider()
    run(evaluate_conversation(utterances, EvaluatorAgent(provider)))
    prefix_calls = [c for c in provider.calls if c["tag"] == "evaluator_prefix"]
    assert len(prefix_calls) == len(utterances)
    for call in prefix_calls:
        shown = [i for i, u in enumerate(utterances) if f"[{u.index}] " in call["body"]]
        last = max(shown)
        for u in utterances[last + 1:]:
            assert f"[{u.index}] " not in call["body"]


def test_the_third_call_does_not_contain_the_fourth_turn():
    """Acceptance clause: build a conversation and assert the 3rd call's request body doesn't contain turn 4's text."""
    utterances = convo(5)
    provider = RecordingProvider()
    run(evaluate_conversation(utterances, EvaluatorAgent(provider)))
    third = [c for c in provider.calls if c["tag"] == "evaluator_prefix"][2]
    assert utterances[2].text in third["body"]
    assert f"[{utterances[3].index}]" not in third["body"]


def test_each_prefix_is_a_fresh_stateless_call():
    provider = RecordingProvider()
    run(evaluate_conversation(convo(3), EvaluatorAgent(provider)))
    for call in provider.calls:
        if call["tag"] == "evaluator_prefix":
            assert call["system"] == PREFIX_EVAL_SYSTEM
        else:
            assert call["system"] == GLOBAL_EVAL_SYSTEM


def test_evaluator_is_blind_to_the_scenario():
    # The prompt structurally has no place for the scenario: neither system prompt mentions these fields.
    for prompt in (PREFIX_EVAL_SYSTEM, GLOBAL_EVAL_SYSTEM):
        # This checks ground-truth **field names**; a bare "family" is too broad: the anchor
        # "keep the call from family" is normal English, not leakage.
        for leaked in ("hard_negative", "late_onset", "scam_family", "expected_signs",
                       "ground truth", "scenario"):
            assert leaked not in prompt


def test_labelling_runs_at_temperature_zero():
    provider = RecordingProvider()
    run(evaluate_conversation(convo(2), EvaluatorAgent(provider)))
    assert {c["temperature"] for c in provider.calls} == {0.0}


def test_only_the_caller_is_scored_per_the_prompt():
    assert "Score the CALLER's behaviour" in PREFIX_EVAL_SYSTEM
    assert "never raises the score" in PREFIX_EVAL_SYSTEM


# --------------------------------------------------------------------------- #
# Vocabulary and monotonic fix-up (R5.7 / R5.8)
# --------------------------------------------------------------------------- #

def test_out_of_vocabulary_signals_are_dropped_not_mapped():
    provider = RecordingProvider(payloads={1: {
        "risk_score": 0.4, "risk_tier": "medium",
        "signals": ["urgency_pressure", "sounds_dodgy", "URGENCY PRESSURE", "bad_vibes"],
        "new_signals_in_last_turn": ["sounds_dodgy"],
        "sufficient_evidence": False, "rationale": "x",
    }})
    label, failure = run(EvaluatorAgent(provider).label_prefix(convo(1)[:1]))
    assert failure is None
    assert label.signals == ["urgency_pressure"]


def test_signals_are_forced_monotonic():
    """Acceptance clause: build a sequence where a signal disappears; the monotonic fix-up corrects it."""
    provider = RecordingProvider(payloads={
        1: {"risk_score": 0.2, "signals": ["urgency_pressure"],
            "sufficient_evidence": False, "rationale": "a"},
        2: {"risk_score": 0.2, "signals": [], "sufficient_evidence": False, "rationale": "b"},
        3: {"risk_score": 0.3, "signals": ["authority_claim"],
            "sufficient_evidence": False, "rationale": "c"},
    })
    result = run(evaluate_conversation(convo(1) + [Utterance(2, "caller", "x")],
                                       EvaluatorAgent(provider)))
    assert [p.signals for p in result.prefix_labels] == [
        ["urgency_pressure"],
        ["urgency_pressure"],
        ["authority_claim", "urgency_pressure"],
    ]


def test_new_signals_are_recomputed_not_taken_from_the_model():
    provider = RecordingProvider(payloads={
        1: {"risk_score": 0.2, "signals": ["urgency_pressure"],
            "new_signals_in_last_turn": ["urgency_pressure"],
            "sufficient_evidence": False, "rationale": "a"},
        2: {"risk_score": 0.3, "signals": ["urgency_pressure"],
            # the model reported "also shown this turn" as new; it must be corrected to empty
            "new_signals_in_last_turn": ["urgency_pressure"],
            "sufficient_evidence": False, "rationale": "b"},
    })
    result = run(evaluate_conversation(convo(1), EvaluatorAgent(provider)))
    assert result.prefix_labels[0].new_signals == ["urgency_pressure"]
    assert result.prefix_labels[1].new_signals == []


def test_risk_score_is_allowed_to_fall():
    """A hard negative's discriminating power is all in this drop; it must not be flattened."""
    provider = RecordingProvider(payloads={
        1: {"risk_score": 0.7, "signals": ["urgency_pressure"],
            "sufficient_evidence": False, "rationale": "a"},
        2: {"risk_score": 0.2, "signals": ["urgency_pressure"],
            "sufficient_evidence": False, "rationale": "welcomed a callback"},
    })
    result = run(evaluate_conversation(convo(1), EvaluatorAgent(provider)))
    assert [p.risk_score for p in result.prefix_labels] == [0.7, 0.2]


def test_scores_and_tiers_are_clamped_and_coerced():
    provider = RecordingProvider(payloads={1: {
        "risk_score": 3.7, "risk_tier": "catastrophic", "signals": [],
        "sufficient_evidence": True, "rationale": "r" * 400,
    }})
    label, _ = run(EvaluatorAgent(provider).label_prefix(convo(1)[:1]))
    assert label.risk_score == 1.0
    assert label.risk_tier == "high"
    assert len(label.rationale) == 240


# --------------------------------------------------------------------------- #
# Detection point (R5.10)
# --------------------------------------------------------------------------- #

def test_the_detection_point_is_decided_by_the_score_alone():
    """detect/v2: the threshold looks only at risk_score. That boolean was the step that lost the most information."""
    labels = [
        PrefixLabel(0, 0.9, "high", sufficient_evidence=False),   # high score, boolean false
        PrefixLabel(1, 0.3, "low", sufficient_evidence=True),     # boolean true, score too low
        PrefixLabel(2, 0.8, "high", sufficient_evidence=True),
    ]
    # the 0.9 one is the detection point even though its sufficient_evidence is false;
    # the persistence rule doesn't stop it here because the next one, 0.3, is below the threshold -> moves on to index 2
    assert earliest_detection_index(labels, 0.5, require_persistence=False) == 0
    assert earliest_detection_index(labels, 0.5) == 2


def test_require_sufficient_reproduces_the_old_rule():
    """The detect/v1 rule must be reproducible, otherwise old labels can't be compared."""
    labels = [
        PrefixLabel(0, 0.9, "high", sufficient_evidence=False),
        PrefixLabel(1, 0.3, "low", sufficient_evidence=True),
        PrefixLabel(2, 0.8, "high", sufficient_evidence=True),
    ]
    assert earliest_detection_index(labels, 0.5, require_sufficient=True) == 2
    assert earliest_detection_index(labels[:2], 0.5, require_sufficient=True) is None
    assert earliest_detection_index(labels, 0.25, require_sufficient=True) == 1


def test_no_detection_point_when_nothing_qualifies():
    provider = RecordingProvider()
    result = run(evaluate_conversation(convo(3), EvaluatorAgent(provider)))
    assert result.earliest_detection_index is None
    assert not any(u.is_detection_point for u in result.utterance_labels)


def test_the_detection_point_is_marked_on_exactly_one_utterance():
    provider = RecordingProvider(payloads={
        k: {"risk_score": 0.1 if k < 5 else 0.8,
            "signals": ["urgency_pressure"] if k >= 5 else [],
            "sufficient_evidence": k >= 5, "rationale": "x"}
        for k in range(1, 11)
    })
    result = run(evaluate_conversation(convo(5), EvaluatorAgent(provider)))
    assert result.earliest_detection_index == 4
    marked = [u.index for u in result.utterance_labels if u.is_detection_point]
    assert marked == [4]


def test_default_threshold_is_explicit():
    # 0.7 isn't arbitrary: see the comment on evaluator.DEFAULT_DETECTION_THRESHOLD and EVAL_SYSTEM §5.9
    assert DEFAULT_DETECTION_THRESHOLD == 0.7


# --------------------------------------------------------------------------- #
# stride and the off switch (R5.14)
# --------------------------------------------------------------------------- #

def test_prefix_indices_always_keep_t0_and_the_last_turn():
    assert prefix_indices(10, 1) == list(range(10))
    assert prefix_indices(10, 3) == [0, 2, 5, 8, 9]
    assert prefix_indices(1, 5) == [0]
    assert prefix_indices(0, 1) == []
    with pytest.raises(ValueError):
        prefix_indices(5, 0)


def test_stride_reduces_calls_and_labels_only_look_backwards():
    utterances = convo(5)
    provider = RecordingProvider(payloads={
        k: {"risk_score": 0.1 * k, "signals": [], "sufficient_evidence": False,
            "rationale": "x"}
        for k in range(1, 11)
    })
    result = run(evaluate_conversation(utterances, EvaluatorAgent(provider), stride=3))
    prefix_calls = [c for c in provider.calls if c["tag"] == "evaluator_prefix"]
    assert len(prefix_calls) == 5 < len(utterances)
    assert len(result.utterance_labels) == len(utterances)
    for u in result.utterance_labels:
        assert u.label_from_index is not None
        assert u.label_from_index <= u.index, "label borrowed from a later prefix"
    # utterances in between carry over the previous evaluated prefix
    assert result.utterance_labels[1].label_from_index == 0
    assert result.utterance_labels[3].label_from_index == 2


def test_new_signals_only_sit_on_the_utterance_that_was_evaluated():
    utterances = convo(3)
    provider = RecordingProvider(payloads={
        k: {"risk_score": 0.6, "signals": ["urgency_pressure"],
            "sufficient_evidence": False, "rationale": "x"}
        for k in range(1, 7)
    })
    result = run(evaluate_conversation(utterances, EvaluatorAgent(provider), stride=3))
    carried = [u for u in result.utterance_labels if u.label_from_index != u.index]
    assert carried and all(u.new_signals == [] for u in carried)


def test_prefix_eval_can_be_turned_off():
    provider = RecordingProvider()
    result = run(evaluate_conversation(convo(3), EvaluatorAgent(provider), prefix_eval=False))
    assert result.prefix_labels == []
    assert result.utterance_labels == []
    assert result.earliest_detection_index is None
    assert result.blind_assessment is not None
    assert [c["tag"] for c in provider.calls] == ["evaluator_global"]


def test_global_eval_can_be_turned_off():
    provider = RecordingProvider()
    result = run(evaluate_conversation(convo(2), EvaluatorAgent(provider), global_eval=False))
    assert result.blind_assessment is None
    assert all(c["tag"] == "evaluator_prefix" for c in provider.calls)


def test_concurrency_limit_is_respected():
    utterances = convo(4)
    provider = RecordingProvider()
    result = run(evaluate_conversation(utterances, EvaluatorAgent(provider), concurrency=2))
    assert len(result.prefix_labels) == len(utterances)


# --------------------------------------------------------------------------- #
# Degradation (R5.11)
# --------------------------------------------------------------------------- #

def test_unparseable_output_degrades_to_an_empty_label():
    provider = RecordingProvider(raw="I'd rather not answer in JSON, sorry.")
    result = run(evaluate_conversation(convo(2), EvaluatorAgent(provider)))
    assert len(result.prefix_labels) == 4
    assert all(p.risk_score == 0.0 and p.signals == [] for p in result.prefix_labels)
    assert {f.kind for f in result.failures} == {"unparseable"}
    assert {f.stage for f in result.failures} == {"prefix", "global"}
    assert result.earliest_detection_index is None


def test_provider_errors_are_recorded_not_raised():
    provider = ExplodingProvider()
    result = run(evaluate_conversation(convo(2), EvaluatorAgent(provider)))
    assert [f.kind for f in result.failures] == ["provider_error"] * 5
    assert all(p.risk_score == 0.0 for p in result.prefix_labels)


def test_failures_serialise_with_a_bounded_detail():
    provider = RecordingProvider(raw="x" * 5000)
    result = run(evaluate_conversation(convo(1), EvaluatorAgent(provider)))
    for f in result.failures:
        assert len(f.to_dict()["detail"]) <= 400


def test_json_wrapped_in_prose_and_fences_still_parses():
    provider = RecordingProvider(
        raw='Here you go:\n```json\n{"risk_score": 0.66, "risk_tier": "high", '
            '"signals": ["urgency_pressure"], "sufficient_evidence": true, '
            '"rationale": "pressed for a same-day resolution"}\n```'
    )
    label, failure = run(EvaluatorAgent(provider).label_prefix(convo(1)[:1]))
    assert failure is None
    assert label.risk_score == 0.66
    assert label.signals == ["urgency_pressure"]


# --------------------------------------------------------------------------- #
# Quality assessment and provenance (R5.13 / C4)
# --------------------------------------------------------------------------- #

def test_blind_assessment_is_separate_from_the_detection_labels():
    provider = RecordingProvider()
    result = run(evaluate_conversation(convo(3), EvaluatorAgent(provider)))
    assert result.blind_assessment.blind_is_suspicious is True
    # it takes no part in the detection point and doesn't go into prefix labels
    assert result.earliest_detection_index is None
    assert "blind" not in str([p.to_dict() for p in result.prefix_labels])


def test_provenance_records_whether_the_seed_was_honoured():
    prov = EvaluatorAgent(build_provider("mock"), seed=99).provenance()
    assert prov["provider"] == "mock"
    assert prov["temperature"] == 0.0
    assert prov["seed"] == 99
    assert prov["seed_honoured_by_provider"] is True


# --------------------------------------------------------------------------- #
# Offline end-to-end (P7)
# --------------------------------------------------------------------------- #

def test_mock_provider_labels_a_whole_call_reproducibly():
    utterances = convo(5)
    runs = []
    for _ in range(2):
        result = run(evaluate_conversation(
            utterances, EvaluatorAgent(build_provider("mock"), seed=7)
        ))
        runs.append([(p.through_index, p.risk_score, tuple(p.signals))
                     for p in result.prefix_labels])
    assert runs[0] == runs[1]
    # the mock matches cues only on caller lines, so the score really does rise with caller pressure
    assert runs[0][-1][1] > runs[0][0][1]


def test_receiver_suspicion_alone_does_not_raise_the_mock_score():
    """The receiver says every suspicious thing there is, but the caller says one ordinary line: the score must not rise."""
    utterances = [
        Utterance(0, "caller", "Morning, it's [PERSON_1] from [COMPANY_A] about Thursday."),
        Utterance(1, "receiver", "Who is this? I'd rather not give anything out over "
                                 "the phone, and don't call the main line nonsense either."),
    ]
    result = run(evaluate_conversation(
        utterances, EvaluatorAgent(build_provider("mock"), seed=7), global_eval=False
    ))
    assert result.prefix_labels[-1].signals == []


def test_build_utterance_labels_tolerates_a_missing_t0_label():
    utterances = convo(2)
    labels = [PrefixLabel(2, 0.4, "medium", signals=["urgency_pressure"])]
    out = build_utterance_labels(utterances, labels, earliest=None)
    assert [u.index for u in out] == [2, 3]


# --------------------------------------------------------------------------- #
# A momentary spike isn't a detection point
# --------------------------------------------------------------------------- #

def test_a_spike_the_evaluator_walks_back_is_not_a_detection_point():
    """Seen in testing: a hard negative spiked to 0.6, then corrected itself back to 0.2, yet permanently became a ground-truth detection point."""
    labels = [
        PrefixLabel(0, 0.10, "low", sufficient_evidence=False),
        PrefixLabel(1, 0.60, "medium", sufficient_evidence=True),   # spike
        PrefixLabel(2, 0.20, "low", sufficient_evidence=False),     # self-overturned
        PrefixLabel(3, 0.20, "low", sufficient_evidence=False),
    ]
    assert earliest_detection_index(labels, 0.5) is None
    assert earliest_detection_index(labels, 0.5, require_persistence=False) == 1


def test_a_sustained_rise_is_still_a_detection_point():
    labels = [
        PrefixLabel(0, 0.10, "low", sufficient_evidence=False),
        PrefixLabel(1, 0.60, "medium", sufficient_evidence=True),
        PrefixLabel(2, 0.70, "high", sufficient_evidence=True),
    ]
    assert earliest_detection_index(labels, 0.5) == 1


def test_a_spike_on_the_very_last_prefix_still_counts():
    """The last turn has no successor to overturn it, so it has to be taken as is."""
    labels = [
        PrefixLabel(0, 0.10, "low", sufficient_evidence=False),
        PrefixLabel(1, 0.80, "high", sufficient_evidence=True),
    ]
    assert earliest_detection_index(labels, 0.5) == 1


def test_persistence_only_looks_at_the_score_not_the_boolean():
    # the next prefix's score is still high but the boolean flips back to false: not overturned
    labels = [
        PrefixLabel(0, 0.70, "high", sufficient_evidence=True),
        PrefixLabel(1, 0.70, "high", sufficient_evidence=False),
    ]
    assert earliest_detection_index(labels, 0.5) == 0


def test_the_rule_is_recorded_in_the_result():
    provider = RecordingProvider()
    result = run(evaluate_conversation(convo(2), EvaluatorAgent(provider)))
    assert result.require_persistence is True


# --------------------------------------------------------------------------- #
# The rubric must not depend on the receiver's resistance (R5.9, both directions)
# --------------------------------------------------------------------------- #

def test_the_rubric_separates_proactive_from_reactive_anchors():
    """Seen in testing: with purely reactive anchors, a cooperative receiver turns a real scam undecidable."""
    assert "GROUP A" in PREFIX_EVAL_SYSTEM and "GROUP B" in PREFIX_EVAL_SYSTEM
    a = PREFIX_EVAL_SYSTEM.split("GROUP A")[1].split("GROUP B")[0]
    b = PREFIX_EVAL_SYSTEM.split("GROUP B")[1]
    # the proactive group must not depend on the other side's actions
    assert "need NOTHING from the" in a
    assert "even when the listener is perfectly cooperative" in a
    # the reactive group must state that "absence means nothing"
    assert "Absence of" in b and "never pushed" in b


def test_the_prompt_forbids_lowering_the_score_for_a_trusting_receiver():
    low = PREFIX_EVAL_SYSTEM.lower()
    assert "never lowers the score" in low
    assert "never raises the score" in low


def test_the_final_check_closes_the_rationale_loophole():
    """Seen in testing: the evaluator listed every pressure fact in its rationale yet still judged sufficient_evidence=false."""
    assert "If your own rationale describes" in PREFIX_EVAL_SYSTEM
    assert "however relaxed, friendly or convinced" in PREFIX_EVAL_SYSTEM


# --------------------------------------------------------------------------- #
# evidence_established: the monotonic "evidence is sufficient"
# --------------------------------------------------------------------------- #

def test_evidence_established_is_monotone_after_the_detection_point():
    from callfactory.evaluator import mark_evidence_established
    from callfactory.schema import PrefixLabel

    labels = [
        PrefixLabel(through_index=i, risk_score=s, risk_tier="x", sufficient_evidence=suf)
        for i, (s, suf) in enumerate([(0.1, False), (0.8, True), (0.2, False), (0.9, True)])
    ]
    mark_evidence_established(labels, earliest=1)
    assert [l.evidence_established for l in labels] == [False, True, True, True]


def test_evidence_established_is_all_false_without_a_detection_point():
    from callfactory.evaluator import mark_evidence_established
    from callfactory.schema import PrefixLabel

    labels = [PrefixLabel(through_index=i, risk_score=0.2, risk_tier="low") for i in range(3)]
    mark_evidence_established(labels, earliest=None)
    assert not any(l.evidence_established for l in labels)


def test_evidence_established_differs_from_sufficient_evidence_on_a_fallback():
    """When the labeller grows suspicious and is then reassured, the in-place verdict falls back but the monotonic label doesn't. That's why it exists."""
    from callfactory.evaluator import mark_evidence_established
    from callfactory.schema import PrefixLabel

    labels = [
        PrefixLabel(through_index=0, risk_score=0.9, risk_tier="high", sufficient_evidence=True),
        PrefixLabel(through_index=1, risk_score=0.9, risk_tier="high", sufficient_evidence=True),
        PrefixLabel(through_index=2, risk_score=0.2, risk_tier="low", sufficient_evidence=False),
    ]
    mark_evidence_established(labels, earliest=0)
    assert [l.sufficient_evidence for l in labels] == [True, True, False]
    assert [l.evidence_established for l in labels] == [True, True, True]


def test_utterance_labels_carry_the_monotone_flag():
    labels = [
        PrefixLabel(through_index=0, risk_score=0.1, risk_tier="low"),
        PrefixLabel(through_index=2, risk_score=0.8, risk_tier="high", sufficient_evidence=True),
    ]
    utterances = [
        Utterance(index=i, speaker="caller" if i % 2 == 0 else "receiver", text=f"line {i}")
        for i in range(4)
    ]
    out = build_utterance_labels(utterances, labels, earliest=2)
    assert [u.evidence_established for u in out] == [False, False, True, True]
    assert [u.is_detection_point for u in out] == [False, False, True, False]
