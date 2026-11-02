"""Acceptance tests for step 1 (provider layer + schema)."""

import asyncio
import json

import pytest

from callfactory.providers import (
    ProviderIndependenceError,
    ProviderRoles,
    build_provider,
    extract_json,
)
from callfactory.schema import (
    LABEL_KEYS,
    RISK_SIGNALS,
    Scenario,
    clean_signals,
    coerce_tier,
)
from tests.factories import make_conversation, make_scenario, make_task


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

def test_scenario_roundtrip_keeps_provenance():
    s = make_scenario()
    back = Scenario.from_dict(s.to_dict())
    assert back.family == s.family
    assert back.provenance.context == "bitcoin investment scam"
    assert back.provenance.bank_version == 1


def test_scenario_without_provenance_roundtrips():
    s = make_scenario(provenance=None)
    assert "provenance" not in s.to_dict()
    assert Scenario.from_dict(s.to_dict()).provenance is None


def test_clean_signals_normalises_and_drops_out_of_vocabulary():
    assert clean_signals(["Urgency-Pressure", "vibes off", "AUTHORITY_CLAIM"]) == [
        "urgency_pressure",
        "authority_claim",
    ]
    assert clean_signals(["urgency_pressure", "urgency_pressure"]) == ["urgency_pressure"]
    assert clean_signals("not a list") == []
    assert clean_signals([1, None, {}]) == []


def test_coerce_tier_falls_back_to_score():
    assert coerce_tier("HIGH", 0.1) == "high"
    assert coerce_tier("catastrophic", 0.8) == "high"
    assert coerce_tier(None, 0.05) == "none"
    assert coerce_tier(None, 0.4) == "medium"


# --------------------------------------------------------------------------- #
# P2 labels separate from inputs
# --------------------------------------------------------------------------- #

def test_transcript_record_has_only_three_fields_per_utterance():
    conv = make_conversation()
    rec = conv.transcript_record()
    for utt in rec["utterances"]:
        assert set(utt) == {"index", "speaker", "text"}


def test_transcript_record_contains_no_label_key_at_any_depth():
    blob = json.dumps(make_conversation().transcript_record())
    for key in LABEL_KEYS:
        assert f'"{key}"' not in blob, key


def test_metadata_record_carries_the_labels_and_joins_on_id():
    conv = make_conversation()
    meta = conv.metadata_record()
    assert meta["conversation_id"] == conv.transcript_record()["conversation_id"]
    assert meta["ground_truth"]["is_scam"] is True
    assert meta["ground_truth"]["scam_family"] == "crypto_investment_pressure"
    assert len(meta["prefix_labels"]) == len(conv.utterances)
    assert meta["earliest_detection_index"] == 4


def test_label_keys_covers_every_top_level_metadata_key_that_is_a_label():
    """If a new metadata field is added without registering it in LABEL_KEYS, this test catches it."""
    meta = make_conversation().metadata_record()
    not_a_label = {"conversation_id", "validation"}
    for key in meta:
        if key not in not_a_label:
            assert key in LABEL_KEYS, f"{key} not registered in LABEL_KEYS; the leakage check would miss it"


def test_detection_round_derives_from_index():
    conv = make_conversation()
    assert conv.earliest_detection_round == 3      # index 4 → round 3
    conv.earliest_detection_index = None
    assert conv.earliest_detection_round is None


def test_rounds_counts_receiver_turns():
    assert make_conversation(rounds=5).rounds == 5
    assert len(make_conversation(rounds=5).utterances) == 10


def test_expected_signs_in_fixtures_are_in_vocabulary():
    for sign in make_scenario().expected_signs:
        assert sign in RISK_SIGNALS


# --------------------------------------------------------------------------- #
# Diversity level-1 fingerprint
# --------------------------------------------------------------------------- #

def test_fingerprint_identifies_task_combination():
    a = make_task()
    b = make_task(outcome="receiver refuses")
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint == make_task().fingerprint
    assert a.fingerprint[0] == "crypto_investment_pressure"


# --------------------------------------------------------------------------- #
# P4 same-source check
# --------------------------------------------------------------------------- #

def test_evaluator_sharing_provider_with_caller_is_rejected():
    p = build_provider("openai", model="gpt-4.1-mini")
    roles = ProviderRoles(caller=p, receiver=p, evaluator=p)
    assert len(roles.violations()) == 2
    with pytest.raises(ProviderIndependenceError):
        roles.enforce()


def test_explicit_override_returns_warnings_instead_of_raising():
    p = build_provider("openai", model="gpt-4.1-mini")
    roles = ProviderRoles(caller=p, receiver=p, evaluator=p)
    warnings = roles.enforce(allow_same_provider=True)
    assert len(warnings) == 2


def test_distinct_providers_pass():
    roles = ProviderRoles(
        caller=build_provider("openai", model="gpt-4.1-mini"),
        receiver=build_provider("openai", model="gpt-4.1-mini"),
        evaluator=build_provider("anthropic", model="claude-sonnet-5-5"),
        gatekeeper=build_provider("fireworks", model="x"),
    )
    assert roles.enforce() == []


def test_gatekeeper_sharing_with_evaluator_is_flagged():
    ev = build_provider("anthropic", model="claude-sonnet-5-5")
    roles = ProviderRoles(
        caller=build_provider("openai", model="gpt-4.1-mini"),
        receiver=build_provider("openai", model="gpt-4.1-mini"),
        evaluator=ev,
        gatekeeper=build_provider("anthropic", model="claude-sonnet-5-5"),
    )
    issues = roles.violations()
    assert len(issues) == 1 and "gatekeeper" in issues[0]


def test_all_mock_is_allowed_for_offline_development():
    m = build_provider("mock")
    assert ProviderRoles(caller=m, receiver=m, evaluator=m, gatekeeper=m).enforce() == []


def test_describe_and_dedup():
    p = build_provider("mock")
    q = build_provider("anthropic", model="claude-sonnet-5-5")
    roles = ProviderRoles(caller=p, receiver=p, evaluator=q)
    assert len(roles.all_providers()) == 2
    assert roles.describe()["evaluator"] == "anthropic/claude-sonnet-5-5"
    assert roles.describe()["gatekeeper"] is None


# --------------------------------------------------------------------------- #
# mock support for the newer agents
# --------------------------------------------------------------------------- #

def call_mock(tag, content, system="sys"):
    m = build_provider("mock")
    return asyncio.run(
        m.complete(system=system, messages=[{"role": "user", "content": content}], tag=tag)
    )


def test_mock_scenario_author_returns_valid_structured_scenarios():
    raw = call_mock(
        "scenario_author",
        "Produce exactly 4 scenarios.\nCONTEXT: crypto investment fraud",
    )
    data = json.loads(raw)
    assert len(data["scenarios"]) == 4
    for s in data["scenarios"]:
        assert s["category"] in ("scam", "legitimate")
        assert all(sig in RISK_SIGNALS for sig in s["expected_signs"])
        # a hard negative must have a disambiguator (R1.4)
        if s["hard_negative"]:
            assert s["disambiguator"]
        # only scams may have late_onset true
        if s["late_onset"]:
            assert s["category"] == "scam"


def test_mock_scenario_author_uses_the_human_context():
    raw = call_mock("scenario_author", "Produce exactly 2 scenarios.\nCONTEXT: utility impersonation")
    families = [s["family"] for s in json.loads(raw)["scenarios"]]
    assert all(f.startswith("utility_impersonation") for f in families), families


def test_mock_gatekeeper_craft_rewards_spoken_markers():
    flat = json.loads(call_mock("gatekeeper_craft", "[0] CALLER: I am calling regarding your account."))
    spoken = json.loads(call_mock(
        "gatekeeper_craft",
        "[0] CALLER: sorry -- uh, I mean, hang on -- sorry, I mean the other thing",
    ))
    assert spoken["realism"] > flat["realism"]
    assert "inconsistencies" in spoken


def test_mock_gatekeeper_detection_penalises_both_extremes():
    too_early = json.loads(call_mock("gatekeeper_detection", "earliest_detection_index: 0\nsignals: [x]"))
    no_signal = json.loads(call_mock("gatekeeper_detection", "earliest_detection_index: None\nis_scam: True"))
    useful = json.loads(call_mock("gatekeeper_detection", "earliest_detection_index: 5\nsignals: [x]"))
    assert too_early["detection_value"] <= 2
    assert no_signal["detection_value"] <= 2
    assert useful["detection_value"] >= 3


def test_mock_is_deterministic_across_instances():
    args = ("caller", "go", "INTENT: suspicious")
    assert call_mock(*args) == call_mock(*args)


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('here: {"a": [1,2]} done') == {"a": [1, 2]}
