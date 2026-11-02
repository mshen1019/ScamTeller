"""Scenario bank I/O and admission checks."""

import json

import pytest

from callfactory.scenarios.bank import (
    BankError,
    latest_bank,
    load_bank,
    validate_scenario_dict,
    write_bank,
)
from callfactory.seed_scenarios import build_seed_scenarios


def good(**over):
    base = {
        "scenario_id": "scn-t-000",
        "category": "scam",
        "family": "test_family",
        "pretext": "claims something plausible about the receiver's account",
        "org_kind": "a fictional service company",
        "expected_signs": ["urgency_pressure"],
        "plausible_outcomes": ["receiver refuses", "receiver stalls"],
        "hard_negative": False,
        "disambiguator": "",
        "late_onset": False,
        "notes": "",
    }
    base.update(over)
    return base


# --------------------------- Admission checks ---------------------------- #

def test_valid_scenario_passes():
    assert validate_scenario_dict(good()) == []


def test_missing_fields_are_reported():
    errs = validate_scenario_dict({"category": "scam"})
    assert any("scenario_id" in e for e in errs)
    assert any("family" in e for e in errs)


def test_out_of_vocabulary_signal_is_rejected():
    errs = validate_scenario_dict(good(expected_signs=["urgency_pressure", "vibes_off"]))
    assert any("closed vocabulary" in e for e in errs)


def test_bad_category_and_family_shape():
    assert any("category" in e for e in validate_scenario_dict(good(category="maybe")))
    assert any("snake_case" in e for e in validate_scenario_dict(good(family="Test Family")))


def test_hard_negative_without_disambiguator_is_rejected():
    """R1.4: a hard negative without a disambiguator is just an ordinary negative."""
    errs = validate_scenario_dict(
        good(category="legitimate", hard_negative=True, expected_signs=["urgency_pressure"])
    )
    assert any("disambiguator" in e for e in errs)


def test_hard_negative_with_empty_signs_is_rejected():
    errs = validate_scenario_dict(
        good(category="legitimate", hard_negative=True, expected_signs=[], disambiguator="x" * 20)
    )
    assert any("expected_signs" in e for e in errs)


def test_hard_negative_must_be_legitimate():
    errs = validate_scenario_dict(good(hard_negative=True, disambiguator="x" * 20))
    assert any("legitimate" in e for e in errs)


def test_late_onset_only_for_scam():
    errs = validate_scenario_dict(good(category="legitimate", late_onset=True))
    assert any("late_onset" in e for e in errs)


def test_outcomes_count_bounds():
    assert any("plausible_outcomes" in e for e in validate_scenario_dict(good(plausible_outcomes=["only one"])))


# --------------------------- Disk round trip ----------------------------- #

def test_write_then_load_roundtrip(tmp_path):
    path = write_bank(build_seed_scenarios(), tmp_path, "seed", 1)
    bank = load_bank(path)
    assert len(bank) == 25
    assert bank.name == "seed" and bank.version == 1
    assert len(bank.scams) == 13
    assert len(bank.hard_negatives) == 7
    assert len(bank.plain_negatives) == 5


def test_provenance_is_stamped_on_write(tmp_path):
    path = write_bank([good()], tmp_path, "b", 3, author_model="claude-sonnet-5-5", context="bitcoin")
    s = load_bank(path).scenarios[0]
    assert s.provenance.author_model == "claude-sonnet-5-5"
    assert s.provenance.context == "bitcoin"
    assert s.provenance.bank_version == 3


def test_human_authored_is_marked(tmp_path):
    path = write_bank([good()], tmp_path, "b", 1, author_model="human")
    assert load_bank(path).scenarios[0].provenance.edited_by_human is True


def test_write_refuses_to_overwrite_a_version(tmp_path):
    write_bank([good()], tmp_path, "b", 1)
    with pytest.raises(BankError, match="version already exists"):
        write_bank([good()], tmp_path, "b", 1)
    write_bank([good()], tmp_path, "b", 2)        # bumping the version is fine


def test_write_rejects_duplicate_family(tmp_path):
    with pytest.raises(BankError, match="duplicate family"):
        write_bank([good(), good(scenario_id="scn-t-001")], tmp_path, "b", 1)


def test_write_rejects_invalid_scenario(tmp_path):
    with pytest.raises(BankError, match="closed vocabulary"):
        write_bank([good(expected_signs=["nope"])], tmp_path, "b", 1)


def test_load_rejects_bad_filename(tmp_path):
    p = tmp_path / "notabank.jsonl"
    p.write_text("{}\n")
    with pytest.raises(BankError, match="file name"):
        load_bank(p)


def test_load_reports_line_number_on_bad_record(tmp_path):
    p = tmp_path / "b.v1.jsonl"
    p.write_text(json.dumps(good()) + "\n" + json.dumps(good(family="X Y")) + "\n")
    with pytest.raises(BankError, match="b.v1.jsonl:2"):
        load_bank(p)


def test_load_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "b.v1.jsonl"
    p.write_text(json.dumps(good()) + "\n" + json.dumps(good(family="other")) + "\n")
    with pytest.raises(BankError, match="duplicate scenario_id"):
        load_bank(p)


def test_latest_bank_picks_highest_version(tmp_path):
    write_bank([good()], tmp_path, "seed", 1)
    write_bank([good(family="second")], tmp_path, "seed", 7)
    write_bank([good(family="third")], tmp_path, "seed", 2)
    assert latest_bank(tmp_path).version == 7


def test_latest_bank_requires_name_when_ambiguous(tmp_path):
    write_bank([good()], tmp_path, "a", 1)
    write_bank([good(family="other")], tmp_path, "b", 1)
    with pytest.raises(BankError, match="multiple bank names"):
        latest_bank(tmp_path)
    assert latest_bank(tmp_path, "b").name == "b"


# --------------------------- Filtering and gaps --------------------------- #

def test_filter_families(tmp_path):
    bank = load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))
    only = bank.filter_families(include=["tech_support_impersonation", "pharmacy_refill_ready"])
    assert sorted(only.families) == ["pharmacy_refill_ready", "tech_support_impersonation"]
    without = bank.filter_families(exclude=["tech_support_impersonation"])
    assert "tech_support_impersonation" not in without.families
    assert len(without) == len(bank) - 1


def test_filter_rejects_unknown_family(tmp_path):
    bank = load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))
    with pytest.raises(BankError, match="no such families"):
        bank.filter_families(include=["no_such_family"])


def test_coverage_gaps_reports_unpaired_signals(tmp_path):
    bank = load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))
    gaps = bank.coverage_gaps()
    # the seed bank really does have gaps; this is what review-bank shows a human
    assert "refusal_to_verify" in gaps["signals_without_hard_negative"]
    assert gaps["scam_families_without_paired_hard_negative"] == []
    assert "tech_support_impersonation" in gaps["late_onset_capable"]


def test_coverage_gaps_flags_a_scam_family_with_no_paired_hard_negative(tmp_path):
    # secrecy_request has no hard negative covering it in the seed bank
    scns = build_seed_scenarios() + [
        good(scenario_id="scn-x", family="lonely_scam", expected_signs=["secrecy_request"])
    ]
    bank = load_bank(write_bank(scns, tmp_path, "seed", 1))
    assert "lonely_scam" in bank.coverage_gaps()["scam_families_without_paired_hard_negative"]


def test_scenario_cannot_declare_a_style_level_signal():
    """script_like_repetition has nothing to do with the call's purpose; it comes from the caller persona, and scenarios shouldn't declare it."""
    errs = validate_scenario_dict(good(expected_signs=["script_like_repetition"]))
    assert any("style-level signal" in e for e in errs)
    # mixed in with others it must still be caught
    errs = validate_scenario_dict(
        good(expected_signs=["urgency_pressure", "script_like_repetition"])
    )
    assert any("style-level signal" in e for e in errs)


def test_style_signals_are_not_counted_as_scenario_gaps(tmp_path):
    """Before the bug fix this reported script_like_repetition as a gap, which is a false gap."""
    bank = load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))
    gaps = bank.coverage_gaps()
    assert "script_like_repetition" not in gaps["signals_never_used_by_scam"]
    assert gaps["signals_never_used_by_scam"] == []


def test_persona_library_covers_the_style_signals(tmp_path):
    from callfactory.personas import CALLER_PERSONAS

    bank = load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))
    gaps = bank.coverage_gaps(CALLER_PERSONAS)
    assert gaps["style_signals_covered_by_personas"] == ["script_like_repetition"]
    assert gaps["style_signals_with_no_persona"] == []


def test_missing_persona_coverage_is_reported(tmp_path):
    from callfactory.personas import CALLER_PERSONAS

    bank = load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))
    bare = [c for c in CALLER_PERSONAS if not c.style_signals]
    gaps = bank.coverage_gaps(bare)
    assert gaps["style_signals_with_no_persona"] == ["script_like_repetition"]


# --------------------------------------------------------------------------- #
# Admission checks for compliant outcomes and "no usable evidence" scenarios
# --------------------------------------------------------------------------- #

def test_compliant_outcomes_must_be_a_subset_of_the_outcomes():
    errs = validate_scenario_dict(good(
        plausible_outcomes=["a receiver refuses outright", "a receiver stalls for time"],
        compliant_outcomes=["a receiver agrees to everything"],
    ))
    assert any("subset" in e for e in errs)


def test_only_scam_scenarios_may_have_compliant_outcomes():
    errs = validate_scenario_dict(good(
        category="legitimate", expected_signs=[],
        plausible_outcomes=["the business is concluded", "the receiver asks to be rung back"],
        compliant_outcomes=["the business is concluded"],
    ))
    assert any("only makes sense for scam" in e for e in errs)


def test_no_observable_evidence_cannot_declare_a_strong_signal():
    """Self-contradiction: a strong signal is enough on its own, so it can't also be "no usable evidence"."""
    errs = validate_scenario_dict(good(
        no_observable_evidence=True,
        expected_signs=["unverified_identity", "threat_of_consequence"],
    ))
    assert any("threat_of_consequence" in e and "conflicts" in e for e in errs)


def test_no_observable_evidence_with_only_weak_signals_is_accepted():
    assert validate_scenario_dict(good(
        no_observable_evidence=True, expected_signs=["unverified_identity"],
    )) == []


def test_no_observable_evidence_is_scam_only():
    errs = validate_scenario_dict(good(
        category="legitimate", no_observable_evidence=True, expected_signs=[],
    ))
    assert any("can only be used for scam" in e for e in errs)


def test_no_observable_evidence_scenario_cannot_have_compliant_outcomes():
    errs = validate_scenario_dict(good(
        no_observable_evidence=True, expected_signs=[],
        compliant_outcomes=["a receiver refuses outright"],
    ))
    assert any("never asks for anything" in e for e in errs)


def test_the_seed_bank_carries_the_new_fields_through_a_roundtrip(tmp_path):
    bank = load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))
    recon = [s for s in bank.scams if s.no_observable_evidence]
    assert len(recon) == 1
    assert recon[0].eval_only is True
    assert recon[0].compliant_outcomes == []
    others = [s for s in bank.scams if not s.no_observable_evidence]
    assert all(s.compliant_outcomes for s in others)
    assert all(set(s.compliant_outcomes) <= set(s.plausible_outcomes) for s in others)


def test_the_seed_bank_has_resistance_outcomes_on_both_sides():
    """Shortcut found in testing: originally 5/6 scam outcomes were resisting and 0/5 legitimate ones were, so a model could score by reading the receiver."""
    from callfactory.seed_scenarios import LEGIT_OUTCOMES, SCAM_OUTCOMES
    shared = {
        "receiver insists on calling back on a number they look up themselves",
    }
    assert shared <= set(SCAM_OUTCOMES)
    # the legitimate side needs word-for-word identical receiver behaviour; the only difference is the caller's reaction
    assert any("calling back on a number they look up" in o for o in LEGIT_OUTCOMES)
    assert any("suspicious throughout" in o for o in LEGIT_OUTCOMES)
