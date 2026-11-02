"""Acceptance tests for the scenario sampler (component 2)."""

from collections import Counter

import pytest

from callfactory.personas import CALLER_PERSONAS, RECEIVER_PERSONAS
from callfactory.scenarios import load_bank, sample_tasks, write_bank
from callfactory.scenarios.sampler import COMPLIANCE_SKEPTICISM, derive_seed
from callfactory.seed_scenarios import build_seed_scenarios


@pytest.fixture()
def bank(tmp_path):
    return load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))


# --------------------------- Reproducibility (P5 / R2.1) ---------------- #

def test_same_inputs_give_identical_tasks(bank):
    a, _ = sample_tasks(bank, 20, seed=7)
    b, _ = sample_tasks(bank, 20, seed=7)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


def test_different_seed_gives_different_tasks(bank):
    a, _ = sample_tasks(bank, 20, seed=7)
    b, _ = sample_tasks(bank, 20, seed=8)
    assert [t.scenario.family for t in a] != [t.scenario.family for t in b]


def test_bank_version_participates_in_the_derived_seed():
    """Same seed and index under a different bank version must derive a different seed, otherwise data from a new bank collides with old batches."""
    assert derive_seed(1, 0, 1) != derive_seed(1, 0, 2)
    assert derive_seed(1, 0) != derive_seed(0, 1)
    assert len({derive_seed(s, i, v) for s in range(4) for i in range(8) for v in range(3)}) == 96


# --------------------------- Stratification (R2.2 / R2.3) --------------- #

def test_ratios_are_honoured(bank):
    tasks, warnings = sample_tasks(
        bank, 100, seed=3, scam_ratio=0.5, hard_negative_ratio=0.4, late_onset_ratio=0.3
    )
    assert len(tasks) == 100
    assert warnings == []
    scams = [t for t in tasks if t.scenario.category == "scam"]
    legits = [t for t in tasks if t.scenario.category == "legitimate"]
    assert len(scams) == 50
    assert sum(1 for t in legits if t.scenario.hard_negative) == 20
    assert sum(1 for t in scams if t.late_onset) == 15


def test_non_default_scam_ratio(bank):
    tasks, _ = sample_tasks(bank, 100, seed=3, scam_ratio=0.8)
    assert sum(1 for t in tasks if t.scenario.category == "scam") == 80


def test_stratification_spreads_families_evenly(bank):
    """Pure random would swamp small families; round-robin usage counts must differ by <= 1."""
    tasks, _ = sample_tasks(bank, 96, seed=11)
    for category in ("scam",):
        counts = Counter(
            t.scenario.family for t in tasks if t.scenario.category == category
        )
        assert max(counts.values()) - min(counts.values()) <= 1
        assert len(counts) == 13          # all 13 scam families used


def test_ratio_deviation_warns_instead_of_failing_silently(bank):
    small = bank.filter_families(include=["tech_support_impersonation", "pharmacy_refill_ready"])
    tasks, warnings = sample_tasks(small, 10, seed=1, hard_negative_ratio=0.5)
    codes = {w.code for w in warnings}
    # no hard negative scenarios after filtering; the quota must be moved and a warning raised
    assert "layer.empty" in codes
    assert all(not t.scenario.hard_negative for t in tasks)


def test_late_onset_only_lands_on_capable_scenarios(bank):
    tasks, _ = sample_tasks(bank, 80, seed=5, late_onset_ratio=0.5)
    for t in tasks:
        if t.late_onset:
            assert t.scenario.late_onset, f"{t.scenario.family} doesn't suit slow burn but is marked late_onset"
            assert t.scenario.category == "scam"


def test_insufficient_late_onset_capacity_warns(bank):
    """100% slow burn requested but only 4 scenarios in the bank suit it: must warn rather than pretend it's met."""
    tasks, warnings = sample_tasks(bank, 40, seed=5, scam_ratio=1.0, late_onset_ratio=1.0)
    assert any(w.code == "late_onset.insufficient" for w in warnings)
    assert all(t.scenario.late_onset for t in tasks if t.late_onset)


# --------------------------- Persona cross product (R2.5) --------------- #

def test_persona_triples_are_unique(bank):
    tasks, warnings = sample_tasks(bank, 200, seed=9)
    triples = [
        (t.scenario.scenario_id, t.caller_persona.persona_id, t.receiver_persona.persona_id)
        for t in tasks
    ]
    assert len(set(triples)) == len(triples)
    assert not any(w.code == "personas.exhausted" for w in warnings)


def test_same_scenario_reused_with_different_personas(bank):
    one = bank.filter_families(include=["tech_support_impersonation"])
    tasks, _ = sample_tasks(one, 30, seed=2, scam_ratio=1.0)
    assert len({t.scenario.scenario_id for t in tasks}) == 1
    assert len({(t.caller_persona.persona_id, t.receiver_persona.persona_id) for t in tasks}) == 30


def test_persona_exhaustion_warns(bank):
    one = bank.filter_families(include=["tech_support_impersonation"])
    tasks, warnings = sample_tasks(
        one, 6, seed=2, scam_ratio=1.0,
        callers=CALLER_PERSONAS[:2], receivers=RECEIVER_PERSONAS[:2],
    )
    assert any(w.code == "personas.exhausted" for w in warnings)
    assert len(tasks) == 6


def test_both_persona_libraries_are_used(bank):
    tasks, _ = sample_tasks(bank, 144, seed=4)
    assert len({t.caller_persona.persona_id for t in tasks}) == 12
    assert len({t.receiver_persona.persona_id for t in tasks}) == 12


# --------------------------- Other --------------------------------------- #

def test_rounds_within_bounds_and_directives_long_enough(bank):
    tasks, _ = sample_tasks(bank, 40, seed=6, min_rounds=5, max_rounds=9)
    for t in tasks:
        assert 5 <= t.target_rounds <= 9
        assert len(t.turn_directives) >= 2 * t.target_rounds


def test_outcome_comes_from_the_scenario(bank):
    tasks, _ = sample_tasks(bank, 40, seed=6)
    for t in tasks:
        assert t.outcome in t.scenario.plausible_outcomes


def test_fingerprint_distinguishes_tasks(bank):
    tasks, _ = sample_tasks(bank, 50, seed=6)
    assert len({t.fingerprint for t in tasks}) == 50


def test_include_and_exclude_family(bank):
    tasks, _ = sample_tasks(bank, 20, seed=1, include_family=["prize_notification"], scam_ratio=1.0)
    assert {t.scenario.family for t in tasks} == {"prize_notification"}
    tasks, _ = sample_tasks(bank, 40, seed=1, exclude_family=["prize_notification"])
    assert "prize_notification" not in {t.scenario.family for t in tasks}


def test_rejects_bad_arguments(bank):
    with pytest.raises(ValueError):
        sample_tasks(bank, 0, seed=1)
    with pytest.raises(ValueError):
        sample_tasks(bank, 5, seed=1, min_rounds=9, max_rounds=4)
    with pytest.raises(ValueError):
        sample_tasks(bank, 5, seed=1, callers=[])


def test_task_metadata_records_bank_version(bank):
    tasks, _ = sample_tasks(bank, 3, seed=1)
    for t in tasks:
        assert t.bank_version == bank.version
        assert t.derived_seed == derive_seed(1, t.index, bank.version)


def test_small_batch_rounding_does_not_warn(bank):
    """5 scams at 30% slow burn can only give 2/5 = 0.40; that's rounding and shouldn't warn."""
    tasks, warnings = sample_tasks(bank, 10, seed=2024, min_rounds=5, max_rounds=9)
    assert [w.code for w in warnings] == []
    assert len(tasks) == 10


def test_large_batch_still_catches_real_ratio_drift(bank):
    """On large batches the tolerance returns to its strict value, and real deviations must be caught."""
    only_prize = bank.filter_families(include=["prize_notification", "pharmacy_refill_ready"])
    tasks, warnings = sample_tasks(only_prize, 100, seed=1, scam_ratio=0.5, late_onset_ratio=0.5)
    # prize_notification doesn't suit slow burn, so the 50% target is bound to miss
    assert any(w.code == "late_onset.insufficient" for w in warnings)
    assert any(w.code == "ratio.late_onset" for w in warnings)


# --------------------------------------------------------------------------- #
# Compliant outcomes: breaking the "receiver resists = scam" shortcut
# --------------------------------------------------------------------------- #

def test_compliance_only_lands_on_credulous_receivers(bank):
    """Having a skepticism=high persona "won over" fights the persona and must not happen."""
    tasks, _ = sample_tasks(bank, 60, 11, compliance_ratio=0.3)
    complied = [t for t in tasks if t.outcome in t.scenario.compliant_outcomes]
    assert complied, "there should be compliant samples"
    assert all(t.receiver_persona.skepticism in COMPLIANCE_SKEPTICISM for t in complied)
    assert all(t.scenario.category == "scam" for t in complied)


def test_compliance_ratio_is_respected(bank):
    tasks, warnings = sample_tasks(bank, 60, 11, compliance_ratio=0.25)
    scams = [t for t in tasks if t.scenario.category == "scam"]
    complied = [t for t in scams if t.outcome in t.scenario.compliant_outcomes]
    assert abs(len(complied) / len(scams) - 0.25) <= 0.1
    assert not [w for w in warnings if w.code == "ratio.compliance"]


def test_zero_compliance_ratio_produces_none(bank):
    tasks, _ = sample_tasks(bank, 30, 11, compliance_ratio=0.0)
    assert not [t for t in tasks if t.outcome in t.scenario.compliant_outcomes]


def test_an_unreachable_compliance_ratio_warns_instead_of_forcing(bank):
    # only high-skepticism receivers: not a single compliant sample can be assigned
    stubborn = [r for r in RECEIVER_PERSONAS if r.skepticism == "high"]
    tasks, warnings = sample_tasks(bank, 20, 5, receivers=stubborn, compliance_ratio=0.5)
    assert not [t for t in tasks if t.outcome in t.scenario.compliant_outcomes]
    assert [w for w in warnings if w.code == "compliance.insufficient"]


def test_non_compliance_samples_never_draw_a_compliant_outcome(bank):
    tasks, _ = sample_tasks(bank, 40, 3, compliance_ratio=0.2)
    scams = [t for t in tasks if t.scenario.category == "scam"]
    complied = {t.index for t in scams if t.outcome in t.scenario.compliant_outcomes}
    for t in scams:
        if t.index not in complied:
            assert t.outcome not in t.scenario.compliant_outcomes


def test_changing_the_compliance_ratio_does_not_reshuffle_personas(bank):
    """Outcomes are drawn with a separate rng: changing a ratio shouldn't swap the personas too."""
    a, _ = sample_tasks(bank, 30, 9, compliance_ratio=0.0)
    b, _ = sample_tasks(bank, 30, 9, compliance_ratio=0.5)
    assert [(t.caller_persona.persona_id, t.receiver_persona.persona_id) for t in a] == \
           [(t.caller_persona.persona_id, t.receiver_persona.persona_id) for t in b]
    assert [t.target_rounds for t in a] == [t.target_rounds for t in b]


def test_the_no_evidence_family_is_sampled_like_any_other_scam(bank):
    tasks, _ = sample_tasks(bank, 60, 4)
    recon = [t for t in tasks if t.scenario.no_observable_evidence]
    assert recon, "the no-decidable-evidence family should be sampled"
    assert all(t.scenario.category == "scam" for t in recon)
    assert all(t.scenario.eval_only for t in recon)
    assert all(t.outcome in t.scenario.plausible_outcomes for t in recon)


# --------------------------- Family-balance mode --------------------------- #

def test_balanced_mode_gives_every_family_the_same_count(bank):
    n = len({s.family for s in bank.scenarios})
    tasks, _ = sample_tasks(bank, n * 3, seed=11, balance_families=True)
    counts = Counter(t.scenario.family for t in tasks)
    assert len(counts) == n
    assert set(counts.values()) == {3}


def test_balanced_mode_spreads_the_remainder_by_one(bank):
    n = len({s.family for s in bank.scenarios})
    tasks, _ = sample_tasks(bank, n * 2 + 3, seed=11, balance_families=True)
    counts = Counter(t.scenario.family for t in tasks)
    assert sorted(counts.values())[0] == 2 and sorted(counts.values())[-1] == 3
    assert sum(1 for v in counts.values() if v == 3) == 3


def test_existing_counts_are_topped_up_to_the_average(bank):
    """Already-sampled families only get the gap filled; that's all "top up to the mean" means."""
    families = sorted({s.family for s in bank.scenarios})
    existing = {families[0]: 4, families[1]: 2}
    n = len(families)
    tasks, _ = sample_tasks(
        bank, n * 5 - 6, seed=11, balance_families=True, existing_counts=existing
    )
    counts = Counter(t.scenario.family for t in tasks)
    final = {f: counts.get(f, 0) + existing.get(f, 0) for f in families}
    assert set(final.values()) == {5}
    assert counts[families[0]] == 1 and counts[families[1]] == 3


def test_a_family_already_past_the_average_gets_no_quota(bank):
    families = sorted({s.family for s in bank.scenarios})
    tasks, warnings = sample_tasks(
        bank, len(families), seed=11, balance_families=True,
        existing_counts={families[0]: 50},
    )
    counts = Counter(t.scenario.family for t in tasks)
    assert counts[families[0]] == 0
    assert any(w.code == "balance.already_full" for w in warnings)
    assert any(w.code == "balance.uneven" for w in warnings)


def test_balanced_mode_reports_the_composition_instead_of_ratio_deviations(bank):
    tasks, warnings = sample_tasks(bank, 60, seed=11, balance_families=True)
    codes = {w.code for w in warnings}
    assert "balance.composition" in codes
    # in this mode the ratios are an **outcome**; faulting them against target ratios only produces noise
    assert not {"ratio.scam", "ratio.hard_negative"} & codes


def test_balanced_mode_still_keeps_the_persona_triple_unique(bank):
    tasks, _ = sample_tasks(bank, 100, seed=11, balance_families=True)
    assert len({t.fingerprint for t in tasks}) == len(tasks)


def test_balanced_mode_is_reproducible(bank):
    kw = dict(balance_families=True, existing_counts={"prize_notification": 3})
    a, _ = sample_tasks(bank, 50, seed=11, **kw)
    b, _ = sample_tasks(bank, 50, seed=11, **kw)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


def test_balanced_mode_does_not_group_one_family_together(bank):
    """Items of the same family sitting together make batch generation advance family by family, and resuming skews with it."""
    tasks, _ = sample_tasks(bank, 75, seed=11, balance_families=True)
    families = [t.scenario.family for t in tasks]
    runs = sum(1 for a, b in zip(families, families[1:]) if a == b)
    assert runs <= 5
