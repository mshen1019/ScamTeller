"""The scenario author agent (component 1): dedup context, the three gates, outcome vocabulary set by the dataset."""

from __future__ import annotations

import asyncio
import json

import pytest

from callfactory.author import (
    PRETEXT_DUP,
    author_scenarios,
    duplicate_reasons,
    merge_into_bank,
    normalise,
)
from callfactory.prompts import SCENARIO_AUTHOR_SYSTEM, scenario_author_user_message
from callfactory.providers import build_provider
from callfactory.providers.base import LLMProvider, ProviderConfig, ProviderError
from callfactory.scenarios import load_bank, write_bank
from callfactory.seed_scenarios import COMPLIANT_SCAM_OUTCOMES, build_seed_scenarios


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def bank(tmp_path):
    return load_bank(write_bank(build_seed_scenarios(), tmp_path, "seed", 1))


class Scripted(LLMProvider):
    def __init__(self, payloads):
        super().__init__(ProviderConfig(name="mock", model="scripted-1"))
        self.payloads = list(payloads)
        self.messages: list[str] = []

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag):
        self.messages.append(messages[-1]["content"])
        return self.payloads.pop(0) if self.payloads else json.dumps({"scenarios": []})


def one(**over):
    base = {
        "category": "scam",
        "family": "parcel_customs_fee",
        "pretext": "says a held parcel needs a customs charge paid before release",
        "org_kind": "a parcel courier",
        "expected_signs": ["urgency_pressure", "unusual_payment_channel"],
        "hard_negative": False,
        "disambiguator": "",
        "late_onset": True,
        "notes": "",
    }
    base.update(over)
    return base


def payload(*items):
    return json.dumps({"scenarios": list(items)})


# --------------------------- Dedup context (R1.1) ----------------------- #

def test_existing_families_and_pretexts_are_sent_to_the_agent(bank):
    provider = Scripted([payload(one())])
    run(author_scenarios(provider, bank, 1))
    sent = provider.messages[0]
    assert "tech_support_impersonation" in sent
    assert "do not repeat these kinds of call" in sent
    assert bank.scenarios[0].pretext[:40] in sent


def test_context_and_gaps_reach_the_prompt(bank):
    provider = Scripted([payload(one())])
    run(author_scenarios(provider, bank, 1, context="bitcoin investment scam",
                         gaps={"signals_without_hard_negative": ["secrecy_request"]}))
    sent = provider.messages[0]
    assert "bitcoin investment scam" in sent
    assert "secrecy_request" in sent
    assert "no hard negative to balance" in sent


def test_the_system_prompt_carries_the_closed_vocabulary_and_the_boundary():
    assert "refusal_to_verify" in SCENARIO_AUTHOR_SYSTEM
    assert "script_like_repetition" in SCENARIO_AUTHOR_SYSTEM     # explicitly says not to write it into scenarios
    assert "Never put it in expected_signs" in SCENARIO_AUTHOR_SYSTEM
    assert "SAFETY BOUNDARY" in SCENARIO_AUTHOR_SYSTEM
    assert "Hard negatives are the most valuable" in SCENARIO_AUTHOR_SYSTEM


def test_the_user_message_asks_for_an_exact_count():
    msg = scenario_author_user_message(["a_b"], [], 7)
    assert "exactly 7" in msg


# --------------------------- The three gates ----------------------------- #

def test_an_out_of_vocabulary_signal_drops_the_whole_record(bank):
    provider = Scripted([payload(one(expected_signs=["urgency_pressure", "bad_vibes"]))])
    result = run(author_scenarios(provider, bank, 1, max_attempts=1))
    assert result.accepted == []
    assert any("closed vocabulary" in r for rej in result.rejected for r in rej.reasons)


def test_a_style_level_signal_is_rejected(bank):
    provider = Scripted([payload(one(expected_signs=["script_like_repetition"]))])
    result = run(author_scenarios(provider, bank, 1, max_attempts=1))
    assert result.accepted == []
    assert any("style-level signal" in r for rej in result.rejected for r in rej.reasons)


def test_a_hard_negative_without_a_disambiguator_is_rejected(bank):
    provider = Scripted([payload(one(
        category="legitimate", hard_negative=True, family="genuine_courier_desk",
        pretext="rings about a delivery the receiver actually arranged last week",
        expected_signs=["urgency_pressure"], disambiguator="",
    ))])
    result = run(author_scenarios(provider, bank, 1, max_attempts=1))
    assert result.accepted == []
    assert any("disambiguator" in r for rej in result.rejected for r in rej.reasons)


def test_a_duplicate_family_name_is_rejected(bank):
    provider = Scripted([payload(one(family="tech_support_impersonation"))])
    result = run(author_scenarios(provider, bank, 1, max_attempts=1))
    assert any("already exists" in r for rej in result.rejected for r in rej.reasons)


def test_a_near_duplicate_pretext_is_rejected(bank):
    existing = bank.scenarios[0].pretext
    provider = Scripted([payload(one(family="totally_new_name", pretext=existing))])
    result = run(author_scenarios(provider, bank, 1, max_attempts=1))
    assert any("near-duplicate" in r for rej in result.rejected for r in rej.reasons)


def test_the_curated_bank_passes_its_own_dedup(bank):
    """The 25 hand-written scenarios must not be judged duplicates of each other. This is the regression test for the dedup threshold:
    whoever tightens it too far turns this red first."""
    scns = bank.scenarios
    for i, s in enumerate(scns):
        others = [x for j, x in enumerate(scns) if j != i]
        reasons = duplicate_reasons(
            {"family": s.family, "pretext": s.pretext},
            [o.family for o in others], [o.pretext for o in others],
        )
        assert reasons == [], f"{s.family} judged a duplicate by its own bank: {reasons}"


def test_similar_family_names_with_different_premises_coexist():
    """Similar names aren't duplicates. investment_pressure and crypto_investment_pressure are a deliberate pair."""
    assert duplicate_reasons(
        {"family": "crypto_investment_pressure",
         "pretext": "pushes a wallet top-up into a token sale said to close within the hour"},
        ["investment_pressure"],
        ["pitches a closing-today position in a fund the receiver never enquired about"],
    ) == []


def test_a_one_word_difference_in_the_pretext_is_caught():
    """Trigrams miss this; bigrams catch it."""
    a = "says a held item needs a small release charge settled before it is sent on"
    b = "says a held item needs a small release charge settled before it is posted on"
    assert duplicate_reasons({"family": "x_y", "pretext": a}, [], [b])


def test_records_accepted_in_one_call_dedup_against_each_other(bank):
    same = "says a held parcel needs a customs charge paid before release"
    provider = Scripted([payload(
        one(family="parcel_customs_fee", pretext=same),
        one(family="parcel_release_fee", pretext=same),
    )])
    result = run(author_scenarios(provider, bank, 2, max_attempts=1))
    assert len(result.accepted) == 1
    assert any("near-duplicate" in r for rej in result.rejected for r in rej.reasons)


# --------------------------- Outcome vocabulary -------------------------- #

def test_the_outcome_vocabulary_comes_from_the_dataset_not_the_author():
    rec = normalise(one(plausible_outcomes=["the caller gets the money"]), bank_name="b", seq=0)
    assert "the caller gets the money" not in rec["plausible_outcomes"]
    assert rec["compliant_outcomes"] == [
        o for o in COMPLIANT_SCAM_OUTCOMES if o in rec["plausible_outcomes"]
    ]
    assert rec["compliant_outcomes"], "a scam must carry compliant outcomes, otherwise the R2.8 shortcut comes back"


def test_legitimate_scenarios_get_no_compliant_outcomes():
    rec = normalise(one(category="legitimate", expected_signs=[]), bank_name="b", seq=0)
    assert rec["compliant_outcomes"] == []
    assert any("suspicious throughout" in o for o in rec["plausible_outcomes"])


def test_hard_negative_and_late_onset_are_forced_consistent():
    # late_onset only means something for scam, hard_negative only for legitimate
    legit = normalise(one(category="legitimate", late_onset=True, expected_signs=[]),
                      bank_name="b", seq=0)
    assert legit["late_onset"] is False
    scam = normalise(one(category="scam", hard_negative=True), bank_name="b", seq=0)
    assert scam["hard_negative"] is False


def test_family_names_are_normalised():
    assert normalise(one(family="Parcel Customs-Fee"), bank_name="b", seq=0)["family"] == \
        "parcel_customs_fee"


# --------------------------- Failure and retry --------------------------- #

def test_it_retries_until_it_has_enough(bank):
    provider = Scripted([
        payload(one(family="dup_one", pretext=bank.scenarios[0].pretext)),
        payload(one(family="fresh_family_here",
                    pretext="claims a council tax rebate is waiting but the bank details on file failed")),
    ])
    result = run(author_scenarios(provider, bank, 1, max_attempts=3))
    assert len(result.accepted) == 1 and result.attempts == 2


def test_unparseable_output_does_not_crash(bank):
    provider = Scripted(["not json at all", "still not json"])
    result = run(author_scenarios(provider, bank, 1, max_attempts=2))
    assert result.accepted == []
    assert any("parse failed" in rej.family for rej in result.rejected)


def test_a_provider_error_stops_cleanly(bank):
    class Boom(LLMProvider):
        async def _complete(self, **kw):
            raise ProviderError("down", status=500)

    result = run(author_scenarios(
        Boom(ProviderConfig(name="mock", model="boom", max_retries=0)), bank, 2))
    assert result.accepted == []
    assert any("call failed" in rej.family for rej in result.rejected)


# --------------------------- Committing ---------------------------------- #

def test_merge_renumbers_and_keeps_order(bank):
    accepted = [normalise(one(family="brand_new_family",
                              pretext="claims an insurance refund is pending a small verification"),
                          bank_name="seed", seq=99)]
    merged = merge_into_bank(bank, accepted, bank_name="seed")
    assert len(merged) == len(bank) + 1
    assert [r["scenario_id"] for r in merged][:2] == ["scn-seed-000", "scn-seed-001"]
    assert merged[-1]["family"] == "brand_new_family"


def test_the_merged_bank_writes_and_reloads(tmp_path, bank):
    accepted = [normalise(one(family="brand_new_family",
                              pretext="claims an insurance refund is pending a small verification"),
                          bank_name="seed", seq=99)]
    accepted[0]["provenance"] = {"author_model": "m", "bank_version": 2, "context": "ctx"}
    path = write_bank(merge_into_bank(bank, accepted, bank_name="seed"), tmp_path, "seed", 2)
    reloaded = load_bank(path)
    assert len(reloaded) == len(bank) + 1
    new = [s for s in reloaded.scenarios if s.family == "brand_new_family"][0]
    assert new.provenance.author_model == "m"
    assert new.compliant_outcomes


def test_an_authored_bank_reloads(tmp_path, bank):
    """Provenance written by the author must be readable by Scenario.from_dict.

    A test with hand-built provenance can't catch this: if the author writes one extra key the bank won't load,
    and that only blows up after commit."""
    result = run(author_scenarios(build_provider("mock"), bank, 2, context="parcel fee"))
    path = write_bank(merge_into_bank(bank, result.accepted, bank_name="seed"),
                      tmp_path, "seed", 2)
    reloaded = load_bank(path)
    assert len(reloaded) == len(bank) + len(result.accepted)
    new = [s for s in reloaded.scenarios if s.provenance
           and s.provenance.author_model != "human"]
    assert new and all(s.provenance.rubric_version.startswith("scenario-author/")
                       for s in new)


def test_the_offline_mock_path_produces_usable_records(bank):
    result = run(author_scenarios(build_provider("mock"), bank, 3, context="parcel fee"))
    assert result.accepted
    assert all(r["family"] and r["pretext"] for r in result.accepted)
