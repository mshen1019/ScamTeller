"""Hand-written seed scenario bank.

The scenario author agent (component 1) is there to **extend** the bank, not build it from scratch: R1.1 requires existing scenarios to be passed in as
dedup context, so there has to be a starting point. This is it: 13 scam families (one of which deliberately shows no decidable evidence), 5 ordinary legitimate calls,
7 hard negatives.

A hard negative's ``disambiguator`` is mandatory (R1.4): it states how it behaves like a real organisation when challenged.
Without it, a hard negative degrades into an ordinary negative and loses its value against false alarms.

Every scam family's signal combination should overlap with at least one hard negative, otherwise the model learns that signal
by itself as grounds for an alarm. ``review-bank`` reports the gaps.
"""

from __future__ import annotations

from typing import Any

#: Outcomes where the receiver is **won over**. Listed separately because the sampler allocates them by persona skepticism,
#: and they go into Scenario.compliant_outcomes for bench slicing.
#:
#: Each one ends at the moment of "agreeing" (P8): never any steps, numbers, transfers or installs.
#: Act out the action itself and this data becomes a script someone could follow.
COMPLIANT_SCAM_OUTCOMES = [
    "receiver is talked round and agrees to do what is asked; the call ends on that "
    "agreement, before anything is actually done",
    "receiver cooperates readily throughout and the call ends with the caller saying "
    "they will ring back later to finish it off",
    "receiver hesitates, is reassured by the caller, and agrees; the call ends there",
]

SCAM_OUTCOMES = [
    "receiver grows suspicious and ends the call without giving anything up",
    "receiver insists on calling back on a number they look up themselves",
    "receiver goes along part of the way, then stalls and says they will check with someone first",
    "receiver stays on the line, confused, and the call ends unresolved",
    "receiver challenges the caller directly and the caller keeps pushing until the receiver hangs up",
    "receiver asks a question the caller cannot answer and the caller abruptly wraps up",
    *COMPLIANT_SCAM_OUTCOMES,
]

#: Legitimate calls need resisting outcomes too, otherwise "receiver resists" still points one way, at scam.
#: Especially the last two: the receiver's behaviour is **word-for-word identical** to some scam outcomes, and only the caller's reaction differs,
#: which forces the discriminating information back onto the caller. That's exactly what R5.9 wants.
LEGIT_OUTCOMES = [
    "the business is concluded and both sides sign off normally",
    "the receiver asks to be called back at a better time",
    "the receiver declines politely and the caller accepts it",
    "a misunderstanding is cleared up and the call ends cordially",
    "the receiver verifies the caller through a detail only the real organisation would know, then proceeds",
    "the receiver is suspicious throughout and ends the call without dealing with it; the caller does not push",
    "receiver insists on calling back on a number they look up themselves, and the caller agrees readily",
    "receiver asks a question the caller cannot answer, and the caller offers to find out and call back",
]

SEED_SCENARIOS: list[dict[str, Any]] = [
    # ---------------------------- scam ----------------------------------- #
    {
        "family": "tech_support_impersonation",
        "pretext": "claims the receiver's home computer is sending out infected traffic",
        "org_kind": "a computer security company",
        "expected_signs": ["authority_claim", "remote_access_request", "urgency_pressure", "unverified_identity"],
        "late_onset": True,
    },
    {
        "family": "bank_fraud_alert_spoof",
        "pretext": "claims a suspicious charge was just attempted on the receiver's account",
        "org_kind": "a bank's fraud team",
        "expected_signs": ["urgency_pressure", "personal_info_request", "callback_evasion", "threat_of_consequence"],
        "late_onset": True,
    },
    {
        "family": "delivery_redelivery_fee",
        "pretext": "says a parcel is held and a small handling fee must be settled by phone",
        "org_kind": "a courier company",
        "expected_signs": ["unusual_payment_channel", "urgency_pressure", "inconsistent_details"],
        "late_onset": True,
    },
    {
        "family": "utility_disconnection_threat",
        "pretext": "warns the power will be cut within the hour over an unpaid balance",
        "org_kind": "a utility company",
        "expected_signs": ["threat_of_consequence", "urgency_pressure", "unusual_payment_channel", "refusal_to_verify"],
        "late_onset": True,
    },
    {
        "family": "prize_notification",
        "pretext": "announces the receiver has won a regional draw they do not remember entering",
        "org_kind": "a promotions agency",
        "expected_signs": ["too_good_to_be_true", "personal_info_request", "urgency_pressure"],
    },
    {
        "family": "family_emergency_pretext",
        "pretext": "claims to be helping a relative who is in trouble and cannot talk right now",
        "org_kind": "no organisation, a stranger acting as an intermediary",
        "expected_signs": ["emotional_manipulation", "secrecy_request", "urgency_pressure", "unverified_identity"],
    },
    {
        "family": "advance_fee_job_offer",
        "pretext": "offers remote work that starts as soon as an onboarding cost is cleared",
        "org_kind": "a recruitment agency",
        "expected_signs": ["too_good_to_be_true", "unusual_payment_channel", "inconsistent_details"],
        "late_onset": True,
    },
    {
        "family": "crypto_investment_pressure",
        "pretext": "pitches a digital-asset position closing today that the receiver never enquired about",
        "org_kind": "a digital-asset investment desk",
        "expected_signs": ["too_good_to_be_true", "urgency_pressure", "secrecy_request", "unusual_payment_channel"],
        "late_onset": True,
    },
    {
        "family": "investment_pressure",
        "pretext": "pitches a closing-today position in a fund the receiver never enquired about",
        "org_kind": "an investment firm",
        "expected_signs": ["too_good_to_be_true", "urgency_pressure", "callback_evasion"],
        "late_onset": True,
    },
    {
        "family": "government_benefit_verification",
        "pretext": "says a benefits record is incomplete and must be confirmed over the phone today",
        "org_kind": "a public benefits office",
        "expected_signs": ["authority_claim", "personal_info_request", "threat_of_consequence", "refusal_to_verify"],
        "late_onset": True,
    },
    {
        "family": "subscription_refund_overpayment",
        "pretext": "claims a subscription was double-billed and a refund must be arranged live on the call",
        "org_kind": "a software retailer",
        "expected_signs": ["remote_access_request", "inconsistent_details", "urgency_pressure"],
        "late_onset": True,
    },
    {
        "family": "property_management_arrears",
        "pretext": "says a service charge is overdue and access to the building may be affected",
        "org_kind": "a property management company",
        "expected_signs": ["threat_of_consequence", "authority_claim", "unusual_payment_channel"],
        "late_onset": True,
    },
    # ------------------------ legitimate, plain -------------------------- #
    {
        "family": "dental_appointment_reminder",
        "pretext": "confirms next week's check-up and offers to move it",
        "org_kind": "a dental practice",
        "category": "legitimate",
    },
    {
        "family": "pharmacy_refill_ready",
        "pretext": "lets the receiver know a repeat prescription is ready for collection",
        "org_kind": "a pharmacy",
        "category": "legitimate",
    },
    {
        "family": "market_research_survey",
        "pretext": "asks whether the receiver has five minutes for a short consumer survey",
        "org_kind": "a research agency",
        "category": "legitimate",
    },
    {
        "family": "school_absence_check",
        "pretext": "checks why a pupil was marked absent this morning",
        "org_kind": "a school office",
        "category": "legitimate",
    },
    {
        "family": "contractor_scheduling",
        "pretext": "arranges a window for a boiler service visit the receiver already booked",
        "org_kind": "a maintenance contractor",
        "category": "legitimate",
    },
    # ---------------------- legitimate, hard negative -------------------- #
    {
        "family": "genuine_bank_fraud_team",
        "pretext": "flags a real declined transaction and asks the receiver to confirm whether it was theirs",
        "org_kind": "the receiver's own bank, calling for real",
        "category": "legitimate",
        "hard_negative": True,
        "expected_signs": ["urgency_pressure", "unverified_identity", "personal_info_request"],
        "disambiguator": "happily waits while the receiver hangs up and calls back on the number from their own card; never asks the receiver to prove their identity in order to proceed",
    },
    {
        "family": "genuine_utility_arrears",
        "pretext": "gives notice of a real overdue balance and the date supply would be interrupted",
        "org_kind": "the receiver's actual energy supplier",
        "category": "legitimate",
        "hard_negative": True,
        "expected_signs": ["threat_of_consequence", "urgency_pressure"],
        "disambiguator": "quotes the account figure from the receiver's last bill, offers a payment plan, takes no payment on the call and directs the receiver to their own online account",
    },
    {
        "family": "genuine_it_helpdesk",
        "pretext": "follows up on a support ticket the receiver opened at work this morning",
        "org_kind": "the receiver's own employer's IT helpdesk",
        "category": "legitimate",
        "hard_negative": True,
        "expected_signs": ["remote_access_request", "authority_claim"],
        "disambiguator": "cites the ticket number the receiver filed and the colleague who escalated it; offers to reschedule through the internal portal if the receiver would rather verify first",
    },
    {
        "family": "genuine_courier_customs",
        "pretext": "explains a real import charge on a parcel the receiver is expecting",
        "org_kind": "a courier's customs desk",
        "category": "legitimate",
        "hard_negative": True,
        "expected_signs": ["unusual_payment_channel", "urgency_pressure"],
        "disambiguator": "gives the tracking reference the receiver can already see, and directs payment through the tracking page at the receiver's own pace rather than on the call",
    },
    {
        "family": "genuine_charity_renewal",
        "pretext": "thanks the receiver for a lapsed monthly donation and asks about renewing",
        "org_kind": "a charity the receiver has given to before",
        "category": "legitimate",
        "hard_negative": True,
        "expected_signs": ["emotional_manipulation"],
        "disambiguator": "takes no for an answer the first time it is given and offers to remove the receiver from the calling list",
    },
    {
        "family": "genuine_prize_draw_followup",
        "pretext": "confirms a minor prize from a draw the receiver really did enter at a local event",
        "org_kind": "a local business that ran the draw",
        "category": "legitimate",
        "hard_negative": True,
        "expected_signs": ["too_good_to_be_true", "personal_info_request"],
        "disambiguator": "nothing to pay, collection in person at the shop, and the caller can name the event and date the receiver attended",
    },
    # --- scam, but no decidable evidence at any point (tests restraint, eval only by default) --- #
    {
        "family": "reconnaissance_detail_confirmation",
        "pretext": "says it is updating records and only ever tries to confirm that the "
                   "receiver still lives at the address, then rings off",
        "org_kind": "an unnamed records department",
        "category": "scam",
        # Not even one weak signal declares a strong one: this scenario is defined by "the listener has nothing to go on"
        "expected_signs": ["unverified_identity"],
        "no_observable_evidence": True,
        "eval_only": True,
        "plausible_outcomes": [
            "receiver confirms the address without thinking much of it and the call ends",
            "receiver asks who is calling, gets a vague answer, and the call ends politely",
            "receiver declines to confirm anything and the caller rings off without pushing",
        ],
        "notes": "The caller is casing the target, but the call contains no decidable evidence — alarming on this one is a false alarm. "
                 "The ground-truth earliest_detection_index is expected to be None and is excluded from latency statistics",
    },
    {
        "family": "genuine_broker_compliance_check",
        "pretext": "runs a routine suitability review on an investment account the receiver actually holds",
        "org_kind": "the receiver's own brokerage, compliance team",
        "category": "legitimate",
        "hard_negative": True,
        "expected_signs": ["too_good_to_be_true", "authority_claim", "personal_info_request"],
        "disambiguator": "makes no recommendation and sells nothing; reads back holdings the receiver can see in their own statement and offers to complete the review in writing instead",
        "notes": "Pairs with crypto_investment_pressure / investment_pressure, so the model does not learn 'mentions investing' as an alarm signal",
    },
]


def build_seed_scenarios(bank_name: str = "seed") -> list[dict[str, Any]]:
    """Fill in default fields and produce scenario records ready for the bank."""
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(SEED_SCENARIOS):
        category = raw.get("category", "scam")
        outcomes = list(
            raw.get("plausible_outcomes")
            or (SCAM_OUTCOMES if category == "scam" else LEGIT_OUTCOMES)
        )
        rec: dict[str, Any] = {
            "scenario_id": f"scn-{bank_name}-{i:03d}",
            "category": category,
            "family": raw["family"],
            "pretext": raw["pretext"],
            "org_kind": raw["org_kind"],
            "expected_signs": list(raw.get("expected_signs", [])),
            "plausible_outcomes": outcomes,
            "hard_negative": raw.get("hard_negative", False),
            "disambiguator": raw.get("disambiguator", ""),
            "late_onset": raw.get("late_onset", False),
            "compliant_outcomes": list(raw.get("compliant_outcomes") or (
                [o for o in COMPLIANT_SCAM_OUTCOMES if o in outcomes]
                if category == "scam" else []
            )),
            "no_observable_evidence": raw.get("no_observable_evidence", False),
            "eval_only": raw.get("eval_only", False),
            "notes": raw.get("notes", ""),
        }
        out.append(rec)
    return out
