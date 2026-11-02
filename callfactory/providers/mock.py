"""Offline deterministic provider.

Not a quality substitute: it exists so the whole pipeline, the validators and the tests run with no key and no network,
and so CI can assert reproducibility. Dialogue comes from a template bank; each agent's structured output is rule-generated.

Not a quality substitute for a real model -- it exists so the whole pipeline,
the validators and the tests can run with no API key and no network, and so CI
can assert reproducibility. Dialogue comes from template banks; evaluator output
comes from keyword matching over the prefix it is shown.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from .base import LLMProvider, ProviderConfig

CALLER_SUSPICIOUS = [
    "Hello, am I speaking with the account holder? I'm calling from [COMPANY_A], it's about an urgent matter on your file.",
    "Right, so -- I do need to deal with this today, the window closes at five and after that it's out of my hands.",
    "I understand the hesitation, but I can't put you through to anyone else, this has to be settled with me on this call.",
    "Ma'am -- sir -- if we don't resolve it now there'll be a charge applied to [ACCOUNT_REF], and I'd rather spare you that.",
    "I just need you to confirm a couple of details so I know I'm talking to the right person. Nothing sensitive.",
    "No, don't call the main line, they'll only route you back to me and you'll have lost the slot.",
    "Look, I've got your file open in front of me. It says [AMOUNT_1] outstanding. I'm trying to help you here.",
    "Between us, I shouldn't really be offering this, so I'd keep it off social media and so on.",
]

CALLER_LEGIT = [
    "Hi, this is [PERSON_1] calling from [COMPANY_A] -- is now an all right time for a quick word?",
    "No rush at all. It's about the appointment on the fourteenth, we've had a cancellation earlier if that suits better.",
    "Of course, take your time. I can hold, or I can ring back this afternoon, whichever is easier.",
    "That's absolutely fine, and if you'd rather check first, the number on your last statement reaches the same desk.",
    "So on our side it's showing [AMOUNT_1], which should match what you've got. Does that look right to you?",
    "I've made a note of that. Nothing else needed from you today.",
    "Understood, I'll take you off the list for this one. Sorry to have bothered you.",
    "Lovely, I'll confirm that in writing and you'll have it by Thursday.",
]

RECEIVER_LINES = [
    "Sorry, who is this again? I didn't catch the company.",
    "Hang on -- the signal's not great, say that again?",
    "I wasn't expecting a call about this, if I'm honest.",
    "Right. Right, I see. And what is it you need from me exactly?",
    "I'd rather not give anything out over the phone, to be blunt.",
    "Can I ring you back? I've got something on the hob.",
    "What do you mean, today? Nobody's written to me about any of this.",
    "Okay, okay. So just so I've got it straight -- you're saying there's money owing?",
    "Hmm. I don't know. Let me think about it.",
    "You know what, I think I'll leave it there. Thanks anyway.",
]

SIGNAL_CUES = {
    "urgency_pressure": ("urgent", "today", "right now", "closes at", "window", "immediately"),
    "authority_claim": ("officer", "department", "on behalf of", "i'm calling from", "my file"),
    "unverified_identity": ("who is this", "didn't catch", "account holder"),
    "secrecy_request": ("between us", "shouldn't really", "off social media", "don't tell"),
    "unusual_payment_channel": ("voucher", "transfer", "card details over", "settle it on this call"),
    "threat_of_consequence": ("charge applied", "cut off", "disconnect", "out of my hands", "penalty"),
    "personal_info_request": ("confirm a couple of details", "date of birth", "your details"),
    "remote_access_request": ("remote", "install", "screen", "let me in"),
    "refusal_to_verify": ("can't put you through", "don't call the main line", "has to be settled with me"),
    "callback_evasion": ("don't call", "lost the slot", "route you back to me"),
    "too_good_to_be_true": ("won", "prize", "guaranteed", "free"),
    "emotional_manipulation": ("spare you", "trying to help you", "i'd hate for"),
}


class MockProvider(LLMProvider):
    """Deterministic given (seed, tag, conversation prefix)."""

    supports_seed = True

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

    def _rng(self, seed: int | None, *parts: str) -> random.Random:
        key = "|".join(parts).encode("utf-8")
        digest = int(hashlib.sha256(key).hexdigest()[:12], 16)
        return random.Random((seed or 0) * 1_000_003 + digest)

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        turn_count = sum(1 for m in messages if m["role"] == "assistant")
        rng = self._rng(seed, tag or "", str(turn_count), last_user[-200:])

        if tag == "scenario_author":
            return self._author_scenarios(last_user)
        if tag == "gatekeeper_craft":
            return self._gatekeeper_craft(last_user, rng)
        if tag == "gatekeeper_detection":
            return self._gatekeeper_detection(last_user, rng)
        if tag == "evaluator_prefix":
            return self._evaluate_prefix(last_user)
        if tag == "evaluator_global":
            # Judged from the transcript it was shown, not from the scenario label.
            hits = json.loads(self._evaluate_prefix(last_user))["signals"]
            return json.dumps(
                {
                    "coherence": round(rng.uniform(3.5, 5.0), 2),
                    "naturalness": round(rng.uniform(3.0, 4.8), 2),
                    "repetition": round(rng.uniform(0.0, 0.3), 2),
                    "role_consistency": True,
                    "notes": "mock provider global evaluation",
                    "blind_is_suspicious": len(hits) >= 3,
                }
            )
        if tag == "receiver":
            line = RECEIVER_LINES[turn_count % len(RECEIVER_LINES)]
            if turn_count >= 4 and rng.random() < 0.08:
                line += " [END_CALL]"
            return line
        bank = CALLER_SUSPICIOUS if "INTENT: suspicious" in system else CALLER_LEGIT
        return bank[turn_count % len(bank)]

    @staticmethod
    def _evaluate_prefix(prefix_text: str) -> str:
        # Only the caller's behaviour counts; a receiver's questions are not signals.
        turns = [t for t in re.split(r"\n(?=\[\d+\] )", prefix_text.strip()) if t.strip()]
        caller_turns = [t.lower() for t in turns if "] CALLER:" in t.upper()]
        lowered = "\n".join(caller_turns)
        signals = sorted(
            name for name, cues in SIGNAL_CUES.items() if any(c in lowered for c in cues)
        )
        score = min(1.0, 0.18 * len(signals))
        tier = "none" if score < 0.1 else "low" if score < 0.35 else "medium" if score < 0.6 else "high"
        # New signals can only come from the final turn, and only if a caller spoke it.
        final = turns[-1].lower() if turns else ""
        new_signals = (
            sorted(name for name in signals if any(c in final for c in SIGNAL_CUES[name]))
            if "] CALLER:" in final.upper()
            else []
        )
        return json.dumps(
            {
                "risk_score": round(score, 3),
                "risk_tier": tier,
                "signals": signals,
                "new_signals_in_last_turn": new_signals,
                "sufficient_evidence": score >= 0.55,
                "rationale": f"mock: {len(signals)} cue group(s) matched",
            }
        )

    # --------------------------------------------------------------------- #
    # Structured output for the newer agents
    # --------------------------------------------------------------------- #

    @staticmethod
    def _slug(text: str) -> str:
        """Squeeze a family prefix out of the human context, to test whether the context is actually used."""
        words = re.findall(r"[a-z]{3,}", text.lower())
        stop = {"the", "and", "for", "with", "scenario", "scenarios", "generate",
                "exactly", "produce", "new", "call", "phone", "about"}
        picked = [w for w in words if w not in stop][:2]
        return "_".join(picked) if picked else "generic"

    #: Premise templates that deliberately differ from each other: near-identical pretexts aren't what the author agent should produce,
    #: and the author's near-duplicate check would (rightly) block them, so the offline pipeline wouldn't run.
    MOCK_PRETEXTS = [
        "says a held item needs a small release charge settled before it is sent on",
        "claims a benefits record is incomplete and must be confirmed over the phone",
        "offers a closing position in a fund the receiver never enquired about",
        "warns that a utility account is about to be cut off over an unpaid balance",
        "reports suspicious activity and wants the account secured during this call",
        "says a relative is in trouble somewhere and needs help arranged quietly",
        "confirms an appointment slot and reads back the details on file",
        "follows up on a repair visit and asks which morning suits",
    ]

    def _author_scenarios(self, user_msg: str) -> str:
        m = re.search(r"exactly (\d+)", user_msg)
        count = int(m.group(1)) if m else 3
        ctx = ""
        cm = re.search(r"CONTEXT:\s*(.+)", user_msg)
        if cm:
            ctx = cm.group(1).strip()
        base = self._slug(ctx) if ctx else "mock_family"
        out = []
        for i in range(count):
            is_scam = i % 3 != 2
            hard_neg = (not is_scam) and i % 2 == 1
            out.append({
                "category": "scam" if is_scam else "legitimate",
                "family": f"{base}_{i}",
                "pretext": self.MOCK_PRETEXTS[i % len(self.MOCK_PRETEXTS)],
                "org_kind": "a fictional service company",
                "expected_signs": (
                    ["urgency_pressure", "authority_claim"] if is_scam
                    else (["urgency_pressure"] if hard_neg else [])
                ),
                "plausible_outcomes": [
                    "receiver refuses and ends the call",
                    "receiver stalls and says they will check first",
                ],
                "hard_negative": hard_neg,
                "disambiguator": (
                    "genuine: welcomes a callback on a number the receiver looks up"
                    if hard_neg else ""
                ),
                "late_onset": is_scam and i % 4 == 0,
                "notes": "mock-authored",
            })
        return json.dumps({"scenarios": out})

    @staticmethod
    def _gatekeeper_craft(user_msg: str, rng: random.Random) -> str:
        """Realism / Consistency. Ground truth is not visible here (R6.3 segmented masking)."""
        turns = [t for t in re.split(r"\n(?=\[\d+\] )", user_msg.strip()) if t.strip()]
        # Crude proxy: the more spoken-language markers, the more it sounds like a real call
        markers = sum(user_msg.lower().count(k) for k in ("--", " uh", "sorry", "hang on", "i mean"))
        realism = 2 if markers == 0 else 3 if markers < 3 else 4 if markers < 8 else 5
        return json.dumps({
            "realism": realism,
            "consistency": rng.choice([4, 4, 5]),
            "inconsistencies": [],
            "realism_notes": f"mock: {markers} spoken-language markers / {len(turns)} turns",
        })

    @staticmethod
    def _gatekeeper_detection(user_msg: str, rng: random.Random) -> str:
        """Detection Value. Prefix labels may be seen here (R6.3)."""
        early = "earliest_detection_index: 0" in user_msg or "earliest_detection_index: 1" in user_msg
        none = "earliest_detection_index: None" in user_msg
        is_scam_hint = "signals" in user_msg and not none
        if none and "is_scam: True" in user_msg:
            value = 1          # scam with no signal at all -> no value
        elif early:
            value = 2          # decidable at T0/T1 -> no discriminative value
        elif is_scam_hint:
            value = 4
        else:
            value = 3
        return json.dumps({
            "detection_value": value,
            "detection_notes": "mock: scored by how early the detection point is",
        })
