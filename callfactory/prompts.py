"""Prompt construction (where the information boundaries of components 3 / 4 live).

This layer is the only enforcement point for P1 (information asymmetry), so it deliberately touches no provider and no I/O:
each of the three views comes from a pure function, so boundary violations are caught right here by unit tests.

* the caller sees everything about the scenario + its own persona + the outcome intent + the target round count
* the receiver sees **only** its own persona, plus what has been said aloud
* the evaluator (component 5) sees only ``utterances[:k+1]``

None of ``category`` / ``family`` / ``pretext`` / ``expected_signs`` may appear in the receiver prompt
(R4.1); otherwise its wariness becomes a restatement of the label, and the prefix labels lose their meaning with it.
"""

from __future__ import annotations

import hashlib

from typing import Any

from .schema import (
    RISK_SIGNALS,
    SCENARIO_SIGNALS,
    CallerPersona,
    GenerationTask,
    ReceiverPersona,
    Utterance,
)

PLACEHOLDER_RULE = (
    "Use bracketed placeholders for every proper noun and every number that "
    "would identify someone: [PERSON_1], [PERSON_2], [COMPANY_A], [COMPANY_B], "
    "[PERSON_3], [PHONE_1], [ACCOUNT_REF], [AMOUNT_1], [AMOUNT_2], [CITY_1], "
    "[URL_1], [CODE_1]. "
    "Never invent a real company, agency, bank, brand or person."
)

SAFETY_RULE = (
    "SAFETY BOUNDARIES (these are absolute):\n"
    "- Everything is fictional. No real organisations, people, phone numbers, "
    "domains, payment rails or account identifiers.\n"
    "- Stay at the level of observable conversational behaviour: urgency, "
    "authority claims, vagueness, evasion when challenged. Do NOT produce "
    "working fraud procedure -- no usable steps for extracting a credential, "
    "one-time code, card number or remote-access session, and no payment "
    "instructions a listener could follow.\n"
    "- Any sensitive value stays a placeholder; never read out digits.\n"
    "- **If the other person agrees to do what is asked, the call ENDS on that "
    "agreement.** Never walk through how anything is done: no steps to follow, no "
    "digits, no codes read out, no transfer, download or install procedure. The "
    "agreement itself is what the transcript records; the mechanics are out of scope.\n"
    "- Never mention datasets, labels, models, prompts or that this is generated."
)

SPEECH_RULE = (
    "Write spoken English, not prose: contractions, fillers (uh, I mean, sort of), "
    "self-corrections, sentences that break off with --, the occasional single-word "
    "turn. No stage directions, no asterisks, no quotation marks, no speaker label. "
    "Output only the words spoken in this one turn, normally one to four sentences."
)


# --------------------------------------------------------------------------- #
# Caller (component 3)
# --------------------------------------------------------------------------- #

def caller_system_prompt(task: GenerationTask) -> str:
    """R3.1: scenario + persona + outcome intent + target round count, all of it."""
    scn = task.scenario
    persona: CallerPersona = task.caller_persona
    intent = "suspicious" if scn.category == "scam" else "legitimate"
    signs = ", ".join(scn.expected_signs) or "none -- this is a straightforward call"

    lines = [
        "You are role-playing ONE SIDE of a fictional phone call. The transcripts "
        "train scam-detection models, so the dialogue must be realistic and the two "
        "sides must stay independent: you never see the other side's instructions.",
        "",
        "You are the CALLER. You speak only as the caller.",
        "",
        # The mock provider picks its template bank from this line; if the wording changes, update providers/mock.py too.
        f"SCENARIO INTENT: {intent}",
        f"You present yourself as {persona.role} at [COMPANY_A], {scn.org_kind}.",
        f"Reason for the call: you {scn.pretext}.",
        "",
        "WHO YOU ARE ON THE PHONE:",
        f"- Speaking style: {persona.speech_style}",
        f"- How you apply pressure: {persona.pressure_profile}",
        f"- When the other person challenges you, your manner: {persona.challenge_response}",
        f"- How practised you are: {persona.competence}",
    ]
    if persona.competence.startswith("green"):
        # Seen in testing: with only "makes mistakes", inexperienced personas played the scenario as achieving nothing,
        # and samples became poisoned labels: "the label says scam, the content has no scam". An inexperienced scammer is still a scammer.
        lines.append(
            "  Being inexperienced, you occasionally contradict a detail you gave "
            "earlier, or have to go back and correct yourself. You are still "
            "pursuing the call's purpose throughout -- clumsiness slows you down "
            "and makes you leak inconsistencies, it does not make you give up. "
            "Whatever else happens, the behaviour listed under warning signs must "
            "actually appear in what you say."
        )
    if persona.style_signals:
        lines.append(
            "  You habitually fall back on the same rehearsed phrasing, and when "
            "interrupted you return to the opener rather than answering."
        )

    lines += ["", f"Warning signs to display when it arises naturally: {signs}"]

    if scn.category == "scam":
        if task.late_onset:
            # R3.3: slow burn must really be slow. Early warning only has value if the first few turns show nothing unusual.
            lines.append(
                "SLOW BURN: open the call as an entirely ordinary, unremarkable "
                "enquiry. For the first few exchanges there must be nothing a "
                "listener could object to. Let the pressure appear only later, once "
                "there is some rapport."
            )
        else:
            lines.append(
                "The pressure is present early, though you still sound professional."
            )
        lines.append(
            "You are not a cartoon. You sound plausible; the warning signs leak out "
            "through evasion and insistence, never through announcing bad intent."
        )
    else:
        lines.append(
            "You are genuinely who you say you are, and you behave like it."
        )
        if scn.hard_negative:
            # The whole value of a hard negative is in this line: it looks like a scam on the surface but doesn't behave like one when challenged.
            lines.append(
                "This call will sound alarming on the surface. What separates you "
                f"from a fraudster is how you behave when doubted: {scn.disambiguator}"
            )
    if scn.notes:
        lines.append(f"Note about this kind of call: {scn.notes}")

    compliant = task.outcome in scn.compliant_outcomes
    if compliant:
        lines.append(
            "The other person is going to be won over on this call. Do not soften "
            "what you are doing to get there -- the warning signs still appear; it "
            "is simply that this particular person does not act on them. Stop the "
            "moment they agree."
        )

    lines += [
        "",
        f"How the call should end up: {task.outcome}",
        f"Aim for roughly {task.target_rounds} exchanges before that resolution.",
        "",
        PLACEHOLDER_RULE,
        "",
        SAFETY_RULE,
        "",
        SPEECH_RULE,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Receiver (component 4)
# --------------------------------------------------------------------------- #

def receiver_system_prompt(task: GenerationTask) -> str:
    """R4.1: persona only. No scenario field may appear here."""
    persona: ReceiverPersona = task.receiver_persona
    exposure = {
        "none": "You have no particular experience of calls like this.",
    }.get(
        persona.prior_exposure,
        f"Relevant past experience of yours: {persona.prior_exposure}. "
        "It colours how you react, but you do not lecture anyone about it.",
    )

    return "\n".join([
        "You are role-playing ONE SIDE of a fictional phone call. You are the person "
        "who PICKED UP the phone. You did not expect this call and you have no idea "
        "who is on the line.",
        "",
        # R4.2: a preset stance would make the receiver a restater of the label rather than a source of evidence.
        "You know nothing whatsoever about who is calling or why, beyond what has "
        "actually been said aloud on this call. Do not assume good faith and do not "
        "assume bad faith. Form your view only from what you hear, as it is said.",
        "",
        "WHO YOU ARE:",
        f"- How sceptical you are by nature: {persona.skepticism}",
        f"- How comfortable you are with technology: {persona.tech_familiarity}",
        f"- How you are feeling right now: {persona.emotional_state}",
        f"- What you are in the middle of doing: {persona.situation}",
        f"- How you talk: {persona.speech_style}",
        f"- {exposure}",
        "",
        "You may mishear, get confused, ask who is calling, ask how you can verify "
        "any of this, wander off the subject, give in, stall, get irritated, or "
        "decide you have had enough. React like a real person in your situation on "
        "an ordinary afternoon, not like an expert and not like a trained listener.",
        "",
        # Seen in testing: without this section, 20/20 conversations were hung up by the receiver as soon as they were long enough, and slow-burn scenarios never got to burn slowly.
        "You are not trying to get off the phone. People stay on calls they find "
        "odd -- out of politeness, curiosity, confusion, or because they want to "
        "know what this is actually about. Keep asking, keep pushing back, let "
        "yourself be talked round and then doubt it again.",
        "",
        "End the call only when you have genuinely decided it is finished: the "
        "matter is settled, or you have concluded you will not deal with it this "
        "way and you say so. That the call is taking a while is not a reason. "
        "When you do end it, say your goodbye and finish your turn with the exact "
        "token [END_CALL].",
        "",
        "Never reveal real personal data. If you give a detail at all, give the "
        "placeholder form: [PERSON_1], [ACCOUNT_REF], [AMOUNT_1], [CITY_1], [PHONE_1].",
        "",
        SAFETY_RULE,
        "",
        SPEECH_RULE,
    ])


# --------------------------------------------------------------------------- #
# Per-turn user message
# --------------------------------------------------------------------------- #

def turn_user_message(
    directive: str | None,
    *,
    is_opening: bool,
    is_last: bool,
    role: str,
) -> str:
    """R3.7 / R3.8: per-turn colour and wrap-up cues. Turns with a None directive are about 1/4."""
    parts: list[str] = []
    if is_opening:
        parts.append("The receiver has just picked up. Speak first.")
    else:
        parts.append(f"Your turn as the {role}. Respond to what was just said.")
    if directive:
        parts.append(f"Colour for this turn: {directive}")
    if is_last:
        parts.append(
            "This is the final stretch of the call -- bring it to its natural close "
            "in this turn or the next."
        )
    return " ".join(parts)


def format_transcript(utterances: list[Utterance]) -> str:
    return "\n".join(f"[{u.index}] {u.speaker.upper()}: {u.text}" for u in utterances)


# --------------------------------------------------------------------------- #
# Evaluator (component 5)
# --------------------------------------------------------------------------- #

#: Blind labelling: these two prompts structurally have no place for the scenario; the evaluator gets only what was said aloud (R5.5).
#: So "don't tell it the answer" doesn't rely on discipline; there's simply no field to put the answer in.

PREFIX_EVAL_SYSTEM = f"""You annotate scam risk in phone calls for an early-warning detector.

You are shown a PARTIAL transcript: turns 1..k of a call that may continue. Judge
ONLY the evidence present in those turns. Do not guess what comes next, and do not
let a hunch about the call's genre stand in for evidence actually spoken.

Score the CALLER's behaviour, and only the caller's. The person who picked up may
be suspicious, rude or alarmed: that is their reaction, not a risk signal, and it
never raises the score. Equally -- and this is missed more often -- a person who
is trusting, cheerful, chatty or entirely cooperative never LOWERS the score. The
risk lives in what the caller does. Whether this particular listener happens to
believe it changes nothing.

Signal vocabulary (use these exact strings, nothing else):
{", ".join(RISK_SIGNALS)}

Return ONLY a JSON object, no prose, no code fence:
{{"risk_score": <float 0..1>,
  "risk_tier": "none"|"low"|"medium"|"high",
  "signals": [<every signal evidenced anywhere in the prefix>],
  "new_signals_in_last_turn": [<signals first evidenced by the final turn shown>],
  "sufficient_evidence": <true|false>,
  "rationale": "<at most 240 characters, point at the evidence>"}}

SETTING "sufficient_evidence" -- this is the field that decides the detection
point, so it needs anchors rather than instinct:

true  = a careful listener who had heard ONLY these turns would already have
        reasonable grounds to treat the call as likely fraudulent and to stop
        cooperating. It does NOT require certainty, and it does NOT require the
        caller to have asked for money yet.
false = the call is so far only unusual, unexpected, pushy, or badly explained.

NOT sufficient on its own: urgency; an unexpected call; an unfamiliar
department; a caller who sounds disorganised or nervous; a receiver who is
suspicious or hostile. Legitimate organisations do all of this.

SUFFICIENT as soon as THE CALLER has done any one of the things in either group
below. Read the subject of every line as the caller: what the person who picked
up does is never what makes evidence sufficient.

GROUP A -- things the caller does unprompted. These need NOTHING from the
listener, so they hold even when the listener is perfectly cooperative:
- on a call the listener did not initiate, the caller wants a payment, a fee or a
  charge settled now or today, by a route the caller chose
- the caller asks for a one-time code, a password, a card or account detail, or
  remote access to a device, at all
- the caller wants money or details for something whose origin it cannot state
  (it cannot say who sent the parcel, who filed the claim, which transaction it
  means) and keeps going anyway
- the caller asks the listener to keep the call from family, a colleague, or
  their bank
- the caller threatens a consequence -- suspension, charge, legal step, a lost
  slot -- if the matter is not settled during this call

GROUP B -- things that only surface when the listener pushes back. Absence of
these means nothing if the listener never pushed:
- the caller deflected, stalled or refused when the listener asked to verify
  through a channel of the listener's choosing (a callback number, the branch,
  the main line, a letter, ringing back later, involving a relative).
  NOT this line: the listener declining the caller's own suggested channel, or
  the listener deciding to go and check independently. That is the listener
  being careful; it changes nothing either way.
- the caller contradicted itself about who it is or why it is calling, and let
  the contradiction stand
- the caller insisted that nobody else and no later time can deal with the matter
- the caller required personal details before it would say what the call is about

Also sufficient when two or more of the weaker cues hold together, for example
urgency plus an authority claim plus refusing to say what the call concerns.

Before you answer, check three things:
1. Every signal you list must be something THE CALLER said or did. If you are
   describing the listener's caution, refusal, or plan to verify, drop it.
2. "sufficient_evidence" does NOT wait for the caller to ask for money, a code
   or access. A caller that has already evaded verification, threatened a
   consequence, or contradicted its own purpose is sufficient on its own, even
   if it never gets round to asking for anything.
3. If your own rationale describes the caller pressing for payment, withholding
   what the call is about, or asking for a credential, then
   "sufficient_evidence" is true -- however relaxed, friendly or convinced the
   person on the other end sounds.
"""

GLOBAL_EVAL_SYSTEM = """You rate the quality of a synthetic phone-call transcript.

You see the whole call. Judge craft, not risk. Return ONLY a JSON object:
{"coherence": <1..5, do the turns follow from each other>,
 "naturalness": <1..5, does it sound like speech rather than writing>,
 "repetition": <0..1, how much of the call is near-restatement of earlier turns>,
 "role_consistency": <true|false, does each side stay in its own role throughout>,
 "notes": "<at most 240 characters>",
 "blind_is_suspicious": <true|false, your own read of whether the caller was a fraudster>}

"blind_is_suspicious" is your independent judgement from the transcript alone. You
have not been told the answer, and it is used only to cross-check label quality --
never as the detection label itself.
"""


def prefix_eval_user_message(utterances: list[Utterance]) -> str:
    return (
        f"Partial transcript, turns 1..{len(utterances)}. The call may or may not "
        f"continue.\n\n{format_transcript(utterances)}\n\nAnnotate this prefix."
    )


def global_eval_user_message(utterances: list[Utterance]) -> str:
    return f"Complete transcript:\n\n{format_transcript(utterances)}\n\nRate it."


# --------------------------------------------------------------------------- #
# rubric version
# --------------------------------------------------------------------------- #

#: Version of the prefix-labelling rubric. **Bump it whenever PREFIX_EVAL_SYSTEM changes**.
#:
#: The hand-written version states intent; the ``rubric_fingerprint()`` hash catches oversights: if the prompt changes and the
#: version isn't bumped, two records have the same version but different fingerprints, which is easy to spot.
#:
#: Why it must be recorded in every record: the same transcript can get different labels under different rubrics
#: (in testing: the same 20 conversations, one rubric change, detection rate 0.80 -> 0.90). Without the version,
#: mixing two batches into one training set gives systematically contradictory labels, and nothing warns about it.
PREFIX_EVAL_RUBRIC_VERSION = "prefix-eval/v3"

#: v1 = only says "what doesn't count as evidence"; v2 = adds positive anchors; v3 = anchors name their subject + proactive/reactive groups.
RUBRIC_CHANGELOG = {
    "prefix-eval/v1": "only listed insufficient cases, vague positive anchors; scam detection rate 0.4 in testing",
    "prefix-eval/v2": "added six positive anchors; detection rate 0.8, but the anchor sentences confused their subject, "
                      "counting the receiver's refusals as the caller's evasion; hard negative false alarms 0.25",
    "prefix-eval/v3": "each anchor names its subject, split into proactive/reactive groups, added 'receiver cooperation doesn't lower the score'"
                      "; on fixed dialogues detection rate 0.9, false alarms 0, two detection points earlier, none lost",
}


def rubric_fingerprint() -> str:
    """Short hash of the current rubric text. Covers both evaluator prompts."""
    digest = hashlib.sha256(
        (PREFIX_EVAL_SYSTEM + "\x00" + GLOBAL_EVAL_SYSTEM).encode("utf-8")
    )
    return digest.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Scenario author (component 1)
# --------------------------------------------------------------------------- #

#: Rubric version for the scenario author. Bump it when the prompt below changes; committed provenance records it.
#: v2: adds the rule "a family must not be named after the signals it is meant to show". Under v1, 4 of 30 candidates
#: wrote expected_signs into the family name, which hands the model a family -> label shortcut.
SCENARIO_AUTHOR_VERSION = "scenario-author/v2"

SCENARIO_AUTHOR_SYSTEM = f"""You write scenario records for a synthetic phone-call
dataset used to train early-warning scam detectors. You do not write dialogue --
you write the premise a later agent will act out.

Each record is a structured object, never prose. Return ONLY JSON:
{{"scenarios": [{{
  "category": "scam" | "legitimate",
  "family": "<snake_case, 2-4 words, the KIND of call>",
  "pretext": "<one clause, third person, why this caller is ringing>",
  "org_kind": "<the kind of organisation the caller claims to be, generic>",
  "expected_signs": [<from the vocabulary below; [] for an ordinary legitimate call>],
  "hard_negative": <true|false>,
  "disambiguator": "<required when hard_negative: how it behaves when doubted>",
  "late_onset": <true|false, scam only>,
  "notes": "<one line: what this family is for, or what it pairs with>"
}}]}}

SIGNAL VOCABULARY -- use these exact strings and nothing else:
{", ".join(SCENARIO_SIGNALS)}
(There is one more signal, script_like_repetition, which belongs to a caller's
speaking style rather than to the call's purpose. Never put it in expected_signs.)

WHAT MAKES A GOOD RECORD

1. The `family` names a kind of call, not an instance: `parcel_customs_fee`, not
   `woman_gets_parcel_call`. No brand, company, agency, product or person names
   anywhere in the record -- the caller claims a generic KIND of organisation.
   It must also NOT be named after the warning sign it demonstrates.
   `caller_swaps_story_under_questioning`, `mystery_official_wont_say_who`,
   `refuses_any_verification` all restate `expected_signs` in the name, and a
   model can then read the label off the family instead of off the call. Name
   what the call is ABOUT (`tax_refund_claim`, `safe_account_transfer`) and
   leave the behaviour in `expected_signs`. One exception: a scam pattern whose
   recognised name IS the behaviour (`insider_investigation_secrecy`) may keep
   it, and then `notes` must say that is why.
2. The `pretext` is the reason for the call in one clause, specific enough that
   two different writers would stage the same call: "says a parcel is held at a
   depot and a small customs fee must be paid before it is released".
3. `expected_signs` lists what a listener could OBSERVE, not what is true. A scam
   whose signs are only `authority_claim` is not detectable; give the ones the
   caller's behaviour will actually show.
4. **Hard negatives are the most valuable records you can write.** A hard
   negative is a GENUINE call that sounds alarming: a real bank's fraud team
   ringing unprompted, a real IT desk asking to connect to a machine, a real
   surgery confirming a date of birth. It must declare the signs it will trip,
   AND a `disambiguator` saying what it does when doubted -- something a
   fraudster would not do: welcomes a callback on a number the listener looks up,
   quotes a reference only the real organisation holds, offers to put it in
   writing, proceeds without the listener proving anything. Without that, it is
   just an ordinary negative and worth far less.
5. `late_onset` (scam only) means the call can be played with an entirely
   ordinary opening and pressure appearing several exchanges later. Most scams
   can; set it when the premise genuinely allows a dull first minute.
6. An ordinary legitimate call has `expected_signs: []`, `hard_negative: false`:
   an appointment reminder, a delivery slot, a survey. These stop the detector
   from treating any unexpected call as suspicious, so some are needed.

SAFETY BOUNDARY (absolute)
Write the premise at the level of observable conversational behaviour. Never
write a pretext that embeds a working fraud procedure: no real payment rails,
no steps for extracting a credential or one-time code, no named transfer or
gift-card mechanics, no instructions a reader could follow. "Asks for a fee by
an unusual route" is a premise; naming the route and the steps is not.
No real organisations and no real people, anywhere.
"""


def scenario_author_user_message(
    existing_families: list[str],
    existing_pretexts: list[str],
    count: int,
    *,
    context: str = "",
    gaps: dict[str, Any] | None = None,
) -> str:
    """R1.1: existing scenarios must be passed in as dedup context, otherwise the same kinds get generated again and again."""
    parts = [f"Write exactly {count} NEW scenario records."]
    if context:
        # R1.9: context is at the granularity of key behaviour keywords, which the agent expands into a full structure
        parts.append(
            f"CONTEXT: {context}\n"
            "Expand that into complete scenario records. It names the behaviour or "
            "the kind of call wanted; you decide the premise, the signs and the "
            "organisation kind."
        )
    if gaps:
        wanted = []
        if gaps.get("signals_never_used_by_scam"):
            wanted.append(
                "signals no scam scenario covers yet: "
                + ", ".join(gaps["signals_never_used_by_scam"])
            )
        if gaps.get("signals_without_hard_negative"):
            wanted.append(
                "signals with no hard negative to balance them (a legitimate call "
                "that trips the same sign is worth more than another scam here): "
                + ", ".join(gaps["signals_without_hard_negative"])
            )
        if gaps.get("scam_families_without_paired_hard_negative"):
            wanted.append(
                "scam families with no paired hard negative: "
                + ", ".join(gaps["scam_families_without_paired_hard_negative"][:12])
            )
        if wanted:
            parts.append("COVERAGE GAPS worth closing:\n- " + "\n- ".join(wanted))

    parts.append(
        "ALREADY IN THE BANK -- do not repeat these kinds of call, and do not "
        "rename one of them:\n" + ", ".join(sorted(existing_families))
    )
    if existing_pretexts:
        shown = existing_pretexts[:40]
        parts.append(
            "Existing premises, for the same reason:\n"
            + "\n".join(f"- {p}" for p in shown)
        )
    parts.append(
        "Return the JSON object only. Every record must differ from the above in "
        "what the call is ABOUT, not merely in wording."
    )
    return "\n\n".join(parts)
