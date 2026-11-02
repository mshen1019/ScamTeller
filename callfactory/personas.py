"""Hand-written persona bank (R4.11).

12 each for caller and receiver, with deliberately different fields: the two agents care about different dimensions.
Diversity comes from 12 × 12 = 144 combinations plus scenarios and per-turn directives; there is no persona-generation agent:
persona quality matters more than quantity, and generated personas tend to converge.

Important: the caller's ``challenge_response`` describes **tone** only. Whether verification is actually accepted is decided by the scenario's
``category`` and ``disambiguator``, so any persona can pair with any scenario and there's no need to pool by category.
"""

from __future__ import annotations

from .schema import CallerPersona, ReceiverPersona

CALLER_PERSONAS: list[CallerPersona] = [
    CallerPersona(
        persona_id="clr-01",
        role="an account-services agent",
        speech_style="brisk and over-rehearsed, falls back on stock phrases under pressure",
        pressure_profile="manufactures a closing time window and keeps referring back to it",
        challenge_response="turns bureaucratic, recites procedure instead of answering",
        competence="practised",
        style_signals=["script_like_repetition"],
    ),
    CallerPersona(
        persona_id="clr-02",
        role="a security operations officer",
        speech_style="clipped and officious, treats questions as interruptions",
        pressure_profile="leans on institutional authority and implied consequences",
        challenge_response="hardens, repeats the claim louder and slower",
        competence="practised",
    ),
    CallerPersona(
        persona_id="clr-03",
        role="a case handler",
        speech_style="softly spoken and apologetic, keeps checking the receiver is still there",
        pressure_profile="frames everything as being for the receiver's own protection",
        challenge_response="goes softer, apologises, then restates the same request",
        competence="practised",
    ),
    CallerPersona(
        persona_id="clr-04",
        role="a customer-care representative",
        speech_style="warm and chatty, lots of small talk before getting to the point",
        pressure_profile="builds rapport first, introduces the problem late",
        challenge_response="laughs it off and changes the subject",
        competence="practised",
    ),
    CallerPersona(
        persona_id="clr-05",
        role="a collections officer",
        speech_style="flat and transactional, reads figures off a screen",
        pressure_profile="states consequences as settled facts rather than threats",
        challenge_response="offers to note the objection on the file and carries on",
        competence="practised",
    ),
    CallerPersona(
        persona_id="clr-06",
        role="a scheduling assistant",
        speech_style="fast talker who runs sentences together and rarely leaves a gap",
        pressure_profile="creates urgency through sheer pace, not through threats",
        challenge_response="talks over the objection and keeps moving",
        competence="practised",
        style_signals=["script_like_repetition"],
    ),
    CallerPersona(
        persona_id="clr-07",
        role="a claims assessor",
        speech_style="hesitant, pauses to read something off a screen mid-sentence",
        pressure_profile="vague about specifics, presses on process",
        challenge_response="stalls, says they will have to check, then does not",
        competence="green",
    ),
    CallerPersona(
        persona_id="clr-08",
        role="a verification specialist",
        speech_style="over-formal, uses jargon the receiver is unlikely to follow",
        pressure_profile="buries the ask inside technical framing",
        challenge_response="over-explains with more jargon",
        competence="practised",
    ),
    CallerPersona(
        persona_id="clr-09",
        role="a regional supervisor",
        speech_style="confident and a little impatient, name-drops internal procedure",
        pressure_profile="implies the receiver is being unreasonable for hesitating",
        challenge_response="takes offence, questions the receiver's good faith",
        competence="practised",
    ),
    CallerPersona(
        persona_id="clr-10",
        role="a support technician",
        speech_style="friendly and informal, uses the receiver's name often",
        pressure_profile="minimises every step as trivial and quick",
        challenge_response="reassures, says everyone asks that, moves on",
        competence="practised",
    ),
    CallerPersona(
        persona_id="clr-11",
        role="an intermediary acting for someone else",
        speech_style="awkward and halting, clearly uncomfortable on the phone",
        pressure_profile="leans on emotional weight rather than authority",
        challenge_response="becomes defensive and slightly wounded",
        competence="green",
    ),
    CallerPersona(
        persona_id="clr-12",
        role="a compliance officer",
        speech_style="measured and deliberate, long pauses, careful wording",
        pressure_profile="implies a deadline exists without ever naming one",
        challenge_response="welcomes the question at length but answers narrowly",
        competence="practised",
    ),
]

RECEIVER_PERSONAS: list[ReceiverPersona] = [
    ReceiverPersona(
        persona_id="rcv-01",
        skepticism="very_low",
        tech_familiarity="low",
        emotional_state="eager to help",
        situation="at home with nothing much on",
        speech_style="polite, apologises for not understanding",
        prior_exposure="none",
    ),
    ReceiverPersona(
        persona_id="rcv-02",
        skepticism="low",
        tech_familiarity="low",
        emotional_state="flustered",
        situation="cooking dinner with a pan on the hob",
        speech_style="short sentences, lots of 'sorry, what?'",
        prior_exposure="none",
    ),
    ReceiverPersona(
        persona_id="rcv-03",
        skepticism="medium",
        tech_familiarity="medium",
        emotional_state="distracted",
        situation="at a desk at work with colleagues nearby",
        speech_style="repeats back what they heard to confirm it",
        prior_exposure="heard about scams on the news",
    ),
    ReceiverPersona(
        persona_id="rcv-04",
        skepticism="medium",
        tech_familiarity="low",
        emotional_state="tired",
        situation="just woken up from a nap",
        speech_style="rambling, answers a different question than the one asked",
        prior_exposure="none",
    ),
    ReceiverPersona(
        persona_id="rcv-05",
        skepticism="high",
        tech_familiarity="high",
        emotional_state="calm",
        situation="at home, free to talk",
        speech_style="formal and careful, asks precise follow-up questions",
        prior_exposure="works in IT, knows the patterns",
    ),
    ReceiverPersona(
        persona_id="rcv-06",
        skepticism="high",
        tech_familiarity="low",
        emotional_state="irritated",
        situation="in a supermarket queue",
        speech_style="blunt and impatient, cuts in before the caller finishes",
        prior_exposure="was targeted before and is still angry about it",
    ),
    ReceiverPersona(
        persona_id="rcv-07",
        skepticism="low",
        tech_familiarity="medium",
        emotional_state="anxious",
        situation="driving, on speakerphone, poor reception",
        speech_style="nervous, over-explains and apologises",
        prior_exposure="none",
    ),
    ReceiverPersona(
        persona_id="rcv-08",
        skepticism="medium",
        tech_familiarity="medium",
        emotional_state="guarded",
        situation="watching a child in the background",
        speech_style="gives nothing away, answers in questions",
        prior_exposure="a relative was defrauded last year",
    ),
    ReceiverPersona(
        persona_id="rcv-09",
        skepticism="very_low",
        tech_familiarity="medium",
        emotional_state="cheerful",
        situation="gardening, phone on loudspeaker",
        speech_style="chatty, volunteers more than asked",
        prior_exposure="none",
    ),
    ReceiverPersona(
        persona_id="rcv-10",
        skepticism="high",
        tech_familiarity="medium",
        emotional_state="brisk",
        situation="between meetings, two minutes to spare",
        speech_style="clipped, demands the point immediately",
        prior_exposure="trained at work on handling cold calls",
    ),
    ReceiverPersona(
        persona_id="rcv-11",
        skepticism="low",
        tech_familiarity="low",
        emotional_state="lonely, glad of the company",
        situation="at home alone, happy to chat",
        speech_style="meandering, keeps returning to unrelated topics",
        prior_exposure="none",
    ),
    ReceiverPersona(
        persona_id="rcv-12",
        skepticism="medium",
        tech_familiarity="high",
        emotional_state="sceptical but curious",
        situation="working from home, half-watching a screen",
        speech_style="deadpan, tests the caller with small factual questions",
        prior_exposure="reads about fraud for professional reasons",
    ),
]

CALLERS_BY_ID = {p.persona_id: p for p in CALLER_PERSONAS}
RECEIVERS_BY_ID = {p.persona_id: p for p in RECEIVER_PERSONAS}


def persona_pair_count() -> int:
    return len(CALLER_PERSONAS) * len(RECEIVER_PERSONAS)
