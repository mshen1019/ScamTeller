"""Record builders for tests."""

from callfactory.schema import (
    BlindAssessment,
    CallerPersona,
    Conversation,
    DiversityScore,
    GenerationTask,
    PrefixLabel,
    ReceiverPersona,
    Scenario,
    ScenarioProvenance,
    Utterance,
    UtteranceLabel,
)


def make_scenario(**over):
    base = dict(
        scenario_id="scn-test-001",
        category="scam",
        family="crypto_investment_pressure",
        pretext="pitches a position closing today in a fund the receiver never enquired about",
        org_kind="an investment firm",
        expected_signs=["too_good_to_be_true", "urgency_pressure"],
        plausible_outcomes=["receiver refuses", "receiver stalls to check first"],
        hard_negative=False,
        disambiguator="",
        late_onset=False,
        provenance=ScenarioProvenance(
            author_model="mock-dialogue-1", bank_version=1, context="bitcoin investment scam"
        ),
    )
    base.update(over)
    return Scenario(**base)


def make_task(**over):
    base = dict(
        task_id="task-1-00000",
        index=0,
        seed=42,
        derived_seed=1193669,
        bank_version=1,
        scenario=make_scenario(),
        caller_persona=CallerPersona(
            persona_id="clr-03",
            role="an account-services agent",
            speech_style="brisk and over-rehearsed",
            pressure_profile="manufactures a closing time window",
            challenge_response="deflects and insists the matter stays on this call",
        ),
        receiver_persona=ReceiverPersona(
            persona_id="rcv-05",
            skepticism="medium",
            tech_familiarity="low",
            emotional_state="distracted",
            situation="cooking dinner",
            speech_style="short sentences, asks for repeats",
        ),
        outcome="receiver stalls to check first",
        target_rounds=5,
        turn_directives=["Cut in before the other finishes.", None],
    )
    base.update(over)
    return GenerationTask(**base)


CALLER_LINES = [
    "Good afternoon, am I through to the account holder?",
    "There's a note on the file I wanted to go over with you, nothing alarming.",
    "I can hold while you fetch it -- take as long as you need.",
    "If it's easier I can put that in writing and we pick it up another day.",
    "That matches what's showing here, so I'll close the query off.",
]
RECEIVER_LINES = [
    "Sorry, who did you say you were calling from?",
    "A note? Nobody's written to me about anything.",
    "Hang on, the pan's going -- give me a second.",
    "In writing would be better, I'm hopeless on the phone.",
    "Right, well, thanks for letting me know. Bye now.",
]


def make_conversation(rounds=5, with_labels=True, **over):
    utterances = []
    for i in range(rounds):
        utterances.append(Utterance(len(utterances), "caller", CALLER_LINES[i % 5]))
        utterances.append(Utterance(len(utterances), "receiver", RECEIVER_LINES[i % 5]))
    conv = Conversation(
        conversation_id="run-42-00000",
        utterances=utterances,
        task=make_task(),
        generation={"seed": 42, "caller_model": "mock-dialogue-1"},
        **over,
    )
    if with_labels:
        conv.prefix_labels = [
            PrefixLabel(
                through_index=u.index,
                risk_score=min(1.0, 0.1 * (u.index + 1)),
                risk_tier="low",
                signals=["urgency_pressure"],
                sufficient_evidence=u.index >= 4,
            )
            for u in utterances
        ]
        conv.earliest_detection_index = 4
        conv.utterance_labels = [
            UtteranceLabel(
                index=u.index,
                speaker=u.speaker,
                risk_score=min(1.0, 0.1 * (u.index + 1)),
                risk_tier="low",
                cumulative_signals=["urgency_pressure"],
                is_detection_point=u.index == 4,
                label_from_index=u.index,
            )
            for u in utterances
        ]
        conv.blind_assessment = BlindAssessment(
            coherence=4.2, naturalness=4.0, blind_is_suspicious=True
        )
        conv.diversity = DiversityScore(fingerprint_seen=0, max_text_similarity=0.11)
    return conv
