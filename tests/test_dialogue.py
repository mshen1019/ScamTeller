"""Components 3 / 4: prompt boundaries, perspective re-rolling, the alternation loop."""

from __future__ import annotations

import asyncio

import pytest

from callfactory.agents import CallerAgent, ReceiverAgent, clean_utterance
from callfactory.dialogue import DialogueFailure, run_dialogue
from callfactory.prompts import (
    caller_system_prompt,
    format_transcript,
    receiver_system_prompt,
    turn_user_message,
)
from callfactory.providers import build_provider
from callfactory.providers.base import LLMProvider, ProviderConfig
from callfactory.schema import Utterance

from factories import make_scenario, make_task


# --------------------------------------------------------------------------- #
# Test provider
# --------------------------------------------------------------------------- #

class ScriptedProvider(LLMProvider):
    """Emit preset lines by tag in turn, and record every request so request shapes can be asserted."""

    def __init__(self, caller_lines, receiver_lines):
        super().__init__(ProviderConfig(name="scripted", model="scripted-1"))
        self.lines = {"caller": list(caller_lines), "receiver": list(receiver_lines)}
        self.calls: list[dict] = []

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag):
        self.calls.append(
            {"system": system, "messages": messages, "seed": seed, "tag": tag}
        )
        bank = self.lines[tag]
        n = sum(1 for c in self.calls if c["tag"] == tag) - 1
        return bank[n] if n < len(bank) else bank[-1]


def agents_for(task, provider, *, seed=7):
    return (
        CallerAgent(provider, task, seed=seed),
        ReceiverAgent(provider, task, seed=seed),
    )


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Prompt boundaries (P1 / R4.1)
# --------------------------------------------------------------------------- #

def test_receiver_prompt_leaks_nothing_about_the_scenario():
    task = make_task()
    prompt = receiver_system_prompt(task)
    scn = task.scenario
    for leaked in (scn.category, scn.family, scn.pretext, scn.org_kind, scn.scenario_id):
        assert leaked.lower() not in prompt.lower(), f"receiver prompt leaked {leaked!r}"
    for sign in scn.expected_signs:
        assert sign not in prompt
    assert task.outcome.lower() not in prompt.lower()
    for word in ("scam", "fraudster", "suspicious", "legitimate", "hard negative"):
        assert word not in prompt.lower()


def test_receiver_prompt_leaks_nothing_for_a_hard_negative_either():
    task = make_task(
        scenario=make_scenario(
            category="legitimate",
            family="genuine_broker_compliance_check",
            hard_negative=True,
            disambiguator="quotes a ticket number only the real firm would hold",
            expected_signs=["urgency_pressure"],
        )
    )
    prompt = receiver_system_prompt(task)
    assert "ticket number" not in prompt
    assert "urgency_pressure" not in prompt


def test_receiver_prompt_carries_the_whole_persona():
    task = make_task()
    prompt = receiver_system_prompt(task)
    p = task.receiver_persona
    for field in (p.skepticism, p.tech_familiarity, p.emotional_state,
                  p.situation, p.speech_style):
        assert field in prompt
    assert "[END_CALL]" in prompt


def test_receiver_prompt_forbids_presuming_either_way():
    prompt = receiver_system_prompt(make_task()).lower()
    assert "do not assume good faith" in prompt
    assert "do not assume bad faith" in prompt


def test_caller_prompt_has_scenario_persona_outcome_and_rounds():
    task = make_task()
    prompt = caller_system_prompt(task)
    assert task.scenario.pretext in prompt
    assert task.scenario.org_kind in prompt
    assert task.caller_persona.role in prompt
    assert task.caller_persona.pressure_profile in prompt
    assert task.caller_persona.challenge_response in prompt
    assert task.outcome in prompt
    assert str(task.target_rounds) in prompt
    for sign in task.scenario.expected_signs:
        assert sign in prompt


def test_caller_prompt_marks_intent_for_the_mock_provider():
    # providers/mock.py picks its template bank from this wording; keep them in sync if it changes.
    assert "SCENARIO INTENT: suspicious" in caller_system_prompt(make_task())
    legit = make_task(scenario=make_scenario(category="legitimate", expected_signs=[]))
    assert "SCENARIO INTENT: legitimate" in caller_system_prompt(legit)


def test_late_onset_tells_the_caller_to_open_unremarkably():
    plain = caller_system_prompt(make_task(late_onset=False))
    slow = caller_system_prompt(make_task(late_onset=True))
    assert "SLOW BURN" in slow
    assert "SLOW BURN" not in plain
    assert "pressure is present early" in plain


def test_hard_negative_caller_is_told_how_to_hold_up_under_challenge():
    task = make_task(
        scenario=make_scenario(
            category="legitimate",
            hard_negative=True,
            disambiguator="welcomes a callback on a number the receiver looks up",
            expected_signs=["urgency_pressure"],
        )
    )
    prompt = caller_system_prompt(task)
    assert "welcomes a callback" in prompt
    assert "genuinely who you say you are" in prompt


def test_green_caller_is_told_to_slip_up():
    task = make_task(**{})
    task.caller_persona.competence = "green"
    assert "contradict a detail" in caller_system_prompt(task)


def test_style_signal_persona_is_told_to_fall_back_on_the_script():
    task = make_task()
    task.caller_persona.style_signals = ["script_like_repetition"]
    assert "rehearsed phrasing" in caller_system_prompt(task)


def test_turn_user_message_shape():
    opening = turn_user_message(None, is_opening=True, is_last=False, role="caller")
    assert "Speak first" in opening
    mid = turn_user_message("Mishear a word.", is_opening=False, is_last=False, role="receiver")
    assert "Mishear a word." in mid and "receiver" in mid
    last = turn_user_message(None, is_opening=False, is_last=True, role="caller")
    assert "natural close" in last


# --------------------------------------------------------------------------- #
# Perspective re-rolling (R3.2 / R4.9)
# --------------------------------------------------------------------------- #

def test_each_side_sees_its_own_turns_as_assistant():
    task = make_task()
    provider = ScriptedProvider(["c"], ["r"])
    caller, receiver = agents_for(task, provider)
    history = [
        Utterance(0, "caller", "caller one"),
        Utterance(1, "receiver", "receiver one"),
    ]
    cm = caller.messages(history, "go")
    rm = receiver.messages(history, "go")
    assert [m["role"] for m in cm] == ["assistant", "user"]
    assert [m["role"] for m in rm] == ["user", "assistant", "user"]
    assert cm[0]["content"] == "caller one"
    assert rm[0]["content"] == "caller one"


def test_consecutive_same_role_turns_are_merged():
    task = make_task()
    provider = ScriptedProvider(["c"], ["r"])
    caller, _ = agents_for(task, provider)
    history = [
        Utterance(0, "caller", "one"),
        Utterance(1, "caller", "two"),
        Utterance(2, "receiver", "three"),
    ]
    msgs = caller.messages(history, "go")
    assert [m["role"] for m in msgs] == ["assistant", "user"]
    assert msgs[0]["content"] == "one\ntwo"
    assert "three" in msgs[1]["content"] and "go" in msgs[1]["content"]


def test_the_two_sides_do_not_sample_in_lockstep():
    task = make_task()
    provider = ScriptedProvider(["c"], ["r"])
    caller, receiver = agents_for(task, provider, seed=7)
    assert caller.seed != receiver.seed


def test_agents_never_see_each_others_instructions():
    task = make_task()
    provider = ScriptedProvider(["c"] * 20, ["r"] * 20)
    caller, receiver = agents_for(task, provider)
    run(run_dialogue(task, caller, receiver, min_rounds=5, max_rounds=5))
    for call in provider.calls:
        other = receiver.system if call["tag"] == "caller" else caller.system
        assert call["system"] != other
        assert call["system"] == (caller.system if call["tag"] == "caller" else receiver.system)


# --------------------------------------------------------------------------- #
# Cleaning layer (R3.6)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,expected",
    [
        ('CALLER: Good afternoon.', "Good afternoon."),
        ('[3] Receiver: Who is this?', "Who is this?"),
        ('"Just a quick word."', "Just a quick word."),
        ("(pause) I see.", "I see."),
        ("*sighs* Fine.", "Fine."),
        ("Line  one\n\nline   two", "Line one line two"),
    ],
)
def test_clean_utterance_strips_formatting_tells(raw, expected):
    text, ends = clean_utterance(raw)
    assert text == expected
    assert ends is False


def test_clean_utterance_detects_and_removes_the_end_token():
    text, ends = clean_utterance("Right, I'm going to go. [END_CALL]")
    assert ends is True
    assert "END_CALL" not in text
    assert text == "Right, I'm going to go."


def test_clean_utterance_handles_an_empty_reply():
    assert clean_utterance("   ") == ("", False)
    assert clean_utterance(None) == ("", False)


# --------------------------------------------------------------------------- #
# Alternation loop (R4.4 / R4.7 / R4.8)
# --------------------------------------------------------------------------- #

def test_strict_caller_first_alternation_and_round_count():
    task = make_task(target_rounds=6)
    provider = ScriptedProvider([f"c{i}" for i in range(20)], [f"r{i}" for i in range(20)])
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    assert result.rounds == 6
    assert len(result.utterances) == 12
    assert [u.speaker for u in result.utterances[:4]] == [
        "caller", "receiver", "caller", "receiver"
    ]
    assert [u.index for u in result.utterances] == list(range(12))
    assert result.ended_by == "target_rounds"


def test_target_rounds_is_clamped_into_the_configured_range():
    task = make_task(target_rounds=19)
    provider = ScriptedProvider(["c"] * 40, ["r"] * 40)
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=6))
    assert result.rounds == 6


def test_hangup_before_min_rounds_is_ignored():
    task = make_task(target_rounds=6)
    provider = ScriptedProvider(
        ["c"] * 20,
        ["I'm done with this. [END_CALL]"] + ["r"] * 20,
    )
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    assert result.rounds >= 5
    assert any("wanted to hang up" in n for n in result.notes)
    assert "END_CALL" not in result.utterances[1].text


def test_hangup_is_honoured_once_the_target_is_in_sight():
    """The gate is max(min_rounds, target-1): in testing with min_rounds every conversation collapsed to the minimum length."""
    task = make_task(target_rounds=8)
    receiver_lines = ["r"] * 6 + ["That's enough, goodbye. [END_CALL]"] + ["r"] * 20
    provider = ScriptedProvider(["c"] * 40, receiver_lines)
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    assert result.ended_by == "end_call"
    assert result.rounds == 7 == result.hangup_from
    assert result.utterances[-1].speaker == "receiver"


def test_an_early_hangup_no_longer_collapses_the_call_to_min_rounds():
    task = make_task(target_rounds=12)
    # wants to hang up every round
    provider = ScriptedProvider(["c"] * 40, ["Right, I'm going. [END_CALL]"] * 40)
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    assert result.rounds == 11            # target-1, not min_rounds
    assert result.ended_by == "end_call"
    assert len([n for n in result.notes if "wanted to hang up" in n]) == 10


def test_the_target_is_still_the_ceiling_when_nobody_hangs_up():
    task = make_task(target_rounds=6)
    provider = ScriptedProvider(["c"] * 40, ["r"] * 40)
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    assert result.rounds == 6
    assert result.ended_by == "target_rounds"
    assert result.to_dict()["hangup_allowed_from_round"] == 5


def test_a_caller_end_token_is_noted_but_does_not_end_the_call():
    task = make_task(target_rounds=5)
    provider = ScriptedProvider(["Goodbye then. [END_CALL]"] + ["c"] * 20, ["r"] * 20)
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    assert result.rounds == 5
    assert any("caller emitted the end token" in n for n in result.notes)


def test_one_empty_reply_is_filled_in_and_the_call_continues():
    task = make_task(target_rounds=5)
    provider = ScriptedProvider(["c"] * 20, ["", "", "r"] + ["r"] * 20)
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    assert result.empty_fallbacks == 1
    assert result.rounds == 5
    assert result.utterances[1].text.startswith("Sorry, say that again")
    assert any("returned nothing" in n for n in result.notes)


def test_persistent_empty_replies_fail_the_task():
    """When one side keeps collapsing, the other side's normal turns must not reset the count."""
    task = make_task(target_rounds=5)
    provider = ScriptedProvider(["c"] * 20, [""] * 40)
    with pytest.raises(DialogueFailure, match="receiver"):
        run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))


def test_scattered_empty_replies_do_not_fail_the_task():
    task = make_task(target_rounds=5)
    provider = ScriptedProvider(
        ["c", "c", "", "", "c", "c", "c", "c", "c", "c"],
        ["", "", "r", "r", "", "", "r", "r", "r", "r"],
    )
    result = run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    assert result.rounds == 5
    # both sides went empty, but neither stayed empty: not a collapse.
    assert result.empty_fallbacks == 3
    assert {n.split()[2] for n in result.notes} == {"caller", "receiver"}


def test_rejects_an_impossible_round_range():
    task = make_task()
    provider = ScriptedProvider(["c"], ["r"])
    with pytest.raises(ValueError):
        run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=4))


def test_directives_are_injected_per_utterance():
    task = make_task(target_rounds=5)
    task.turn_directives = ["DIRECTIVE_ZERO", None] + [None] * 20
    provider = ScriptedProvider(["c"] * 20, ["r"] * 20)
    run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    first = provider.calls[0]["messages"][-1]["content"]
    second = provider.calls[1]["messages"][-1]["content"]
    assert "DIRECTIVE_ZERO" in first
    assert "DIRECTIVE_ZERO" not in second


def test_the_last_stretch_gets_a_wrap_up_hint():
    task = make_task(target_rounds=5)
    provider = ScriptedProvider(["c"] * 20, ["r"] * 20)
    run(run_dialogue(task, *agents_for(task, provider), min_rounds=5, max_rounds=20))
    hints = [c for c in provider.calls if "natural close" in c["messages"][-1]["content"]]
    assert len(hints) == 2          # once for each side on the last round
    assert provider.calls.index(hints[0]) == 8


# --------------------------------------------------------------------------- #
# Offline reproducibility (P7 / P5)
# --------------------------------------------------------------------------- #

def test_mock_provider_runs_the_whole_loop_reproducibly():
    task = make_task(target_rounds=6)
    texts = []
    for _ in range(2):
        provider = build_provider("mock")
        result = run(
            run_dialogue(task, *agents_for(task, provider, seed=task.derived_seed),
                         min_rounds=5, max_rounds=20)
        )
        # mock's receiver hangs up by itself after min_rounds, so only the lower bound and reproducibility are asserted.
        assert 5 <= result.rounds <= 6
        texts.append([u.text for u in result.utterances])
    assert texts[0] == texts[1]
    assert all(t.strip() for t in texts[0])


def test_format_transcript_is_indexed_and_role_tagged():
    out = format_transcript([Utterance(0, "caller", "hi"), Utterance(1, "receiver", "yes?")])
    assert out == "[0] CALLER: hi\n[1] RECEIVER: yes?"
