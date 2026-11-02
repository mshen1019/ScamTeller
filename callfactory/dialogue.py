"""The alternating dialogue loop (driver for components 3 / 4).

This only handles "how the two agents take turns and when the call is over"; no labelling, no writing to disk.

Three rules are worth spelling out:

* **Strict caller-first alternation** (R4.7). Even indices are always the caller, odd ones always the receiver,
  so ``index // 2`` is the round; prefix labels and round-level latency both rely on this invariant.
* **Hang-up requests are ignored until close to ``target_rounds``** (R4.4). The gate is
  ``max(min_rounds, target_rounds - 1)``, not ``min_rounds``: in testing with min_rounds,
  20/20 conversations were hung up in the first round long enough (14 of them at exactly 5 rounds), so the sampler's 5-10 rounds
  had no effect and slow-burn scenarios had no time to hold their pressure back.
* **An empty reply is first patched with a filler line; the task fails only after too many in a row** (R4.8). A single empty reply is usually sampling jitter,
  and dropping the whole conversation for it is too costly; a run of them means that side has really collapsed and carrying on only produces junk.
  The count is **per role**: while one side keeps returning nothing the other is still talking normally, so a global count would keep being
  reset by the other side's successful turns, and a conversation made entirely of filler lines would still "succeed".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents import CallerAgent, ReceiverAgent
from .schema import GenerationTask, Utterance

#: The line substituted for an empty reply. Deliberately bland: it is there to keep the call going, not to add content.
FALLBACK_LINES = {
    "caller": "Sorry, I lost you there for a second. Are you still with me?",
    "receiver": "Sorry, say that again?",
}


class DialogueFailure(RuntimeError):
    """This task can't produce a usable conversation; the caller should record it in failures.jsonl."""


@dataclass
class DialogueResult:
    utterances: list[Utterance]
    ended_by: str                      # end_call | target_rounds | max_rounds
    rounds: int
    target_rounds: int = 0
    hangup_from: int = 0
    empty_fallbacks: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ended_by": self.ended_by,
            "rounds": self.rounds,
            "target_rounds": self.target_rounds,
            "hangup_allowed_from_round": self.hangup_from,
            "empty_fallbacks": self.empty_fallbacks,
            "notes": list(self.notes),
        }


async def run_dialogue(
    task: GenerationTask,
    caller: CallerAgent,
    receiver: ReceiverAgent,
    *,
    min_rounds: int = 5,
    max_rounds: int = 20,
    empty_retries: int = 1,
    max_consecutive_fallbacks: int = 2,
) -> DialogueResult:
    if min_rounds < 1 or max_rounds < min_rounds:
        raise ValueError("requires 1 <= min_rounds <= max_rounds")

    target = max(min_rounds, min(max_rounds, task.target_rounds))
    #: Round from which the receiver may hang up. Leaves one round of slack so it can wrap up naturally before the target length.
    hangup_from = max(min_rounds, target - 1)
    directives = task.turn_directives
    utterances: list[Utterance] = []
    notes: list[str] = []
    fallbacks = 0
    consecutive: dict[str, int] = {"caller": 0, "receiver": 0}
    ended_by = "max_rounds"

    for index in range(2 * max_rounds):
        agent = caller if index % 2 == 0 else receiver
        round_index = index // 2
        directive = directives[index] if index < len(directives) else None

        text, wants_end = "", False
        for attempt in range(empty_retries + 1):
            text, wants_end = await agent.speak(
                utterances,
                directive=directive if attempt == 0 else None,
                is_last=round_index >= target - 1,
            )
            if text:
                break

        if not text:
            text = FALLBACK_LINES[agent.role]
            fallbacks += 1
            consecutive[agent.role] += 1
            notes.append(f"index {index}: {agent.role} returned nothing, filled in")
            if consecutive[agent.role] > max_consecutive_fallbacks:
                raise DialogueFailure(
                    f"{agent.role} returned nothing {consecutive[agent.role]} times in a row "
                    f"(index {index}), giving up on this task"
                )
        else:
            consecutive[agent.role] = 0

        utterances.append(Utterance(index=index, speaker=agent.role, text=text))
        rounds = sum(1 for u in utterances if u.speaker == "receiver")

        if wants_end:
            if agent.role != "receiver":
                # Only the receiver may hang up (R4.4). Any token the caller emits has already been cleaned out.
                notes.append(f"index {index}: caller emitted the end token, ignored")
            elif rounds >= hangup_from:
                ended_by = "end_call"
                break
            else:
                notes.append(
                    f"round {rounds}: receiver wanted to hang up before "
                    f"round {hangup_from} (target {target}), call continues"
                )

        if rounds >= target:
            ended_by = "target_rounds"
            break

    rounds = sum(1 for u in utterances if u.speaker == "receiver")
    if rounds < min_rounds:
        raise DialogueFailure(
            f"only {rounds} rounds produced, below min_rounds={min_rounds}"
        )
    return DialogueResult(
        utterances=utterances,
        ended_by=ended_by,
        rounds=rounds,
        target_rounds=target,
        hangup_from=hangup_from,
        empty_fallbacks=fallbacks,
        notes=notes,
    )
