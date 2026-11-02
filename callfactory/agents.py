"""The Caller / Receiver agents themselves (components 3 / 4).

The two agents are **two independent LLM sessions**, each with its own system prompt, and neither sees the other's instructions.
All they share is what has been said aloud, each re-rolled from its own point of view (R3.2): its own turns as ``assistant``,
the other side's as ``user``. That is exactly why the receiver's wariness can count as evidence: it can only come from what it heard.

The cleaning layer (``clean_utterance``) is the output-side safety net: models now and then add speaker prefixes, quotes or stage directions,
and once those are written into a transcript they become formatting cues a downstream model can exploit.
"""

from __future__ import annotations

import re

from .prompts import caller_system_prompt, receiver_system_prompt, turn_user_message
from .providers import LLMProvider
from .schema import GenerationTask, Utterance

END_TOKEN = "[END_CALL]"

_STRIP_PREFIX = re.compile(
    r"^\s*(?:\[\d+\]\s*)?(?:caller|receiver|agent|me|you)\s*[:\-]\s*", re.IGNORECASE
)
_STAGE_DIRECTION = re.compile(
    r"[\*\(]\s*(?:pause|pauses|sighs?|laughs?|background|silence|beat)[^\*\)]*[\*\)]",
    re.IGNORECASE,
)


def clean_utterance(text: str) -> tuple[str, bool]:
    """Normalise one raw model output. Returns ``(text, wants_to_end)``."""
    text = (text or "").strip()
    ends = END_TOKEN.lower() in text.lower()
    text = re.sub(re.escape(END_TOKEN), "", text, flags=re.IGNORECASE)
    text = _STAGE_DIRECTION.sub("", text)
    text = _STRIP_PREFIX.sub("", text)
    text = text.strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    return text, ends


class DialogueAgent:
    """One side of the call."""

    role: str = "caller"
    #: Sample the two sides at different temperatures (R4.6).
    seed_offset: int = 0

    def __init__(
        self,
        provider: LLMProvider,
        task: GenerationTask,
        *,
        system: str,
        seed: int | None,
        temperature: float = 0.9,
    ):
        self.provider = provider
        self.task = task
        self.system = system
        self.seed = None if seed is None else seed + self.seed_offset
        self.temperature = temperature

    def messages(self, history: list[Utterance], user_message: str) -> list[dict[str, str]]:
        """Re-roll the history from this side's point of view (R3.2) and merge adjacent same-role messages (R4.9)."""
        messages: list[dict[str, str]] = []
        for utt in history:
            role = "assistant" if utt.speaker == self.role else "user"
            if messages and messages[-1]["role"] == role:
                # Some backends reject consecutive messages with the same role.
                messages[-1]["content"] += "\n" + utt.text
            else:
                messages.append({"role": role, "content": utt.text})
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n\n({user_message})"
        else:
            messages.append({"role": "user", "content": user_message})
        return messages

    async def speak(
        self,
        history: list[Utterance],
        *,
        directive: str | None = None,
        is_last: bool = False,
    ) -> tuple[str, bool]:
        user_message = turn_user_message(
            directive,
            is_opening=not history,
            is_last=is_last,
            role=self.role,
        )
        raw = await self.provider.complete(
            system=self.system,
            messages=self.messages(history, user_message),
            seed=None if self.seed is None else self.seed + len(history),
            temperature=self.temperature,
            tag=self.role,
        )
        return clean_utterance(raw)


class CallerAgent(DialogueAgent):
    role = "caller"
    seed_offset = 0

    def __init__(
        self,
        provider: LLMProvider,
        task: GenerationTask,
        *,
        seed: int | None = None,
        temperature: float = 0.9,
    ):
        super().__init__(
            provider,
            task,
            system=caller_system_prompt(task),
            seed=seed,
            temperature=temperature,
        )


class ReceiverAgent(DialogueAgent):
    role = "receiver"
    seed_offset = 101

    def __init__(
        self,
        provider: LLMProvider,
        task: GenerationTask,
        *,
        seed: int | None = None,
        temperature: float = 0.9,
    ):
        super().__init__(
            provider,
            task,
            system=receiver_system_prompt(task),
            seed=seed,
            temperature=temperature,
        )
