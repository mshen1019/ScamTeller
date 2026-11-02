"""Provider abstraction: one async ``complete`` call plus retry policy.

Deliberately thin. Every backend is reachable over plain HTTP, so the only
runtime dependency is ``httpx`` (sync + async in one client) instead of three
vendor SDKs that each want a different version of the same transport.
"""

from __future__ import annotations

import abc
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class ProviderError(RuntimeError):
    """Non-retryable provider failure (bad request, auth, exhausted retries)."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.9
    max_tokens: int = 400
    max_retries: int = 5
    timeout: float = 60.0
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    extra_body: dict[str, Any] = field(default_factory=dict)


class LLMProvider(abc.ABC):
    """A single-shot chat completion endpoint."""

    #: Whether the backend accepts a seed and makes sampling reproducible.
    supports_seed: bool = False
    #: Whether the backend accepts ``temperature``. The Claude Messages API stopped
    #: accepting it (non-default values 400 since 2026-10-21), so diversity and
    #: label reproducibility there have to come from somewhere else -- every record
    #: states which knobs actually applied (C4 / N5).
    supports_temperature: bool = True

    def __init__(self, config: ProviderConfig):
        self.config = config
        #: Cumulative usage for this provider. Tracked per provider, not per conversation: under concurrency many conversations share
        #: one provider, so "per-conversation usage" from snapshot differences bleeds between them. The batch total is accurate.
        self.usage: dict[str, int] = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

    def _add_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.usage["requests"] += 1
        self.usage["input_tokens"] += int(input_tokens or 0)
        self.usage["output_tokens"] += int(output_tokens or 0)

    @property
    def model(self) -> str:
        return self.config.model

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        tag: str | None = None,
    ) -> str:
        """Return assistant text, retrying transient failures with jittered backoff."""
        cfg = self.config
        last: Exception | None = None
        for attempt in range(cfg.max_retries + 1):
            try:
                return await self._complete(
                    system=system,
                    messages=list(messages),
                    temperature=cfg.temperature if temperature is None else temperature,
                    max_tokens=cfg.max_tokens if max_tokens is None else max_tokens,
                    seed=seed,
                    tag=tag,
                )
            except ProviderError as exc:
                last = exc
                if not exc.retryable or attempt == cfg.max_retries:
                    raise
            except (asyncio.TimeoutError, OSError) as exc:  # transport-level
                last = exc
                if attempt == cfg.max_retries:
                    raise ProviderError(f"{cfg.name}: transport failure: {exc}") from exc
            delay = min(cfg.max_backoff, cfg.initial_backoff * (2**attempt))
            await asyncio.sleep(delay * (0.5 + random.random()))
        raise ProviderError(f"{cfg.name}: retries exhausted: {last}")

    @abc.abstractmethod
    async def _complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None,
        tag: str | None,
    ) -> str: ...

    async def aclose(self) -> None:
        return None


def resolve_api_key(explicit: str | None, env_names: Sequence[str]) -> str | None:
    if explicit:
        return explicit
    for name in env_names:
        value = os.environ.get(name)
        if value:
            return value
    return None


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose or fences often enough that parsing the whole
    string is not worth attempting first.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    return json.loads(match.group(0))
