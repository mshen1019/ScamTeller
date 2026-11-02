"""Per-provider judges. A judge = one stateless model, made streaming by ``ReplayAdapter``.

**The only thing here worth a few extra lines is prefix caching.** In testing, 84% of this data's input tokens are
cacheable (the system prompt alone is a third of it, resent 912 times). Turning it on cuts the bill by about half,
and **not one byte of request content changes**: caching only changes billing and latency, it doesn't give the model memory, so P3's
statelessness and baseline comparability are both unaffected. That sentence is worth writing here because it's easy to misread.

To actually hit, three things must all hold:

1. prompt layout = system (fixed) + transcript (growing) + question (last); see ``judge_prompt``
2. a call's prefixes **run back to back**: the cache TTL is only minutes; the runner takes care of this
3. Anthropic needs explicit ``cache_control`` breakpoints; OpenAI / DeepSeek / Gemini are automatic

Every provider's usage fields include cache counts, so ``cache_report()`` can prove afterwards whether it actually hit;
otherwise "I turned caching on" is just a slogan.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..providers.base import ProviderError, extract_json
from ..schema import Utterance
from . import judge_prompt as JP
from .protocol import Verdict


def _httpx():
    try:
        import httpx  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ProviderError("httpx required: pip install httpx") from exc
    return httpx


@dataclass
class JudgeUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    parse_failures: int = 0

    #: We were bitten by this: Anthropic's ``input_tokens`` **excludes** the hit and write parts, and using it as the denominator
    #: gives a "hit rate" of 1107%. The denominator must be the sum of all three.
    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cached_tokens + self.cache_write_tokens

    def to_dict(self) -> dict[str, Any]:
        billed = self.total_input
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "total_input_tokens": self.total_input,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_hit_share": round(self.cached_tokens / billed, 4) if billed else 0.0,
            "parse_failures": self.parse_failures,
        }


class BaseJudge:
    """Shared parts: timing, JSON parsing, client-side validation, usage accumulation."""

    name = "base"
    #: Fallback verdict when parsing fails. **No retry, no guessing**: it's recorded as a failure,
    #: otherwise "the model didn't answer in format" would quietly count as "it judged no alarm".
    FALLBACK = Verdict(risk_score=0.0, is_alarm=False, rationale="(parse failure)")
    #: Times the output cap truncated a reply. Counted separately from "didn't answer in format": the first means tuning max_output, the second fixing the prompt.
    truncated = 0

    #: Output budget. **Reasoning tokens come out of this too**, so it can't be sized by "how long the JSON is":
    #: in testing opus's thinking ate 145/200 and deepseek's ate 200/200 (returning an empty string,
    #: finish_reason=length). At 200, deepseek failed to parse on 36/37.
    MAX_OUTPUT = 1200

    def __init__(self, model: str, *, timeout: float = 120.0, max_output: int | None = None):
        self.model = model
        self.max_output = max_output or self.MAX_OUTPUT
        self.usage = JudgeUsage()
        self._client = _httpx().AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    def cache_report(self) -> dict[str, Any]:
        return {**self.usage.to_dict(), "truncated": self.truncated}

    async def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        httpx = _httpx()
        try:
            resp = await self._client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name} timed out: {exc}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"{self.name} transport failed: {exc}", retryable=True) from exc
        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.name} HTTP {resp.status_code}: {resp.text[:300]}",
                status=resp.status_code,
                retryable=resp.status_code in {408, 429, 500, 502, 503, 504, 529},
            )
        return resp.json()

    def _verdict(self, text: str, elapsed_ms: float, *, finish: str | None = None) -> Verdict:
        if finish in ("length", "MAX_TOKENS"):
            self.truncated += 1
        try:
            data = extract_json(text)
        except (ValueError, TypeError):
            self.usage.parse_failures += 1
            v = Verdict(**{**self.FALLBACK.__dict__})
            v.latency_ms = elapsed_ms
            return v
        score = data.get("risk_score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        return Verdict(
            risk_score=score,
            is_alarm=bool(data.get("alarm")),
            signals=[s for s in (data.get("signals") or []) if isinstance(s, str)],
            rationale=str(data.get("why", ""))[:200],
            latency_ms=elapsed_ms,
        )


class MockJudge(BaseJudge):
    """Offline deterministic judge. Exists so the runner and scoring run with no key and no network."""

    CUES = ("urgent", "right now", "don't tell", "can't put you through",
            "voucher", "transfer", "confirm a couple of details", "charge applied",
            "won", "prize", "remote", "no, don't call")

    def __init__(self, model: str = "mock-judge-1", threshold_hits: int = 2):
        self.name = "mock"
        self.model = model
        self.usage = JudgeUsage()
        self.threshold_hits = threshold_hits
        self._client = None

    async def aclose(self) -> None:
        return None

    async def judge(self, call_id: str, prefix: Sequence[Utterance]) -> Verdict:
        caller = " ".join(u.text.lower() for u in prefix if u.speaker == "caller")
        hits = sum(1 for c in self.CUES if c in caller)
        self.usage.requests += 1
        self.usage.input_tokens += len(JP.user_message(prefix)) // 4
        self.usage.output_tokens += 20
        score = min(1.0, 0.25 * hits)
        return Verdict(
            risk_score=score,
            is_alarm=hits >= self.threshold_hits,
            signals=["urgency_pressure"] if hits else [],
            rationale=f"mock: {hits} cue(s)",
        )


class AnthropicJudge(BaseJudge):
    """Explicit ``cache_control``: the breakpoint sits at the end of the transcript, so the next, longer prefix hits it."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "claude-haiku-5-5", *, minimal_reasoning: bool = False,
                 cache: bool = True, **kw):
        super().__init__(model, **kw)
        self.name = f"anthropic/{model}"
        self.minimal = minimal_reasoning
        self.cache = cache
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")

    async def judge(self, call_id: str, prefix: Sequence[Utterance]) -> Verdict:
        transcript = f"CALL SO FAR:\n{JP.render_transcript(prefix)}"
        content: list[dict[str, Any]] = [{"type": "text", "text": transcript}]
        if self.cache:
            content[0]["cache_control"] = {"type": "ephemeral"}
        content.append({"type": "text", "text": JP.QUESTION})
        system: list[dict[str, Any]] = [{"type": "text", "text": JP.SYSTEM}]
        if self.cache:
            system[0]["cache_control"] = {"type": "ephemeral"}
        body: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_output,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        if self.minimal:
            # The shape of C2 differs by provider and model, and the error message is the right answer:
            #   haiku-5-5  accepts thinking.type=disabled
            #   opus-5-5   rejects it -> "Use thinking.type.adaptive and output_config.effort"
            #              in testing thinking_tokens is 0 at output_config.effort=low
            # So for opus C2 isn't "reasoning off" but "reasoning at its lowest", and the report must say so:
            # these two models can't be put on a strictly equal footing with Jev's "never reasons".
            if "opus" in self.model or "sonnet-5-5" in self.model:
                body["output_config"] = {"effort": "low"}
            else:
                body["thinking"] = {"type": "disabled"}
        t0 = time.monotonic()
        data = await self._post(self.URL, {
            "content-type": "application/json", "x-api-key": self.key,
            "anthropic-version": "2023-06-01",
        }, body)
        ms = (time.monotonic() - t0) * 1000
        u = data.get("usage") or {}
        self.usage.requests += 1
        self.usage.input_tokens += int(u.get("input_tokens") or 0)
        self.usage.output_tokens += int(u.get("output_tokens") or 0)
        self.usage.cached_tokens += int(u.get("cache_read_input_tokens") or 0)
        self.usage.cache_write_tokens += int(u.get("cache_creation_input_tokens") or 0)
        text = "".join(b.get("text", "") for b in data.get("content") or []
                       if b.get("type") == "text")
        return self._verdict(text, ms, finish=data.get("stop_reason"))


class OpenAICompatJudge(BaseJudge):
    """OpenAI and DeepSeek. Both cache **automatically**, so nothing is marked here; only the hit counts are read."""

    def __init__(self, model: str = "gpt-6-luna", *, base_url: str = "https://api.openai.com/v1",
                 key_env: str = "OPENAI_API_KEY", minimal_reasoning: bool = False,
                 strict_schema: bool = True, **kw):
        super().__init__(model, **kw)
        self.name = f"{base_url.split('//')[-1].split('.')[1]}/{model}"
        self.base_url = base_url.rstrip("/")
        self.minimal = minimal_reasoning
        self.strict = strict_schema
        self.key = os.environ.get(key_env, "")

    async def judge(self, call_id: str, prefix: Sequence[Utterance]) -> Verdict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JP.SYSTEM},
                {"role": "user", "content": JP.user_message(prefix)},
            ],
            "max_completion_tokens": self.max_output,
            # Requests from the same call use the same cache key to raise the hit rate (in testing the first round races the cache write)
            "prompt_cache_key": f"bench-{call_id}",
        }
        if self.strict:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "verdict", "strict": True,
                                "schema": JP.RESPONSE_SCHEMA},
            }
        else:
            # DeepSeek doesn't support strict schemas -> json_object mode + client-side validation
            body["response_format"] = {"type": "json_object"}
            body.pop("max_completion_tokens")
            body.pop("prompt_cache_key", None)
            body["max_tokens"] = self.max_output
        if self.minimal:
            if "deepseek" in self.base_url:
                body["reasoning_effort"] = None
                body.pop("reasoning_effort")
                body["thinking"] = {"type": "disabled"}
            else:
                # gpt-6-luna accepts "none"; gpt-6.1-sol only takes low/medium/high/xhigh,
                # error text: does not support 'none' with this model.
                body["reasoning_effort"] = "low" if "sol" in self.model else "none"
        t0 = time.monotonic()
        data = await self._post(f"{self.base_url}/chat/completions", {
            "content-type": "application/json", "authorization": f"Bearer {self.key}",
        }, body)
        ms = (time.monotonic() - t0) * 1000
        u = data.get("usage") or {}
        det = u.get("prompt_tokens_details") or {}
        self.usage.requests += 1
        self.usage.input_tokens += int(u.get("prompt_tokens") or 0)
        self.usage.output_tokens += int(u.get("completion_tokens") or 0)
        self.usage.cached_tokens += int(
            det.get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0)
        self.usage.cache_write_tokens += int(det.get("cache_write_tokens") or 0)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        return self._verdict(text, ms, finish=choice.get("finish_reason"))


class GeminiJudge(BaseJudge):
    """Gemini REST. Implicit caching applies automatically; the hit count is in ``usageMetadata.cachedContentTokenCount``."""

    def __init__(self, model: str = "gemini-3.1-flash-lite", *, minimal_reasoning: bool = False, **kw):
        super().__init__(model, **kw)
        self.name = f"gemini/{model}"
        self.minimal = minimal_reasoning
        self.key = os.environ.get("GEMINI_API_KEY", "")

    async def judge(self, call_id: str, prefix: Sequence[Utterance]) -> Verdict:
        gen: dict[str, Any] = {"maxOutputTokens": self.max_output,
                               "responseMimeType": "application/json"}
        if self.minimal:
            gen["thinkingConfig"] = {"thinkingBudget": 0}
        body = {
            "systemInstruction": {"parts": [{"text": JP.SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": JP.user_message(prefix)}]}],
            "generationConfig": gen,
        }
        t0 = time.monotonic()
        data = await self._post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {"content-type": "application/json", "x-goog-api-key": self.key}, body)
        ms = (time.monotonic() - t0) * 1000
        u = data.get("usageMetadata") or {}
        self.usage.requests += 1
        self.usage.input_tokens += int(u.get("promptTokenCount") or 0)
        self.usage.output_tokens += int(u.get("candidatesTokenCount") or 0)
        self.usage.cached_tokens += int(u.get("cachedContentTokenCount") or 0)
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        return self._verdict(text, ms, finish=cand.get("finishReason"))


class JevJudge(BaseJudge):
    """Jev. ``system_one(state, questions)`` is its only inference method.

    It has **no session and no streaming interface**, so "one score per round" is also implemented as one call per round carrying the accumulated
    state: token cost is likewise O(N²), the same shape as LLM replay. The earlier claim in REQUIREMENTS §6
    that "Jev is natively streaming with no O(N²) cost" is corrected accordingly.
    """

    def __init__(self, model: str = "jev-1.13.0", **kw):
        self.name = f"jev/{model}"
        self.model = model
        self.usage = JudgeUsage()
        self._client = None
        self._sdk = None

    async def aclose(self) -> None:
        return None

    def _client_or_raise(self):
        if self._sdk is None:
            try:
                from typesafe_sdk import TypeSafeClient  # noqa: PLC0415
            except ModuleNotFoundError as exc:
                raise ProviderError(
                    "typesafe_sdk missing: pip install typesafe-sdk==0.7.4"
                ) from exc
            self._sdk = TypeSafeClient(api_key=os.environ.get("JEV_API_KEY", ""))
        return self._sdk

    #: Question name. ``system_one``'s questions are a ``Mapping[str, Question]`` and answers come back under the same name,
    #: so this string is both the request key and the key for reading the answer.
    QUESTION_NAME = "scam_evidence_sufficient"

    async def judge(self, call_id: str, prefix: Sequence[Utterance]) -> Verdict:
        client = self._client_or_raise()
        state = f"CALL SO FAR:\n{JP.render_transcript(prefix)}"
        t0 = time.monotonic()
        # questions must be a dict, not a list: SDK 0.7.4's normalize_questions calls
        # ``questions.items()``. Only found on the first real Jev run: passing a list is an immediate AttributeError.
        res = client.system_one(
            state=state,
            questions={self.QUESTION_NAME: {"type": "noul",
                                            "instructions": JP.JEV_QUESTION}},
        )
        ms = (time.monotonic() - t0) * 1000

        # The probability is in answers[name].noul. The response also lifts each question name to the top level, so both paths are tried;
        # the dict-shaped response is kept as a fallback in case a minor SDK version changes the wrapper.
        p = None
        answers = getattr(res, "answers", None)
        if isinstance(answers, dict):
            a = answers.get(self.QUESTION_NAME)
            p = getattr(a, "noul", None) if a is not None else None
            if p is None and isinstance(a, dict):
                p = a.get("noul")
        if p is None:
            a = getattr(res, self.QUESTION_NAME, None)
            p = getattr(a, "noul", None) if a is not None else None
        if p is None and isinstance(res, dict):
            a = (res.get("answers") or {}).get(self.QUESTION_NAME) or {}
            if isinstance(a, dict):
                p = a.get("noul", a.get("probability"))

        self.usage.requests += 1
        # Use the real usage the server reports, not a character-count estimate
        u = getattr(res, "usage", None)
        if u is not None:
            self.usage.input_tokens += int(getattr(u, "input_tokens", 0) or 0)
            self.usage.output_tokens += int(getattr(u, "output_tokens", 0) or 0)
        else:
            self.usage.input_tokens += len(state) // 4 + 276
        if p is None:
            self.usage.parse_failures += 1
        return Verdict(
            risk_score=float(p) if p is not None else 0.0,
            is_alarm=bool(p is not None and float(p) >= 0.5),
            rationale="", latency_ms=ms,
        )


#: Command-line name -> constructor. One expensive and one cheap per provider; prices in EVAL_SYSTEM §6.
JUDGES: dict[str, Any] = {
    # mock has no "reasoning level" to speak of; drop that argument rather than error. It exists to make the whole pipeline runnable
    "mock": lambda **kw: MockJudge(
        **{k: v for k, v in kw.items() if k not in ("minimal_reasoning", "cache")}),
    "claude-haiku": lambda **kw: AnthropicJudge("claude-haiku-5-5", **kw),
    "claude-opus": lambda **kw: AnthropicJudge("claude-opus-5-5", **kw),
    # sonnet-5-5 is what generated these dialogues; as a baseline it must be marked separately as "same model as the generator"
    "claude-sonnet": lambda **kw: AnthropicJudge("claude-sonnet-5-5", **kw),
    "gpt-luna": lambda **kw: OpenAICompatJudge("gpt-6-luna", **kw),
    "gpt-sol": lambda **kw: OpenAICompatJudge("gpt-6.1-sol", **kw),
    "gemini-flash-lite": lambda **kw: GeminiJudge("gemini-3.1-flash-lite", **kw),
    "gemini-argon": lambda **kw: GeminiJudge("gemini-4-argon", **kw),
    "deepseek-flash": lambda **kw: OpenAICompatJudge(
        "deepseek-flash", base_url="https://api.deepseek.com",
        key_env="DEEPSEEK_API_KEY", strict_schema=False, **kw),
    "deepseek-pro": lambda **kw: OpenAICompatJudge(
        "deepseek-v4-pro", base_url="https://api.deepseek.com",
        key_env="DEEPSEEK_API_KEY", strict_schema=False, **kw),
    "jev": lambda **kw: JevJudge(**{k: v for k, v in kw.items() if k != "minimal_reasoning"}),
}
