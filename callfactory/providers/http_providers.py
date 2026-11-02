"""OpenAI-compatible and Anthropic backends over HTTP."""

from __future__ import annotations

from typing import Any

from .base import RETRYABLE_STATUS, LLMProvider, ProviderConfig, ProviderError, resolve_api_key


def _httpx():
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import guard
        raise ProviderError(
            "httpx is required for network providers: pip install httpx"
        ) from exc
    return httpx


class _HTTPProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        httpx = _httpx()
        self._client = httpx.AsyncClient(timeout=config.timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        httpx = _httpx()
        try:
            resp = await self._client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"timeout calling {url}", retryable=True) from exc
        except httpx.TransportError as exc:
            raise ProviderError(f"transport error calling {url}: {exc}", retryable=True) from exc
        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.config.name} HTTP {resp.status_code}: {resp.text[:400]}",
                status=resp.status_code,
                retryable=resp.status_code in RETRYABLE_STATUS,
            )
        return resp.json()


class OpenAICompatProvider(_HTTPProvider):
    """``/chat/completions`` -- OpenAI, Fireworks, Together, vLLM, Ollama, ...

    OpenAI and Fireworks both honour ``seed`` on a best-effort basis; it narrows
    run-to-run drift but is not a hard reproducibility guarantee.
    """

    supports_seed = True
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = resolve_api_key(
            config.api_key, ("OPENAI_API_KEY", "FIREWORKS_API_KEY", "LLM_API_KEY")
        )

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag) -> str:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self.config.extra_body,
        }
        if seed is not None:
            body["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = await self._post(f"{self.base_url}/chat/completions", headers, body)
        usage = data.get("usage") or {}
        self._add_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(data)[:300]}") from exc


class FireworksProvider(OpenAICompatProvider):
    DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


class AnthropicProvider(_HTTPProvider):
    """Anthropic Messages API.

    Two parameters this project would like are simply not available here:

    * **no seed** -- runs are not bit-reproducible, so every record carries
      ``seed_honoured_by_provider: false`` (N5).
    * **no temperature** -- the Messages API rejects non-default ``temperature`` /
      ``top_p`` / ``top_k`` with a 400 (measured on claude-sonnet-5-5:
      ``temperature`` 0.9 -> 400, 1.0 -> accepted because it is the default).
      Dialogue diversity therefore comes from the task cross product and the
      per-turn directives, never from the sampler.

    Thinking is off by default (``between_tools``): these turns are role-play, not
    reasoning, and in a measured probe leaving it on roughly doubled the output
    tokens per turn (122 vs 63) for no gain. Override via ``extra_body``.
    """

    supports_seed = False
    supports_temperature = False
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"
    DEFAULT_THINKING: dict[str, Any] = {"type": "between_tools"}

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = resolve_api_key(config.api_key, ("ANTHROPIC_API_KEY",))

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag) -> str:
        # Send neither temperature nor seed: the first returns 400, and this API has no second.
        body: dict[str, Any] = {
            "model": self.config.model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": dict(self.DEFAULT_THINKING),
            **self.config.extra_body,
        }
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        data = await self._post(f"{self.base_url}/v1/messages", headers, body)
        usage = data.get("usage") or {}
        self._add_usage(usage.get("input_tokens"), usage.get("output_tokens"))
        try:
            parts = [b.get("text", "") for b in data["content"] if b.get("type") == "text"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {str(data)[:300]}") from exc
        return "".join(parts).strip()
