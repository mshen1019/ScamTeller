"""Provider registry."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider, ProviderConfig, ProviderError, extract_json
from .roles import ProviderIndependenceError, ProviderRoles
from .http_providers import AnthropicProvider, FireworksProvider, OpenAICompatProvider
from .mock import MockProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAICompatProvider,
    "openai-compat": OpenAICompatProvider,
    "fireworks": FireworksProvider,
    "anthropic": AnthropicProvider,
    "mock": MockProvider,
}

#: Sensible starting points only -- model names move, pass ``--model`` to pin one.
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "openai-compat": "gpt-4.1-mini",
    "fireworks": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "anthropic": "claude-sonnet-5-5",
    "mock": "mock-dialogue-1",
}


def build_provider(
    name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **config_kwargs: Any,
) -> LLMProvider:
    key = name.lower()
    if key not in PROVIDERS:
        raise ValueError(
            f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}"
        )
    config = ProviderConfig(
        name=key,
        model=model or DEFAULT_MODELS[key],
        base_url=base_url,
        api_key=api_key,
        **config_kwargs,
    )
    return PROVIDERS[key](config)


__all__ = [
    "LLMProvider",
    "ProviderRoles",
    "ProviderIndependenceError",
    "ProviderConfig",
    "ProviderError",
    "PROVIDERS",
    "DEFAULT_MODELS",
    "build_provider",
    "extract_json",
]
