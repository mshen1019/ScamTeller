"""Exercise the real HTTP paths against a local stub, since CI has no API keys."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from callfactory.providers import build_provider
from callfactory.providers.base import ProviderError

pytest.importorskip("httpx")

STATE = {"fail_next": 0, "requests": []}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):  # keep pytest output clean
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        STATE["requests"].append({"path": self.path, "body": body, "headers": dict(self.headers)})
        if STATE["fail_next"] > 0:
            STATE["fail_next"] -= 1
            self._send(429, {"error": "slow down"})
            return
        if "/wrong" in self.path:
            self._send(404, {"error": "no such route"})
        elif self.path.endswith("/chat/completions"):
            self._send(200, {"choices": [{"message": {"content": "openai-style reply"}}]})
        elif self.path.endswith("/v1/messages"):
            self._send(200, {"content": [{"type": "text", "text": "anthropic-style reply"}]})
        else:
            self._send(404, {"error": "no such route"})

    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def server():
    STATE["fail_next"] = 0
    STATE["requests"].clear()
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def call(provider, **kwargs):
    async def go():
        try:
            return await provider.complete(
                system="sys", messages=[{"role": "user", "content": "hi"}], **kwargs
            )
        finally:
            await provider.aclose()

    return asyncio.run(go())


def test_openai_compatible_request_and_response(server):
    provider = build_provider("openai", model="m1", base_url=server, api_key="k", max_tokens=99)
    assert call(provider, seed=7) == "openai-style reply"
    req = STATE["requests"][-1]
    assert req["path"].endswith("/chat/completions")
    assert req["body"]["model"] == "m1"
    assert req["body"]["seed"] == 7
    assert req["body"]["max_tokens"] == 99
    assert req["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert req["headers"]["Authorization"] == "Bearer k"


def test_fireworks_uses_its_own_default_base_url():
    provider = build_provider("fireworks")
    assert "fireworks.ai" in provider.base_url


def test_anthropic_request_shape(server):
    provider = build_provider("anthropic", model="claude-x", base_url=server, api_key="k")
    assert call(provider, seed=7) == "anthropic-style reply"
    req = STATE["requests"][-1]
    assert req["path"].endswith("/v1/messages")
    assert req["body"]["system"] == "sys"
    assert "seed" not in req["body"], "the Messages API has no seed parameter"
    assert req["headers"]["x-api-key"] == "k"
    assert req["headers"]["anthropic-version"]


def test_429_is_retried(server):
    STATE["fail_next"] = 2
    provider = build_provider(
        "openai", model="m", base_url=server, api_key="k", max_retries=3, initial_backoff=0.0
    )
    assert call(provider) == "openai-style reply"
    assert len(STATE["requests"]) == 3


def test_404_is_not_retried(server):
    provider = build_provider(
        "openai", model="m", base_url=f"{server}/wrong", api_key="k",
        max_retries=3, initial_backoff=0.0,
    )
    with pytest.raises(ProviderError) as exc:
        call(provider)
    assert exc.value.status == 404
    assert len(STATE["requests"]) == 1


def test_retries_are_exhausted_then_raised(server):
    STATE["fail_next"] = 99
    provider = build_provider(
        "openai", model="m", base_url=server, api_key="k", max_retries=2, initial_backoff=0.0
    )
    with pytest.raises(ProviderError):
        call(provider)
    assert len(STATE["requests"]) == 3


# --------------------------------------------------------------------------- #
# Claude's Messages API no longer accepts temperature (400 in testing) and has no seed
# --------------------------------------------------------------------------- #

def test_anthropic_request_carries_no_sampling_params(server):
    provider = build_provider("anthropic", base_url=server, api_key="k", temperature=0.9)
    assert call(provider) == "anthropic-style reply"
    body = STATE["requests"][-1]["body"]
    for banned in ("temperature", "top_p", "top_k", "seed"):
        assert banned not in body, f"{banned} makes Claude return 400"
    assert body["thinking"] == {"type": "between_tools"}
    assert provider.supports_temperature is False
    assert provider.supports_seed is False


def test_anthropic_thinking_can_be_overridden(server):
    provider = build_provider(
        "anthropic", base_url=server, api_key="k",
        extra_body={"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
    )
    call(provider)
    body = STATE["requests"][-1]["body"]
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low"}


def test_openai_compat_still_sends_temperature_and_seed(server):
    provider = build_provider("openai", base_url=server, api_key="k", temperature=0.0)
    call(provider, seed=11)
    body = STATE["requests"][-1]["body"]
    assert body["temperature"] == 0.0
    assert body["seed"] == 11
    assert provider.supports_temperature is True
