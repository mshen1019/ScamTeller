"""Binding roles to providers, and the P4 same-source check.

P4: the judge and the judged must not be the same model. The prototype scam-convo-gen defaulted to
``evaluator_provider or caller_provider``, i.e. the same model grading itself. This turns that
into an explicit check that refuses to start by default.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import LLMProvider


class ProviderIndependenceError(RuntimeError):
    pass


@dataclass
class ProviderRoles:
    caller: LLMProvider
    receiver: LLMProvider
    evaluator: LLMProvider
    gatekeeper: LLMProvider | None = None
    scenario_author: LLMProvider | None = None

    def identity(self, provider: LLMProvider | None) -> tuple[str, str] | None:
        if provider is None:
            return None
        return (provider.config.name, provider.model)

    def violations(self) -> list[str]:
        """Return descriptions of P4 violations. Mock providers don't count as violating (offline development)."""
        out: list[str] = []
        ev = self.identity(self.evaluator)
        if ev and ev[0] != "mock":
            if ev == self.identity(self.caller):
                out.append(
                    f"evaluator and caller share a source ({ev[0]}/{ev[1]}): labels produced by the party being labelled"
                )
            if ev == self.identity(self.receiver):
                out.append(
                    f"evaluator and receiver share a source ({ev[0]}/{ev[1]}): labels produced by the party being labelled"
                )
        gk = self.identity(self.gatekeeper)
        if gk and gk[0] != "mock" and gk == ev:
            out.append(
                f"gatekeeper and evaluator share a source ({gk[0]}/{gk[1]}): scores and labels from the same source, "
                "so the Detection Value axis is self-confirmation"
            )
        return out

    def enforce(self, *, allow_same_provider: bool = False) -> list[str]:
        """Raise on a P4 violation; if explicitly allowed, return the warnings for the caller to record in metadata."""
        issues = self.violations()
        if issues and not allow_same_provider:
            raise ProviderIndependenceError(
                "violates P4 (judge and judged must come from different sources):\n  - "
                + "\n  - ".join(issues)
                + "\nTo allow it explicitly pass allow_same_provider=True / --allow-same-provider; "
                "this is recorded in every record's metadata."
            )
        return issues

    def all_providers(self) -> list[LLMProvider]:
        seen: dict[int, LLMProvider] = {}
        for p in (self.caller, self.receiver, self.evaluator,
                  self.gatekeeper, self.scenario_author):
            if p is not None:
                seen[id(p)] = p
        return list(seen.values())

    def describe(self) -> dict[str, str | None]:
        def fmt(p: LLMProvider | None) -> str | None:
            return None if p is None else f"{p.config.name}/{p.model}"

        return {
            "caller": fmt(self.caller),
            "receiver": fmt(self.receiver),
            "evaluator": fmt(self.evaluator),
            "gatekeeper": fmt(self.gatekeeper),
            "scenario_author": fmt(self.scenario_author),
        }
