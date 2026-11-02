"""The scenario author agent (component 1).

Its job is to **extend** the scenario bank, not build one from scratch, so existing scenarios must be passed in as dedup context
(R1.1); otherwise the second call generates more of what the first one did.

Three gates, in a fixed order:

1. **Schema hard rules** (``validate_scenario_dict``): out-of-vocabulary signals, hard negatives missing a ``disambiguator``,
   and scenarios declaring style-level signals are dropped whole, with no guesswork repairs (R1.3).
2. **Semantic near-duplicates**: the family name and the pretext are each compared with existing scenarios (including ones accepted in this batch), and anything over the threshold is rejected
   (R1.2). Checking only the schema and not duplicates, the agent would rewrite "courier impersonation" as "courier fee scam" and hand it in again.
3. **The outcome vocabulary is set by the dataset design, not by the author.** The author writes only the premise; compliant and resisting outcomes are
   attached here uniformly, so the R2.8 property ("break the 'receiver resists = scam' shortcut") holds across the **whole bank**,
   not just in the hand-written seed bank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from .prompts import (
    SCENARIO_AUTHOR_SYSTEM,
    SCENARIO_AUTHOR_VERSION,
    scenario_author_user_message,
)
from .providers import LLMProvider, ProviderError, extract_json
from .scenarios.bank import ScenarioBank, validate_scenario_dict
from .seed_scenarios import COMPLIANT_SCAM_OUTCOMES, LEGIT_OUTCOMES, SCAM_OUTCOMES
from .validators import similarity

#: Near-duplicate threshold for pretexts.
#:
#: Family names are **checked for exact duplicates only, no fuzzy matching**; testing showed it's the wrong tool: catching
#: parcel_delivery_fee vs parcel_delivery_charge (unigram 0.50) would need the threshold down at 0.5,
#: which would also reject investment_pressure vs crypto_investment_pressure (0.67),
#: a pair that coexists in the seed bank on purpose (general investment pressure / a crypto-specific scenario).
#: Name similarity measures naming habits, not duplication; leave the semantic judgement to the pretext.
PRETEXT_DUP = 0.5

#: Maximum number of scenarios to ask for per call. In testing, asking for 25 at once pushed output to the max_tokens limit, the JSON was truncated,
#: and all three rounds were wasted. Several rounds have a side benefit too: each round's accepted items join the next round's dedup context.
PER_CALL_MAX = 10
#: Cap on a single reply. A record is about 150-200 tokens; leave plenty of headroom.
MAX_TOKENS = 16000


@dataclass
class Rejection:
    family: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "reasons": self.reasons}


@dataclass
class AuthorResult:
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    attempts: int = 0
    raw_proposed: int = 0
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "accepted": len(self.accepted),
            "rejected": len(self.rejected),
            "attempts": self.attempts,
            "proposed": self.raw_proposed,
            "model": self.model,
            "usage": dict(self.usage),
            "rejections": [r.to_dict() for r in self.rejected],
        }


def _norm_family(name: Any) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalise(raw: dict[str, Any], *, bank_name: str, seq: int) -> dict[str, Any]:
    """Complete a model-supplied object into a bankable record. The outcome vocabulary is attached here; see the module docstring."""
    category = "scam" if str(raw.get("category", "scam")).lower() == "scam" else "legitimate"
    outcomes = list(SCAM_OUTCOMES if category == "scam" else LEGIT_OUTCOMES)
    return {
        "scenario_id": f"scn-{bank_name}-{seq:03d}",
        "category": category,
        "family": _norm_family(raw.get("family")),
        "pretext": str(raw.get("pretext", "")).strip(),
        "org_kind": str(raw.get("org_kind", "")).strip(),
        "expected_signs": [
            str(s).strip().lower().replace(" ", "_").replace("-", "_")
            for s in (raw.get("expected_signs") or [])
            if isinstance(s, str)
        ],
        "plausible_outcomes": outcomes,
        "hard_negative": bool(raw.get("hard_negative", False)) and category == "legitimate",
        "disambiguator": str(raw.get("disambiguator", "")).strip(),
        "late_onset": bool(raw.get("late_onset", False)) and category == "scam",
        "compliant_outcomes": (
            [o for o in COMPLIANT_SCAM_OUTCOMES if o in outcomes] if category == "scam" else []
        ),
        "no_observable_evidence": False,
        "eval_only": False,
        "notes": str(raw.get("notes", "")).strip(),
    }


def duplicate_reasons(
    record: dict[str, Any],
    families: Sequence[str],
    pretexts: Sequence[str],
) -> list[str]:
    """R1.2: semantic near-duplicate check. Family names checked for exact duplicates, pretexts for near-duplicates."""
    out: list[str] = []
    family = record["family"]
    if family in set(families):
        out.append(f"family {family!r} already exists")
        return out
    for other in pretexts:
        # Trigrams miss near-duplicates that "differ by one word" (changing one word only touches 3/9 trigrams,
        # so similarity drops to 0.33); bigrams are more sensitive to short-range differences, so take the larger of the two.
        score = max(
            similarity(record["pretext"], other, n=3),
            similarity(record["pretext"], other, n=2),
        )
        if score >= PRETEXT_DUP:
            out.append(
                f"pretext is a near-duplicate of an existing scenario (jaccard={score:.2f}): {other[:70]!r}"
            )
            break
    return out


async def author_scenarios(
    provider: LLMProvider,
    bank: ScenarioBank | None,
    count: int,
    *,
    context: str = "",
    gaps: dict[str, Any] | None = None,
    pending: Sequence[dict[str, Any]] = (),
    bank_name: str = "seed",
    max_attempts: int = 6,
    temperature: float = 1.0,
    seed: int | None = None,
) -> AuthorResult:
    """Generate ``count`` **new** scenarios. If short, ask for another round, up to ``max_attempts`` rounds."""
    existing = list(bank.scenarios) if bank else []
    families = [s.family for s in existing]
    pretexts = [s.pretext for s in existing]
    next_seq = len(existing)

    # Candidates generated but not yet committed must join the dedup context too. They aren't in the bank, so the bank can't see
    # them; without passing them in, the second extension round rewrites the same kinds as the first, and the dedup check still
    # lets them through, because it compares against the bank.
    for rec in pending:
        families.append(str(rec.get("family", "")))
        pretexts.append(str(rec.get("pretext", "")))
    next_seq += len(pending)

    result = AuthorResult(model=provider.model)
    for attempt in range(1, max_attempts + 1):
        need = count - len(result.accepted)
        if need <= 0:
            break
        result.attempts = attempt
        ask = min(need, PER_CALL_MAX)
        message = scenario_author_user_message(
            families, pretexts, ask, context=context, gaps=gaps
        )
        try:
            raw = await provider.complete(
                system=SCENARIO_AUTHOR_SYSTEM,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                seed=seed,
                max_tokens=MAX_TOKENS,
                tag="scenario_author",
            )
        except ProviderError as exc:
            result.rejected.append(Rejection("(call failed)", [f"{type(exc).__name__}: {exc}"]))
            break
        try:
            payload = extract_json(raw)
        except (ValueError, TypeError) as exc:
            # Report truncation separately from "the model didn't answer in format": the first means raising max_tokens or asking for fewer per call,
            # the second means fixing the prompt. Lumping them into one "parse failed" sends people to fix the wrong thing.
            truncated = raw.rstrip().endswith(("," , '"', ":", "{", "[")) or (
                raw.count("{") > raw.count("}")
            )
            why = "reply truncated (raise max_tokens or ask for fewer per call)" if truncated else "not valid JSON"
            result.rejected.append(Rejection("(parse failed)", [f"{why}: {exc}"]))
            continue

        proposed = payload.get("scenarios")
        if not isinstance(proposed, list):
            result.rejected.append(Rejection("(wrong shape)", ["no scenarios list at the top level"]))
            continue
        result.raw_proposed += len(proposed)

        for item in proposed:
            if len(result.accepted) >= count:
                break
            if not isinstance(item, dict):
                result.rejected.append(Rejection("(not an object)", ["not a JSON object"]))
                continue
            record = normalise(item, bank_name=bank_name, seq=next_seq)
            errors = validate_scenario_dict(record)
            errors += duplicate_reasons(record, families, pretexts)
            if errors:
                result.rejected.append(Rejection(record["family"] or "(unnamed)", errors))
                continue
            record["provenance"] = {
                "author_model": provider.model,
                "bank_version": (bank.version if bank else 0) + 1,
                "context": context,
                "rubric_version": SCENARIO_AUTHOR_VERSION,
                "edited_by_human": False,
            }
            result.accepted.append(record)
            # Items accepted in this batch join the dedup context too, so one call doesn't repeat itself
            families.append(record["family"])
            pretexts.append(record["pretext"])
            next_seq += 1

    result.usage = dict(provider.usage)
    return result


def merge_into_bank(
    bank: ScenarioBank | None,
    accepted: Sequence[dict[str, Any]],
    *,
    bank_name: str = "seed",
) -> list[dict[str, Any]]:
    """Existing scenarios + new ones, with scenario_id renumbered in bank order.

    Renumbered rather than kept: scenario_id is only a stable handle inside the bank, while ``(bank_version, seed, index)``
    is the provenance anchor for data, so renumbering doesn't cost generated data its traceability: that data's
    metadata stores a **full copy** of the scenario, not a pointer.
    """
    out: list[dict[str, Any]] = [scn.to_dict() for scn in (bank.scenarios if bank else [])]
    out.extend(dict(record) for record in accepted)
    for i, record in enumerate(out):
        record["scenario_id"] = f"scn-{bank_name}-{i:03d}"
    return out
