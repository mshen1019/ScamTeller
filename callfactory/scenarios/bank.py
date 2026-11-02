"""Reading, writing, versioning and admission checks for scenario banks.

Banks are versioned plain-text JSONL (R1.5 / R1.7): every write produces a new version file and nothing is rewritten in place;
after a manual edit, bumping the version number makes it available to the sampler. File names look like ``seed.v3.jsonl``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..schema import (
    CATEGORIES,
    RISK_SIGNALS,
    SCENARIO_SIGNALS,
    STRONG_SIGNALS,
    STYLE_ONLY_SIGNALS,
    Scenario,
    ScenarioProvenance,
)

BANK_FILE_RE = re.compile(r"^(?P<name>.+)\.v(?P<version>\d+)\.jsonl$")


class BankError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Admission checks
# --------------------------------------------------------------------------- #

REQUIRED_FIELDS = (
    "scenario_id", "category", "family", "pretext", "org_kind",
    "expected_signs", "plausible_outcomes",
)


def validate_scenario_dict(data: Any) -> list[str]:
    """Return a list of errors. An empty list means the scenario can be admitted.

    These are hard rules with no guesswork repairs: a model output that fails is dropped whole (R1.3).
    """
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["not a JSON object"]

    for f in REQUIRED_FIELDS:
        if f not in data:
            errors.append(f"missing field {f!r}")
    if errors:
        return errors

    if data["category"] not in CATEGORIES:
        errors.append(f"category must be one of {CATEGORIES}, got {data['category']!r}")
    if not isinstance(data["family"], str) or not re.fullmatch(r"[a-z][a-z0-9_]*", data["family"] or ""):
        errors.append(f"family must be snake_case, got {data.get('family')!r}")
    for f in ("pretext", "org_kind"):
        if not isinstance(data[f], str) or len(data[f].strip()) < 8:
            errors.append(f"{f} too short or not a string")

    signs = data["expected_signs"]
    if not isinstance(signs, list):
        errors.append("expected_signs must be a list")
    else:
        bad = [s for s in signs if s not in RISK_SIGNALS]
        if bad:
            errors.append(f"expected_signs contains words outside the closed vocabulary: {bad}")
        style = [s for s in signs if s in STYLE_ONLY_SIGNALS]
        if style:
            errors.append(
                f"expected_signs contains style-level signals {style}: these have nothing to do with the call's purpose, "
                "come from the caller persona's style_signals, and don't belong in a scenario"
            )

    outcomes = data["plausible_outcomes"]
    # The cap is 10: the seed bank's shared outcome pool has to hold both resisting and compliant outcomes.
    # The scenario author agent still produces 2-4 per R1; this just doesn't block the hand-written bank.
    if not isinstance(outcomes, list) or not 2 <= len(outcomes) <= 10:
        errors.append("plausible_outcomes needs 2-10 items")

    compliant = data.get("compliant_outcomes") or []
    if not isinstance(compliant, list):
        errors.append("compliant_outcomes must be a list")
    else:
        if compliant and data["category"] != "scam":
            errors.append("compliant_outcomes only makes sense for scam: compliance on a legitimate call is just ordinary business")
        stray = [o for o in compliant if o not in (outcomes if isinstance(outcomes, list) else [])]
        if stray:
            errors.append(f"compliant_outcomes must be a subset of plausible_outcomes; extra: {stray}")

    if data.get("no_observable_evidence"):
        if data["category"] != "scam":
            errors.append("no_observable_evidence can only be used for scam scenarios")
        strong = [s for s in (signs if isinstance(signs, list) else []) if s in STRONG_SIGNALS]
        if strong:
            # Self-contradiction: declares a signal that is "enough on its own" yet says there's no usable evidence throughout
            errors.append(
                f"no_observable_evidence conflicts with strong signals {strong}: "
                "one of these is enough on its own, so a scenario can't declare it and also claim there's no usable evidence"
            )
        if compliant:
            errors.append("a no_observable_evidence scenario shouldn't have compliant outcomes: it never asks for anything")

    hard_negative = data.get("hard_negative", False)
    if hard_negative:
        if data["category"] != "legitimate":
            errors.append("hard_negative can only be used for legitimate scenarios")
        if not signs:
            errors.append("a hard_negative's expected_signs must not be empty, otherwise it's just an ordinary negative")
        if not str(data.get("disambiguator", "")).strip():
            # R1.4
            errors.append("a hard_negative must fill in disambiguator: how it behaves like a real organisation when challenged")
    if data.get("late_onset") and data["category"] != "scam":
        errors.append("late_onset can only be used for scam scenarios")
    return errors


# --------------------------------------------------------------------------- #
# Bank object
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioBank:
    name: str
    version: int
    scenarios: list[Scenario]
    path: Path | None = None

    def __len__(self) -> int:
        return len(self.scenarios)

    @property
    def families(self) -> list[str]:
        return [s.family for s in self.scenarios]

    def by_category(self, category: str) -> list[Scenario]:
        return [s for s in self.scenarios if s.category == category]

    @property
    def scams(self) -> list[Scenario]:
        return self.by_category("scam")

    @property
    def hard_negatives(self) -> list[Scenario]:
        return [s for s in self.scenarios if s.hard_negative]

    @property
    def plain_negatives(self) -> list[Scenario]:
        return [s for s in self.scenarios if s.category == "legitimate" and not s.hard_negative]

    def filter_families(
        self, include: Iterable[str] | None = None, exclude: Iterable[str] | None = None
    ) -> "ScenarioBank":
        """R2.4 targeted generation. An empty ``include`` means no restriction."""
        inc = set(include or ())
        exc = set(exclude or ())
        unknown = (inc | exc) - set(self.families)
        if unknown:
            raise BankError(f"no such families in the bank: {sorted(unknown)}")
        kept = [
            s for s in self.scenarios
            if (not inc or s.family in inc) and s.family not in exc
        ]
        if not kept:
            raise BankError("no scenarios left after filtering")
        return ScenarioBank(self.name, self.version, kept, self.path)

    def coverage_gaps(self, callers: Iterable[Any] = ()) -> dict[str, Any]:
        """R1.12: report coverage gaps for a human to decide what to add. Zero model calls.

        Scenario gaps are computed only for **scenario-level** signals. Style-level signals (``STYLE_ONLY_SIGNALS``) have nothing to do with
        the call's purpose and come from the caller persona; counting them together reports false gaps. Pass the persona bank to check both.
        """
        covered = {sig for s in self.scams for sig in s.expected_signs}
        hn_signs = {sig for s in self.hard_negatives for sig in s.expected_signs}
        unpaired = [
            s.family for s in self.scams
            if not (set(s.expected_signs) & hn_signs)
        ]
        persona_signals = {
            sig for c in callers for sig in getattr(c, "style_signals", [])
        }
        compliant = [s.family for s in self.scams if s.compliant_outcomes]
        return {
            "signals_never_used_by_scam": sorted(set(SCENARIO_SIGNALS) - covered),
            "scam_families_with_compliant_outcomes": compliant,
            # Scam families with no compliant outcome: they only produce "receiver resists" samples,
            # so a model can score by reading the receiver (a shortcut seen in testing). That's a gap to fill.
            "scam_families_without_compliant_outcome": [
                s.family for s in self.scams
                if not s.compliant_outcomes and not s.no_observable_evidence
            ],
            "no_observable_evidence_families": [
                s.family for s in self.scams if s.no_observable_evidence
            ],
            "eval_only_families": [s.family for s in self.scenarios if s.eval_only],
            "signals_without_hard_negative": sorted(covered - hn_signs),
            "scam_families_without_paired_hard_negative": unpaired,
            "late_onset_capable": [s.family for s in self.scams if s.late_onset],
            "style_signals_covered_by_personas": sorted(
                persona_signals & set(STYLE_ONLY_SIGNALS)
            ),
            "style_signals_with_no_persona": sorted(
                set(STYLE_ONLY_SIGNALS) - persona_signals
            ) if callers else None,
        }


# --------------------------------------------------------------------------- #
# Disk
# --------------------------------------------------------------------------- #

def bank_path(directory: str | Path, name: str, version: int) -> Path:
    return Path(directory) / f"{name}.v{version}.jsonl"


def load_bank(path: str | Path) -> ScenarioBank:
    path = Path(path)
    m = BANK_FILE_RE.match(path.name)
    if not m:
        raise BankError(f"file name doesn't match <name>.v<N>.jsonl: {path.name}")
    if not path.exists():
        raise BankError(f"scenario bank not found: {path}")

    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    seen_families: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BankError(f"{path.name}:{lineno} JSON parse failed: {exc}") from exc
        errors = validate_scenario_dict(data)
        if errors:
            raise BankError(f"{path.name}:{lineno} invalid scenario: " + "; ".join(errors))
        if data["scenario_id"] in seen_ids:
            raise BankError(f"{path.name}:{lineno} duplicate scenario_id: {data['scenario_id']}")
        if data["family"] in seen_families:
            raise BankError(f"{path.name}:{lineno} duplicate family: {data['family']}")
        seen_ids.add(data["scenario_id"])
        seen_families.add(data["family"])
        scenarios.append(Scenario.from_dict(data))

    if not scenarios:
        raise BankError(f"scenario bank is empty: {path}")
    return ScenarioBank(
        name=m.group("name"), version=int(m.group("version")),
        scenarios=scenarios, path=path,
    )


def latest_bank(directory: str | Path, name: str | None = None) -> ScenarioBank:
    """Load the highest-versioned bank in a directory. With no ``name``, the directory must hold only one bank name."""
    directory = Path(directory)
    found: dict[str, list[int]] = {}
    for p in directory.glob("*.jsonl"):
        m = BANK_FILE_RE.match(p.name)
        if m:
            found.setdefault(m.group("name"), []).append(int(m.group("version")))
    if not found:
        raise BankError(f"no scenario bank under {directory} (<name>.v<N>.jsonl)")
    if name is None:
        if len(found) > 1:
            raise BankError(f"multiple bank names in the directory {sorted(found)}; pick one with --bank-name")
        name = next(iter(found))
    if name not in found:
        raise BankError(f"no bank named {name!r}; available: {sorted(found)}")
    return load_bank(bank_path(directory, name, max(found[name])))


def write_bank(
    scenarios: Iterable[dict[str, Any] | Scenario],
    directory: str | Path,
    name: str,
    version: int,
    *,
    author_model: str = "human",
    context: str = "",
    overwrite: bool = False,
) -> Path:
    """Write out a new version. Refuses to overwrite an existing version by default (R1.5)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = bank_path(directory, name, version)
    if path.exists() and not overwrite:
        raise BankError(f"version already exists, not rewriting in place: {path} (bump the version number)")

    records: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for item in scenarios:
        data = item.to_dict() if isinstance(item, Scenario) else dict(item)
        data.setdefault("hard_negative", False)
        data.setdefault("disambiguator", "")
        data.setdefault("late_onset", False)
        data.setdefault("notes", "")
        errors = validate_scenario_dict(data)
        if errors:
            raise BankError(f"{data.get('family')!r} invalid: " + "; ".join(errors))
        if data["family"] in seen_families:
            raise BankError(f"duplicate family: {data['family']}")
        seen_families.add(data["family"])
        if not data.get("provenance"):
            data["provenance"] = ScenarioProvenance(
                author_model=author_model, bank_version=version, context=context,
                edited_by_human=(author_model == "human"),
            ).to_dict()
        else:
            data["provenance"]["bank_version"] = version
        records.append(data)

    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path
