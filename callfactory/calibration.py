"""Splitting the calibration set and computing agreement (R6.5).

Two steps, and the order can't be reversed:

1. **Split first, then look at the data.** `make_split` uses a fixed seed to set aside part of the items as ``holdout``
   before anyone (ourselves included) analyses the batch. The rubric is tuned only against ``tune`` and checked once at the end
   on ``holdout``. The split has to come before analysis, otherwise "held out" is just a slogan: once all the data
   has been seen before splitting, the holdout set has already taken part in the decisions.
2. **Then compute agreement.** Align human and machine labels and give the numbers R6.5 asks for:
   Cohen's kappa on ``is_scam``, the distribution of detection-point differences (positive = machine later than human,
   negative = machine earlier), and agreement sliced by hard negative / compliant / slow-burn.

Note what low agreement means: **the rubric's anchors aren't clear enough; fix the rubric, not the threshold**.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

SPLIT_FILE = "_calibration_split.json"


@dataclass
class CalibrationSplit:
    seed: int
    tune: list[str]
    holdout: list[str]
    created_at: float = field(default_factory=time.time)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __contains__(self, conversation_id: str) -> bool:
        return conversation_id in set(self.tune) | set(self.holdout)


def split_path(run: str | Path) -> Path:
    return Path(run) / SPLIT_FILE


def load_split(run: str | Path) -> CalibrationSplit | None:
    path = split_path(run)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationSplit(
        seed=data["seed"], tune=list(data["tune"]), holdout=list(data["holdout"]),
        created_at=data.get("created_at", 0.0), note=data.get("note", ""),
    )


def make_split(
    run: str | Path,
    conversation_ids: Sequence[str],
    *,
    holdout: int,
    seed: int,
    note: str = "",
    overwrite: bool = False,
) -> CalibrationSplit:
    """Draw ``holdout`` items as the holdout set. Refuses to overwrite an existing one.

    Refusing is deliberate: redrawing the holdout set swaps out items that have already been seen,
    which turns the holdout set into a second tuning set.
    """
    path = split_path(run)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Redrawing the holdout set swaps out items that have been seen, "
            "turning it into a second tuning set; to really start over pass overwrite=True"
        )
    ids = sorted(conversation_ids)
    if not 0 < holdout < len(ids):
        raise ValueError(f"holdout must be within 1..{len(ids) - 1}, got {holdout}")
    rng = random.Random(seed)
    picked = sorted(rng.sample(ids, holdout))
    split = CalibrationSplit(
        seed=seed,
        tune=[i for i in ids if i not in set(picked)],
        holdout=picked,
        note=note,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(split.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return split


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #

@dataclass
class AgreementReport:
    n: int = 0
    #: Binary (scam or not)
    both_scam: int = 0
    both_legit: int = 0
    machine_only: int = 0              # machine says scam, human says not
    human_only: int = 0                # human says scam, machine says not
    unsure: int = 0                    # human marked "unsure"; not counted in kappa
    kappa: float | None = None
    #: Detection points (items where both sides marked an alarm point)
    both_flagged: int = 0
    deltas: list[int] = field(default_factory=list)   # machine - human, positive = machine later
    median_delta: float | None = None
    machine_earlier: int = 0
    machine_later: int = 0
    exact: int = 0
    #: Only one side marked an alarm point
    machine_flagged_only: int = 0
    human_flagged_only: int = 0
    neither_flagged: int = 0

    def summary(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("deltas")
        out["mean_abs_delta_utterances"] = (
            round(statistics.mean(abs(d) for d in self.deltas), 2) if self.deltas else None
        )
        return out


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Cohen's kappa between two binary annotators. Returns None when everything agrees with no variation."""
    if len(a) != len(b):
        raise ValueError("the two label lists must have the same length")
    n = len(a)
    if n == 0:
        return None
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        # Both sides put every sample in the same class: kappa is undefined (no variation for "chance agreement")
        return None
    return round((observed - expected) / (1 - expected), 3)


def agreement(
    machine: dict[str, dict[str, Any]],
    human: dict[str, Any],
    *,
    only: Sequence[str] | None = None,
) -> AgreementReport:
    """``machine`` maps conversation_id -> metadata record; ``human`` maps id -> HumanLabel."""
    report = AgreementReport()
    keys = [k for k in sorted(human) if k in machine]
    if only is not None:
        allowed = set(only)
        keys = [k for k in keys if k in allowed]

    binary_m: list[bool] = []
    binary_h: list[bool] = []
    for key in keys:
        meta, label = machine[key], human[key]
        report.n += 1
        m_scam = bool((meta.get("ground_truth") or {}).get("is_scam"))
        h_scam = getattr(label, "is_scam", None)
        if h_scam is None:
            report.unsure += 1
        else:
            binary_m.append(m_scam)
            binary_h.append(h_scam)
            if m_scam and h_scam:
                report.both_scam += 1
            elif not m_scam and not h_scam:
                report.both_legit += 1
            elif m_scam:
                report.machine_only += 1
            else:
                report.human_only += 1

        m_index = meta.get("earliest_detection_index")
        h_index = getattr(label, "detection_index", None)
        if m_index is not None and h_index is not None:
            report.both_flagged += 1
            delta = m_index - h_index
            report.deltas.append(delta)
            if delta < 0:
                report.machine_earlier += 1
            elif delta > 0:
                report.machine_later += 1
            else:
                report.exact += 1
        elif m_index is not None:
            report.machine_flagged_only += 1
        elif h_index is not None:
            report.human_flagged_only += 1
        else:
            report.neither_flagged += 1

    report.kappa = cohen_kappa(binary_m, binary_h)
    if report.deltas:
        report.median_delta = statistics.median(report.deltas)
    return report


def render_agreement(report: AgreementReport, *, label: str = "") -> str:
    head = f"agreement{(' — ' + label) if label else ''}: {report.n} human-labelled"
    if not report.n:
        return head + " (no human labels to align yet)"
    lines = [head]
    lines.append(
        f"  scam or not: agree {report.both_scam + report.both_legit}"
        f" (scam {report.both_scam} / legitimate {report.both_legit})"
        f", machine says scam but human says not {report.machine_only}"
        f", human says scam but machine says not {report.human_only}"
        + (f", human unsure {report.unsure}" if report.unsure else "")
    )
    lines.append(f"  Cohen's kappa: {report.kappa if report.kappa is not None else 'undefined (no variation)'}")
    if report.both_flagged:
        lines.append(
            f"  detection point (both alarmed on {report.both_flagged}): median difference {report.median_delta:+.1f} utterances"
            f", machine earlier {report.machine_earlier} / exact {report.exact} / machine later {report.machine_later}"
        )
    lines.append(
        f"  machine alarmed only {report.machine_flagged_only}"
        f" | human alarmed only {report.human_flagged_only}"
        f" | neither alarmed {report.neither_flagged}"
    )
    if report.human_flagged_only > report.machine_flagged_only:
        lines.append("  ↑ machine is more conservative than the human: the rubric's anchors aren't enough yet; fix the rubric, not the threshold (R6.5)")
    elif report.machine_flagged_only > report.human_flagged_only:
        lines.append("  ↑ machine is more aggressive than the human: check whether the receiver's reactions are being counted as caller behaviour")
    return "\n".join(lines)
