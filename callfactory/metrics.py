"""Batch-level guardrail metrics.

Only metrics that should already show problems at **generation** time live here; bench-stage metrics (latency, AUC,
per-family slices) wait for the harness (§6).

The key one is detection position: when `earliest_detection_index` collapses towards the start of the call, however good the false-alarm rate looks,
the batch has lost its discriminative value (R7.5: samples on which every model alarms at T0 are marked non-discriminative).
An offline mock batch once had a mean detection round of 2.0 and was useless, while all four headline numbers were perfect.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

#: Warn when the median detection position falls below this fraction (of the whole call).
COLLAPSE_MEDIAN = 0.25
#: Upper bound on the share of detection points that land on the first utterance.
T0_SHARE = 0.3


@dataclass
class DetectionPositionReport:
    n_scam: int = 0
    n_detected: int = 0
    median_fraction: float | None = None
    t0_count: int = 0
    very_early_count: int = 0          # within the first 20%
    fractions: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "n_scam": self.n_scam,
            "n_detected": self.n_detected,
            "detection_rate": round(self.n_detected / self.n_scam, 3) if self.n_scam else 0.0,
            "median_position_in_call": self.median_fraction,
            "t0_count": self.t0_count,
            "very_early_count": self.very_early_count,
            "warnings": list(self.warnings),
        }

    def render(self) -> str:
        if not self.n_detected:
            return f"detection position: {self.n_detected}/{self.n_scam} have a detection point, no position to compute"
        line = (
            f"detection position: median {self.median_fraction:.0%} of call"
            f" (T0 {self.t0_count}/{self.n_detected}, first 20% {self.very_early_count}/{self.n_detected})"
        )
        return "\n".join([line, *(f"  ! {w}" for w in self.warnings)])


def detection_positions(
    samples: Iterable[tuple[int | None, int, bool]]
) -> DetectionPositionReport:
    """``samples`` are ``(earliest_detection_index, num_utterances, is_scam)``.

    Scams only: a detection point on a legitimate call is a false alarm, which the false-alarm rate covers; it isn't mixed into position stats.
    """
    report = DetectionPositionReport()
    for index, n_utt, is_scam in samples:
        if not is_scam:
            continue
        report.n_scam += 1
        if index is None or n_utt <= 0:
            continue
        report.n_detected += 1
        fraction = (index + 1) / n_utt
        report.fractions.append(fraction)
        if index == 0:
            report.t0_count += 1
        if fraction <= 0.2:
            report.very_early_count += 1

    if report.fractions:
        report.median_fraction = round(statistics.median(report.fractions), 3)
        if report.median_fraction < COLLAPSE_MEDIAN:
            report.warnings.append(
                f"median detection position {report.median_fraction:.0%} < {COLLAPSE_MEDIAN:.0%}: "
                "detection points are collapsing towards the start of the call, and the batch is losing discriminative value (R7.5)"
            )
        if report.t0_count / report.n_detected > T0_SHARE:
            report.warnings.append(
                f"{report.t0_count}/{report.n_detected} decidable at the first utterance: "
                "samples decidable at the opening can't tell models under test apart"
            )
    return report


def label_shift(
    before: Sequence[int | None], after: Sequence[int | None]
) -> dict[str, int]:
    """Label shift on the same samples before and after a rubric change.

    ``before``/``after`` give ``earliest_detection_index`` in the same order.
    On fixed transcripts this is the net effect of the rubric change. Comparing across batches doesn't work:
    Claude has no seed, so every rerun is a different dialogue.
    """
    if len(before) != len(after):
        raise ValueError("the two label lists must have the same length")
    out = {"gained": 0, "lost": 0, "earlier": 0, "later": 0, "unchanged": 0}
    for a, b in zip(before, after):
        if a is None and b is None:
            out["unchanged"] += 1
        elif a is None:
            out["gained"] += 1
        elif b is None:
            out["lost"] += 1
        elif b < a:
            out["earlier"] += 1
        elif b > a:
            out["later"] += 1
        else:
            out["unchanged"] += 1
    return out
