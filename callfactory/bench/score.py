"""Scoring (REQUIREMENTS §7.3 / §7.5 / §7.8). Pure functions, no network.

There's only one core constraint, but it decides the shape of the whole table: **latency and false-alarm rate must be reported together**.
On latency alone, "alarm at utterance 0 on every call" is optimal; on false-alarm rate alone, "never alarm" is optimal.
So nothing here outputs a single overall score: there is no "bench score", only a set of numbers that must be read together.

Two degenerate baselines always appear in the table (``trivial_baselines``), so it's obvious at a glance whether a system
beats "doing nothing": prefix-level always-no scores 81.6% on this eval.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Area under the PR curve (average precision). Meaningless with no positives; returns None."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    pos = sum(labels)
    if not pos or pos == len(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = 0
    total = 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i]:
            tp += 1
            total += tp / rank
    return round(total / pos, 4)


def balanced_accuracy(tp: int, fn: int, tn: int, fp: int) -> float | None:
    if (tp + fn) == 0 or (tn + fp) == 0:
        return None
    return round(0.5 * (tp / (tp + fn) + tn / (tn + fp)), 4)


@dataclass
class PrefixScore:
    n: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    average_precision: float | None = None

    @property
    def accuracy(self) -> float | None:
        return round((self.tp + self.tn) / self.n, 4) if self.n else None

    @property
    def balanced_accuracy(self) -> float | None:
        return balanced_accuracy(self.tp, self.fn, self.tn, self.fp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "accuracy": self.accuracy, "balanced_accuracy": self.balanced_accuracy,
            "average_precision": self.average_precision,
        }


@dataclass
class CallScore:
    """Call level. How the denominators are cut is the easiest thing to get wrong here, so each class is counted separately."""

    scam_detectable: int = 0          # is_scam and ground truth has a detection point
    scam_detected: int = 0
    scam_no_gt_point: int = 0         # is_scam but ground truth has no detection point (R7.8)
    scam_no_gt_point_alarmed: int = 0  # alarming on these counts as a false alarm (restraint test)
    hard_negative: int = 0
    hard_negative_alarmed: int = 0
    plain: int = 0
    plain_alarmed: int = 0
    latencies: list[int] = field(default_factory=list)   # in utterances, signed

    def to_dict(self) -> dict[str, Any]:
        lat = sorted(self.latencies)
        return {
            "detection_rate": (round(self.scam_detected / self.scam_detectable, 4)
                               if self.scam_detectable else None),
            "detected": self.scam_detected, "detectable": self.scam_detectable,
            "hard_negative_false_alarm_rate": (
                round(self.hard_negative_alarmed / self.hard_negative, 4)
                if self.hard_negative else None),
            "hard_negative_alarmed": self.hard_negative_alarmed,
            "hard_negative_n": self.hard_negative,
            "plain_false_alarm_rate": (round(self.plain_alarmed / self.plain, 4)
                                       if self.plain else None),
            "plain_alarmed": self.plain_alarmed, "plain_n": self.plain,
            # R7.8: latency can't be computed for this slice (ground truth is None), only "alarmed or not"
            "no_gt_point_n": self.scam_no_gt_point,
            "no_gt_point_alarmed": self.scam_no_gt_point_alarmed,
            "latency_median_utterances": (statistics.median(lat) if lat else None),
            "latency_median_rounds": (round(statistics.median(lat) / 2, 2) if lat else None),
            "latency_n": len(lat),
            "earlier_than_label": sum(1 for x in lat if x < 0),
            "later_than_label": sum(1 for x in lat if x > 0),
            "same_as_label": sum(1 for x in lat if x == 0),
        }


def _gt(meta: dict[str, Any]) -> dict[str, Any]:
    return meta.get("ground_truth") or {}


def _fired(step: dict[str, Any], threshold: float, *, use_model_alarm: bool = False) -> bool:
    """The decision is made **at scoring time**; the ``fired`` stored in the run file isn't trusted.

    That makes the threshold a real knob: change it and recompute without rerunning the model. With ``use_model_alarm=True``
    the model's own boolean is used instead; that's the vendor's chosen operating point, reported as its own row.
    """
    if use_model_alarm:
        return bool(step.get("is_alarm"))
    return float(step.get("risk_score", 0.0)) >= threshold


def _alarm_index(run: dict[str, Any], threshold: float, *,
                 use_model_alarm: bool = False) -> int | None:
    for step in run.get("steps") or []:
        if _fired(step, threshold, use_model_alarm=use_model_alarm):
            return step["index"]
    return None


def score_system(
    runs: Sequence[dict[str, Any]],
    metas: dict[str, dict[str, Any]],
    *,
    threshold: float,
    exclude: Iterable[str] = (),
    use_model_alarm: bool = False,
) -> dict[str, Any]:
    """All metrics for one system.

    ``runs`` is the list of ``replay_call`` outputs; ``exclude`` lists conversation_ids to leave out
    (for the version of the table "without unruled labels").
    """
    skip = set(exclude)
    prefix = PrefixScore()
    call = CallScore()
    scores: list[float] = []
    labels: list[int] = []
    pairs_total = pairs_right = 0

    for run in runs:
        cid = run["conversation_id"]
        if cid in skip or cid not in metas:
            continue
        meta = metas[cid]
        gt = _gt(meta)
        by_index = {s["index"]: s for s in run["steps"]}

        # ---- prefix level: compare only on **labelled** prefixes; elsewhere there's no comparable ground truth ----
        for pl in meta.get("prefix_labels") or []:
            k = pl["through_index"]
            step = by_index.get(k)
            if step is None:
                continue
            want = bool(pl.get("evidence_established"))
            got = _fired(step, threshold, use_model_alarm=use_model_alarm)
            prefix.n += 1
            if want and got: prefix.tp += 1
            elif want and not got: prefix.fn += 1
            elif not want and got: prefix.fp += 1
            else: prefix.tn += 1
            scores.append(float(step["risk_score"]))
            labels.append(int(want))

        # ---- pair: the prefix before the detection point / the detection-point prefix, two utterances apart with the label flipped ----
        d = meta.get("earliest_detection_index")
        if d is not None:
            idxs = [pl["through_index"] for pl in meta.get("prefix_labels") or []]
            if d in idxs:
                pos = idxs.index(d)
                if pos > 0:
                    before, at = by_index.get(idxs[pos - 1]), by_index.get(d)
                    if before and at:
                        pairs_total += 1
                        if (not _fired(before, threshold, use_model_alarm=use_model_alarm)
                                and _fired(at, threshold, use_model_alarm=use_model_alarm)):
                            pairs_right += 1

        # ---- call level ----
        alarm = _alarm_index(run, threshold, use_model_alarm=use_model_alarm)
        if gt.get("is_scam"):
            if d is None:
                call.scam_no_gt_point += 1
                call.scam_no_gt_point_alarmed += int(alarm is not None)
            else:
                call.scam_detectable += 1
                if alarm is not None:
                    call.scam_detected += 1
                    call.latencies.append(alarm - d)
        elif gt.get("hard_negative"):
            call.hard_negative += 1
            call.hard_negative_alarmed += int(alarm is not None)
        else:
            call.plain += 1
            call.plain_alarmed += int(alarm is not None)

    prefix.average_precision = average_precision(scores, labels)
    return {
        "threshold": threshold,
        "decision": "model_alarm" if use_model_alarm else f"risk_score>={threshold}",
        "prefix_level": prefix.to_dict(),
        "call_level": call.to_dict(),
        "detection_point_pairs": {
            "n": pairs_total,
            "correct": pairs_right,
            "accuracy": round(pairs_right / pairs_total, 4) if pairs_total else None,
        },
    }


def slice_scores(
    runs: Sequence[dict[str, Any]],
    metas: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Computed separately for each slice R7.3 asks for: compliant / slow-burn / hard negative / the unruled no-evidence slice."""
    def ids(pred) -> set[str]:
        return {c for c, m in metas.items() if pred(_gt(m))}

    cuts = {
        "compliance": ids(lambda g: g.get("is_scam") and g.get("compliance_intended")),
        "late_onset": ids(lambda g: g.get("is_scam") and g.get("late_onset")),
        "hard_negative": ids(lambda g: g.get("hard_negative")),
        "no_observable_evidence": ids(lambda g: g.get("no_observable_evidence")),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, keep in cuts.items():
        if not keep:
            continue
        out[name] = score_system(
            [r for r in runs if r["conversation_id"] in keep],
            metas, threshold=threshold,
        )
    return out


def trivial_baselines(
    metas: dict[str, dict[str, Any]], *, exclude: Iterable[str] = ()
) -> dict[str, Any]:
    """Scores for always-no / always-yes. Any system that can't beat the former is useless."""
    skip = set(exclude)
    n = pos = 0
    for cid, meta in metas.items():
        if cid in skip:
            continue
        for pl in meta.get("prefix_labels") or []:
            n += 1
            pos += int(bool(pl.get("evidence_established")))
    if not n:
        return {}
    return {
        "always_no": {"prefix_accuracy": round((n - pos) / n, 4),
                      "balanced_accuracy": 0.5, "detection_rate": 0.0,
                      "hard_negative_false_alarm_rate": 0.0},
        "always_yes": {"prefix_accuracy": round(pos / n, 4),
                       "balanced_accuracy": 0.5, "detection_rate": 1.0,
                       "hard_negative_false_alarm_rate": 1.0},
        "prefixes": n, "positives": pos,
    }


def discrimination_feedback(
    by_system: dict[str, Sequence[dict[str, Any]]],
    metas: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """R7.5: the two kinds of sample sent back to the gatekeeper.

    * every system alarms at T0 -> the sample has **no discriminative value**; keeping it measures nothing
    * no system alarms while ground truth says decidable -> **suspected label error**, sent for human review first
    """
    if not by_system:
        return {"no_discrimination": [], "suspected_label_error": []}
    per: dict[str, list[int | None]] = {}
    for runs in by_system.values():
        for run in runs:
            per.setdefault(run["conversation_id"], []).append(run["alarm_index"])
    n_sys = len(by_system)
    no_disc, suspect = [], []
    for cid, alarms in per.items():
        if len(alarms) < n_sys:
            continue
        if all(a == 0 for a in alarms):
            no_disc.append(cid)
        if all(a is None for a in alarms):
            if (metas.get(cid, {}) or {}).get("earliest_detection_index") is not None:
                suspect.append(cid)
    return {"no_discrimination": sorted(no_disc),
            "suspected_label_error": sorted(suspect)}
