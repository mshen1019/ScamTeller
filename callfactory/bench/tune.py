"""Pick one alarm threshold per system on a train subset.

**The objective is fixed before seeing any eval results**, which matters more than which objective is chosen: coming back after eval
to pick an objective that "works well" is tuning on eval, and that is exactly why we require baseline and finetuned
models to be treated the same.

The chosen objective (``OBJECTIVE``):

    subject to a hard-negative false-alarm rate ≤ 0.10 **and a detection rate > 0**, take the threshold with the highest detection rate;
    on ties take the largest threshold (more conservative is better).
    If no such threshold exists, the system is judged **unable to reach this operating point** (``met_budget=False``),
    and falls back to "the point with the fewest false alarms among those with detection rate > 0" as the reportable operating point.

The "detection rate > 0" condition was forced by a test, not thought of up front: the first version read "highest detection
under the false-alarm budget", and on a synthetic case it picked threshold 0.95: zero detections, zero false alarms, **meeting the
budget by never alarming**. That is exactly the degenerate strategy this project has guarded against since §1; letting it in through the objective's back door
would record "this model can't do it" as "this model passes".

Why "constrained maximum detection" and not "highest balanced accuracy": this product's failure modes are asymmetric.
Users won't tolerate a detector that alarms on a quarter of real calls, while balanced accuracy counts that
the same as "missing a quarter of scams". The constraint sits on the false-alarm side because that's the side where users uninstall the app.

0.10 is a **chosen product tolerance, not a measurement**. Changing it changes the operating point, and the tables must say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

OBJECTIVE = "max detection rate subject to hard_negative_false_alarm_rate <= 0.10"
HARD_NEG_BUDGET = 0.10
#: Candidate thresholds. Step 0.05; finer is pointless: in testing the models' scores themselves come in multiples of about 0.05.
GRID = [round(0.05 * i, 2) for i in range(1, 20)]


@dataclass
class TuneResult:
    system: str
    threshold: float
    detection_rate: float | None
    hard_negative_false_alarm_rate: float | None
    plain_false_alarm_rate: float | None
    met_budget: bool
    curve: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system, "objective": OBJECTIVE,
            "hard_negative_budget": HARD_NEG_BUDGET,
            "threshold": self.threshold,
            "on_tuning_set": {
                "detection_rate": self.detection_rate,
                "hard_negative_false_alarm_rate": self.hard_negative_false_alarm_rate,
                "plain_false_alarm_rate": self.plain_false_alarm_rate,
                "met_budget": self.met_budget,
            },
            "curve": self.curve,
        }


def _call_level(runs: Sequence[dict[str, Any]], metas: dict[str, dict[str, Any]],
                threshold: float) -> dict[str, Any]:
    """Recompute the three call-level numbers at a given threshold. **No new requests**: the scores are already in the run file."""
    det = detectable = hn = hn_fa = pl = pl_fa = 0
    for run in runs:
        meta = metas.get(run["conversation_id"])
        if not meta:
            continue
        gt = meta.get("ground_truth") or {}
        # Score only. Mixing in the model's own is_alarm short-circuits the threshold into a no-op; we've seen it.
        fired = any(s["risk_score"] >= threshold for s in run["steps"])
        if gt.get("is_scam"):
            if meta.get("earliest_detection_index") is not None:
                detectable += 1
                det += int(fired)
        elif gt.get("hard_negative"):
            hn += 1
            hn_fa += int(fired)
        else:
            pl += 1
            pl_fa += int(fired)
    return {
        "threshold": threshold,
        "detection_rate": round(det / detectable, 4) if detectable else None,
        "hard_negative_false_alarm_rate": round(hn_fa / hn, 4) if hn else None,
        "plain_false_alarm_rate": round(pl_fa / pl, 4) if pl else None,
    }


def tune(system: str, runs: Sequence[dict[str, Any]],
         metas: dict[str, dict[str, Any]]) -> TuneResult:
    curve = [_call_level(runs, metas, t) for t in GRID]
    usable = [p for p in curve if (p["detection_rate"] or 0.0) > 0.0]
    ok = [p for p in usable
          if p["hard_negative_false_alarm_rate"] is not None
          and p["hard_negative_false_alarm_rate"] <= HARD_NEG_BUDGET]
    if ok:
        best = max(ok, key=lambda p: ((p["detection_rate"] or 0.0), p["threshold"]))
        met = True
    else:
        # Budget not reachable. Report a point that **still alarms**, not "never alarm":
        # the latter is faking a pass with the degenerate strategy.
        pool = usable or curve
        # `x or 1.0` is wrong here: **0.0 is falsy**, so a genuine zero-false-alarm point would be treated as 1.0
        # and sorted last. This project has already been bitten by the "falsy / short-circuit" class in three places (§5.9's 0.5 threshold,
        # bench's is_alarm short-circuit), so None is checked explicitly here.
        best = min(pool, key=lambda p: (
            1.0 if p["hard_negative_false_alarm_rate"] is None
            else p["hard_negative_false_alarm_rate"], -p["threshold"]))
        met = False
    return TuneResult(
        system=system, threshold=best["threshold"],
        detection_rate=best["detection_rate"],
        hard_negative_false_alarm_rate=best["hard_negative_false_alarm_rate"],
        plain_false_alarm_rate=best["plain_false_alarm_rate"],
        met_budget=met, curve=curve,
    )
