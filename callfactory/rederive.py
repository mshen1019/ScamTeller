"""Recompute only the detection point, without relabelling (zero model calls).

How this differs from ``relabel`` is the easiest thing to mix up in the whole pipeline, so to be explicit:

* ``relabel`` -- **relabels**. The dialogue is untouched, the evaluator reruns, and ``risk_score`` /
  ``signals`` / ``sufficient_evidence`` may all change. Costs money. Use it to change the rubric.
* ``rederive`` -- **changes only the derivation rule**. Prefix labels stay byte-identical; only
  ``earliest_detection_index`` and what derives from it,
  ``is_detection_point`` / ``evidence_established``, are recomputed. Zero model calls, zero dialogue noise.
  Use it to change the threshold or whether sufficient_evidence is required.

Both refuse to write back to the source directory: the old labels are the other half of the comparison.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .batch import OutputPaths
from .evaluator import (
    DEFAULT_DETECTION_THRESHOLD,
    build_utterance_labels,
    earliest_detection_index,
    mark_evidence_established,
)
from .metrics import DetectionPositionReport, detection_positions, label_shift
from .relabel import read_run
from .schema import PrefixLabel, Utterance


@dataclass
class RederiveStats:
    total: int = 0
    shift: dict[str, int] = field(default_factory=dict)
    before_rule: list[str] = field(default_factory=list)
    rule: str = ""
    threshold: float = 0.0
    positions: DetectionPositionReport | None = None
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "rule": self.rule,
            "threshold": self.threshold,
            "shift": dict(self.shift),
            "before_rule": sorted(set(self.before_rule)),
            "positions": self.positions.summary() if self.positions else None,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _prefix_labels(meta: dict[str, Any]) -> list[PrefixLabel]:
    out: list[PrefixLabel] = []
    for d in meta.get("prefix_labels") or []:
        out.append(PrefixLabel(
            through_index=d["through_index"],
            risk_score=d["risk_score"],
            risk_tier=d.get("risk_tier", ""),
            signals=list(d.get("signals") or []),
            new_signals=list(d.get("new_signals") or []),
            sufficient_evidence=bool(d.get("sufficient_evidence", False)),
            rationale=d.get("rationale", ""),
        ))
    return out


def rederive_run(
    src: str | Path,
    dst: str | Path,
    *,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    require_persistence: bool = True,
    require_sufficient: bool = False,
    progress: Callable[[str], None] | None = None,
) -> RederiveStats:
    started = time.monotonic()
    src_paths, convs, metas = read_run(src)
    dst_paths = OutputPaths.resolve(dst)
    if dst_paths.root.resolve() == src_paths.root.resolve():
        raise ValueError(
            f"refusing to write back to the source directory {src_paths.root}: "
            "the old labels are the other half of the comparison; overwrite them and there's nothing to compare"
        )
    dst_paths.mkdirs()

    stats = RederiveStats(
        rule="detect/v1" if require_sufficient else "detect/v2", threshold=threshold
    )
    before: list[int | None] = []
    after: list[int | None] = []
    out_metas: list[dict[str, Any]] = []
    positions: list[tuple[int | None, int, bool]] = []

    by_id = {c["conversation_id"]: c for c in convs}
    for cid, meta in metas.items():
        conv = by_id.get(cid)
        if conv is None:
            continue
        utterances = [
            Utterance(index=u["index"], speaker=u["speaker"], text=u["text"])
            for u in conv.get("utterances") or []
        ]
        labels = _prefix_labels(meta)
        old = meta.get("earliest_detection_index")
        new = earliest_detection_index(
            labels, threshold,
            require_persistence=require_persistence,
            require_sufficient=require_sufficient,
        )
        mark_evidence_established(labels, new)

        out = dict(meta)
        out["prefix_labels"] = [l.to_dict() for l in labels]
        out["utterance_labels"] = [
            u.to_dict() for u in build_utterance_labels(utterances, labels, new)
        ]
        out["earliest_detection_index"] = new
        out["earliest_detection_round"] = None if new is None else new // 2 + 1

        generation = dict(out.get("generation") or {})
        evaluation = dict(generation.get("evaluation") or {})
        stats.before_rule.append(str(evaluation.get("detection_rule") or "detect/v1"))
        # Keep the other half of the comparison in the data, so there's no need to open two directories to see what changed.
        generation["rederived_from"] = {
            "run_id": generation.get("run_id"),
            "detection_rule": evaluation.get("detection_rule") or "detect/v1",
            "threshold": evaluation.get("threshold"),
            "require_sufficient": evaluation.get("require_sufficient", True),
            "earliest_detection_index": old,
        }
        evaluation.update({
            "threshold": threshold,
            "require_persistence": require_persistence,
            "require_sufficient": require_sufficient,
            "detection_rule": stats.rule,
        })
        generation["evaluation"] = evaluation
        out["generation"] = generation

        before.append(old)
        after.append(new)
        positions.append((new, len(utterances), bool((out.get("ground_truth") or {}).get("is_scam"))))
        out_metas.append(out)
        stats.total += 1
        if progress and stats.total % 200 == 0:
            progress(f"  recomputed {stats.total}")

    with dst_paths.conversations.open("w", encoding="utf-8") as ch:
        for conv in convs:
            ch.write(json.dumps(conv, ensure_ascii=False) + "\n")
    with dst_paths.metadata.open("w", encoding="utf-8") as mh:
        for meta in out_metas:
            mh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    stats.shift = label_shift(before, after)
    stats.positions = detection_positions(positions)
    stats.elapsed_s = time.monotonic() - started
    return stats
