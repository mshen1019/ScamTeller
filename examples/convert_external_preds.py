"""Convert an external detector's per-prefix predictions into the bench run format.

Used for the original ScamTeller training run, whose predictions were written by a separate
evaluation pipeline (records with ``uid = "<conversation_id>:prefix:<through_index>"``, ``probs``,
``pred`` and ``gt``). scamteller.score writes the bench format directly and does not need this.

Two key mappings, both backed by that run's provenance.json:
  * targets_all {"A": 149, "B": 763} matches the 149 positive prefixes in our eval -> **A = positive**
  * the prefix:NNN in uid is through_index (checked item by item: calib-101-00002 = [0,1,3,5,7,9])

So probs.A is the risk_score, and pred=="A" is the model's own operating point (argmax).

alarm_index is written at 0.5, consistent with the six baseline run files; the decision is recomputed at scoring time,
and this field is only used by discrimination_feedback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REF_THRESHOLD = 0.5


def load_labels(split: Path) -> dict[str, dict[int, bool]]:
    out: dict[str, dict[int, bool]] = {}
    for line in (split / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        out[m["conversation_id"]] = {
            p["through_index"]: bool(p.get("evidence_established"))
            for p in (m.get("prefix_labels") or [])
        }
    return out


def convert(src: Path, labels: dict[str, dict[int, bool]]) -> tuple[list[dict], dict]:
    by_call: dict[str, list[dict]] = {}
    gt_mismatch = 0
    bad_uid = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        uid = r.get("uid", "")
        cid, _, tail = uid.partition(":prefix:")
        if not cid or not tail.isdigit():
            bad_uid += 1
            continue
        k = int(tail)
        score = float((r.get("probs") or {}).get("A", 0.0))
        # their gt should agree with our evidence_established; if not, the alignment is wrong
        theirs = (r.get("gt") == "A")
        ours = labels.get(cid, {}).get(k)
        if ours is None or ours != theirs:
            gt_mismatch += 1
        by_call.setdefault(cid, []).append({
            "index": k,
            "risk_score": round(score, 6),
            "is_alarm": r.get("pred") == "A",   # the vendor's/trainer's own operating point (argmax)
            "signals": [],
            "rationale": "",
            "latency_ms": None,
            "fresh": True,
            "fired": score >= REF_THRESHOLD,
        })

    runs = []
    for cid, steps in by_call.items():
        steps.sort(key=lambda s: s["index"])
        alarm = next((s["index"] for s in steps if s["risk_score"] >= REF_THRESHOLD), None)
        runs.append({
            "conversation_id": cid,
            "alarm_index": alarm,
            "alarm_round": None if alarm is None else alarm // 2 + 1,
            "steps": steps,
        })
    runs.sort(key=lambda r: r["conversation_id"])
    return runs, {"calls": len(runs), "prefixes": sum(len(r["steps"]) for r in runs),
                  "gt_mismatch": gt_mismatch, "bad_uid": bad_uid}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("pairs", nargs="+", help="name=path/to/predictions.jsonl")
    args = ap.parse_args()

    labels = load_labels(Path(args.split))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for pair in args.pairs:
        name, _, path = pair.partition("=")
        runs, stats = convert(Path(path), labels)
        dst = out_dir / f"{name}.jsonl"
        with dst.open("w", encoding="utf-8") as fh:
            for r in runs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        flag = "" if stats["gt_mismatch"] == 0 and stats["bad_uid"] == 0 else "  ← problem"
        print(f"{name:26s} {stats['calls']:4d} calls {stats['prefixes']:4d} prefixes "
              f"| gt mismatch {stats['gt_mismatch']:3d} | uid parse failures {stats['bad_uid']:3d}{flag}")
        if flag:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
