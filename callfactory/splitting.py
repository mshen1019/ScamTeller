"""train / eval split (N8's ``split``).

The gatekeeper only decides admission; splitting is a separate command, and this is that command.

Four hard rules; the first two are pitfalls this project has hit:

1. **Mixed rubric versions are refused.** Seen in practice: four batches on disk came from three rubrics, and at the time
   the metadata couldn't tell them apart at all. Put two batches into one training set and the same scam behaviour is labelled
   "insufficient evidence" in one half and "decidable at round 3" in the other; the model gets systematically contradictory labels and nobody is warned.
2. **Near-duplicate texts must stay on the same side.** Otherwise a high eval score may just be memorising the same call from train.
   Similarity uses the project's own word-trigram Jaccard, the same measure as Diversity.
3. ``eval_only`` scenarios go to eval only (e.g. the "scam by intent but no decidable evidence" kind: it tests restraint
   and shouldn't take part in training).
4. The class make-up of both sides must be reported: a split with every hard negative on one side is meaningless,
   because the ability to avoid false alarms can no longer be measured.

``by="family_holdout"`` gives three splits: whole families held out as **test** (generalisation to **unseen call
types**), a small share of every remaining family as **eval** (timing precision on seen types),
and the rest as train. The two kinds of generalisation ask different questions, so they are two datasets, not one.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .validators import similarity

SPLIT_MANIFEST = "_split.json"
#: Near-duplicate threshold. Matches the validators' default repetition threshold.
NEAR_DUPLICATE = 0.5


@dataclass
class SplitPlan:
    train: list[str] = field(default_factory=list)
    eval: list[str] = field(default_factory=list)
    #: Only used by by="family_holdout": the test side of whole held-out families
    test: list[str] = field(default_factory=list)
    by: str = "stratified"
    seed: int = 0
    ratio: float = 0.8
    holdout_families: list[str] = field(default_factory=list)
    eval_ratio: float = 0.1
    rubric_versions: list[str] = field(default_factory=list)
    duplicate_groups: int = 0
    forced_to_eval: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SplitError(RuntimeError):
    pass


def _caller_text(conv: dict[str, Any]) -> str:
    return " ".join(
        u["text"] for u in conv.get("utterances", []) if u.get("speaker") == "caller"
    )


def _layer(meta: dict[str, Any]) -> str:
    """Stratification key.

    Compliance (receiver won over) is its own stratum: it is currently the slice with the lowest detection rate (0.40 vs 0.67 in testing),
    and without stratifying, the 14 compliant samples land at random; the eval side might end up with one, and that's exactly the kind most in need of testing.
    """
    gt = meta.get("ground_truth") or {}
    if gt.get("is_scam"):
        parts = ["scam"]
        if gt.get("late_onset"):
            parts.append("late")
        if gt.get("compliance_intended"):
            parts.append("compliant")
        return "_".join(parts)
    return "hard_negative" if gt.get("hard_negative") else "plain"


def _family(meta: dict[str, Any]) -> str:
    return ((meta.get("task") or {}).get("scenario") or {}).get("family") \
        or (meta.get("ground_truth") or {}).get("scam_family") or "?"


def _family_kind(meta: dict[str, Any]) -> str:
    """The family's own class (excluding sample-level strata such as slow-burn/compliant)."""
    gt = meta.get("ground_truth") or {}
    if gt.get("is_scam"):
        return "scam"
    return "hard_negative" if gt.get("hard_negative") else "plain"


def rubric_of(meta: dict[str, Any]) -> str:
    evaluator = ((meta.get("generation") or {}).get("agents") or {}).get("evaluator") or {}
    return evaluator.get("rubric_version") or "(not recorded)"


def _duplicate_groups(
    convs: dict[str, dict[str, Any]],
    metas: dict[str, dict[str, Any]],
    threshold: float,
) -> list[list[str]]:
    """Group near-duplicate conversations. Compared pairwise within a family; near-duplicates across families aren't realistic."""
    by_family: dict[str, list[str]] = {}
    for cid, meta in metas.items():
        family = (meta.get("ground_truth") or {}).get("scam_family") or "?"
        by_family.setdefault(family, []).append(cid)

    parent: dict[str, str] = {cid: cid for cid in metas}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for members in by_family.values():
        texts = {cid: _caller_text(convs[cid]) for cid in members if cid in convs}
        ordered = sorted(texts)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if similarity(texts[a], texts[b]) >= threshold:
                    union(a, b)

    groups: dict[str, list[str]] = {}
    for cid in metas:
        groups.setdefault(find(cid), []).append(cid)
    return [sorted(g) for g in groups.values()]


def plan_split(
    convs: dict[str, dict[str, Any]],
    metas: dict[str, dict[str, Any]],
    *,
    ratio: float = 0.8,
    seed: int = 0,
    by: str = "stratified",
    forced_eval: Iterable[str] = (),
    threshold: float = NEAR_DUPLICATE,
    allow_mixed_rubric: bool = False,
    holdout_family_ratio: float = 0.1,
    eval_ratio: float = 0.1,
) -> SplitPlan:
    if by not in ("stratified", "family", "family_holdout"):
        raise ValueError("by must be stratified / family / family_holdout")
    if by == "family_holdout":
        if not 0.0 <= holdout_family_ratio < 1.0:
            raise ValueError(f"holdout_family_ratio must be in [0,1), got {holdout_family_ratio}")
        if not 0.0 < eval_ratio < 1.0:
            raise ValueError(f"eval_ratio must be in (0,1), got {eval_ratio}")
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0,1), got {ratio}")
    ids = sorted(metas)
    if not ids:
        raise SplitError("no data to split")

    plan = SplitPlan(by=by, seed=seed, ratio=ratio, eval_ratio=eval_ratio)
    plan.rubric_versions = sorted({rubric_of(metas[c]) for c in ids})
    if len(plan.rubric_versions) > 1:
        message = (
            f"this data comes from {len(plan.rubric_versions)} rubrics: {plan.rubric_versions}. "
            "Mixing them into one training set gives systematically contradictory labels; first bring them to one "
            "rubric with relabel, or pass allow_mixed_rubric=True explicitly and accept the consequences"
        )
        if not allow_mixed_rubric:
            raise SplitError(message)
        plan.warnings.append("rubric.mixed: " + message)

    # eval_only scenarios and explicitly named items are pinned to the eval side
    pinned = set(forced_eval)
    for cid in ids:
        if ((metas[cid].get("ground_truth") or {}).get("eval_only")):
            pinned.add(cid)
    plan.forced_to_eval = sorted(pinned & set(ids))

    groups = _duplicate_groups(convs, metas, threshold)
    plan.duplicate_groups = sum(1 for g in groups if len(g) > 1)

    # Stratify: allocate scam / scam_late / hard_negative / plain separately, so no stratum lands entirely on one side
    rng = random.Random(seed)
    if by == "family_holdout":
        _plan_family_holdout(
            plan, groups, metas, pinned, rng,
            holdout_family_ratio=holdout_family_ratio, eval_ratio=eval_ratio,
        )
    elif by == "family":
        # Family-disjoint split: tests generalisation to **unseen scam types**, stricter than a random split
        families = sorted({(metas[c].get("ground_truth") or {}).get("scam_family") or "?" for c in ids})
        rng.shuffle(families)
        n_train = max(1, round(len(families) * ratio))
        train_families = set(families[:n_train])
        for group in groups:
            fams = {(metas[c].get("ground_truth") or {}).get("scam_family") or "?" for c in group}
            target = "train" if fams <= train_families else "eval"
            if set(group) & pinned:
                target = "eval"
            (plan.train if target == "train" else plan.eval).extend(group)
    else:
        buckets: dict[str, list[list[str]]] = {}
        pinned_per_layer: Counter[str] = Counter()
        for group in groups:
            key = _layer(metas[group[0]])
            if set(group) & pinned:
                plan.eval.extend(group)
                pinned_per_layer[key] += len(group)
                continue
            buckets.setdefault(key, []).append(group)
        for key in sorted(buckets):
            layer = buckets[key]
            rng.shuffle(layer)
            free = sum(len(g) for g in layer)
            # ratio is computed on **all items in the stratum**; items pinned to eval use up eval's quota rather than adding to it,
            # otherwise --ratio 0.8 with 24 pinned items would actually split 0.68.
            want = min(free, round((free + pinned_per_layer[key]) * ratio))
            taken = 0
            for group in layer:
                # A group is the smallest indivisible unit, so the ratio is filled by item count
                if taken + len(group) <= want or taken == 0:
                    plan.train.extend(group)
                    taken += len(group)
                else:
                    plan.eval.extend(group)

    plan.train.sort()
    plan.eval.sort()
    plan.test.sort()
    plan.stats = _stats(plan, metas)
    plan.warnings.extend(_balance_warnings(plan, metas))
    return plan


def _plan_family_holdout(
    plan: SplitPlan,
    groups: Sequence[Sequence[str]],
    metas: dict[str, dict[str, Any]],
    pinned: set[str],
    rng: random.Random,
    *,
    holdout_family_ratio: float,
    eval_ratio: float,
) -> None:
    """Hold out whole families as test, and sample ``eval_ratio`` of each remaining family as eval.

    Why two and not one: generalisation to **unseen call types** and the precision of the detection moment on
    **seen types** are two different questions. Merged into one, a model that is precise on familiar types but blind to a new
    playbook gets its score averaged out to "not bad", and that is exactly the failure mode this dataset should expose.

    Holdout is stratified by family class (scam / hard negative / plain); otherwise six randomly drawn families could easily
    all be scams, and the test side would only measure misses, not false alarms.
    """
    kinds: dict[str, list[str]] = {}
    for family in sorted({_family(m) for m in metas.values()}):
        sample = next(m for m in metas.values() if _family(m) == family)
        kinds.setdefault(_family_kind(sample), []).append(family)

    holdout: set[str] = set()
    for kind in sorted(kinds):
        pool = sorted(kinds[kind])
        rng.shuffle(pool)
        want = round(len(pool) * holdout_family_ratio)
        if want >= len(pool):      # the whole class held out -> train has none of it; not allowed
            want = len(pool) - 1
            plan.warnings.append(
                f"holdout.kind_exhausted: {kind} has only {len(pool)} families; "
                f"the holdout ratio would move the whole class into test, reduced to {want}"
            )
        if want == 0 and len(pool) >= 1:
            plan.warnings.append(
                f"holdout.kind_empty: {kind} has {len(pool)} families; "
                f"at {holdout_family_ratio:.0%} none are held out, so test has none of this class"
            )
        holdout.update(pool[:want])
    plan.holdout_families = sorted(holdout)

    # Pass 1: every item of a held-out family goes to test (near-duplicate groups are within a family anyway, so they never cross sides)
    rest: dict[str, list[list[str]]] = {}
    for group in groups:
        family = _family(metas[group[0]])
        if family in holdout:
            plan.test.extend(group)
        else:
            rest.setdefault(family, []).append(list(group))

    # Pass 2: sample eval_ratio by item count within each family. Pinned groups go first and **use up** that family's
    # eval quota; adding on top would make the eval side uneven across families, and an equal share per family is the point of this side.
    for family in sorted(rest):
        members = rest[family]
        total = sum(len(g) for g in members)
        want = max(1, round(total * eval_ratio))
        rng.shuffle(members)
        members.sort(key=lambda g: 0 if set(g) & pinned else 1)
        taken = 0
        for group in members:
            forced = bool(set(group) & pinned)
            if forced or taken < want:
                plan.eval.extend(group)
                taken += len(group)
            else:
                plan.train.extend(group)


def _side_stats(cids: Sequence[str], metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    layers = Counter(_layer(metas[c]) for c in cids)
    scam_layers = {k: v for k, v in layers.items() if k.startswith("scam")}
    scams = sum(scam_layers.values())

    # ``no_observable_evidence`` declares a **scenario premise**: the caller never produces decidable evidence.
    # It isn't a property of the actual dialogue: nothing in the caller prompt requires the caller to honour it, and in testing
    # 16 of 17 refused a verification channel when pressed, which is exactly the behaviour the rubric's GROUP B calls "sufficient
    # on its own". So this slice **stays** in the detection-rate denominator and is only reported separately in addition: taking it out
    # as "correct" would record an open question awaiting human ruling as a score (EVAL_SYSTEM §5.7).
    no_evidence = [
        c for c in cids
        if (metas[c].get("ground_truth") or {}).get("is_scam")
        and (metas[c].get("ground_truth") or {}).get("no_observable_evidence")
    ]
    detected = sum(
        1 for c in cids
        if (metas[c].get("ground_truth") or {}).get("is_scam")
        and metas[c].get("earliest_detection_index") is not None
    )
    ne_detected = sum(
        1 for c in no_evidence if metas[c].get("earliest_detection_index") is not None
    )
    return {
        "n": len(cids),
        "scam": scams,
        "scam_no_evidence": len(no_evidence),
        "no_evidence_detected": ne_detected,
        "scam_late": sum(v for k, v in scam_layers.items() if "late" in k),
        "hard_negative": layers.get("hard_negative", 0),
        "plain": layers.get("plain", 0),
        "families": len({(metas[c].get("ground_truth") or {}).get("scam_family") for c in cids}),
        "scam_detection_rate": round(detected / scams, 3) if scams else None,
        "compliance": sum(
            1 for c in cids if (metas[c].get("ground_truth") or {}).get("compliance_intended")
        ),
    }


def _stats(plan: SplitPlan, metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {"train": _side_stats(plan.train, metas), "eval": _side_stats(plan.eval, metas)}
    if plan.test:
        out["test"] = _side_stats(plan.test, metas)
    return out


def _balance_warnings(plan: SplitPlan, metas: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    train, ev = plan.stats["train"], plan.stats["eval"]
    test = plan.stats.get("test")
    if test:
        if not test["hard_negative"] and not test["plain"]:
            out.append(
                "balance.no_negative_in_test: the test side is all scam; it can only measure misses, "
                "not whether there are false alarms on unseen legitimate calls"
            )
        if not test["scam"]:
            out.append("balance.no_scam_in_test: no scam on the test side")
    if not ev["n"]:
        out.append("split.empty_eval: the eval side is empty")
        return out
    if not ev["hard_negative"]:
        out.append(
            "balance.no_hard_negative_in_eval: no hard negative on the eval side; "
            "false-alarm avoidance can't be measured, and that's the most important half of this dataset"
        )
    if not train["hard_negative"]:
        out.append("balance.no_hard_negative_in_train: no hard negative on the train side")
    for side, label in ((train, "train"), (ev, "eval")):
        if side["n"] and not side["scam"]:
            out.append(f"balance.no_scam_in_{label}: no scam on the {label} side")
    if train["compliance"] and not ev["compliance"]:
        out.append(
            "balance.no_compliance_in_eval: no compliant samples on the eval side; "
            "detection when the receiver cooperates can't be measured, and that's currently the slice with the lowest detection rate"
        )
    return out


def write_split(
    out_root: str | Path,
    name: str,
    plan: SplitPlan,
    convs: dict[str, dict[str, Any]],
    metas: dict[str, dict[str, Any]],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write train/ and eval/ in the two-file form of C1/C2."""
    root = Path(out_root)
    paths: dict[str, Path] = {}
    sides: list[tuple[str, list[str]]] = [("train", plan.train), ("eval", plan.eval)]
    if plan.test:
        sides.append(("test", plan.test))
    for side, cids in sides:
        target = root / side / name
        if target.exists() and any(target.iterdir()) and not overwrite:
            raise SplitError(f"{target} already has content; pass overwrite=True to overwrite")
        target.mkdir(parents=True, exist_ok=True)
        with (target / "conversations.jsonl").open("w", encoding="utf-8") as ch, \
             (target / "metadata.jsonl").open("w", encoding="utf-8") as mh:
            for cid in cids:
                ch.write(json.dumps(convs[cid], ensure_ascii=False) + "\n")
                mh.write(json.dumps(metas[cid], ensure_ascii=False) + "\n")
        (target / SPLIT_MANIFEST).write_text(
            json.dumps({"side": side, **plan.to_dict()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (target / "README.md").write_text(
            _readme(name, side, plan, metas), encoding="utf-8"
        )
        paths[side] = target
    return paths


def _readme(name: str, side: str, plan: SplitPlan, metas: dict[str, dict[str, Any]]) -> str:
    """Every split ships its own notes: whoever gets the data shouldn't have to dig through chat logs to learn what the labels are."""
    st = plan.stats[side]
    rubric = ", ".join(plan.rubric_versions)
    roles = {
        "train": "**Training set.**",
        "eval": (
            "**Eval (development set).** "
            f"{plan.eval_ratio:.0%} of every family is sampled into it, so it measures whether the detection moment is right on "
            "**call types already seen** — you can look at it repeatedly and use it to tune models."
        ),
        "test": (
            "**Test (held-out set).** The "
            f"{len(plan.holdout_families)} scenario families on this side **never appear in train or eval**, "
            "so it measures generalisation to **unseen call types**. Run it only once at the end; "
            "once you have tuned anything on it, it is just a second development set."
        ),
    }.get(side, "")
    no_evidence_row = ""
    if st["scam_no_evidence"]:
        no_evidence_row = (
            f"| of which scams whose premise declares \"no decidable evidence\" ⚠️ | {st['scam_no_evidence']}, "
            f"{st['no_evidence_detected']} detected — **this slice is an open question, see below** |"
        )
    holdout = ""
    if side == "test" and plan.holdout_families:
        holdout = "\nHeld-out families: " + ", ".join(f"`{f}`" for f in plan.holdout_families) + "\n"
    return f"""# {name} — {side}

**Dataset version** `{name}` · **Labelling rubric** `{rubric}`

These are two different version numbers: the dataset version says which release of this data it is, the rubric version says which revision of the labelling rules.
The two are not in step and should not be.

{roles}
{holdout}
## What is in here

{st['n']} synthetic phone conversations, each labelled with **the round from which the evidence is sufficient to call it a scam**
(`earliest_detection_index`). This is a **moment**, not a class — it is for training and evaluating
early-warning detectors: not "is this a scam?" but "what is the earliest round at which it can be called?".

| | |
|---|---|
| scam | {st['scam']} (late-onset {st['scam_late']} / receiver persuaded {st['compliance']}) |
| hard negative (genuine but looks like a scam) | {st['hard_negative']} |
| ordinary legitimate call | {st['plain']} |
| scenario families | {st['families']} |
| scam detection rate in the labels | {st['scam_detection_rate']} |
{no_evidence_row}

## Two files, joined on conversation_id

* `conversations.jsonl` — **the only input a model may see.** Each utterance has only
  `index` / `speaker` / `text`; no label field exists at any depth.
* `metadata.jsonl` — labels, scenario, ground truth, generation provenance. **Must never enter model input.**

The recommended training target is `prefix_labels[].evidence_established` (equivalently
`utterance_labels[].evidence_established`): a **monotonic** "the evidence is now sufficient" —
false before `earliest_detection_index`, true from it onward, never falling back. The
`sufficient_evidence` in the same record, by contrast, is a per-prefix **in-place** verdict that **may fall back**
(8.3% of conversations across the whole corpus go true and then false), so using it as the target teaches the model to retract warnings.
`evidence_established` is a label derived with hindsight; **do not use it as a feature**.

## What to know before using it

* **Do not mix data from different rubric versions in one training set.** The same scam behaviour would be
  labelled "insufficient evidence" in part of the data and "decidable at round 3" in another. The `split` command rejects such a corpus outright;
  to merge other batches, first bring them to the same rubric with `callfactory relabel`.
* **This version of the labels is conservative.** The corpus-wide detection rate is 0.593, only 0.417 on the "receiver persuaded" samples,
  and not a single false alarm on 366 hard negatives. A model trained on it inherits a bias in this direction:
  **it alarms later than an attentive listener would**, and more so the more cooperative the other side is. This is a known, directional bias, not noise
  — once human calibration produces the next rubric version, the **same transcripts** can be relabelled and the gap measured.
* **Latency and false-alarm rate must be read together.** Looking only at latency, "alarm immediately on every call" is optimal; looking only at false-alarm rate,
  "never alarm" is optimal. Hard negatives exist to measure the latter.
* ⚠️ **The `no_observable_evidence` slice is an open question; do not treat it as settled labels.**
  The flag declares a **scenario premise** (the caller never produces decidable evidence), but nothing in the caller prompt
  requires the caller to honour it: in practice 16 of the 17 such samples refuse when the receiver asks for a callback number —
  exactly the behaviour the rubric's GROUP B lists as "individually sufficient" — while the label still says "insufficient evidence".
  These samples are `eval_only`, so they go to eval only and never to training, and therefore **do not contaminate the training set**; but on eval
  the latency figures for this slice should not be taken at face value until a human has adjudicated them. See `docs/EVAL_SYSTEM.md` §5.7.

The full design notes are in `docs/EVAL_SYSTEM.md`.
"""
