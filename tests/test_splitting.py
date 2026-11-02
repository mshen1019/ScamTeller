"""train / eval split: refuse mixed rubrics, keep near-duplicates on one side, stratified balance."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from callfactory.splitting import SplitError, plan_split, rubric_of, write_split


def rec(cid, *, is_scam=True, hard_negative=False, late=False, compliant=False,
        eval_only=False, no_evidence=False, family="f1", detect=3,
        rubric="prefix-eval/v3", text="unique words here"):
    meta = {
        "conversation_id": cid,
        "ground_truth": {
            "is_scam": is_scam, "hard_negative": hard_negative, "late_onset": late,
            "compliance_intended": compliant, "eval_only": eval_only,
            "no_observable_evidence": no_evidence, "scam_family": family,
        },
        "earliest_detection_index": detect,
        "generation": {"agents": {"evaluator": {"rubric_version": rubric}}},
    }
    conv = {
        "conversation_id": cid,
        "utterances": [
            {"index": 0, "speaker": "caller", "text": text},
            {"index": 1, "speaker": "receiver", "text": "mm, right, I see, go on then"},
        ],
    }
    return conv, meta


def corpus(specs):
    convs, metas = {}, {}
    for cid, kw in specs:
        c, m = rec(cid, **kw)
        convs[cid], metas[cid] = c, m
    return convs, metas


def balanced(n=40):
    specs = []
    for i in range(n):
        kind = i % 4
        specs.append((f"c{i:03d}", dict(
            is_scam=kind < 2, hard_negative=kind == 2, late=kind == 1,
            family=f"fam{i % 8}", text=f"caller line number {i} with distinct wording {i*7}",
        )))
    return corpus(specs)


# --------------------------- rubric gate ---------------------------------- #

def test_mixed_rubrics_are_refused():
    convs, metas = corpus([("a", {}), ("b", {"rubric": "prefix-eval/v2"})])
    with pytest.raises(SplitError, match="contradictory"):
        plan_split(convs, metas)


def test_mixed_rubrics_can_be_forced_with_a_warning():
    convs, metas = corpus([("a", {}), ("b", {"rubric": "prefix-eval/v2"})])
    plan = plan_split(convs, metas, allow_mixed_rubric=True)
    assert any(w.startswith("rubric.mixed") for w in plan.warnings)
    assert len(plan.rubric_versions) == 2


def test_unstamped_rubric_shows_up_as_such():
    convs, metas = corpus([("a", {})])
    metas["a"]["generation"]["agents"]["evaluator"].pop("rubric_version")
    assert rubric_of(metas["a"]) == "(not recorded)"


# --------------------------- Near-duplicates ----------------------------- #

def test_near_duplicates_stay_on_the_same_side():
    same = "I do need to settle this today, the window closes at five and then it's gone"
    convs, metas = corpus([
        ("dup1", {"text": same, "family": "fam1"}),
        ("dup2", {"text": same, "family": "fam1"}),
        *[(f"x{i}", {"text": f"wholly different sentence {i} about nothing much at all",
                     "family": "fam2"}) for i in range(8)],
    ])
    plan = plan_split(convs, metas, ratio=0.5, seed=1)
    assert plan.duplicate_groups >= 1
    train, ev = set(plan.train), set(plan.eval)
    assert ({"dup1", "dup2"} <= train) or ({"dup1", "dup2"} <= ev)


def test_cross_family_similarity_is_not_merged():
    same = "I do need to settle this today, the window closes at five and then it's gone"
    convs, metas = corpus([
        ("a", {"text": same, "family": "fam1"}),
        ("b", {"text": same, "family": "fam2"}),
    ])
    plan = plan_split(convs, metas, ratio=0.5, seed=1)
    assert plan.duplicate_groups == 0


# --------------------------- Pinned to eval ------------------------------ #

def test_eval_only_scenarios_never_reach_train():
    convs, metas = balanced()
    metas["c000"]["ground_truth"]["eval_only"] = True
    plan = plan_split(convs, metas, seed=3)
    assert "c000" in plan.eval and "c000" not in plan.train
    assert "c000" in plan.forced_to_eval


def test_forced_ids_go_to_eval():
    convs, metas = balanced()
    plan = plan_split(convs, metas, seed=3, forced_eval=["c001", "c002"])
    assert {"c001", "c002"} <= set(plan.eval)


def test_pinned_items_eat_the_eval_quota_rather_than_adding_to_it():
    """--ratio 0.8 must really split 0.8, with the pinned items counted in the eval quota."""
    convs, metas = balanced(40)
    loose = plan_split(convs, metas, ratio=0.8, seed=5)
    pinned = plan_split(convs, metas, ratio=0.8, seed=5,
                        forced_eval=[f"c{i:03d}" for i in range(8)])
    assert abs(len(loose.train) - len(pinned.train)) <= 3
    assert abs(len(pinned.train) / 40 - 0.8) <= 0.1


# --------------------------- Stratification and balance ------------------ #

def test_both_sides_get_hard_negatives_and_scams():
    convs, metas = balanced()
    plan = plan_split(convs, metas, ratio=0.75, seed=7)
    for side in ("train", "eval"):
        assert plan.stats[side]["scam"] > 0
        assert plan.stats[side]["hard_negative"] > 0
    assert not plan.warnings


def test_a_missing_hard_negative_in_eval_warns():
    convs, metas = corpus([
        *[(f"s{i}", {"family": f"f{i}"}) for i in range(8)],
        ("hn", {"is_scam": False, "hard_negative": True, "family": "hnf"}),
    ])
    plan = plan_split(convs, metas, ratio=0.9, seed=1)
    assert any("no_hard_negative_in_eval" in w for w in plan.warnings)


def test_compliance_is_stratified_so_eval_keeps_some():
    specs = []
    for i in range(40):
        specs.append((f"c{i:03d}", dict(
            compliant=i % 4 == 0, family=f"fam{i % 5}",
            text=f"a quite different opening line {i} for this particular call {i*3}",
        )))
    convs, metas = corpus(specs)
    plan = plan_split(convs, metas, ratio=0.8, seed=2)
    assert plan.stats["eval"]["compliance"] >= 1
    assert plan.stats["train"]["compliance"] >= 1


def test_missing_compliance_in_eval_warns():
    convs, metas = corpus([
        *[(f"p{i}", {"compliant": True, "family": "f1",
                     "text": f"line {i} quite distinct here {i*9}"}) for i in range(4)],
        *[(f"q{i}", {"family": "f2", "text": f"other line {i} also distinct {i*11}"})
          for i in range(4)],
    ])
    plan = plan_split(convs, metas, ratio=1.0 - 1e-9, seed=1)
    assert any("no_compliance_in_eval" in w or "empty_eval" in w for w in plan.warnings)


# --------------------------- Disjoint families ---------------------------- #

def test_family_disjoint_split_shares_no_family():
    convs, metas = balanced(40)
    plan = plan_split(convs, metas, by="family", ratio=0.7, seed=4)
    fam = lambda cid: metas[cid]["ground_truth"]["scam_family"]
    assert not {fam(c) for c in plan.train} & {fam(c) for c in plan.eval}


def test_an_unknown_split_mode_is_rejected():
    convs, metas = balanced(8)
    with pytest.raises(ValueError):
        plan_split(convs, metas, by="vibes")


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.2, 1.5])
def test_an_impossible_ratio_is_rejected(ratio):
    convs, metas = balanced(8)
    with pytest.raises(ValueError):
        plan_split(convs, metas, ratio=ratio)


def test_an_empty_corpus_is_rejected():
    with pytest.raises(SplitError):
        plan_split({}, {})


# --------------------------- Writing to disk ----------------------------- #

def test_write_split_produces_the_two_file_contract(tmp_path):
    convs, metas = balanced(20)
    plan = plan_split(convs, metas, ratio=0.75, seed=9)
    paths = write_split(tmp_path, "v1", plan, convs, metas)
    for side in ("train", "eval"):
        root = paths[side]
        got = [json.loads(l) for l in (root / "conversations.jsonl").read_text().splitlines()]
        gotm = [json.loads(l) for l in (root / "metadata.jsonl").read_text().splitlines()]
        assert len(got) == len(gotm) == plan.stats[side]["n"]
        assert {c["conversation_id"] for c in got} == {m["conversation_id"] for m in gotm}
        manifest = json.loads((root / "_split.json").read_text())
        assert manifest["side"] == side and manifest["seed"] == 9


def test_writing_twice_is_refused(tmp_path):
    convs, metas = balanced(20)
    plan = plan_split(convs, metas, seed=9)
    write_split(tmp_path, "v1", plan, convs, metas)
    with pytest.raises(SplitError, match="already has content"):
        write_split(tmp_path, "v1", plan, convs, metas)
    write_split(tmp_path, "v1", plan, convs, metas, overwrite=True)


def test_the_split_is_deterministic():
    convs, metas = balanced(40)
    a = plan_split(convs, metas, seed=42)
    b = plan_split(convs, metas, seed=42)
    assert a.train == b.train and a.eval == b.eval
    c = plan_split(convs, metas, seed=43)
    assert (a.train, a.eval) != (c.train, c.eval)


def test_each_side_ships_a_readme_naming_both_versions(tmp_path):
    convs, metas = balanced(20)
    plan = plan_split(convs, metas, ratio=0.75, seed=9)
    paths = write_split(tmp_path, "ds7-rubric-v3", plan, convs, metas)
    for side in ("train", "eval"):
        text = (paths[side] / "README.md").read_text(encoding="utf-8")
        assert "ds7-rubric-v3" in text and "prefix-eval/v3" in text
        assert "two different version numbers" in text
        assert "Do not mix data from different rubric versions in one training set" in text
        assert "Must never enter model input" in text


# --------------------------- Three-way split (family holdout) ------------ #

def family_corpus(per_kind=10, per_family=10):
    """``per_kind`` families per class, ``per_family`` items per family, texts dissimilar to each other."""
    specs = []
    n = 0
    for kind in ("scam", "hard_negative", "plain"):
        for f in range(per_kind):
            for i in range(per_family):
                n += 1
                specs.append((f"{kind[:2]}{f:02d}-{i:02d}", dict(
                    is_scam=kind == "scam",
                    hard_negative=kind == "hard_negative",
                    compliant=kind == "scam" and i % 4 == 0,
                    family=f"{kind}_fam{f:02d}",
                    text=f"caller utterance {n} mentioning {kind} topic {f} variant {i} uniquely",
                )))
    return corpus(specs)


def fam_of(metas, cid):
    return metas[cid]["ground_truth"]["scam_family"]


def test_holdout_families_appear_only_in_test():
    convs, metas = family_corpus()
    plan = plan_split(convs, metas, by="family_holdout", seed=3)
    assert plan.holdout_families
    held = set(plan.holdout_families)
    assert {fam_of(metas, c) for c in plan.test} == held
    assert not {fam_of(metas, c) for c in plan.train} & held
    assert not {fam_of(metas, c) for c in plan.eval} & held


def test_the_holdout_is_stratified_across_the_three_kinds():
    """Six randomly drawn families could easily all be scams, and then test can't measure false alarms."""
    convs, metas = family_corpus()
    plan = plan_split(convs, metas, by="family_holdout", seed=11)
    kinds = {f.split("_fam")[0] for f in plan.holdout_families}
    assert kinds == {"scam", "hard_negative", "plain"}
    assert len(plan.holdout_families) == 3          # 10 families per class × 10%


def test_every_remaining_family_contributes_to_eval():
    convs, metas = family_corpus()
    plan = plan_split(convs, metas, by="family_holdout", seed=5, eval_ratio=0.1)
    held = set(plan.holdout_families)
    train_fams = {fam_of(metas, c) for c in plan.train}
    eval_fams = {fam_of(metas, c) for c in plan.eval}
    assert eval_fams == train_fams                   # every eval family has been seen
    assert eval_fams == {f for f in {fam_of(metas, c) for c in metas} if f not in held}
    per_family = Counter(fam_of(metas, c) for c in plan.eval)
    assert set(per_family.values()) == {1}           # 10% of 10 items per family


def test_the_three_sides_partition_the_corpus():
    convs, metas = family_corpus(per_kind=4, per_family=6)
    plan = plan_split(convs, metas, by="family_holdout", seed=7, holdout_family_ratio=0.25)
    assert plan.test
    allocated = plan.train + plan.eval + plan.test
    assert sorted(allocated) == sorted(metas)
    assert len(allocated) == len(set(allocated))


def test_eval_only_samples_go_to_eval_not_test_or_train():
    convs, metas = family_corpus(per_kind=4, per_family=6)
    target = next(c for c in sorted(metas) if metas[c]["ground_truth"]["is_scam"])
    metas[target]["ground_truth"]["eval_only"] = True
    plan = plan_split(convs, metas, by="family_holdout", seed=2, holdout_family_ratio=0.25)
    if fam_of(metas, target) in set(plan.holdout_families):
        pytest.skip("this family is held out; eval_only pinning doesn't apply")
    assert target in plan.eval


def test_an_all_scam_test_side_warns():
    specs = [(f"s{i:02d}", dict(family=f"scam_fam{i // 5:02d}",
                                text=f"only scam families here number {i} spoken plainly"))
             for i in range(40)]
    convs, metas = corpus(specs)
    plan = plan_split(convs, metas, by="family_holdout", seed=1)
    assert any(w.startswith("balance.no_negative_in_test") for w in plan.warnings)


def test_a_kind_too_small_to_hold_out_is_reported():
    convs, metas = family_corpus(per_kind=2, per_family=4)
    plan = plan_split(convs, metas, by="family_holdout", seed=1, holdout_family_ratio=0.1)
    assert any(w.startswith("holdout.kind_empty") for w in plan.warnings)
    assert not plan.test


def test_holdout_never_takes_a_whole_kind():
    convs, metas = family_corpus(per_kind=2, per_family=4)
    plan = plan_split(convs, metas, by="family_holdout", seed=1, holdout_family_ratio=0.9)
    assert any(w.startswith("holdout.kind_exhausted") for w in plan.warnings)
    train_kinds = {fam_of(metas, c).split("_fam")[0] for c in plan.train}
    assert train_kinds == {"scam", "hard_negative", "plain"}


def test_write_split_writes_three_sides(tmp_path):
    convs, metas = family_corpus(per_kind=4, per_family=6)
    plan = plan_split(convs, metas, by="family_holdout", seed=6, holdout_family_ratio=0.25)
    written = write_split(tmp_path, "ds-three", plan, convs, metas)
    assert set(written) == {"train", "eval", "test"}
    for side, path in written.items():
        lines = (path / "conversations.jsonl").read_text().splitlines()
        assert len(lines) == len(getattr(plan, side))
    readme = (written["test"] / "README.md").read_text()
    assert "unseen call types" in readme
    for family in plan.holdout_families:
        assert family in readme


def test_the_family_holdout_split_is_deterministic():
    convs, metas = family_corpus(per_kind=4, per_family=6)
    kw = dict(by="family_holdout", seed=8, holdout_family_ratio=0.25)
    a = plan_split(convs, metas, **kw)
    b = plan_split(convs, metas, **kw)
    assert (a.train, a.eval, a.test, a.holdout_families) == (b.train, b.eval, b.test, b.holdout_families)


# --------------------------- Detection-rate denominator ------------------- #

def test_no_evidence_samples_stay_in_the_denominator_but_are_reported_apart():
    """`no_observable_evidence` is a property of the premise, so it must **not** be deducted from the detection rate automatically.

    We got this wrong twice: first the slice was mixed into the denominator without comment, and eval's detection rate looked like a quality drop;
    then it was taken out of the denominator as "correct", when the caller in 16 of those 17 had in fact refused a verification channel
    when pressed, which records an open question as a score. Now it stays in the denominator and is reported separately.
    """
    specs = [(f"d{i}", dict(family="fam_detectable", detect=3 if i < 3 else None,
                            text=f"caller presses for payment in wording {i} distinctly"))
             for i in range(4)]
    specs += [(f"n{i}", dict(family="fam_no_evidence", no_evidence=True, detect=None,
                             text=f"caller only confirms the address variant {i} and rings off"))
              for i in range(5)]
    convs, metas = corpus(specs)
    plan = plan_split(convs, metas, ratio=0.5, seed=1)
    seen = 0
    for side in ("train", "eval"):
        st = plan.stats[side]
        if not st["scam"]:
            continue
        # the detection-rate denominator is every scam, including those 5
        detected = round(st["scam_detection_rate"] * st["scam"])
        assert detected <= st["scam"]
        if st["scam_no_evidence"]:
            seen += st["scam_no_evidence"]
            assert st["no_evidence_detected"] == 0
    assert seen == 5


def test_the_readme_separates_restraint_from_detection(tmp_path):
    specs = [("d0", dict(family="fam_d", detect=3, text="caller demands a fee at once plainly")),
             ("n0", dict(family="fam_n", no_evidence=True, detect=None,
                         text="caller merely checks the address then hangs up quietly"))]
    convs, metas = corpus(specs)
    plan = plan_split(convs, metas, ratio=0.5, seed=1)
    written = write_split(tmp_path, "ds-restraint", plan, convs, metas)
    text = "".join((written[s] / "README.md").read_text() for s in written)
    assert "scams whose premise declares \"no decidable evidence\"" in text
    assert "open question" in text
