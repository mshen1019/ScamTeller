"""Calibration-set split and agreement (R6.5)."""

from __future__ import annotations

import json

import pytest

from callfactory.calibration import (
    agreement,
    cohen_kappa,
    load_split,
    make_split,
    render_agreement,
)
from callfactory.labelling import HumanLabel


def ids(n=10):
    return [f"c-{i:03d}" for i in range(n)]


# --------------------------- Split --------------------------------------- #

def test_the_split_is_deterministic_and_disjoint(tmp_path):
    a = make_split(tmp_path / "a", ids(20), holdout=5, seed=7)
    b = make_split(tmp_path / "b", ids(20), holdout=5, seed=7)
    assert a.holdout == b.holdout
    assert len(a.holdout) == 5 and len(a.tune) == 15
    assert not set(a.tune) & set(a.holdout)
    assert set(a.tune) | set(a.holdout) == set(ids(20))


def test_a_different_seed_gives_a_different_holdout(tmp_path):
    a = make_split(tmp_path / "a", ids(20), holdout=5, seed=7)
    b = make_split(tmp_path / "b", ids(20), holdout=5, seed=8)
    assert a.holdout != b.holdout


def test_resplitting_is_refused(tmp_path):
    make_split(tmp_path, ids(20), holdout=5, seed=7)
    with pytest.raises(FileExistsError, match="second tuning set"):
        make_split(tmp_path, ids(20), holdout=5, seed=99)
    assert make_split(tmp_path, ids(20), holdout=6, seed=99, overwrite=True).holdout


def test_the_split_round_trips(tmp_path):
    made = make_split(tmp_path, ids(12), holdout=4, seed=3, note="hello")
    loaded = load_split(tmp_path)
    assert loaded.tune == made.tune and loaded.holdout == made.holdout
    assert loaded.seed == 3 and loaded.note == "hello"
    assert "c-000" in loaded


def test_no_split_file_returns_none(tmp_path):
    assert load_split(tmp_path / "nothing") is None


@pytest.mark.parametrize("holdout", [0, 10, 11])
def test_an_impossible_holdout_is_rejected(tmp_path, holdout):
    with pytest.raises(ValueError):
        make_split(tmp_path / str(holdout), ids(10), holdout=holdout, seed=1)


# --------------------------- kappa --------------------------------------- #

def test_kappa_endpoints():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, False, True, False], [False, True, False, True]) == -1.0
    # everything in one class -> no variation for "chance agreement", kappa undefined
    assert cohen_kappa([True] * 4, [True] * 4) is None
    assert cohen_kappa([], []) is None


def test_kappa_is_below_raw_agreement_when_classes_are_skewed():
    machine = [True] * 9 + [False]
    human = [True] * 8 + [False, True]
    k = cohen_kappa(machine, human)
    assert k is not None and k < 0.8        # raw agreement 0.8, kappa lower


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


# --------------------------- Agreement ----------------------------------- #

def machine_rec(cid, is_scam, detect, **gt):
    return {
        "conversation_id": cid,
        "ground_truth": {"is_scam": is_scam, **gt},
        "earliest_detection_index": detect,
    }


def test_agreement_counts_both_axes():
    machine = {
        "a": machine_rec("a", True, 5),
        "b": machine_rec("b", True, None),
        "c": machine_rec("c", False, None),
        "d": machine_rec("d", False, 3),
    }
    human = {
        "a": HumanLabel("a", True, 3, 9),       # human earlier
        "b": HumanLabel("b", True, 7, 9),       # only the human alarms
        "c": HumanLabel("c", False, None, 9),
        "d": HumanLabel("d", False, None, 9),   # only the machine alarms
    }
    r = agreement(machine, human)
    assert r.n == 4
    assert r.both_scam == 2 and r.both_legit == 2
    assert r.machine_only == 0 and r.human_only == 0
    assert r.kappa == 1.0
    assert r.both_flagged == 1 and r.deltas == [2] and r.median_delta == 2
    assert r.machine_later == 1 and r.machine_earlier == 0
    assert r.human_flagged_only == 1 and r.machine_flagged_only == 1
    assert r.neither_flagged == 1


def test_unsure_labels_are_excluded_from_kappa():
    machine = {"a": machine_rec("a", True, 1), "b": machine_rec("b", False, None)}
    human = {"a": HumanLabel("a", None, 1, 5), "b": HumanLabel("b", False, None, 5)}
    r = agreement(machine, human)
    assert r.unsure == 1
    assert r.both_scam == 0 and r.both_legit == 1


def test_only_restricts_the_scope():
    machine = {k: machine_rec(k, True, 1) for k in ("a", "b")}
    human = {k: HumanLabel(k, True, 1, 5) for k in ("a", "b")}
    assert agreement(machine, human, only=["a"]).n == 1


def test_conversations_without_a_human_label_are_ignored():
    machine = {k: machine_rec(k, True, 1) for k in ("a", "b", "c")}
    human = {"a": HumanLabel("a", True, 1, 5)}
    assert agreement(machine, human).n == 1


def test_the_render_names_which_side_is_more_conservative():
    machine = {k: machine_rec(k, True, None) for k in ("a", "b")}
    human = {k: HumanLabel(k, True, 3, 9) for k in ("a", "b")}
    text = render_agreement(agreement(machine, human), label="tune")
    assert "machine is more conservative than the human" in text
    assert "fix the rubric, not the threshold" in text

    machine2 = {k: machine_rec(k, True, 1) for k in ("a", "b")}
    human2 = {k: HumanLabel(k, True, None, 9) for k in ("a", "b")}
    assert "machine is more aggressive than the human" in render_agreement(agreement(machine2, human2))


def test_an_empty_report_renders_safely():
    assert "no human labels to align yet" in render_agreement(agreement({}, {}))
