"""Relabelling fixed dialogues under a new rubric (R6.7), and the detection-position guardrail."""

from __future__ import annotations

import asyncio
import json

import pytest

from callfactory.batch import OutputPaths, run_batch
from callfactory.metrics import detection_positions, label_shift
from callfactory.pipeline import GenerationSettings
from callfactory.providers import ProviderRoles, build_provider
from callfactory.providers.base import LLMProvider, ProviderConfig, ProviderError
from callfactory.relabel import read_run, relabel_run

from factories import make_task


def run(coro):
    return asyncio.run(coro)


SETTINGS = GenerationSettings(min_rounds=5, max_rounds=6)


def seed_run(tmp_path, n=3):
    caller = build_provider("mock")
    roles = ProviderRoles(caller=caller, receiver=caller, evaluator=build_provider("mock"))
    tasks = [
        make_task(task_id=f"task-7-{i:05d}", index=i, seed=7, derived_seed=1000 + i,
                  target_rounds=5)
        for i in range(n)
    ]
    run(run_batch(tasks, roles, output=tmp_path / "src", run_id="src", settings=SETTINGS))
    return tmp_path / "src"


class FixedLabeller(LLMProvider):
    """Give each prefix a hard-coded score, to check that relabelling really changes the labels."""

    def __init__(self, score, suff):
        super().__init__(ProviderConfig(name="mock", model="fixed-1"))
        self.score, self.suff = score, suff

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag):
        if tag == "evaluator_global":
            return json.dumps({"coherence": 4, "naturalness": 4, "repetition": 0.1,
                               "role_consistency": True, "notes": "fixed",
                               "blind_is_suspicious": True})
        return json.dumps({"risk_score": self.score, "risk_tier": "high",
                           "signals": ["urgency_pressure"], "new_signals_in_last_turn": [],
                           "sufficient_evidence": self.suff, "rationale": "fixed"})


# --------------------------------------------------------------------------- #
# Core guarantee: dialogue unchanged, labels changed
# --------------------------------------------------------------------------- #

def test_the_transcripts_come_through_byte_identical(tmp_path):
    src = seed_run(tmp_path)
    stats = run(relabel_run(src, tmp_path / "out", FixedLabeller(0.9, True)))
    assert stats.relabelled == 3
    a = (OutputPaths.resolve(src).conversations).read_text()
    b = (OutputPaths.resolve(tmp_path / "out").conversations).read_text()
    assert json.loads(a.splitlines()[0]) == json.loads(b.splitlines()[0])
    assert len(a.splitlines()) == len(b.splitlines())


def test_relabelling_changes_the_labels(tmp_path):
    src = seed_run(tmp_path)
    run(relabel_run(src, tmp_path / "out", FixedLabeller(0.9, True)))
    metas = [json.loads(l) for l in (tmp_path / "out" / "metadata.jsonl").read_text().splitlines()]
    assert all(m["earliest_detection_index"] == 0 for m in metas)
    assert all(p["risk_score"] == 0.9 for m in metas for p in m["prefix_labels"])


def test_the_old_label_is_kept_in_the_record(tmp_path):
    src = seed_run(tmp_path)
    old = {m["conversation_id"]: m["earliest_detection_index"]
           for m in (json.loads(l) for l in (src / "metadata.jsonl").read_text().splitlines())}
    run(relabel_run(src, tmp_path / "out", FixedLabeller(0.9, True)))
    for line in (tmp_path / "out" / "metadata.jsonl").read_text().splitlines():
        m = json.loads(line)
        prev = m["generation"]["relabelled_from"]
        assert prev["earliest_detection_index"] == old[m["conversation_id"]]
        assert prev["run_id"] == "src"


def test_the_task_and_ground_truth_survive(tmp_path):
    src = seed_run(tmp_path)
    run(relabel_run(src, tmp_path / "out", FixedLabeller(0.9, True)))
    before = json.loads((src / "metadata.jsonl").read_text().splitlines()[0])
    after = json.loads((tmp_path / "out" / "metadata.jsonl").read_text().splitlines()[0])
    assert before["task"] == after["task"]
    assert before["ground_truth"] == after["ground_truth"]
    assert before["generation"]["derived_seed"] == after["generation"]["derived_seed"]


def test_the_new_rubric_version_is_stamped(tmp_path):
    from callfactory.prompts import PREFIX_EVAL_RUBRIC_VERSION, rubric_fingerprint
    src = seed_run(tmp_path)
    stats = run(relabel_run(src, tmp_path / "out", FixedLabeller(0.9, True)))
    assert stats.to_rubric == PREFIX_EVAL_RUBRIC_VERSION
    m = json.loads((tmp_path / "out" / "metadata.jsonl").read_text().splitlines()[0])
    ev = m["generation"]["agents"]["evaluator"]
    assert ev["rubric_version"] == PREFIX_EVAL_RUBRIC_VERSION
    assert ev["rubric_fingerprint"] == rubric_fingerprint()
    # provenance on the dialogue side must not be overwritten
    assert m["generation"]["agents"]["caller"]["model"] == "mock-dialogue-1"


def test_writing_over_the_source_is_refused(tmp_path):
    src = seed_run(tmp_path)
    with pytest.raises(ValueError, match="cannot be the source directory"):
        run(relabel_run(src, src, FixedLabeller(0.9, True)))


def test_a_missing_run_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_run(tmp_path / "nope")


def test_label_shift_is_reported(tmp_path):
    src = seed_run(tmp_path)
    stats = run(relabel_run(src, tmp_path / "hot", FixedLabeller(0.9, True)))
    assert stats.shift["gained"] + stats.shift["earlier"] + stats.shift["unchanged"] == 3
    cold = run(relabel_run(src, tmp_path / "cold", FixedLabeller(0.0, False)))
    assert cold.shift["lost"] + cold.shift["unchanged"] == 3


def test_a_provider_failure_is_recorded_not_raised(tmp_path):
    class Boom(LLMProvider):
        async def _complete(self, **kw):
            raise ProviderError("down", status=500)

    src = seed_run(tmp_path)
    stats = run(relabel_run(src, tmp_path / "out", Boom(
        ProviderConfig(name="mock", model="boom", max_retries=0))))
    # evaluator failures degrade to empty labels inside evaluate_conversation, so data is still produced
    assert stats.relabelled + stats.failed == 3
    assert stats.eval_failures > 0


def test_global_eval_can_be_skipped_and_the_old_one_reused(tmp_path):
    src = seed_run(tmp_path)
    run(relabel_run(src, tmp_path / "out", FixedLabeller(0.9, True), global_eval=False))
    m = json.loads((tmp_path / "out" / "metadata.jsonl").read_text().splitlines()[0])
    assert m["blind_assessment"] is not None       # reused, not paying for another call


# --------------------------------------------------------------------------- #
# Detection-position guardrail
# --------------------------------------------------------------------------- #

def test_position_warns_when_detection_collapses_to_the_start():
    report = detection_positions([(0, 10, True), (0, 12, True), (1, 10, True)])
    assert report.median_fraction is not None and report.median_fraction < 0.25
    assert any("collapsing" in w for w in report.warnings)
    assert any("first utterance" in w for w in report.warnings)


def test_a_healthy_spread_raises_no_warning():
    report = detection_positions([(5, 12, True), (7, 14, True), (4, 10, True)])
    assert report.warnings == []
    assert 0.3 < report.median_fraction < 0.7


def test_legitimate_samples_are_excluded_from_the_position_stats():
    report = detection_positions([(0, 10, False), (5, 12, True)])
    assert report.n_scam == 1 and report.n_detected == 1


def test_undetected_scams_count_in_the_denominator_only():
    report = detection_positions([(None, 10, True), (5, 12, True)])
    assert report.n_scam == 2 and report.n_detected == 1
    assert report.summary()["detection_rate"] == 0.5


def test_label_shift_counts_every_direction():
    assert label_shift([None, 5, 9, 3, None], [4, None, 5, 7, None]) == {
        "gained": 1, "lost": 1, "earlier": 1, "later": 1, "unchanged": 1,
    }


def test_label_shift_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        label_shift([1], [1, 2])


# --------------------------------------------------------------------------- #
# rederive: only the derivation rule changes, zero model calls
# --------------------------------------------------------------------------- #

def _write_run(tmp_path, records, convs):
    import json
    root = tmp_path / "src"
    root.mkdir()
    (root / "conversations.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in convs) + "\n", encoding="utf-8")
    (root / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    return root


def _one(cid, scores, suff, *, old_detect=None, is_scam=True):
    conv = {"conversation_id": cid, "utterances": [
        {"index": i, "speaker": "caller" if i % 2 == 0 else "receiver", "text": f"line {i} of {cid}"}
        for i in range(len(scores))
    ]}
    meta = {
        "conversation_id": cid,
        "earliest_detection_index": old_detect,
        "ground_truth": {"is_scam": is_scam, "hard_negative": False, "scam_family": "f"},
        "prefix_labels": [
            {"through_index": i, "risk_score": s, "risk_tier": "x", "signals": [],
             "new_signals": [], "sufficient_evidence": b, "rationale": ""}
            for i, (s, b) in enumerate(zip(scores, suff))
        ],
        "utterance_labels": [],
        "generation": {"run_id": "src", "evaluation": {
            "threshold": 0.5, "require_sufficient": True, "detection_rule": "detect/v1"}},
    }
    return meta, conv


def test_rederive_fires_on_score_alone(tmp_path):
    from callfactory.rederive import rederive_run

    # 0.7 all the way, but sufficient_evidence is all false: no detection point under detect/v1
    meta, conv = _one("c0", [0.1, 0.7, 0.7, 0.7], [False] * 4, old_detect=None)
    root = _write_run(tmp_path, [meta], [conv])
    stats = rederive_run(root, tmp_path / "out", threshold=0.7)
    assert stats.shift["gained"] == 1
    out = [json.loads(l) for l in (tmp_path / "out" / "metadata.jsonl").read_text().splitlines()]
    assert out[0]["earliest_detection_index"] == 1
    assert out[0]["earliest_detection_round"] == 1
    assert [u["evidence_established"] for u in out[0]["utterance_labels"]] == [False, True, True, True]
    assert out[0]["generation"]["evaluation"]["detection_rule"] == "detect/v2"
    assert out[0]["generation"]["rederived_from"]["earliest_detection_index"] is None


def test_rederive_leaves_the_prefix_labels_untouched(tmp_path):
    """This is its entire difference from relabel: not one byte of the labelling may change."""
    from callfactory.rederive import rederive_run

    meta, conv = _one("c0", [0.3, 0.9, 0.2], [False, True, False], old_detect=1)
    before = json.loads(json.dumps(meta["prefix_labels"]))
    root = _write_run(tmp_path, [meta], [conv])
    rederive_run(root, tmp_path / "out", threshold=0.7)
    out = [json.loads(l) for l in (tmp_path / "out" / "metadata.jsonl").read_text().splitlines()]
    for a, b in zip(before, out[0]["prefix_labels"]):
        for key in ("through_index", "risk_score", "signals", "sufficient_evidence", "rationale"):
            assert a[key] == b[key]


def test_rederive_can_reproduce_the_old_rule(tmp_path):
    from callfactory.rederive import rederive_run

    meta, conv = _one("c0", [0.1, 0.7, 0.7, 0.7], [False] * 4, old_detect=None)
    root = _write_run(tmp_path, [meta], [conv])
    stats = rederive_run(root, tmp_path / "out", threshold=0.7, require_sufficient=True)
    assert stats.shift["unchanged"] == 1 and stats.rule == "detect/v1"


def test_rederive_refuses_to_overwrite_its_source(tmp_path):
    from callfactory.rederive import rederive_run

    meta, conv = _one("c0", [0.9], [True], old_detect=0)
    root = _write_run(tmp_path, [meta], [conv])
    with pytest.raises(ValueError, match="refusing to write back to the source directory"):
        rederive_run(root, root)
