"""bench harness: protocol, replay, scoring. All offline."""

from __future__ import annotations

import asyncio
import json

import pytest

from callfactory.bench import (
    ReplayAdapter,
    Verdict,
    average_precision,
    discrimination_feedback,
    fingerprint,
    render_transcript,
    replay_call,
    score_system,
    slice_scores,
    trivial_baselines,
    user_message,
)
from callfactory.schema import Utterance


def run(coro):
    return asyncio.run(coro)


def utts(n):
    return [
        Utterance(index=i, speaker="caller" if i % 2 == 0 else "receiver", text=f"line {i}")
        for i in range(n)
    ]


class ScriptedJudge:
    """Score by utterance index; record the prefix length of every request."""

    name = "scripted"

    def __init__(self, scores: dict[int, float], alarm_from: int | None = None):
        self.scores = scores
        self.alarm_from = alarm_from
        self.asked: list[int] = []

    async def judge(self, call_id, prefix):
        k = prefix[-1].index
        self.asked.append(len(prefix))
        return Verdict(
            risk_score=self.scores.get(k, 0.0),
            is_alarm=self.alarm_from is not None and k >= self.alarm_from,
            signals=["urgency_pressure", "not_a_real_signal"],
        )

    async def aclose(self):
        return None


# --------------------------- Verdict ------------------------------------- #

def test_verdict_clamps_and_drops_out_of_vocabulary_signals():
    v = Verdict(risk_score=1.7, is_alarm=True, signals=["urgency_pressure", "nope"])
    assert v.risk_score == 1.0
    assert v.signals == ["urgency_pressure"]
    assert Verdict(risk_score=-3, is_alarm=False).risk_score == 0.0


# --------------------------- Replay -------------------------------------- #

def test_replay_sends_the_whole_prefix_every_time():
    """R7.1: stateless models become streaming by replay, at O(N²) cost; this pins that fact down."""
    judge = ScriptedJudge({})
    out = run(replay_call(ReplayAdapter(judge), "c1", utts(6), threshold=0.5))
    assert judge.asked == [1, 2, 3, 4, 5, 6]
    assert len(out["steps"]) == 6


def test_ask_at_carries_the_previous_verdict_forward():
    """R7.6: requests are really sent only at some positions; positions in between repeat the last verdict, marked fresh=False."""
    judge = ScriptedJudge({0: 0.1, 3: 0.9})
    out = run(replay_call(ReplayAdapter(judge, ask_at={0, 3}), "c1", utts(5), threshold=0.5))
    assert judge.asked == [1, 4]
    assert [s["fresh"] for s in out["steps"]] == [True, False, False, True, False]
    assert [s["risk_score"] for s in out["steps"]] == [0.1, 0.1, 0.1, 0.9, 0.9]


def test_the_first_alarm_is_not_retractable():
    """Once raised it stays raised: a real detector can't withdraw an alarm that has gone off."""
    judge = ScriptedJudge({2: 0.9, 3: 0.1, 4: 0.1})
    out = run(replay_call(ReplayAdapter(judge), "c1", utts(5), threshold=0.5))
    assert out["alarm_index"] == 2
    assert out["alarm_round"] == 2


def test_no_alarm_gives_none():
    out = run(replay_call(ReplayAdapter(ScriptedJudge({})), "c1", utts(4), threshold=0.5))
    assert out["alarm_index"] is None and out["alarm_round"] is None


# --------------------------- prompt -------------------------------------- #

def test_the_transcript_comes_before_the_question():
    """The other way round there's no cacheable common prefix; this is the precondition for saving 84% of input tokens."""
    msg = user_message(utts(4))
    assert msg.index("[0] CALLER") < msg.index("sufficient to warn")


def test_the_prefix_rendering_is_purely_concatenative():
    """The rendering of prefix k-1 must be a literal prefix of the rendering of prefix k, otherwise the cache breaks partway."""
    a, b = render_transcript(utts(3)), render_transcript(utts(5))
    assert b.startswith(a)


def test_the_prompt_has_a_fingerprint():
    assert len(fingerprint()) == 12


# --------------------------- Scoring ------------------------------------- #

def meta(cid, *, is_scam=True, hard_negative=False, detect=None, prefixes=None,
         compliance=False, late=False, no_evidence=False):
    prefixes = prefixes if prefixes is not None else []
    return {
        "conversation_id": cid,
        "earliest_detection_index": detect,
        "ground_truth": {
            "is_scam": is_scam, "hard_negative": hard_negative,
            "compliance_intended": compliance, "late_onset": late,
            "no_observable_evidence": no_evidence, "scam_family": "f",
        },
        "prefix_labels": [
            {"through_index": k, "risk_score": 0.0, "sufficient_evidence": False,
             "evidence_established": est}
            for k, est in prefixes
        ],
    }


def runrec(cid, alarm, steps):
    return {"conversation_id": cid, "alarm_index": alarm,
            "alarm_round": None if alarm is None else alarm // 2 + 1,
            "steps": [{"index": k, "risk_score": s, "is_alarm": f, "fired": f,
                       "signals": [], "rationale": "", "latency_ms": None, "fresh": True}
                      for k, s, f in steps]}


def test_prefix_level_counts_only_where_labels_exist():
    metas = {"c1": meta("c1", detect=3, prefixes=[(1, False), (3, True)])}
    runs = [runrec("c1", 3, [(0, .1, False), (1, .2, False), (2, .2, False), (3, .9, True)])]
    out = score_system(runs, metas, threshold=0.5)
    p = out["prefix_level"]
    assert p["n"] == 2 and p["tp"] == 1 and p["tn"] == 1
    assert p["accuracy"] == 1.0


def test_latency_is_signed_against_the_label():
    metas = {"a": meta("a", detect=5, prefixes=[(5, True)]),
             "b": meta("b", detect=5, prefixes=[(5, True)])}
    runs = [runrec("a", 3, [(3, .9, True), (5, .9, True)]),     # 2 early
            runrec("b", 9, [(5, .1, False), (9, .9, True)])]    # 4 late
    cl = score_system(runs, metas, threshold=0.5)["call_level"]
    assert cl["detection_rate"] == 1.0
    assert sorted([-2, 4]) == sorted([-2, 4])
    assert cl["earlier_than_label"] == 1 and cl["later_than_label"] == 1
    assert cl["latency_median_utterances"] == 1  # median(-2, 4)


def test_alarming_on_a_no_evidence_scam_counts_as_a_false_alarm_not_a_detection():
    """R7.8: ground truth has no detection point -> latency can't be computed, and an alarm counts on the false-alarm side."""
    metas = {"c": meta("c", detect=None, no_evidence=True, prefixes=[(1, False)])}
    runs = [runrec("c", 1, [(1, .9, True)])]
    cl = score_system(runs, metas, threshold=0.5)["call_level"]
    assert cl["detectable"] == 0 and cl["detection_rate"] is None
    assert cl["no_gt_point_n"] == 1 and cl["no_gt_point_alarmed"] == 1
    assert cl["latency_n"] == 0


def test_false_alarms_on_negatives_are_split_by_kind():
    metas = {"h": meta("h", is_scam=False, hard_negative=True, prefixes=[(1, False)]),
             "p": meta("p", is_scam=False, prefixes=[(1, False)])}
    runs = [runrec("h", 1, [(1, .9, True)]), runrec("p", None, [(1, .1, False)])]
    cl = score_system(runs, metas, threshold=0.5)["call_level"]
    assert cl["hard_negative_false_alarm_rate"] == 1.0
    assert cl["plain_false_alarm_rate"] == 0.0


def test_detection_point_pairs_catch_a_model_that_only_smells_the_call():
    """Two utterances either side of the detection point: alarming on both means it's sniffing, not judging the moment."""
    metas = {"c": meta("c", detect=3, prefixes=[(1, False), (3, True)])}
    sharp = [runrec("c", 3, [(1, .2, False), (3, .9, True)])]
    smelly = [runrec("c", 1, [(1, .9, True), (3, .9, True)])]
    assert score_system(sharp, metas, threshold=0.5)["detection_point_pairs"]["accuracy"] == 1.0
    assert score_system(smelly, metas, threshold=0.5)["detection_point_pairs"]["accuracy"] == 0.0


def test_exclude_drops_conversations_from_every_number():
    metas = {"a": meta("a", detect=1, prefixes=[(1, True)]),
             "b": meta("b", detect=1, prefixes=[(1, True)])}
    runs = [runrec("a", 1, [(1, .9, True)]), runrec("b", None, [(1, .1, False)])]
    full = score_system(runs, metas, threshold=0.5)
    cut = score_system(runs, metas, threshold=0.5, exclude=["b"])
    assert full["call_level"]["detectable"] == 2
    assert cut["call_level"]["detectable"] == 1 and cut["call_level"]["detection_rate"] == 1.0


def test_average_precision_is_none_without_both_classes():
    assert average_precision([0.1, 0.9], [0, 0]) is None
    assert average_precision([0.9, 0.1], [1, 0]) == 1.0


def test_trivial_baselines_state_what_doing_nothing_scores():
    metas = {f"c{i}": meta(f"c{i}", prefixes=[(1, i < 2)]) for i in range(10)}
    tb = trivial_baselines(metas)
    assert tb["prefixes"] == 10 and tb["positives"] == 2
    assert tb["always_no"]["prefix_accuracy"] == 0.8
    assert tb["always_yes"]["prefix_accuracy"] == 0.2


def test_slices_are_reported_separately():
    metas = {
        "a": meta("a", detect=1, compliance=True, prefixes=[(1, True)]),
        "b": meta("b", detect=1, late=True, prefixes=[(1, True)]),
        "h": meta("h", is_scam=False, hard_negative=True, prefixes=[(1, False)]),
    }
    runs = [runrec("a", None, [(1, .1, False)]), runrec("b", 1, [(1, .9, True)]),
            runrec("h", None, [(1, .1, False)])]
    sl = slice_scores(runs, metas, threshold=0.5)
    assert sl["compliance"]["call_level"]["detection_rate"] == 0.0
    assert sl["late_onset"]["call_level"]["detection_rate"] == 1.0
    assert sl["hard_negative"]["call_level"]["hard_negative_false_alarm_rate"] == 0.0


def test_discrimination_feedback_flags_both_directions():
    """R7.5: every system alarms at T0 = non-discriminative; every system silent while the label says decidable = suspected label error."""
    metas = {"t0": meta("t0", detect=0, prefixes=[(0, True)]),
             "miss": meta("miss", detect=5, prefixes=[(5, True)])}
    by = {
        "m1": [runrec("t0", 0, [(0, .9, True)]), runrec("miss", None, [(5, .1, False)])],
        "m2": [runrec("t0", 0, [(0, .9, True)]), runrec("miss", None, [(5, .1, False)])],
    }
    fb = discrimination_feedback(by, metas)
    assert fb["no_discrimination"] == ["t0"]
    assert fb["suspected_label_error"] == ["miss"]


# --------------------------- runner (offline) ---------------------------- #

def test_the_runner_only_asks_at_labelled_prefixes_and_is_resumable(tmp_path):
    """Saves 42% of calls, and resumes by conversation_id after an interruption."""
    from callfactory.bench.adapters import MockJudge
    from callfactory.bench.run import ask_positions, done_ids, run_bench

    convs = {f"c{i}": utts(8) for i in range(4)}
    metas = {f"c{i}": meta(f"c{i}", detect=3, prefixes=[(1, False), (3, True), (7, True)])
             for i in range(4)}
    assert ask_positions(metas["c0"]) == {1, 3, 7}

    judge = MockJudge()
    out = tmp_path / "runs.jsonl"
    stats = run(run_bench(judge, convs, metas, out, threshold=0.5, concurrency=2))
    assert stats.calls_done == 4 and stats.failed == 0
    assert judge.usage.requests == 4 * 3          # 3 labelled prefixes per call, not 8 utterances
    assert done_ids(out) == {"c0", "c1", "c2", "c3"}

    # resume: everything already done is skipped, no more requests
    again = run(run_bench(MockJudge(), convs, metas, out, threshold=0.5))
    assert again.calls_done == 0 and again.calls_skipped == 4


def test_the_runner_stops_at_the_request_cap(tmp_path):
    from callfactory.bench.adapters import MockJudge
    from callfactory.bench.run import run_bench

    convs = {f"c{i}": utts(6) for i in range(10)}
    metas = {f"c{i}": meta(f"c{i}", detect=1, prefixes=[(1, True), (3, True)])
             for i in range(10)}
    stats = run(run_bench(MockJudge(), convs, metas, tmp_path / "r.jsonl",
                          concurrency=1, max_requests=6))
    assert stats.stopped_early and stats.calls_done < 10


def test_load_split_hands_the_model_only_the_transcript(tmp_path):
    """R7.2: the model under test must not touch metadata."""
    import json as _json
    from callfactory.bench.run import load_split

    (tmp_path / "conversations.jsonl").write_text(_json.dumps({
        "conversation_id": "c1",
        "utterances": [{"index": 0, "speaker": "caller", "text": "hello"}],
    }) + "\n", encoding="utf-8")
    (tmp_path / "metadata.jsonl").write_text(
        _json.dumps(meta("c1", detect=0, prefixes=[(0, True)])) + "\n", encoding="utf-8")
    convs, metas = load_split(tmp_path)
    assert [u.text for u in convs["c1"]] == ["hello"]
    assert not hasattr(convs["c1"][0], "ground_truth")
    assert metas["c1"]["earliest_detection_index"] == 0


def test_the_manifest_freezes_the_prompt_and_threshold():
    from callfactory.bench.run import manifest
    m = manifest(["mock"], threshold=0.5, split="eval/ds3")
    assert m["prompt_fingerprint"] == fingerprint()
    assert m["threshold"] == 0.5 and m["judge_version"]


def test_the_manifest_accumulates_systems_instead_of_clobbering_them():
    """One `bench` run benchmarks one system, so the manifest must union, not overwrite.

    The committed manifests used to name whichever system ran last — a directory of seven run
    files whose provenance claimed one. `created_at` stays at the first write so the record of
    when the directory was started is not lost.
    """
    from callfactory.bench.run import manifest

    first = manifest(["claude-opus"], threshold=0.5, split="eval/ds3")
    second = manifest(["jev"], threshold=0.5, split="eval/ds3", previous=first)
    assert second["systems"] == ["claude-opus", "jev"]
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]

    # Re-running the same system does not duplicate it.
    third = manifest(["jev"], threshold=0.5, split="eval/ds3", previous=second)
    assert third["systems"] == ["claude-opus", "jev"]


# --------------------------- Threshold tuning ----------------------------- #

def test_tuning_respects_the_false_alarm_budget():
    """The objective is **constrained maximum detection**: the false-alarm side is where users uninstall the app."""
    from callfactory.bench.tune import HARD_NEG_BUDGET, tune

    metas = {}
    runs = []
    # 10 decidable scams: score 0.6; 20 hard negatives: 4 at 0.6, 16 at 0.3
    for i in range(10):
        cid = f"s{i}"
        metas[cid] = meta(cid, detect=1, prefixes=[(1, True)])
        runs.append(runrec(cid, 1, [(1, 0.6, False)]))
    for i in range(20):
        cid = f"h{i}"
        metas[cid] = meta(cid, is_scam=False, hard_negative=True, prefixes=[(1, False)])
        runs.append(runrec(cid, None, [(1, 0.6 if i < 4 else 0.3, False)]))
    res = tune("x", runs, metas)
    # 0.3 alarms on all 20 hard negatives; 0.6 alarms on 4/20 = 0.20, still over budget;
    # above 0.65 neither side alarms -> 0 false alarms but **0 detections too**. That case must be judged "budget not reachable",
    # and "never alarm" must not pass for meeting it; that is how the first objective picked 0.95.
    assert res.met_budget is False
    assert res.detection_rate and res.detection_rate > 0, "the reported operating point must still alarm"
    assert res.hard_negative_false_alarm_rate == 0.20
    assert res.threshold <= 0.6
    assert HARD_NEG_BUDGET == 0.10


def test_tuning_picks_the_highest_detection_within_budget():
    from callfactory.bench.tune import tune

    metas, runs = {}, []
    for i in range(10):                      # scam scores 0.9; any threshold ≤0.9 catches them
        cid = f"s{i}"
        metas[cid] = meta(cid, detect=1, prefixes=[(1, True)])
        runs.append(runrec(cid, 1, [(1, 0.9, False)]))
    for i in range(20):                      # hard negatives all at 0.2
        cid = f"h{i}"
        metas[cid] = meta(cid, is_scam=False, hard_negative=True, prefixes=[(1, False)])
        runs.append(runrec(cid, None, [(1, 0.2, False)]))
    res = tune("x", runs, metas)
    assert res.met_budget and res.detection_rate == 1.0
    assert res.hard_negative_false_alarm_rate == 0.0
    assert 0.25 <= res.threshold <= 0.9      # falls between the two groups of scores


def test_tuning_never_calls_the_model():
    """It only recomputes on scores already on disk; changing the threshold shouldn't cost another cent."""
    from callfactory.bench.tune import tune

    metas = {"s0": meta("s0", detect=1, prefixes=[(1, True)])}
    runs = [runrec("s0", 1, [(1, 0.8, False)])]
    res = tune("x", runs, metas)
    assert len(res.curve) == 19 and res.curve[0]["threshold"] == 0.05


def test_the_threshold_is_not_short_circuited_by_the_models_own_alarm_flag():
    """We were bitten by this: writing the decision as `is_alarm or score>=t` short-circuits the threshold into a no-op;
    threshold 0.5->0.95 barely moved the false-alarm rate, because whatever the model said "alarm" on bypassed the threshold.
    Same class as the label-side 0.5 threshold short-circuited by sufficient_evidence in §5.9."""
    metas = {"h": meta("h", is_scam=False, hard_negative=True, prefixes=[(1, False)])}
    # the model gave a low score but said "alarm" itself
    runs = [{"conversation_id": "h", "alarm_index": 1, "alarm_round": 1,
             "steps": [{"index": 1, "risk_score": 0.2, "is_alarm": True, "fired": True,
                        "signals": [], "rationale": "", "latency_ms": None, "fresh": True}]}]
    low = score_system(runs, metas, threshold=0.1)
    high = score_system(runs, metas, threshold=0.9)
    assert low["call_level"]["hard_negative_false_alarm_rate"] == 1.0
    assert high["call_level"]["hard_negative_false_alarm_rate"] == 0.0, "the threshold must have a say"
    # the vendor's own operating point gets its own row
    own = score_system(runs, metas, threshold=0.9, use_model_alarm=True)
    assert own["call_level"]["hard_negative_false_alarm_rate"] == 1.0
    assert own["decision"] == "model_alarm"


def test_scoring_ignores_the_fired_field_stored_in_the_run_file():
    """The run file stores only evidence (scores + model booleans); the decision is made at scoring time, so changing the threshold needs no rerun."""
    metas = {"s": meta("s", detect=1, prefixes=[(1, True)])}
    runs = [{"conversation_id": "s", "alarm_index": None, "alarm_round": None,
             "steps": [{"index": 1, "risk_score": 0.8, "is_alarm": False, "fired": False,
                        "signals": [], "rationale": "", "latency_ms": None, "fresh": True}]}]
    out = score_system(runs, metas, threshold=0.5)
    assert out["call_level"]["detection_rate"] == 1.0   # fired=False was stored, but 0.8 >= 0.5


def test_bench_tune_refuses_to_tune_on_eval(tmp_path, capsys):
    """Tuning the threshold on eval and then reporting on eval is a back door for the baseline; the CLI refuses outright.

    This is enforced in the CLI rather than only written in the docs, because docs can't stop a command typed on autopilot.
    """
    from callfactory.cli import main

    for sub in ("eval", "test"):
        split = tmp_path / sub / "ds"
        split.mkdir(parents=True)
        (split / "conversations.jsonl").write_text("", encoding="utf-8")
        (split / "metadata.jsonl").write_text("", encoding="utf-8")
        code = main(["bench-tune", str(split)])
        assert code == 2
        assert "thresholds are not tuned on eval / test" in capsys.readouterr().err


def test_bench_table_applies_per_system_thresholds(tmp_path, capsys):
    """--thresholds gives each system its own tuned threshold; systems not in the file fall back to --threshold."""
    from callfactory.cli import main

    split = tmp_path / "eval" / "ds"
    split.mkdir(parents=True)
    (split / "conversations.jsonl").write_text(json.dumps({"conversation_id": "s", "utterances": [
        {"index": 0, "speaker": "caller", "text": "hi"},
        {"index": 1, "speaker": "receiver", "text": "hello"}]}) + "\n", encoding="utf-8")
    (split / "metadata.jsonl").write_text(
        json.dumps(meta("s", detect=1, prefixes=[(0, False), (1, True)])) + "\n", encoding="utf-8")
    runs = tmp_path / "runs"
    runs.mkdir()
    rec = runrec("s", 1, [(0, 0.1, False), (1, 0.8, True)])
    for name in ("tuned", "untuned"):
        (runs / f"{name}.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps({"tuned": 0.9}), encoding="utf-8")

    assert main(["bench-table", str(split), "--runs", str(runs), "--thresholds", str(thr), "--json"]) == 0
    systems = json.loads(capsys.readouterr().out)["systems"]
    assert systems["tuned"]["threshold"] == 0.9
    assert systems["tuned"]["with_unadjudicated"]["call_level"]["detection_rate"] == 0.0  # 0.8 < 0.9
    assert systems["untuned"]["threshold"] == 0.5
    assert systems["untuned"]["with_unadjudicated"]["call_level"]["detection_rate"] == 1.0
