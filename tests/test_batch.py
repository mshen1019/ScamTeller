"""Batch generation and writing: two separate files, resume, failure paths, stats."""

from __future__ import annotations

import asyncio
import json

import pytest

from callfactory.batch import (
    BatchStats,
    Checkpoint,
    JSONLWriter,
    OutputPaths,
    estimate_calls,
    run_batch,
)
from callfactory.dialogue import DialogueFailure
from callfactory.pipeline import GenerationSettings, generate_conversation
from callfactory.providers import ProviderRoles, build_provider
from callfactory.providers.base import LLMProvider, ProviderConfig, ProviderError
from callfactory.validators import leaked_keys

from factories import CALLER_LINES, RECEIVER_LINES, make_task


def run(coro):
    return asyncio.run(coro)


def mock_roles():
    caller = build_provider("mock")
    return ProviderRoles(caller=caller, receiver=caller, evaluator=build_provider("mock"))


def tasks_for(n: int, rounds: int = 5):
    out = []
    for i in range(n):
        task = make_task(
            task_id=f"task-7-{i:05d}", index=i, seed=7, derived_seed=1000 + i,
            target_rounds=rounds,
        )
        out.append(task)
    return out


def read(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


SETTINGS = GenerationSettings(min_rounds=5, max_rounds=6)


# --------------------------------------------------------------------------- #
# Pipeline assembly
# --------------------------------------------------------------------------- #

def test_a_generated_conversation_carries_full_provenance():
    conv = run(generate_conversation(make_task(target_rounds=5), mock_roles(),
                                     settings=SETTINGS, run_id="run-7"))
    gen = conv.generation
    assert conv.conversation_id == "run-7-00000"
    for key in ("bank_version", "seed", "derived_seed", "agents", "rubric_version"):
        assert key in gen, f"C4 requires metadata to record {key}"
    for role in ("caller", "receiver", "evaluator"):
        assert gen["agents"][role]["model"] == "mock-dialogue-1"
        assert gen["agents"][role]["seed_honoured_by_provider"] is True
    assert gen["agents"]["evaluator"]["temperature"] == 0.0
    assert gen["agents"]["caller"]["temperature"] == SETTINGS.temperature


def test_the_two_views_split_cleanly():
    conv = run(generate_conversation(make_task(target_rounds=5), mock_roles(),
                                     settings=SETTINGS, run_id="run-7"))
    transcript, meta = conv.transcript_record(), conv.metadata_record()
    assert leaked_keys(transcript) == set()
    assert transcript["conversation_id"] == meta["conversation_id"]
    assert meta["ground_truth"]["is_scam"] is True
    assert set(transcript["utterances"][0]) == {"index", "speaker", "text"}


def test_validation_findings_ride_along_in_the_metadata():
    conv = run(generate_conversation(make_task(target_rounds=5), mock_roles(),
                                     settings=SETTINGS, run_id="run-7"))
    assert isinstance(conv.validation, list)
    assert conv.metadata_record()["validation"] == conv.validation


# --------------------------------------------------------------------------- #
# Writing to disk (C1 / C3)
# --------------------------------------------------------------------------- #

def test_run_batch_writes_three_files_side_by_side(tmp_path):
    stats = run(run_batch(tasks_for(3), mock_roles(), output=tmp_path / "run-a",
                          run_id="run-a", settings=SETTINGS))
    paths = OutputPaths.resolve(tmp_path / "run-a")
    assert stats.written == 3 and stats.failed == 0
    convs, metas = read(paths.conversations), read(paths.metadata)
    assert len(convs) == len(metas) == 3
    assert {c["conversation_id"] for c in convs} == {m["conversation_id"] for m in metas}
    assert all(leaked_keys(c) == set() for c in convs)
    assert paths.checkpoint.exists()


def test_checkpoint_records_completed_indices(tmp_path):
    run(run_batch(tasks_for(3), mock_roles(), output=tmp_path / "r", run_id="r",
                  settings=SETTINGS))
    data = json.loads((tmp_path / "r" / "_checkpoint.json").read_text())
    assert data["completed"] == [0, 1, 2]
    assert data["run_id"] == "r"


def test_resume_skips_what_is_already_done(tmp_path):
    run(run_batch(tasks_for(2), mock_roles(), output=tmp_path / "r", run_id="r",
                  settings=SETTINGS))
    stats = run(run_batch(tasks_for(4), mock_roles(), output=tmp_path / "r", run_id="r",
                          settings=SETTINGS, resume=True))
    assert stats.skipped_resumed == 2
    assert stats.written == 2
    assert len(read(tmp_path / "r" / "conversations.jsonl")) == 4


def test_a_lost_checkpoint_is_rebuilt_from_metadata(tmp_path):
    """N3: the checkpoint is only an index; the truth is the lines on disk."""
    run(run_batch(tasks_for(3), mock_roles(), output=tmp_path / "r", run_id="r",
                  settings=SETTINGS))
    (tmp_path / "r" / "_checkpoint.json").unlink()
    paths = OutputPaths.resolve(tmp_path / "r")
    cp = Checkpoint.load_or_create(paths, "r", {}, resume=True)
    assert cp.completed == {0, 1, 2}
    assert cp.rebuilt_from_metadata is True


def test_resume_without_the_flag_regenerates(tmp_path):
    run(run_batch(tasks_for(2), mock_roles(), output=tmp_path / "r", run_id="r",
                  settings=SETTINGS))
    stats = run(run_batch(tasks_for(2), mock_roles(), output=tmp_path / "r", run_id="r",
                          settings=SETTINGS))
    assert stats.skipped_resumed == 0
    assert len(read(tmp_path / "r" / "conversations.jsonl")) == 4


def test_output_can_be_a_jsonl_path_with_derived_siblings(tmp_path):
    target = tmp_path / "batch.jsonl"
    run(run_batch(tasks_for(1), mock_roles(), output=target, run_id="r", settings=SETTINGS))
    assert target.exists()
    assert (tmp_path / "batch.metadata.jsonl").exists()
    assert (tmp_path / "batch._checkpoint.json").exists()


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #

class DeadReceiver(LLMProvider):
    """The receiver always returns nothing; run_dialogue should give up on this task."""

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag):
        return "" if tag == "receiver" else "Hello, this is [COMPANY_A] calling."


class FlatProvider(LLMProvider):
    """Both sides keep saying the same line; the repetition check should keep it out of the dataset."""

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag):
        if tag == "receiver":
            return "Sorry, who did you say you were calling from, exactly?"
        if tag == "caller":
            return "I do need to settle this today, the window closes at five."
        return '{"risk_score": 0.1, "risk_tier": "low", "signals": [], ' \
               '"sufficient_evidence": false, "rationale": "flat"}'


class BrokenCaller(LLMProvider):
    async def _complete(self, **kw):
        raise ProviderError("upstream exploded", status=500)


def roles_with(provider):
    return ProviderRoles(caller=provider, receiver=provider, evaluator=build_provider("mock"))


def test_a_dialogue_failure_lands_in_failures_not_in_the_dataset(tmp_path):
    provider = DeadReceiver(ProviderConfig(name="mock", model="dead-1"))
    stats = run(run_batch(tasks_for(2), roles_with(provider), output=tmp_path / "r",
                          run_id="r", settings=SETTINGS))
    assert stats.failed == 2 and stats.written == 0
    failures = read(tmp_path / "r" / "failures.jsonl")
    assert {f["stage"] for f in failures} == {"generation"}
    assert "DialogueFailure" in failures[0]["error"]
    assert not (tmp_path / "r" / "conversations.jsonl").read_text().strip()


def test_a_provider_error_fails_only_its_own_sample(tmp_path):
    provider = BrokenCaller(ProviderConfig(name="mock", model="boom-1", max_retries=0))
    stats = run(run_batch(tasks_for(1), roles_with(provider), output=tmp_path / "r",
                          run_id="r", settings=SETTINGS))
    assert stats.failed == 1
    assert "ProviderError" in read(tmp_path / "r" / "failures.jsonl")[0]["error"]


def test_hard_rule_failures_are_not_written_by_default(tmp_path):
    flat = FlatProvider(ProviderConfig(name="mock", model="flat-1"))
    stats = run(run_batch(tasks_for(1), roles_with(flat), output=tmp_path / "r",
                          run_id="r", settings=SETTINGS))
    assert stats.invalid == 1 and stats.written == 0
    failure = read(tmp_path / "r" / "failures.jsonl")[0]
    assert failure["stage"] == "validation"
    assert any(e["code"] == "repetition.excessive" for e in failure["errors"])
    assert not (tmp_path / "r" / "conversations.jsonl").read_text().strip()


def test_keep_invalid_writes_them_anyway(tmp_path):
    flat = FlatProvider(ProviderConfig(name="mock", model="flat-1"))
    stats = run(run_batch(tasks_for(1), roles_with(flat), output=tmp_path / "r",
                          run_id="r", settings=SETTINGS, keep_invalid=True))
    assert stats.invalid == 1 and stats.written == 1
    meta = read(tmp_path / "r" / "metadata.jsonl")[0]
    assert any(i["code"] == "repetition.excessive" for i in meta["validation"])


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def test_stats_report_false_alarms_next_to_detection(tmp_path):
    """R7.3: reporting latency without the false-alarm rate rewards 'always alarm immediately'."""
    summary = BatchStats(
        written=10, scam=5, legitimate=5, hard_negative=4,
        detections_on_scam=4, detection_round_total=12,
        false_alarms_on_legit=2, false_alarms_on_hard_negative=2,
        blind_disagreements=1, rounds_total=70,
    ).summary()
    assert summary["scam_detection_rate"] == 0.8
    assert summary["mean_detection_round_on_detected_scam"] == 3.0
    assert summary["legit_false_alarm_rate"] == 0.4
    assert summary["hard_negative_false_alarm_rate"] == 0.5
    assert summary["blind_disagreement_rate"] == 0.1
    assert summary["mean_rounds"] == 7.0


def test_empty_stats_do_not_divide_by_zero():
    summary = BatchStats().summary()
    assert summary["scam_detection_rate"] == 0.0
    assert summary["mean_detection_round_on_detected_scam"] is None


def test_stats_are_accumulated_from_real_records(tmp_path):
    stats = run(run_batch(tasks_for(3), mock_roles(), output=tmp_path / "r", run_id="r",
                          settings=SETTINGS))
    assert stats.scam == 3            # the fixture's scenario is a scam
    assert stats.written == 3
    assert stats.rounds_total >= 15


# --------------------------------------------------------------------------- #
# Cost estimate (N6 / N1)
# --------------------------------------------------------------------------- #

def test_estimate_counts_every_kind_of_call():
    est = estimate_calls(tasks_for(2, rounds=5), GenerationSettings(min_rounds=5, max_rounds=20))
    assert est["conversations"] == 2
    assert est["dialogue_calls"] == 20            # 2 × 5 rounds × 2 sides
    assert est["prefix_eval_calls"] == 20         # stride 1: once per utterance
    assert est["global_eval_calls"] == 2
    assert est["total_calls"] == 42


def test_stride_and_switches_cut_the_estimate():
    tasks = tasks_for(2, rounds=5)
    base = GenerationSettings(min_rounds=5, max_rounds=20)
    strided = estimate_calls(tasks, GenerationSettings(min_rounds=5, max_rounds=20, eval_stride=3))
    assert strided["prefix_eval_calls"] < estimate_calls(tasks, base)["prefix_eval_calls"]
    off = estimate_calls(tasks, GenerationSettings(min_rounds=5, max_rounds=20,
                                                   prefix_eval=False, global_eval=False))
    assert off["prefix_eval_calls"] == 0 and off["global_eval_calls"] == 0
    assert off["total_calls"] == off["dialogue_calls"]


def test_peak_in_flight_is_bigger_than_concurrency():
    """N1: rate limits must be sized by peak in-flight requests, not by concurrent conversations."""
    est = estimate_calls(tasks_for(4, rounds=5), GenerationSettings(min_rounds=5, max_rounds=20),
                         concurrency=4)
    assert est["peak_in_flight_requests"] == 40 > est["concurrency"]
    capped = estimate_calls(
        tasks_for(4, rounds=5),
        GenerationSettings(min_rounds=5, max_rounds=20, eval_concurrency=2),
        concurrency=4,
    )
    assert capped["peak_in_flight_requests"] == 8


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #

def test_writer_appends_and_survives_reopening(tmp_path):
    path = tmp_path / "out.jsonl"
    with JSONLWriter(path) as w:
        run(w.write({"a": 1}))
    with JSONLWriter(path) as w:
        run(w.write({"b": "naïve café"}))
    lines = read(path)
    assert lines == [{"a": 1}, {"b": "naïve café"}]
    assert "naïve café" in path.read_text(encoding="utf-8")       # not escaped


def test_concurrent_writes_stay_line_atomic(tmp_path):
    path = tmp_path / "out.jsonl"

    async def go():
        with JSONLWriter(path) as w:
            await asyncio.gather(*(w.write({"i": i, "pad": "x" * 500}) for i in range(50)))

    run(go())
    assert sorted(r["i"] for r in read(path)) == list(range(50))


# --------------------------------------------------------------------------- #
# Reproducibility (P5)
# --------------------------------------------------------------------------- #

def test_the_same_tasks_produce_the_same_transcripts(tmp_path):
    for name in ("a", "b"):
        run(run_batch(tasks_for(3), mock_roles(), output=tmp_path / name, run_id="r",
                      settings=SETTINGS))
    a = [c["utterances"] for c in read(tmp_path / "a" / "conversations.jsonl")]
    b = [c["utterances"] for c in read(tmp_path / "b" / "conversations.jsonl")]
    assert a == b


# --------------------------------------------------------------------------- #
# Actual usage (estimate is an estimate; this is measured)
# --------------------------------------------------------------------------- #

class CountingProvider(LLMProvider):
    """Report a fake usage based on response length, to check accumulation and aggregation."""

    #: The lines must differ enough, otherwise the repetition check blocks the whole record and every stat is 0.
    LINES = CALLER_LINES + RECEIVER_LINES

    async def _complete(self, *, system, messages, temperature, max_tokens, seed, tag):
        self._add_usage(100, 20)
        if tag in ("caller", "receiver"):
            return self.LINES[self.usage["requests"] % len(self.LINES)]
        return '{"risk_score": 0.1, "risk_tier": "low", "signals": [], ' \
               '"sufficient_evidence": false, "rationale": "x"}'


def test_usage_is_recorded_per_provider_and_totalled(tmp_path):
    dialogue = CountingProvider(ProviderConfig(name="mock", model="count-1"))
    evaluator = CountingProvider(ProviderConfig(name="mock", model="count-2"))
    roles = ProviderRoles(caller=dialogue, receiver=dialogue, evaluator=evaluator)
    stats = run(run_batch(tasks_for(2), roles, output=tmp_path / "r", run_id="r",
                          settings=SETTINGS))
    assert stats.written == 2
    # caller and receiver share one provider, counted once
    assert set(stats.usage) == {"mock/count-1", "mock/count-2"}
    assert stats.usage["mock/count-1"]["requests"] == 20      # 2 records × 5 rounds × 2 sides
    totals = stats.totals()
    assert totals["requests"] == sum(u["requests"] for u in stats.usage.values())
    assert totals["input_tokens"] == totals["requests"] * 100
    assert stats.summary()["usage_totals"] == totals


def test_usage_stays_empty_when_a_provider_reports_none(tmp_path):
    stats = run(run_batch(tasks_for(1), mock_roles(), output=tmp_path / "r", run_id="r",
                          settings=SETTINGS))
    assert stats.totals()["input_tokens"] == 0     # the offline mock doesn't invent token counts


def test_no_evidence_scams_stay_in_the_detection_denominator():
    """`no_observable_evidence` is a property of the **premise**, not of the actual dialogue.

    Taking them out of the denominator as "correct" would record an open question awaiting human ruling as a score:
    in testing, the caller in 16 of 17 refused a verification channel when pressed.
    """
    from callfactory.batch import BatchStats

    stats = BatchStats(written=10, scam=10, scam_no_evidence=6, no_evidence_detected=0,
                       detections_on_scam=3, detection_round_total=9, rounds_total=60)
    out = stats.summary()
    assert out["scam_detection_rate"] == 0.3           # 3/10, not 3/4
    assert out["no_evidence_detection_rate"] == 0.0    # reported separately, not deducted from the denominator
