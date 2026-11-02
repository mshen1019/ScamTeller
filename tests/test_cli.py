"""CLI (the three zero-model-call commands implemented)."""

import json
from collections import Counter

import pytest

from callfactory.cli import main


def init(tmp_path, **kw):
    args = ["init-bank", "--bank-dir", str(tmp_path)]
    for k, v in kw.items():
        args += [f"--{k.replace('_','-')}", str(v)]
    return main(args)


def test_init_bank_writes_a_v1_bank(tmp_path, capsys):
    assert init(tmp_path) == 0
    out = capsys.readouterr().out
    assert "25 scenarios" in out
    assert (tmp_path / "seed.v1.jsonl").exists()


def test_init_bank_refuses_to_overwrite(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert init(tmp_path) == 1
    assert "version already exists" in capsys.readouterr().err


def test_init_bank_overwrite_flag(tmp_path):
    init(tmp_path)
    assert main(["init-bank", "--bank-dir", str(tmp_path), "--overwrite"]) == 0


def test_review_bank_reports_gaps(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert main(["review-bank", "--bank-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "coverage gaps" in out
    assert "refusal_to_verify" in out            # a real gap in the seed bank
    assert "genuine_it_helpdesk" in out


def test_review_bank_json(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert main(["review-bank", "--bank-dir", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["size"] == 25
    assert "coverage_gaps" in data
    assert len(data["scenarios"]) == 25


def test_plan_prints_tasks_and_makes_no_requests(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert main([
        "plan", "--bank-dir", str(tmp_path),
        "--num-samples", "12", "--seed", "99", "--min-rounds", "5", "--max-rounds", "8",
    ]) == 0
    out = capsys.readouterr().out
    assert "12 tasks" in out
    assert "unique persona triples: 12/12" in out
    assert "ratios: scam" in out


def test_plan_json_shape(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert main(["plan", "--bank-dir", str(tmp_path), "--num-samples", "4", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["tasks"]) == 4
    t = data["tasks"][0]
    assert {"task_id", "scenario", "caller_persona", "receiver_persona", "late_onset"} <= set(t)
    assert t["bank_version"] == 1


def test_plan_is_reproducible_across_invocations(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    main(["plan", "--bank-dir", str(tmp_path), "--num-samples", "6", "--seed", "5", "--json"])
    first = capsys.readouterr().out
    main(["plan", "--bank-dir", str(tmp_path), "--num-samples", "6", "--seed", "5", "--json"])
    assert capsys.readouterr().out == first


def test_plan_family_filter(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    main([
        "plan", "--bank-dir", str(tmp_path), "--num-samples", "5", "--scam-ratio", "1.0",
        "--include-family", "prize_notification", "--json",
    ])
    data = json.loads(capsys.readouterr().out)
    assert {t["scenario"]["family"] for t in data["tasks"]} == {"prize_notification"}


def test_plan_rejects_inverted_rounds(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert main([
        "plan", "--bank-dir", str(tmp_path), "--min-rounds", "9", "--max-rounds", "3"
    ]) == 2


def test_plan_reports_unknown_family(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert main([
        "plan", "--bank-dir", str(tmp_path), "--include-family", "nope", "--num-samples", "2"
    ]) == 1
    assert "no such families" in capsys.readouterr().err


def test_missing_bank_is_reported(tmp_path, capsys):
    assert main(["review-bank", "--bank-dir", str(tmp_path / "empty")]) == 1
    assert "no scenario bank" in capsys.readouterr().err


def test_show_directives(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    main(["plan", "--bank-dir", str(tmp_path), "--num-samples", "2", "--show-directives"])
    assert "outcome:" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# generate / estimate / inspect / validate
# --------------------------------------------------------------------------- #

def generate(tmp_path, out, **kw):
    args = [
        "generate", "--bank-dir", str(tmp_path), "--output", str(out),
        "--run-id", "run-t", "--num-samples", "4", "--seed", "7",
        "--min-rounds", "5", "--max-rounds", "6",
    ]
    for k, v in kw.items():
        flag = f"--{k.replace('_', '-')}"
        args += [flag] if v is True else [flag, str(v)]
    return main(args)


def test_estimate_makes_no_requests(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert main(["estimate", "--bank-dir", str(tmp_path), "--num-samples", "10"]) == 0
    out = capsys.readouterr().out
    assert "peak in-flight requests" in out
    assert "total" in out


def test_estimate_json_has_the_cost_fields(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    main(["estimate", "--bank-dir", str(tmp_path), "--num-samples", "5", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["total_calls"] > 0
    assert data["peak_in_flight_requests"] >= data["concurrency"]


def test_generate_writes_a_run(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    out = tmp_path / "run"
    assert generate(tmp_path, out) == 0
    printed = capsys.readouterr().out
    assert "wrote 4" in printed
    assert "false-alarm rate" in printed            # must be on screen with the detection rate (R7.3)
    for name in ("conversations.jsonl", "metadata.jsonl", "_checkpoint.json"):
        assert (out / name).exists()
    convs = [json.loads(l) for l in (out / "conversations.jsonl").read_text().splitlines()]
    assert len(convs) == 4
    assert all(set(u) == {"index", "speaker", "text"}
               for c in convs for u in c["utterances"])


def test_generate_json_prints_only_stats(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert generate(tmp_path, tmp_path / "run", json=True) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["written"] == 4
    assert "hard_negative_false_alarm_rate" in data


def test_generate_is_reproducible(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    texts = []
    for name in ("a", "b"):
        generate(tmp_path, tmp_path / name)
        capsys.readouterr()
        lines = (tmp_path / name / "conversations.jsonl").read_text().splitlines()
        texts.append([json.loads(l)["utterances"] for l in lines])
    assert texts[0] == texts[1]


def test_generate_refuses_a_self_grading_setup(tmp_path, capsys):
    """P4: when the evaluator shares a source with the caller, refuse to start outright and send no requests."""
    init(tmp_path)
    capsys.readouterr()
    code = generate(tmp_path, tmp_path / "run", provider="openai",
                    model="gpt-4.1-mini", eval_provider="openai", eval_model="gpt-4.1-mini")
    assert code == 3
    assert "violates P4" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_allow_same_provider_records_the_warning(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    code = generate(tmp_path, tmp_path / "run", provider="mock", eval_provider="mock",
                    allow_same_provider=True)
    assert code == 0
    capsys.readouterr()
    # mocks never count as violating, so the trace here is empty; but the field must exist
    meta = json.loads((tmp_path / "run" / "metadata.jsonl").read_text().splitlines()[0])
    assert meta["generation"]["p4_warnings"] == []


def test_generate_resumes(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    out = tmp_path / "run"
    generate(tmp_path, out)
    capsys.readouterr()
    assert main([
        "generate", "--bank-dir", str(tmp_path), "--output", str(out), "--run-id", "run-t",
        "--num-samples", "6", "--seed", "7", "--min-rounds", "5", "--max-rounds", "6",
        "--resume",
    ]) == 0
    printed = capsys.readouterr().out
    assert "resuming" in printed
    assert len((out / "conversations.jsonl").read_text().splitlines()) == 6


def test_no_prefix_eval_leaves_no_labels(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    generate(tmp_path, tmp_path / "run", no_prefix_eval=True)
    capsys.readouterr()
    meta = json.loads((tmp_path / "run" / "metadata.jsonl").read_text().splitlines()[0])
    assert meta["prefix_labels"] == []
    assert meta["earliest_detection_index"] is None
    assert meta["blind_assessment"] is not None


def test_eval_stride_shows_up_in_the_metadata(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    generate(tmp_path, tmp_path / "run", eval_stride=3)
    capsys.readouterr()
    meta = json.loads((tmp_path / "run" / "metadata.jsonl").read_text().splitlines()[0])
    assert meta["generation"]["evaluation"]["stride"] == 3
    n_utt = len(meta["utterance_labels"])
    assert meta["generation"]["evaluation"]["prefixes_labelled"] < n_utt
    assert all(u["label_from_index"] <= u["index"] for u in meta["utterance_labels"])


def test_inspect_summarises_a_run(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    generate(tmp_path, tmp_path / "run")
    capsys.readouterr()
    assert main(["inspect", str(tmp_path / "run")]) == 0
    out = capsys.readouterr().out
    assert "scam_detection_rate" in out
    assert "hard_negative_false_alarms" in out


def test_inspect_one_conversation(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    generate(tmp_path, tmp_path / "run")
    capsys.readouterr()
    assert main(["inspect", str(tmp_path / "run"), "--index", "0"]) == 0
    out = capsys.readouterr().out
    assert "ground truth" in out
    assert "CALLER" in out and "RECEIVER" in out


def test_inspect_rejects_an_unknown_index(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    generate(tmp_path, tmp_path / "run")
    capsys.readouterr()
    assert main(["inspect", str(tmp_path / "run"), "--index", "99"]) == 1


def test_validate_passes_a_clean_run(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    generate(tmp_path, tmp_path / "run")
    capsys.readouterr()
    assert main(["validate", str(tmp_path / "run")]) == 0
    assert "clean" in capsys.readouterr().out


def test_validate_catches_a_label_planted_in_the_transcript(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    generate(tmp_path, tmp_path / "run")
    capsys.readouterr()
    path = tmp_path / "run" / "conversations.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["ground_truth"] = {"is_scam": True}
    path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    assert main(["validate", str(tmp_path / "run")]) == 1
    assert "leakage.label_in_transcript" in capsys.readouterr().out


def test_validate_catches_an_orphan_transcript(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    generate(tmp_path, tmp_path / "run")
    capsys.readouterr()
    meta = tmp_path / "run" / "metadata.jsonl"
    meta.write_text("\n".join(meta.read_text().splitlines()[1:]) + "\n")
    assert main(["validate", str(tmp_path / "run")]) == 1
    assert "pair.missing_metadata" in capsys.readouterr().out


def test_validate_reports_a_missing_run(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "nope")]) == 1
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Candidate scenarios: review, drop, rename
# --------------------------------------------------------------------------- #

def author_dry(tmp_path, num=3, **kw):
    args = ["author-scenarios", "--bank-dir", str(tmp_path), "--num", str(num), "--dry-run"]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return main(args)


def test_dry_run_saves_the_candidates(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert author_dry(tmp_path, 3, context="parcel fee") == 0
    draft = tmp_path / "seed.candidates.jsonl"
    assert draft.exists()
    assert "candidates saved to" in capsys.readouterr().out
    assert len(draft.read_text().splitlines()) >= 1


def test_review_candidates_writes_a_readable_file(tmp_path, capsys):
    init(tmp_path)
    author_dry(tmp_path, 3, context="parcel fee")
    capsys.readouterr()
    assert main(["review-candidates", "--bank-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "candidate scenarios" in out and "pretext" in out
    assert (tmp_path / "seed.candidates.md").exists()


def test_review_candidates_flags_a_signal_named_family(tmp_path, capsys):
    init(tmp_path)
    author_dry(tmp_path, 2, context="parcel fee")
    capsys.readouterr()
    draft = tmp_path / "seed.candidates.jsonl"
    recs = [json.loads(l) for l in draft.read_text().splitlines()]
    recs[0]["family"] = "caller_wont_say_who_they_are"
    draft.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
    main(["review-candidates", "--bank-dir", str(tmp_path)])
    assert "looks named after a signal" in capsys.readouterr().out


def test_commit_candidates_writes_a_new_version(tmp_path, capsys):
    init(tmp_path)
    author_dry(tmp_path, 3, context="parcel fee")
    capsys.readouterr()
    assert main(["commit-candidates", "--bank-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert " new = " in out
    assert (tmp_path / "seed.v2.jsonl").exists()
    assert (tmp_path / "seed.v1.jsonl").exists(), "the old version must be kept (R1.5)"
    assert not (tmp_path / "seed.candidates.jsonl").exists(), "the candidate file should be cleared after commit"


def write_candidates(tmp_path, families):
    """Write the candidate file directly: the mock author's output count depends on dedup, so it's unsuitable for pinning the test's preconditions."""
    recs = []
    for i, fam in enumerate(families):
        recs.append({
            "scenario_id": f"scn-seed-{900 + i:03d}", "category": "scam", "family": fam,
            "pretext": f"claims a distinct thing number {i} that nobody else claims here",
            "org_kind": "a fictional service company",
            "expected_signs": ["urgency_pressure", "authority_claim"],
            "plausible_outcomes": ["receiver refuses outright", "receiver stalls for time"],
            "hard_negative": False, "disambiguator": "", "late_onset": False,
            "compliant_outcomes": [], "no_observable_evidence": False, "eval_only": False,
            "notes": "", "provenance": {"author_model": "test", "bank_version": 2},
        })
    (tmp_path / "seed.candidates.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    return recs


def test_commit_can_drop_and_rename(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    write_candidates(tmp_path, ["drop_me_family", "keep_me_family"])
    drop, keep = "drop_me_family", "keep_me_family"
    code = main(["commit-candidates", "--bank-dir", str(tmp_path),
                 "--drop", drop, "--rename", f"{keep}=renamed_family"])
    assert code == 0
    out = capsys.readouterr().out
    assert f"dropped 1: {drop}" in out
    assert f"renamed {keep} → renamed_family" in out
    families = {json.loads(l)["family"] for l in (tmp_path / "seed.v2.jsonl").read_text().splitlines()}
    assert "renamed_family" in families and drop not in families


def test_commit_rejects_an_unknown_family(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    write_candidates(tmp_path, ["only_family_here"])
    assert main(["commit-candidates", "--bank-dir", str(tmp_path), "--drop", "nope"]) == 2
    assert "no such families in the candidates" in capsys.readouterr().err


def test_commit_rejects_a_malformed_rename(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    write_candidates(tmp_path, ["only_family_here"])
    assert main(["commit-candidates", "--bank-dir", str(tmp_path), "--rename", "nosign"]) == 2
    assert "OLD=NEW" in capsys.readouterr().err


def test_commit_without_candidates_is_reported(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    assert main(["commit-candidates", "--bank-dir", str(tmp_path)]) == 2
    assert "no candidate file" in capsys.readouterr().err


def author_append(tmp_path, num=2, **kw):
    args = ["author-scenarios", "--bank-dir", str(tmp_path),
            "--num", str(num), "--dry-run", "--append"]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return main(args)


def test_append_keeps_the_earlier_candidates(tmp_path, capsys):
    init(tmp_path)
    write_candidates(tmp_path, ["first_round_family", "second_round_family"])
    capsys.readouterr()
    assert author_append(tmp_path, 2, context="ordinary vet reminder") == 0
    out = capsys.readouterr().out
    assert "2 existing candidates" in out
    families = [json.loads(l)["family"]
                for l in (tmp_path / "seed.candidates.jsonl").read_text().splitlines()]
    assert families[:2] == ["first_round_family", "second_round_family"], "earlier candidates must not be overwritten"
    assert len(families) > 2


def test_append_feeds_existing_candidates_into_dedup(tmp_path, capsys):
    """The previous round's candidates aren't in the bank yet, so dedup only sees them if they're passed in explicitly."""
    from callfactory.providers.mock import MockProvider

    init(tmp_path)
    recs = write_candidates(tmp_path, ["already_pending_family"])
    recs[0]["pretext"] = MockProvider.MOCK_PRETEXTS[0]
    (tmp_path / "seed.candidates.jsonl").write_text(
        json.dumps(recs[0], ensure_ascii=False) + "\n", encoding="utf-8")
    capsys.readouterr()
    author_append(tmp_path, 1, context="parcel release charge")
    assert "pretext is a near-duplicate of an existing scenario" in capsys.readouterr().out


def test_append_requires_dry_run(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    code = main(["author-scenarios", "--bank-dir", str(tmp_path),
                 "--num", "1", "--append", "--context", "x"])
    assert code == 2
    assert "--append requires --dry-run" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Command-line wiring for family-balanced sampling / three-way split
# --------------------------------------------------------------------------- #

def test_existing_family_counts_reads_a_run_directory(tmp_path):
    from callfactory.cli import existing_family_counts

    meta = tmp_path / "metadata.jsonl"
    meta.write_text("\n".join(json.dumps(r) for r in [
        {"task": {"scenario": {"family": "a"}}},
        {"task": {"scenario": {"family": "a"}}},
        {"ground_truth": {"scam_family": "b"}},
    ]) + "\n", encoding="utf-8")
    assert existing_family_counts([str(tmp_path)]) == {"a": 2, "b": 1}


def test_existing_family_counts_rejects_a_missing_directory(tmp_path):
    from callfactory.cli import existing_family_counts

    with pytest.raises(FileNotFoundError):
        existing_family_counts([str(tmp_path / "nope")])


def test_plan_balances_families_and_tops_up_from_existing(tmp_path, capsys):
    init(tmp_path)
    capsys.readouterr()
    existing = tmp_path / "prior"
    existing.mkdir()
    (existing / "metadata.jsonl").write_text(
        json.dumps({"task": {"scenario": {"family": "prize_notification"}}}) + "\n",
        encoding="utf-8")
    code = main(["plan", "--bank-dir", str(tmp_path), "--num-samples", "48", "--seed", "3",
                 "--balance-families", "--existing", str(existing), "--json"])
    assert code == 0
    tasks = json.loads(capsys.readouterr().out)["tasks"]
    counts = Counter(t["scenario"]["family"] for t in tasks)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert counts["prize_notification"] == min(counts.values())
