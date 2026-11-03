"""Command-line entry point.

Implemented: ``init-bank`` / ``review-bank`` / ``plan`` / ``estimate`` (zero model calls),
``generate`` / ``inspect`` / ``validate``.
``relabel`` (relabel fixed dialogues under a new rubric), ``label`` (human prefix labelling), ``calibrate`` (calibration split and agreement), ``split`` (train/eval split).
``author-scenarios`` / ``review-candidates`` / ``commit-candidates`` (extend the scenario bank).
Not implemented: ``judge`` / ``review``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .author import author_scenarios, merge_into_bank
from .batch import DEFAULT_CANDIDATE_DIR, OutputPaths, estimate_calls, run_batch
from .envfile import load_env_file
from .personas import CALLER_PERSONAS, RECEIVER_PERSONAS
from .evaluator import DEFAULT_DETECTION_THRESHOLD, EvaluatorAgent
from .pipeline import GenerationSettings
from .calibration import agreement, load_split, make_split, render_agreement
from .labelling import label_session, load_human_labels
from .metrics import detection_positions
from .providers import ProviderIndependenceError, ProviderRoles, build_provider
from .relabel import relabel_run
from .splitting import SplitError, plan_split, write_split
from .scenarios import BankError, latest_bank, load_bank, sample_tasks, write_bank
from .seed_scenarios import build_seed_scenarios
from .validators import validate_record_pair

DEFAULT_BANK_DIR = "datasets/scenario_banks"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="callfactory",
        description="Synthetic phone-call factory: training / eval data for early-warning scam detection",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-bank", help="write the hand-written seed scenarios as a v1 scenario bank")
    init.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    init.add_argument("--bank-name", default="seed")
    init.add_argument("--version", type=int, default=1)
    init.add_argument("--overwrite", action="store_true", help="allow overwriting a file of the same version")

    aut = sub.add_parser(
        "author-scenarios",
        help="extend the scenario bank with an LLM (existing scenarios as dedup context; writes a new version, leaves old ones alone)",
    )
    _bank_args(aut)
    aut.add_argument("--num", type=int, default=10, help="how many to add this time")
    aut.add_argument("--context", default="", help='key behaviour keywords, e.g. "bitcoin investment scam"')
    aut.add_argument("--context-file", default=None, help="read a long context from a file")
    aut.add_argument("--use-gaps", action="store_true",
                     help="also hand review-bank's coverage gaps to the agent, to fill them on purpose")
    aut.add_argument("--provider", default="mock")
    aut.add_argument("--model", default=None)
    aut.add_argument("--env-file", default=None)
    aut.add_argument("--temperature", type=float, default=1.0)
    aut.add_argument("--max-attempts", type=int, default=3)
    aut.add_argument("--version", type=int, default=None, help="version number to write; defaults to current +1")
    aut.add_argument("--dry-run", action="store_true", help="print candidate scenarios without committing them")
    aut.add_argument("--append", action="store_true",
                     help="append this round's candidates to the existing candidate file instead of overwriting it; "
                          "existing candidates also join the dedup context (needs --dry-run)")
    aut.add_argument("--json", action="store_true")

    rc = sub.add_parser(
        "review-candidates",
        help="go through the candidate scenarios saved by --dry-run one by one, and write a readable .md (zero model calls)",
    )
    _bank_args(rc)
    rc.add_argument("--json", action="store_true")

    com = sub.add_parser(
        "commit-candidates",
        help="write the candidate scenarios saved by --dry-run into a new bank version (zero model calls)",
    )
    _bank_args(com)
    com.add_argument("--version", type=int, default=None)
    com.add_argument("--drop", nargs="*", default=None, help="drop these families when committing")
    com.add_argument("--rename", nargs="*", default=None, metavar="OLD=NEW",
                     help="rename on commit, e.g. --rename safe_account_transfer_secrecy=safe_account_transfer")

    rev = sub.add_parser("review-bank", help="review the scenario bank and report coverage gaps (zero model calls)")
    _bank_args(rev)
    rev.add_argument("--json", action="store_true")

    plan = sub.add_parser("plan", help="print the tasks that would be generated, without sending any request")
    _bank_args(plan)
    _sampling_args(plan)
    plan.add_argument("--json", action="store_true")
    plan.add_argument("--show-directives", action="store_true", help="also print per-turn directives")

    est = sub.add_parser("estimate", help="expected call count and peak in-flight requests, zero requests")
    _bank_args(est)
    _sampling_args(est)
    _eval_args(est)
    est.add_argument("--concurrency", type=int, default=4)
    est.add_argument("--json", action="store_true")

    gen = sub.add_parser("generate", help="batch-generate candidate data (transcripts and labels in two files)")
    _bank_args(gen)
    _sampling_args(gen)
    _eval_args(gen)
    gen.add_argument("--env-file", default=None,
                     help="read keys from this .env; if omitted, search upwards from the current directory for .env. "
                          "Variables already exported are not overwritten")
    gen.add_argument("--provider", default="mock", help="caller-side provider")
    gen.add_argument("--model", default=None)
    gen.add_argument("--receiver-provider", default=None, help="defaults to the caller's provider")
    gen.add_argument("--receiver-model", default=None)
    gen.add_argument("--eval-provider", default="mock", help="must come from a different source than the dialogue side (P4)")
    gen.add_argument("--eval-model", default=None)
    gen.add_argument("--allow-same-provider", action="store_true",
                     help="explicitly allow a P4 same-source setup; this is recorded in every record's metadata")
    gen.add_argument("--temperature", type=float, default=0.9)
    gen.add_argument("--output", default=None, help="defaults to datasets/candidates/<run-id>")
    gen.add_argument("--run-id", default=None, help="defaults to run-<seed>")
    gen.add_argument("--concurrency", type=int, default=4, help="conversations in flight at once, not requests")
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--keep-invalid", action="store_true",
                     help="also write hard-rule failures into the candidates; for debugging only")
    gen.add_argument("--quiet", action="store_true")
    gen.add_argument("--json", action="store_true", help="print only the stats JSON")

    rel = sub.add_parser(
        "relabel",
        help="relabel data on disk under the current rubric: dialogue untouched, only the evaluator reruns",
    )
    rel.add_argument("run", help="source run directory")
    rel.add_argument("--out", default=None, help="defaults to <run>-<rubric version>")
    rel.add_argument("--env-file", default=None)
    _eval_args(rel)
    rel.add_argument("--eval-provider", default="mock")
    rel.add_argument("--eval-model", default=None)
    rel.add_argument("--concurrency", type=int, default=4)
    rel.add_argument("--keep-invalid", action="store_true")
    rel.add_argument("--quiet", action="store_true")
    rel.add_argument("--json", action="store_true")

    red = sub.add_parser(
        "rederive",
        help="recompute only the detection point, without relabelling (zero model calls): use it to change the threshold or derivation rule",
    )
    red.add_argument("run", help="source run directory")
    red.add_argument("--out", required=True, help="output directory (must differ from the source)")
    red.add_argument("--threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD,
                     help=f"score line (default {DEFAULT_DETECTION_THRESHOLD})")
    red.add_argument("--require-sufficient", action="store_true",
                     help="go back to detect/v1: also require sufficient_evidence to be true")
    red.add_argument("--no-persistence", action="store_true",
                     help="turn off the persistence rule (not advised: a momentary spike becomes a permanent detection point)")
    red.add_argument("--json", action="store_true")

    lab = sub.add_parser(
        "label",
        help="human prefix labelling (calibration-set collection): reveals round by round, shows no machine labels",
    )
    lab.add_argument("run", help="run directory")
    lab.add_argument("--reveal", choices=("round", "utterance"), default="round")
    lab.add_argument("--only", nargs="*", default=None, help="label only these conversation_ids")
    lab.add_argument("--labeller", default="human")
    lab.add_argument("--status", action="store_true", help="show progress only, don't start labelling")

    cal = sub.add_parser(
        "calibrate",
        help="calibration set: split tune/holdout, or compute agreement between human and machine labels (R6.5)",
    )
    cal.add_argument("run", help="run directory")
    cal.add_argument("--make-split", type=int, default=None, metavar="N",
                     help="draw N items as the holdout set. Must be done before analysing the batch")
    cal.add_argument("--split-seed", type=int, default=20261031)
    cal.add_argument("--overwrite-split", action="store_true")
    cal.add_argument("--include-holdout", action="store_true",
                     help="include the holdout set in agreement; only for the one final validation")
    cal.add_argument("--json", action="store_true")

    spl = sub.add_parser(
        "split",
        help="split candidate data into train / eval (refuses mixed rubrics; near-duplicates stay on one side)",
    )
    spl.add_argument("runs", nargs="+", help="one or more run directories, merged before splitting")
    spl.add_argument("--name", required=True, help="split name, written to datasets/{train,eval}/<name>/")
    spl.add_argument("--out-root", default="datasets")
    spl.add_argument("--ratio", type=float, default=0.8, help="train share")
    spl.add_argument("--seed", type=int, default=0)
    spl.add_argument("--by", choices=("stratified", "family", "family-holdout"),
                     default="stratified",
                     help="stratified=random, stratified by class; family=two-way family-disjoint; "
                          "family-holdout=three-way (whole families held out as test, "
                          "a share of each remaining family as eval)")
    spl.add_argument("--holdout-family-ratio", type=float, default=0.1,
                     help="family-holdout: share of families held out as test (stratified by class)")
    spl.add_argument("--eval-ratio", type=float, default=0.1,
                     help="family-holdout: share of each remaining family sampled into eval")
    spl.add_argument("--threshold", type=float, default=0.5, help="near-duplicate threshold")
    spl.add_argument("--allow-mixed-rubric", action="store_true",
                     help="force mixing data from different rubrics; produces contradictory labels")
    spl.add_argument("--keep-calibration-holdout-in-eval", action="store_true", default=True,
                     help="calibration holdout goes to eval only (on by default)")
    spl.add_argument("--overwrite", action="store_true")
    spl.add_argument("--dry-run", action="store_true", help="print the plan only, write no files")
    spl.add_argument("--json", action="store_true")

    bch = sub.add_parser(
        "bench",
        help="run a detector over split data (asking 'is the evidence enough?' per prefix), resumable, with a budget cap",
    )
    bch.add_argument("split", help="split directory, e.g. datasets/eval/ds3-rubric-v3-detect07")
    bch.add_argument("--system", required=True,
                     help="system under test, see bench.adapters.JUDGES (mock / claude-haiku / "
                          "gpt-sol / gemini-argon / deepseek-flash / jev …)")
    bch.add_argument("--out", default=None, help="defaults to datasets/bench/<split name>/<system>.jsonl")
    bch.add_argument("--cond", choices=("c1", "c2"), default="c1",
                     help="c1=default reasoning, c2=minimal reasoning (on a par with Jev's no reasoning)")
    bch.add_argument("--threshold", type=float, default=0.5,
                     help="alarm threshold. Fixed, never tuned: tuning the threshold on eval is a back door for the baseline")
    bch.add_argument("--no-cache", action="store_true", help="turn off prefix caching (only to compare cost)")
    bch.add_argument("--concurrency", type=int, default=4, help="**conversations** in flight at once")
    bch.add_argument("--limit", type=int, default=None, help="run only the first N calls, for a smoke test")
    bch.add_argument("--max-requests", type=int, default=None, help="request cap; stop when reached")
    bch.add_argument("--env-file", default=None)
    bch.add_argument("--json", action="store_true")

    tbl = sub.add_parser("bench-table", help="score bench results and print the table (zero calls)")
    tbl.add_argument("split", help="split directory")
    tbl.add_argument("--runs", default=None, help="bench results directory, defaults to datasets/bench/<split name>")
    tbl.add_argument("--threshold", type=float, default=0.5)
    tbl.add_argument("--thresholds", default=None,
                     help="per-system thresholds (JSON from bench-tune); systems not in it use --threshold")
    tbl.add_argument("--json", action="store_true")

    tun = sub.add_parser(
        "bench-tune",
        help="pick one threshold per system from bench results on a train subset (zero calls)",
    )
    tun.add_argument("split", help="split directory on the **train side**. An eval split is refused")
    tun.add_argument("--runs", default=None, help="bench results directory, defaults to datasets/bench/<split name>")
    tun.add_argument("--out", default=None,
                     help="where to write thresholds, defaults to <runs>/_thresholds.json")
    tun.add_argument("--json", action="store_true")

    ins = sub.add_parser("inspect", help="read a run directory: stats, or go through dialogues and labels one by one")
    ins.add_argument("run", help="run directory or conversations.jsonl path")
    ins.add_argument("--index", type=int, default=None, help="which one to view (sample_index)")
    ins.add_argument("--conversation-id", default=None)
    ins.add_argument("--json", action="store_true")

    val = sub.add_parser("validate", help="recheck data on disk: leakage, pairing, label structure")
    val.add_argument("run", help="run directory or conversations.jsonl path")
    val.add_argument("--json", action="store_true")
    val.add_argument("--max-show", type=int, default=20)

    return parser


def _sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--min-rounds", type=int, default=5)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--scam-ratio", type=float, default=0.5)
    p.add_argument("--hard-negative-ratio", type=float, default=0.4)
    p.add_argument("--late-onset-ratio", type=float, default=0.3)
    p.add_argument("--compliance-ratio", type=float, default=0.25,
                   help="share of scams where the receiver is won over (breaks the 'receiver resists = scam' shortcut)")
    p.add_argument("--include-family", nargs="*", default=None)
    p.add_argument("--exclude-family", nargs="*", default=None)
    p.add_argument("--balance-families", action="store_true",
                   help="make each family's final count as equal as possible, instead of stratifying by --scam-ratio and similar. "
                        "Class ratios are then set by the bank's make-up")
    p.add_argument("--existing", nargs="*", default=None, metavar="DIR",
                   help="existing data directories (run or split directories); each family's existing count counts towards "
                        "the --balance-families target, and only the gap is filled")


def _eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval-stride", type=int, default=1,
                   help=">1 downsamples prefix labelling (T0 and the last turn are always labelled)")
    p.add_argument("--no-prefix-eval", action="store_true")
    p.add_argument("--no-global-eval", action="store_true")
    p.add_argument("--threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD,
                   help=f"score line for earliest_detection_index (default {DEFAULT_DETECTION_THRESHOLD})")
    p.add_argument("--require-sufficient", action="store_true",
                   help="go back to detect/v1: the detection point also requires sufficient_evidence to be true. "
                        "Only for reproducing old labels")
    p.add_argument("--eval-concurrency", type=int, default=None,
                   help="labelling requests in flight at once within one conversation")


def _bank_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bank", default=None, help="give a <name>.v<N>.jsonl path directly")
    p.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    p.add_argument("--bank-name", default=None, help="required when the directory holds more than one bank name")


def _open_bank(args: argparse.Namespace):
    return load_bank(args.bank) if args.bank else latest_bank(args.bank_dir, args.bank_name)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init-bank":
            return _cmd_init_bank(args)
        if args.command == "author-scenarios":
            return _cmd_author(args)
        if args.command == "review-candidates":
            return _cmd_review_candidates(args)
        if args.command == "commit-candidates":
            return _cmd_commit_candidates(args)
        if args.command == "review-bank":
            return _cmd_review_bank(args)
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "estimate":
            return _cmd_estimate(args)
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "rederive":
            return _cmd_rederive(args)
        if args.command == "relabel":
            return _cmd_relabel(args)
        if args.command == "label":
            return _cmd_label(args)
        if args.command == "calibrate":
            return _cmd_calibrate(args)
        if args.command == "split":
            return _cmd_split(args)
        if args.command == "bench":
            return _cmd_bench(args)
        if args.command == "bench-table":
            return _cmd_bench_table(args)
        if args.command == "bench-tune":
            return _cmd_bench_tune(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
        if args.command == "validate":
            return _cmd_validate(args)
    except ProviderIndependenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except BankError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _cmd_init_bank(args: argparse.Namespace) -> int:
    path = write_bank(
        build_seed_scenarios(args.bank_name),
        args.bank_dir,
        args.bank_name,
        args.version,
        author_model="human",
        context="hand-written seed bank",
        overwrite=args.overwrite,
    )
    bank = load_bank(path)
    print(f"wrote {path} — {len(bank)} scenarios")
    print(
        f"  scam {len(bank.scams)} / hard negative {len(bank.hard_negatives)} "
        f"/ plain negative {len(bank.plain_negatives)}"
    )
    print("next: callfactory review-bank  to see coverage gaps")
    return 0


def _cmd_review_bank(args: argparse.Namespace) -> int:
    bank = _open_bank(args)
    gaps = bank.coverage_gaps(CALLER_PERSONAS)
    if args.json:
        print(json.dumps(
            {
                "bank": bank.name, "version": bank.version, "size": len(bank),
                "scenarios": [s.to_dict() for s in bank.scenarios],
                "coverage_gaps": gaps,
            },
            indent=2, ensure_ascii=False,
        ))
        return 0

    print(f"scenario bank {bank.name} v{bank.version} — {len(bank)} scenarios\n")
    groups = [
        ("SCAM", bank.scams),
        ("LEGITIMATE — hard negative", bank.hard_negatives),
        ("LEGITIMATE — plain", bank.plain_negatives),
    ]
    for title, items in groups:
        print(f"{title}  ({len(items)})")
        for s in items:
            tags = []
            if s.late_onset:
                tags.append("late-onset-capable")
            if s.provenance and s.provenance.author_model != "human":
                tags.append(f"by {s.provenance.author_model}")
            print(f"  {s.family:38s} {','.join(s.expected_signs) or '—'}")
            if tags:
                print(f"  {'':38s} [{' | '.join(tags)}]")
        print()

    print("coverage gaps")
    never = gaps["signals_never_used_by_scam"]
    print(f"  signals no scam scenario uses ({len(never)}): {', '.join(never) or 'none'}")
    unpaired_sig = gaps["signals_without_hard_negative"]
    print(f"  signals with no paired hard negative ({len(unpaired_sig)}): {', '.join(unpaired_sig) or 'none'}")
    if unpaired_sig:
        print("      ↑ the model will learn these signals by themselves as grounds for an alarm, since there are no counter-examples")
    unpaired_fam = gaps["scam_families_without_paired_hard_negative"]
    print(f"  scam families with no paired hard negative at all ({len(unpaired_fam)}): "
          f"{', '.join(unpaired_fam) or 'none'}")
    print(f"  scam families suited to slow burn ({len(gaps['late_onset_capable'])}): "
          f"{', '.join(gaps['late_onset_capable']) or 'none'}")
    print("\noutcome make-up (stops the model scoring by reading the receiver)")
    with_c = gaps["scam_families_with_compliant_outcomes"]
    without_c = gaps["scam_families_without_compliant_outcome"]
    print(f"  scam families with compliant outcomes ({len(with_c)}): {', '.join(with_c) or 'none'}")
    print(f"  scam families that only produce resisting outcomes ({len(without_c)}): {', '.join(without_c) or 'none'}")
    if without_c:
        print("      ↑ in these families' samples the receiver always resists, so the model can read the receiver instead of the caller")
    noev = gaps["no_observable_evidence_families"]
    print(f"  scam by intent but no decidable evidence ({len(noev)}): {', '.join(noev) or 'none'}")
    evonly = gaps["eval_only_families"]
    print(f"  eval only by default, not training ({len(evonly)}): {', '.join(evonly) or 'none'}")
    print("\nstyle-level signals (unrelated to the call's purpose, produced by caller personas, not declared by scenarios)")
    print(f"  covered by personas: {', '.join(gaps['style_signals_covered_by_personas']) or 'none'}")
    missing = gaps.get("style_signals_with_no_persona") or []
    print(f"  no persona can produce: {', '.join(missing) or 'none'}")
    print("\nto add scenarios: callfactory author-scenarios --context \"<key behaviour>\" (not implemented)")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    bank, tasks, warnings = _sample(args)
    if args.json:
        print(json.dumps(
            {
                "bank": bank.name, "bank_version": bank.version,
                "warnings": [w.to_dict() for w in warnings],
                "tasks": [t.to_dict() for t in tasks],
            },
            indent=2, ensure_ascii=False,
        ))
        return 0

    print(f"scenario bank {bank.name} v{bank.version} | seed {args.seed} | {len(tasks)} tasks\n")
    for t in tasks:
        tags = []
        if t.scenario.hard_negative:
            tags.append("hard-neg")
        if t.late_onset:
            tags.append("late-onset")
        print(
            f"{t.index:4d}  {t.scenario.category:11s} {t.scenario.family:34s} "
            f"{t.caller_persona.persona_id}/{t.receiver_persona.persona_id} "
            f"skep={t.receiver_persona.skepticism:9s} rounds={t.target_rounds:2d} "
            f"{' '.join(tags)}"
        )
        if args.show_directives:
            shown = [d for d in t.turn_directives[:4]]
            print(f"      outcome: {t.outcome}")
            for i, d in enumerate(shown):
                print(f"      turn {i}: {d or '(none)'}")

    total = len(tasks) or 1
    scams = [t for t in tasks if t.scenario.category == "scam"]
    legits = [t for t in tasks if t.scenario.category == "legitimate"]
    print(
        f"\nratios: scam {len(scams)}/{total} = {len(scams)/total:.2f}"
        + (f" | hard-neg {sum(1 for t in legits if t.scenario.hard_negative)}/{len(legits)}"
           f" = {sum(1 for t in legits if t.scenario.hard_negative)/len(legits):.2f}" if legits else "")
        + (f" | late-onset {sum(1 for t in scams if t.late_onset)}/{len(scams)}"
           f" = {sum(1 for t in scams if t.late_onset)/len(scams):.2f}" if scams else "")
        + (f" | compliant {sum(1 for t in scams if t.outcome in t.scenario.compliant_outcomes)}/{len(scams)}"
           f" = {sum(1 for t in scams if t.outcome in t.scenario.compliant_outcomes)/len(scams):.2f}"
           if scams else "")
    )
    counts = Counter(t.scenario.family for t in tasks)
    print(f"family coverage: {len(counts)}/{len(bank)}, each used {min(counts.values())}-{max(counts.values())} times")
    print(f"unique persona triples: {len({t.fingerprint for t in tasks})}/{len(tasks)}")
    print(f"persona banks: {len(CALLER_PERSONAS)} caller × {len(RECEIVER_PERSONAS)} receiver")
    for w in warnings:
        print(f"  ! {w.code}: {w.message}")
    return 0


# --------------------------------------------------------------------------- #
# generate / estimate
# --------------------------------------------------------------------------- #

def existing_family_counts(dirs: Sequence[str]) -> dict[str, int]:
    """Count how many items each family already has in the given directories. A directory can be a run directory or a split directory."""
    counts: Counter[str] = Counter()
    for raw in dirs:
        path = Path(raw)
        meta = path if path.suffix == ".jsonl" else path / "metadata.jsonl"
        if not meta.is_file():
            raise FileNotFoundError(f"not found: {meta}")
        for rec in _read_jsonl(meta):
            family = (((rec.get("task") or {}).get("scenario") or {}).get("family")
                      or (rec.get("ground_truth") or {}).get("scam_family"))
            if family:
                counts[family] += 1
    return dict(counts)


def _sample(args: argparse.Namespace):
    bank = _open_bank(args)
    existing = existing_family_counts(args.existing or [])
    tasks, warnings = sample_tasks(
        bank,
        args.num_samples,
        args.seed,
        min_rounds=args.min_rounds,
        max_rounds=args.max_rounds,
        scam_ratio=args.scam_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        late_onset_ratio=args.late_onset_ratio,
        compliance_ratio=args.compliance_ratio,
        include_family=args.include_family,
        exclude_family=args.exclude_family,
        balance_families=args.balance_families,
        existing_counts=existing,
    )
    return bank, tasks, warnings


def _settings(args: argparse.Namespace) -> GenerationSettings:
    return GenerationSettings(
        min_rounds=args.min_rounds,
        max_rounds=args.max_rounds,
        temperature=getattr(args, "temperature", 0.9),
        eval_stride=args.eval_stride,
        prefix_eval=not args.no_prefix_eval,
        global_eval=not args.no_global_eval,
        detection_threshold=args.threshold,
        detection_require_sufficient=getattr(args, "require_sufficient", False),
        eval_concurrency=args.eval_concurrency,
    )


def _cmd_estimate(args: argparse.Namespace) -> int:
    bank, tasks, warnings = _sample(args)
    est = estimate_calls(tasks, _settings(args), concurrency=args.concurrency)
    if args.json:
        print(json.dumps(est, indent=2, ensure_ascii=False))
        return 0
    print(f"scenario bank {bank.name} v{bank.version} | seed {args.seed} | {est['conversations']} conversations\n")
    print(f"  dialogue calls        {est['dialogue_calls']:>7d}   (2 × rounds)")
    print(f"  prefix-label calls    {est['prefix_eval_calls']:>7d}   (stride={args.eval_stride})")
    print(f"  full-quality calls    {est['global_eval_calls']:>7d}")
    print(f"  total                 {est['total_calls']:>7d}")
    print(f"\nconcurrency {est['concurrency']} conversations -> peak in-flight requests about {est['peak_in_flight_requests']}")
    print("  ↑ size rate limits by this number, not by --concurrency (labelling requests inside one conversation are sent concurrently)")
    for w in warnings:
        print(f"  ! {w.code}: {w.message}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    needs_keys = "mock" not in {args.provider, args.eval_provider, args.receiver_provider}
    try:
        env_path, loaded = load_env_file(args.env_file)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if loaded and not args.json:
        # Print variable names only, never values.
        print(f"loaded {env_path}: {', '.join(loaded)}")
    elif needs_keys and env_path is None and not args.json:
        print("no .env found; assuming keys are already exported (--env-file to specify one)", file=sys.stderr)

    bank, tasks, warnings = _sample(args)
    for w in warnings:
        print(f"  ! {w.code}: {w.message}", file=sys.stderr)

    run_id = args.run_id or f"run-{args.seed}"
    output = args.output or str(Path(DEFAULT_CANDIDATE_DIR) / run_id)
    settings = _settings(args)

    caller = build_provider(args.provider, model=args.model, temperature=args.temperature)
    receiver = (
        build_provider(args.receiver_provider, model=args.receiver_model,
                       temperature=args.temperature)
        if args.receiver_provider else caller
    )
    evaluator = build_provider(args.eval_provider, model=args.eval_model, temperature=0.0)
    roles = ProviderRoles(caller=caller, receiver=receiver, evaluator=evaluator)
    p4_warnings = roles.enforce(allow_same_provider=args.allow_same_provider)
    for issue in p4_warnings:
        print(f"  ! P4 explicitly allowed: {issue}", file=sys.stderr)

    est = estimate_calls(tasks, settings, concurrency=args.concurrency)
    if not (args.quiet or args.json):      # --json prints only the stats; don't pollute it
        print(f"run {run_id} → {output}")
        print(f"  {len(tasks)} conversations, about {est['total_calls']} calls expected, "
              f"peak in flight about {est['peak_in_flight_requests']}")
        print(f"  providers: {roles.describe()}\n")

    async def go():
        try:
            return await run_batch(
                tasks, roles,
                output=output,
                run_id=run_id,
                settings=settings,
                concurrency=args.concurrency,
                resume=args.resume,
                keep_invalid=args.keep_invalid,
                p4_warnings=p4_warnings,
                config={
                    "bank": bank.name, "bank_version": bank.version,
                    "seed": args.seed, "num_samples": args.num_samples,
                    "providers": roles.describe(), "settings": settings.to_dict(),
                    "sampling": {
                        "mode": "balance_families" if args.balance_families else "category_ratio",
                        "existing_dirs": list(args.existing or []),
                        "existing_counts": existing_family_counts(args.existing or []),
                        "quotas": dict(Counter(t.scenario.family for t in tasks)),
                    },
                },
                progress=None if args.quiet or args.json else print,
            )
        finally:
            for provider in roles.all_providers():
                await provider.aclose()

    stats = asyncio.run(go())
    summary = stats.summary()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if stats.written else 1

    paths = OutputPaths.resolve(output)
    print(f"\nwrote {stats.written}, failed {stats.failed}, failed hard rules {stats.invalid}, "
          f"took {stats.elapsed_s}s")
    print(f"  {paths.conversations}   transcripts (no labels)")
    print(f"  {paths.metadata}        labels and provenance")
    if stats.failed or stats.invalid:
        print(f"  {paths.failures}        failure reasons")
    print(f"\nmake-up: scam {stats.scam} / legitimate {stats.legitimate}"
          f" (of which hard negative {stats.hard_negative}) / late-onset {stats.late_onset}")
    print(f"mean rounds: {summary['mean_rounds']}")
    print(f"scam detection rate: {summary['scam_detection_rate']} "
          f"(mean detection round {summary['mean_detection_round_on_detected_scam']})")
    if stats.scam_no_evidence:
        print(f"  of which premise declares no decidable evidence: {stats.scam_no_evidence}, "
              f"{stats.no_evidence_detected} detected; this slice is an open question, "
              f"not a score (EVAL_SYSTEM §5.7)")
    print(f"legitimate false-alarm rate: {summary['legit_false_alarm_rate']}"
          f" | hard negative false-alarm rate: {summary['hard_negative_false_alarm_rate']}")
    print(f"evaluator independent-judgement disagreement with ground truth: {summary['blind_disagreement_rate']}")
    print(_position_line(paths))
    if stats.eval_failures:
        print(f"labelling calls failed {stats.eval_failures} times (degraded to empty labels)")
    if any(u.get("requests") for u in stats.usage.values()):
        t = stats.totals()
        print(f"\nactual usage: {t['requests']} requests, "
              f"input {t['input_tokens']:,} / output {t['output_tokens']:,} token")
        for key, u in sorted(stats.usage.items()):
            if u.get("requests"):
                print(f"  {key:40s} {u['requests']:>4d} req  "
                      f"in {u['input_tokens']:>8,}  out {u['output_tokens']:>7,}")
    print("\nnext: callfactory validate <run> / callfactory judge <run> (not implemented)")
    return 0 if stats.written else 1


# --------------------------------------------------------------------------- #
# inspect / validate
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise BankError(f"not found: {path}")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _cmd_inspect(args: argparse.Namespace) -> int:
    paths = OutputPaths.resolve(args.run)
    convs = _read_jsonl(paths.conversations)
    metas = {m["conversation_id"]: m for m in _read_jsonl(paths.metadata)}

    wanted = None
    if args.conversation_id:
        wanted = args.conversation_id
    elif args.index is not None:
        for m in metas.values():
            if (m.get("generation") or {}).get("sample_index") == args.index:
                wanted = m["conversation_id"]
                break
        if wanted is None:
            print(f"error: no data with sample_index={args.index}", file=sys.stderr)
            return 1

    if wanted is None:
        if args.json:
            print(json.dumps(_run_overview(convs, metas), indent=2, ensure_ascii=False))
            return 0
        ov = _run_overview(convs, metas)
        print(f"{paths.root} — {ov['conversations']} conversations\n")
        for key, value in ov.items():
            if key != "conversations":
                print(f"  {key:34s} {value}")
        print("\nto view one: callfactory inspect <run> --index 0")
        return 0

    conv = next((c for c in convs if c["conversation_id"] == wanted), None)
    meta = metas.get(wanted)
    if conv is None or meta is None:
        print(f"error: {wanted} is incomplete (transcript or metadata missing)", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({"transcript": conv, "metadata": meta}, indent=2, ensure_ascii=False))
        return 0
    _print_conversation(conv, meta)
    return 0


def _run_overview(convs: list[dict[str, Any]], metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gts = [m.get("ground_truth") or {} for m in metas.values()]
    detected = [m for m in metas.values() if m.get("earliest_detection_index") is not None]
    scams = [m for m in metas.values() if (m.get("ground_truth") or {}).get("is_scam")]
    hard = [m for m in metas.values() if (m.get("ground_truth") or {}).get("hard_negative")]
    # The slice whose premise declares "no decidable evidence" **stays** in the denominator and is reported separately (see the batch.BatchStats comment)
    no_evidence = [m for m in scams if (m.get("ground_truth") or {}).get("no_observable_evidence")]
    scam_detected = [m for m in scams if m.get("earliest_detection_index") is not None]
    ne_detected = [m for m in no_evidence if m.get("earliest_detection_index") is not None]
    rounds = [len(c.get("utterances") or []) / 2 for c in convs]
    return {
        "conversations": len(convs),
        "metadata_records": len(metas),
        "scam": len(scams),
        "scam_no_evidence": len(no_evidence),
        "no_evidence_detected": len(ne_detected),
        "hard_negative": len(hard),
        "late_onset": sum(1 for g in gts if g.get("late_onset")),
        "mean_rounds": round(sum(rounds) / len(rounds), 2) if rounds else 0.0,
        "scam_detection_rate": round(len(scam_detected) / len(scams), 3) if scams else 0.0,
        "mean_detection_round_on_scam": (
            round(sum(m["earliest_detection_index"] // 2 + 1 for m in scam_detected)
                  / len(scam_detected), 2) if scam_detected else None
        ),
        "hard_negative_false_alarms": sum(
            1 for m in hard if m.get("earliest_detection_index") is not None
        ),
        "families": len({g.get("scam_family") for g in gts}),
        "with_validation_warnings": sum(1 for m in metas.values() if m.get("validation")),
    }


def _print_conversation(conv: dict[str, Any], meta: dict[str, Any]) -> None:
    gt = meta.get("ground_truth") or {}
    task = meta.get("task") or {}
    scn = task.get("scenario") or {}
    gen = meta.get("generation") or {}
    print(f"{conv['conversation_id']}  (sample_index {gen.get('sample_index')})")
    print(f"  ground truth  {'scam' if gt.get('is_scam') else 'legitimate'}"
          f"{', hard negative' if gt.get('hard_negative') else ''}"
          f"{', late onset' if gt.get('late_onset') else ''}  — {gt.get('scam_family')}")
    print(f"  expected      {', '.join(gt.get('expected_signs') or []) or '(none)'}")
    print(f"  personas      {(task.get('caller_persona') or {}).get('persona_id')}"
          f" / {(task.get('receiver_persona') or {}).get('persona_id')}")
    print(f"  outcome       {task.get('outcome')}")
    print(f"  detection     index={meta.get('earliest_detection_index')} "
          f"round={meta.get('earliest_detection_round')}")
    print(f"  providers     {gen.get('agents', {}).get('caller', {}).get('model')} / "
          f"{gen.get('agents', {}).get('evaluator', {}).get('model')}")
    print()
    labels = {u["index"]: u for u in (meta.get("utterance_labels") or [])}
    for utt in conv["utterances"]:
        label = labels.get(utt["index"])
        if label:
            mark = " <== DETECTION POINT" if label.get("is_detection_point") else ""
            new = f" +{','.join(label.get('new_signals') or [])}" if label.get("new_signals") else ""
            rep = (f" ~dup of {label['repetition_of']}"
                   if label.get("repetition_of") is not None else "")
            print(f"{label['risk_score']:>5.2f} {label['risk_tier']:<6}{new}{rep}{mark}")
        print(f"      [{utt['index']:>2}] {utt['speaker'].upper():<8} {utt['text']}")
    for issue in meta.get("validation") or []:
        print(f"  {issue['severity']}: {issue['code']} {issue['message']}")


def _cmd_validate(args: argparse.Namespace) -> int:
    paths = OutputPaths.resolve(args.run)
    convs = _read_jsonl(paths.conversations)
    metas = {m["conversation_id"]: m for m in _read_jsonl(paths.metadata)}

    findings: list[dict[str, Any]] = []
    orphans = [c["conversation_id"] for c in convs if c["conversation_id"] not in metas]
    extra = [cid for cid in metas if cid not in {c["conversation_id"] for c in convs}]
    for cid in orphans:
        findings.append({"conversation_id": cid, "severity": "error",
                         "code": "pair.missing_metadata", "message": "transcript has no matching metadata"})
    for cid in extra:
        findings.append({"conversation_id": cid, "severity": "error",
                         "code": "pair.missing_transcript", "message": "metadata has no matching transcript"})
    for conv in convs:
        meta = metas.get(conv["conversation_id"])
        if meta is None:
            continue
        for issue in validate_record_pair(conv, meta).to_list():
            findings.append({"conversation_id": conv["conversation_id"], **issue})

    errors = [f for f in findings if f["severity"] == "error"]
    if args.json:
        print(json.dumps({"checked": len(convs), "findings": findings},
                         indent=2, ensure_ascii=False))
        return 1 if errors else 0

    print(f"{paths.root} — rechecked {len(convs)}")
    print(f"  ERROR {len(errors)} | WARNING {len(findings) - len(errors)}")
    for f in findings[:args.max_show]:
        print(f"  {f['severity']:7s} {f['code']:32s} {f['conversation_id']}  {f['message']}")
    if len(findings) > args.max_show:
        print(f"  ... {len(findings) - args.max_show} more (adjust with --max-show)")
    if not findings:
        print("  clean: no leakage, pairing complete, label structure valid")
    return 1 if errors else 0


def _position_line(paths: OutputPaths) -> str:
    """Detection-position guardrail: when it collapses towards the start of the call, discriminative value is already being lost (R7.5)."""
    try:
        metas = _read_jsonl(paths.metadata)
    except BankError:
        return ""
    report = detection_positions([
        (m.get("earliest_detection_index"),
         len((m.get("utterance_labels") or [])) or _utt_count(m),
         bool((m.get("ground_truth") or {}).get("is_scam")))
        for m in metas
    ])
    return report.render()


def _utt_count(meta: dict[str, Any]) -> int:
    labels = meta.get("utterance_labels") or []
    return max((l.get("index", 0) for l in labels), default=-1) + 1


def _cmd_relabel(args: argparse.Namespace) -> int:
    try:
        env_path, loaded = load_env_file(args.env_file)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if loaded and not args.json:
        print(f"loaded {env_path}: {', '.join(loaded)}")

    provider = build_provider(args.eval_provider, model=args.eval_model, temperature=0.0)
    rubric = EvaluatorAgent(provider).provenance()["rubric_version"]
    out = args.out or f"{str(args.run).rstrip('/')}-{rubric.replace('/', '-')}"

    async def go():
        try:
            return await relabel_run(
                args.run, out, provider,
                stride=args.eval_stride,
                threshold=args.threshold,
                require_persistence=True,
                require_sufficient=args.require_sufficient,
                global_eval=not args.no_global_eval,
                concurrency=args.concurrency,
                keep_invalid=args.keep_invalid,
                progress=None if args.quiet or args.json else print,
            )
        finally:
            await provider.aclose()

    try:
        stats = asyncio.run(go())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = stats.summary()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if stats.relabelled else 1

    print(f"\nrelabelled {stats.relabelled}/{stats.conversations} -> {out}")
    print(f"rubric: {', '.join(summary['from_rubric'])} -> {stats.to_rubric}")
    shift = stats.shift
    print(f"label shift: detection points gained {shift['gained']} / lost {shift['lost']} / "
          f"earlier {shift['earlier']} / later {shift['later']} / unchanged {shift['unchanged']}")
    for label, rep in (("old", stats.position_before), ("new", stats.position_after)):
        if rep:
            print(f"  {label}: {rep.render()}")
    if stats.failed or stats.invalid:
        print(f"failed {stats.failed}, failed hard rules {stats.invalid}")
    for key, u in stats.usage.items():
        if u.get("requests"):
            print(f"usage: {key} {u['requests']} requests, in {u['input_tokens']:,} / out {u['output_tokens']:,}")
    print("\nDialogue unchanged; the old labels stay in the source directory, so the two can be compared directly.")
    return 0 if stats.relabelled else 1


def _cmd_label(args: argparse.Namespace) -> int:
    paths = OutputPaths.resolve(args.run)
    convs = _read_jsonl(paths.conversations)
    done = load_human_labels(paths.root)

    if args.status:
        print(f"{paths.root} — {len(done)}/{len(convs)} human-labelled")
        if done:
            scams = sum(1 for l in done.values() if l.is_scam)
            flagged = sum(1 for l in done.values() if l.detection_index is not None)
            print(f"  judged scam {scams} | alarm point marked {flagged} | "
                  f"unsure {sum(1 for l in done.values() if l.is_scam is None)}")
        return 0

    # Deliberately doesn't read metadata: this interface shouldn't even have the chance to open machine labels.
    stats = label_session(
        paths.root, convs,
        read_line=lambda prompt: input(prompt),
        write=lambda text: print(text, end=""),
        reveal=args.reveal,
        only=args.only,
        labeller=args.labeller,
    )
    print(f"\nlabelled {stats.labelled} this session, skipped {stats.skipped}, "
          f"{stats.already_done} labelled before, {stats.total} in total")
    print(f"wrote {paths.root / 'human_labels.jsonl'}")
    if stats.quit_early:
        print("Run the same command again to continue where you left off.")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    paths = OutputPaths.resolve(args.run)
    convs = _read_jsonl(paths.conversations)
    ids = [c["conversation_id"] for c in convs]

    if args.make_split is not None:
        try:
            split = make_split(
                paths.root, ids, holdout=args.make_split, seed=args.split_seed,
                note="rubric tuned on tune only; holdout checked once at the end",
                overwrite=args.overwrite_split,
            )
        except (FileExistsError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"split written to {paths.root / '_calibration_split.json'}")
        print(f"  tune {len(split.tune)} | holdout {len(split.holdout)} "
              f"(seed {split.seed})")
        print("  tune the rubric on tune only; keep holdout for one check at the end")
        return 0

    split = load_split(paths.root)
    human = load_human_labels(paths.root)
    if not human:
        print(f"{paths.root} has no human labels yet. Run first: callfactory label {args.run}")
        return 1
    machine = {m["conversation_id"]: m for m in _read_jsonl(paths.metadata)}

    scopes: list[tuple[str, Sequence[str] | None]] = []
    if split is None:
        scopes.append(("all (not split)", None))
    else:
        scopes.append(("tune", split.tune))
        if args.include_holdout:
            scopes.append(("holdout", split.holdout))

    reports = {}
    for label, only in scopes:
        reports[label] = agreement(machine, human, only=only)

    if args.json:
        print(json.dumps(
            {k: v.summary() for k, v in reports.items()}, indent=2, ensure_ascii=False
        ))
        return 0

    print(f"{paths.root} — human-labelled {len(human)}/{len(convs)}\n")
    for label, report in reports.items():
        print(render_agreement(report, label=label))
        print()
    if split is not None and not args.include_holdout:
        pending = sum(1 for k in human if k in set(split.holdout))
        print(f"holdout has {pending} labelled, **not** counted this time; "
              "add --include-holdout for the final validation")
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    convs: dict[str, Any] = {}
    metas: dict[str, Any] = {}
    forced: set[str] = set()
    for run in args.runs:
        paths = OutputPaths.resolve(run)
        for rec in _read_jsonl(paths.conversations):
            convs[rec["conversation_id"]] = rec
        for rec in _read_jsonl(paths.metadata):
            metas[rec["conversation_id"]] = rec
        split = load_split(paths.root)
        if split and args.keep_calibration_holdout_in_eval:
            forced |= set(split.holdout)

    try:
        plan = plan_split(
            convs, metas,
            ratio=args.ratio, seed=args.seed, by=args.by.replace("-", "_"),
            forced_eval=forced, threshold=args.threshold,
            allow_mixed_rubric=args.allow_mixed_rubric,
            holdout_family_ratio=args.holdout_family_ratio,
            eval_ratio=args.eval_ratio,
        )
    except (SplitError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"{len(metas)} candidate records | rubric {', '.join(plan.rubric_versions)}")
        if plan.by == "family_holdout":
            print(f"split mode: family_holdout | held-out families {args.holdout_family_ratio:.0%} "
                  f"| eval share per family {plan.eval_ratio:.0%} | seed {plan.seed}")
        else:
            print(f"split mode: {plan.by} | ratio {plan.ratio} | seed {plan.seed}")
        if plan.duplicate_groups:
            print(f"near-duplicates: {plan.duplicate_groups} groups kept whole on one side")
        if plan.forced_to_eval:
            print(f"pinned to eval: {len(plan.forced_to_eval)}"
                  " (eval_only scenarios + calibration holdout)")
        print()
        if plan.holdout_families:
            print(f"held-out families ({len(plan.holdout_families)}, test only): "
                  + ", ".join(plan.holdout_families))
        for side in ("train", "eval", "test"):
            st = plan.stats.get(side)
            if not st:
                continue
            print(f"  {side:5s} {st['n']:4d}  scam {st['scam']}"
                  f" (slow-burn {st['scam_late']} / compliant {st['compliance']})"
                  f" | hard-neg {st['hard_negative']} | plain {st['plain']}"
                  f" | family {st['families']} | scam detection rate {st['scam_detection_rate']}"
                  + (f" (of which {st['scam_no_evidence']} with a premise declaring no decidable evidence, "
                     f"{st['no_evidence_detected']} detected)"
                     if st['scam_no_evidence'] else ""))
        for w in plan.warnings:
            print(f"  ! {w}")

    if args.dry_run:
        print("\n--dry-run: no files written")
        return 0
    try:
        written = write_split(args.out_root, args.name, plan, convs, metas,
                             overwrite=args.overwrite)
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.json:
        print()
        for side, path in written.items():
            print(f"wrote {path}/  (conversations.jsonl + metadata.jsonl + _split.json + README.md)")
    return 0


def _bench_dir(split: str, runs: str | None) -> Path:
    return Path(runs) if runs else Path("datasets/bench") / Path(split).name


def _cmd_bench(args: argparse.Namespace) -> int:
    from .bench.adapters import JUDGES
    from .bench.run import load_split, manifest, run_bench

    if args.system not in JUDGES:
        print(f"error: unknown system {args.system!r}; choices: {', '.join(sorted(JUDGES))}",
              file=sys.stderr)
        return 2
    if args.system != "mock":
        try:
            env_path, loaded = load_env_file(args.env_file)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if loaded and not args.json:
            print(f"loaded {env_path}: {', '.join(loaded)}")      # variable names only

    try:
        convs, metas = load_split(args.split)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    kw: dict[str, Any] = {}
    if args.system != "jev":
        kw["minimal_reasoning"] = args.cond == "c2"
    if args.system.startswith("claude"):
        kw["cache"] = not args.no_cache
    judge = JUDGES[args.system](**kw)

    out = Path(args.out) if args.out else (
        _bench_dir(args.split, None) / f"{args.system}.{args.cond}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    mf = out.parent / "_manifest.json"
    previous = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else None
    mf.write_text(
        json.dumps(manifest([args.system], threshold=args.threshold, split=args.split,
                            previous=previous),
                   indent=2, ensure_ascii=False), encoding="utf-8")

    async def go():
        try:
            return await run_bench(
                judge, convs, metas, out,
                threshold=args.threshold, concurrency=args.concurrency,
                limit=args.limit, max_requests=args.max_requests,
                progress=None if args.json else print,
            )
        finally:
            await judge.aclose()

    if not args.json:
        n = args.limit or len(convs)
        asked = sum(len(m.get("prefix_labels") or []) for m in list(metas.values())[:n])
        print(f"{judge.name} | {n} calls / about {asked} requests | threshold {args.threshold} "
              f"| {args.cond}{' (no cache)' if args.no_cache else ''}")
    stats = asyncio.run(go())
    summary = stats.summary()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if stats.calls_done or stats.calls_skipped else 1
    u = summary["usage"]
    print(f"\ndone {stats.calls_done} calls (skipped {stats.calls_skipped}, failed {stats.failed})"
          f", took {summary['elapsed_s']}s")
    print(f"requests {u['requests']} | input {u['input_tokens']:,} / output {u['output_tokens']:,}")
    print(f"cache hits {u['cached_tokens']:,} / {u.get('total_input_tokens', 0):,} tokens"
          f" ({u['cache_hit_share']:.0%}) | written {u['cache_write_tokens']:,}")
    print(f"parse failures {u['parse_failures']} (of which truncated by the output cap {u.get('truncated', 0)})")
    if stats.stopped_early:
        print(f"  ! stopped early: {stats.stopped_early}")
    print(f"\nwrote {out}")
    print(f"table: callfactory bench-table {args.split}")
    return 0


def _cmd_bench_tune(args: argparse.Namespace) -> int:
    """Pick thresholds on the train side. Zero model calls: the scores are already in the bench result files.

    **Refuses eval / test directories.** This isn't a convenience check: tuning the threshold on eval and then reporting on eval
    opens a back door for the baseline, and we require the baseline and our own model to be treated the same.
    """
    from .bench.run import load_split
    from .bench.tune import GRID, HARD_NEG_BUDGET, OBJECTIVE, tune

    split = str(args.split)
    if "/eval/" in split or "/test/" in split:
        print("error: thresholds are not tuned on eval / test; use a train-side subset. "
              "The objective is in callfactory/bench/tune.py", file=sys.stderr)
        return 2
    try:
        convs, metas = load_split(split)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    runs_dir = _bench_dir(split, args.runs)
    files = sorted(runs_dir.glob("*.jsonl"))
    if not files:
        print(f"error: no bench results in {runs_dir}", file=sys.stderr)
        return 2

    results = {}
    for f in files:
        name = f.name[: -len(".jsonl")]
        results[name] = tune(name, _read_jsonl(f), metas)

    out_path = Path(args.out) if args.out else runs_dir / "_thresholds.json"
    out_path.write_text(
        json.dumps({n: r.threshold for n, r in sorted(results.items())},
                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps({n: r.to_dict() for n, r in sorted(results.items())},
                         indent=2, ensure_ascii=False))
        return 0

    print(f"{split} | objective: {OBJECTIVE} | hard-neg budget {HARD_NEG_BUDGET}")
    print(f"candidate thresholds {GRID[0]}-{GRID[-1]} (step 0.05)\n")
    head = f"{'system':24s} {'thresh':>6s} {'train det':>9s} {'train-hn FA':>12s} {'met':>6s}"
    print(head); print("-" * len(head))
    for name, r in sorted(results.items()):
        fmt = lambda v, w: (f"{v:>{w}.3f}" if isinstance(v, (int, float)) else f"{'—':>{w}s}")
        print(f"{name:24s} {r.threshold:>6.2f} {fmt(r.detection_rate, 9)} "
              f"{fmt(r.hard_negative_false_alarm_rate, 12)} "
              f"{'yes' if r.met_budget else 'no':>6s}")
    print(f"\nwrote {out_path}")
    print("table on eval with these thresholds (once per system):")
    print("  callfactory bench-table <eval split> --runs <eval results> --threshold <threshold>")
    return 0


def _cmd_bench_table(args: argparse.Namespace) -> int:
    from .bench.run import load_split
    from .bench.score import (discrimination_feedback, score_system, slice_scores,
                              trivial_baselines)

    try:
        convs, metas = load_split(args.split)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    runs_dir = _bench_dir(args.split, args.runs)
    files = sorted(runs_dir.glob("*.jsonl"))
    if not files:
        print(f"error: no bench results in {runs_dir}", file=sys.stderr)
        return 2

    # The unruled slice is excluded separately, giving "with/without" versions: its correct answer isn't settled yet
    unadjudicated = sorted(
        c for c, m in metas.items()
        if ((m.get("ground_truth") or {}).get("no_observable_evidence")))

    by_system: dict[str, list[dict[str, Any]]] = {}
    for f in files:
        by_system[f.name[: -len(".jsonl")]] = _read_jsonl(f)

    per_system: dict[str, float] = {}
    if args.thresholds:
        per_system = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    out: dict[str, Any] = {"split": args.split, "threshold": args.threshold,
                           "trivial_baselines": trivial_baselines(metas),
                           "unadjudicated_excluded": len(unadjudicated), "systems": {}}
    for name, runs in by_system.items():
        thr = per_system.get(name, args.threshold)
        out["systems"][name] = {
            "threshold": thr,
            "with_unadjudicated": score_system(runs, metas, threshold=thr),
            "without_unadjudicated": score_system(runs, metas, threshold=thr,
                                                  exclude=unadjudicated),
            "slices": slice_scores(runs, metas, threshold=thr),
        }
    out["feedback"] = discrimination_feedback(by_system, metas)

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    tb = out["trivial_baselines"]
    thr_note = f"thresholds {args.thresholds}, else {args.threshold}" if per_system else f"threshold {args.threshold}"
    print(f"{args.split} | {thr_note} | prefixes {tb.get('prefixes')} "
          f"(positives {tb.get('positives')})\n")
    head = (f"{'system':24s} {'thr':>5s} {'pfx acc':>7s} {'bal acc':>8s} {'PR-AUC':>7s} "
            f"{'detect':>7s} {'hardneg FA':>11s} {'plain FA':>8s} {'lat med':>8s} {'pair':>6s}")
    print(head); print("-" * len(head))
    print(f"{'(always no)':24s} {'—':>5s} {tb['always_no']['prefix_accuracy']:>7.3f} "
          f"{0.5:>8.3f} {'—':>7s} {0.0:>7.3f} {0.0:>11.3f} {0.0:>8.3f} {'—':>8s} {'—':>6s}")
    print(f"{'(always yes)':24s} {'—':>5s} {tb['always_yes']['prefix_accuracy']:>7.3f} "
          f"{0.5:>8.3f} {'—':>7s} {1.0:>7.3f} {1.0:>11.3f} {1.0:>8.3f} {'T0':>8s} {'—':>6s}")
    for name, s_ in sorted(out["systems"].items()):
        a = s_["with_unadjudicated"]; p = a["prefix_level"]; c = a["call_level"]
        fmt = lambda v, w, d=3: (f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'—':>{w}s}")
        print(f"{name:24s} {s_['threshold']:>5.2f} {fmt(p['accuracy'],7)} {fmt(p['balanced_accuracy'],8)} "
              f"{fmt(p['average_precision'],7)} {fmt(c['detection_rate'],7)} "
              f"{fmt(c['hard_negative_false_alarm_rate'],11)} "
              f"{fmt(c['plain_false_alarm_rate'],8)} "
              f"{fmt(c['latency_median_utterances'],8,1)} "
              f"{fmt(a['detection_point_pairs']['accuracy'],6,2)}")
    fb = out["feedback"]
    print(f"\nback to the gatekeeper (R7.5): non-discriminative {len(fb['no_discrimination'])} | "
          f"suspected label errors {len(fb['suspected_label_error'])}")
    print(f"{len(unadjudicated)} unruled no-evidence samples reported in a separate version (in --json as "
          f"without_unadjudicated)")
    return 0


def _cmd_rederive(args: argparse.Namespace) -> int:
    from .rederive import rederive_run

    try:
        stats = rederive_run(
            args.run, args.out,
            threshold=args.threshold,
            require_persistence=not args.no_persistence,
            require_sufficient=args.require_sufficient,
            progress=None if args.json else print,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = stats.summary()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    print(f"{stats.total}: {', '.join(summary['before_rule'])} -> {stats.rule} "
          f"(threshold {stats.threshold})")
    sh = stats.shift
    print(f"label shift: detections gained {sh['gained']} | lost {sh['lost']} | "
          f"earlier {sh['earlier']} | later {sh['later']} | unchanged {sh['unchanged']}")
    if stats.positions:
        print(stats.positions.render())
    print(f"\nwrote {OutputPaths.resolve(args.out).root}/  (conversations copied as is + metadata recomputed)")
    print("note: this **moves the operating point**, it isn't relabelling; not one byte of the prefix labels changed")
    return 0


def _cmd_author(args: argparse.Namespace) -> int:
    try:
        env_path, loaded = load_env_file(args.env_file)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if loaded and not args.json:
        print(f"loaded {env_path}: {', '.join(loaded)}")

    try:
        bank = _open_bank(args)
    except BankError:
        bank = None          # empty directory: this is the first call that creates the bank
    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8").strip()

    gaps = None
    if args.use_gaps and bank is not None:
        gaps = bank.coverage_gaps(CALLER_PERSONAS)

    name = args.bank_name or (bank.name if bank else "seed")
    draft = Path(args.bank_dir) / f"{name}.candidates.jsonl"

    # Appending must happen on the "don't commit" path: the candidate file is a review queue; committing directly would write this round
    # into the bank while leaving earlier rounds' candidates in the queue, and the two would never match again.
    if args.append and not args.dry_run:
        print("error: --append requires --dry-run (the candidate file is a review queue, not a commit path)",
              file=sys.stderr)
        return 2
    pending: list[dict[str, Any]] = []
    if args.append and draft.exists():
        pending = _read_jsonl(draft)
        if not args.json:
            print(f"{len(pending)} existing candidates, also used as dedup context")

    provider = build_provider(args.provider, model=args.model, temperature=args.temperature)

    async def go():
        try:
            return await author_scenarios(
                provider, bank, args.num,
                context=context, gaps=gaps, pending=pending, bank_name=name,
                max_attempts=args.max_attempts, temperature=args.temperature,
            )
        finally:
            await provider.aclose()

    result = asyncio.run(go())
    summary = result.summary()

    if args.json:
        print(json.dumps(
            {**summary, "scenarios": result.accepted}, indent=2, ensure_ascii=False
        ))
    else:
        print(f"\nproposed {summary['proposed']}, accepted {summary['accepted']}, "
              f"rejected {summary['rejected']} ({summary['attempts']} rounds)\n")
        for rec in result.accepted:
            tags = []
            if rec["hard_negative"]:
                tags.append("hard-neg")
            if rec["late_onset"]:
                tags.append("late-onset")
            print(f"  + {rec['category']:11s} {rec['family']:34s} "
                  f"{','.join(rec['expected_signs']) or '—'}  {' '.join(tags)}")
            print(f"      {rec['pretext']}")
            if rec["disambiguator"]:
                print(f"      when challenged: {rec['disambiguator']}")
        for rej in result.rejected:
            print(f"  - {rej.family:34s} {'; '.join(rej.reasons)}")
        if result.usage.get("requests"):
            u = result.usage
            print(f"\nusage: {u['requests']} requests, in {u['input_tokens']:,} / "
                  f"out {u['output_tokens']:,}")

    if not result.accepted:
        print("\nno new scenarios to commit.", file=sys.stderr)
        return 1
    if args.dry_run:
        # Candidates go to disk too: this generation cost money, and it must still be committable after review;
        # otherwise --dry-run would throw away what was just generated.
        draft.parent.mkdir(parents=True, exist_ok=True)
        rows = pending + result.accepted
        draft.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        if pending:
            print(f"\n--dry-run: scenario bank not written; candidates {len(pending)} + {len(result.accepted)} "
                  f"= {len(rows)} saved to {draft}")
        else:
            print(f"\n--dry-run: scenario bank not written; candidates saved to {draft}")
        print(f"commit after review: callfactory commit-candidates --bank-dir {args.bank_dir}")
        return 0

    version = args.version or ((bank.version if bank else 0) + 1)
    try:
        path = write_bank(
            merge_into_bank(bank, result.accepted, bank_name=name),
            args.bank_dir, name, version,
            author_model=provider.model, context=context,
        )
    except BankError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    total = len(bank) if bank else 0
    print(f"\nwrote {path} — {total} old + {len(result.accepted)} new = "
          f"{total + len(result.accepted)}")
    print("next: callfactory review-bank  to review and see what's still missing")
    return 0


def _cmd_commit_candidates(args: argparse.Namespace) -> int:
    try:
        bank = _open_bank(args)
    except BankError:
        bank = None
    name = args.bank_name or (bank.name if bank else "seed")
    draft = Path(args.bank_dir) / f"{name}.candidates.jsonl"
    if not draft.exists():
        print(f"error: no candidate file {draft}; run author-scenarios --dry-run first", file=sys.stderr)
        return 2
    records = _read_jsonl(draft)
    dropped = set(args.drop or [])
    renames: dict[str, str] = {}
    for pair in (args.rename or []):
        if "=" not in pair:
            print(f"error: --rename must be written OLD=NEW, got {pair!r}", file=sys.stderr)
            return 2
        old_name, new_name = pair.split("=", 1)
        renames[old_name.strip()] = new_name.strip()
    unknown = (dropped | set(renames)) - {r.get("family") for r in records}
    if unknown:
        print(f"error: no such families in the candidates: {sorted(unknown)}", file=sys.stderr)
        return 2

    keep = []
    for rec in records:
        if rec.get("family") in dropped:
            continue
        if rec.get("family") in renames:
            rec = {**rec, "family": renames[rec["family"]]}
        keep.append(rec)
    if not keep:
        print("error: no scenarios left to commit after dropping", file=sys.stderr)
        return 1

    version = args.version or ((bank.version if bank else 0) + 1)
    try:
        path = write_bank(
            merge_into_bank(bank, keep, bank_name=name),
            args.bank_dir, name, version,
            author_model=(keep[0].get("provenance") or {}).get("author_model", "unknown"),
        )
    except BankError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    total = len(bank) if bank else 0
    print(f"wrote {path} — {total} old + {len(keep)} new = {total + len(keep)}")
    if dropped:
        print(f"  dropped {len(records) - len(keep)}: {', '.join(sorted(dropped))}")
    if renames:
        for a, b in sorted(renames.items()):
            print(f"  renamed {a} → {b}")
    draft.unlink()
    print("next: callfactory review-bank  to review and see what's still missing")
    return 0


#: Any of these words in a family name usually means it is named after a **signal** rather than a call type;
#: once family names are coupled to labels, the model can learn a family -> label shortcut.
_SIGNAL_NAMED = (
    "wont_say", "won_t_say", "refuses", "refusing", "dodge", "evasion", "evades",
    "secrecy", "secret", "no_number", "dont_add_up", "doesnt_add_up", "swaps_story",
    "shift", "shifts", "undisclosed", "keeps_dropping", "mystery", "do_not_tell",
    "inconsistent", "withheld", "no_agent", "consistent",
)


def _cmd_review_candidates(args: argparse.Namespace) -> int:
    try:
        bank = _open_bank(args)
    except BankError:
        bank = None
    name = args.bank_name or (bank.name if bank else "seed")
    draft = Path(args.bank_dir) / f"{name}.candidates.jsonl"
    if not draft.exists():
        print(f"error: no candidate file {draft}; run author-scenarios --dry-run first", file=sys.stderr)
        return 2
    records = _read_jsonl(draft)

    gaps = bank.coverage_gaps(CALLER_PERSONAS) if bank else {}
    need_hn = set(gaps.get("signals_without_hard_negative") or [])
    never = set(gaps.get("signals_never_used_by_scam") or [])

    def flags(rec: dict[str, Any]) -> list[str]:
        out = []
        signs = set(rec.get("expected_signs") or [])
        if rec.get("hard_negative") and signs & need_hn:
            out.append(f"fills gap: {','.join(sorted(signs & need_hn))}")
        if rec["category"] == "scam" and signs & never:
            out.append(f"fills gap: {','.join(sorted(signs & never))}")
        hit = [w for w in _SIGNAL_NAMED if w in rec.get("family", "")]
        if hit:
            out.append(f"⚠ family looks named after a signal ({hit[0]})")
        if rec["category"] == "scam" and len(signs) < 2:
            out.append("⚠ too few signals, may not discriminate well")
        return out

    groups = [
        ("SCAM", [r for r in records if r["category"] == "scam"]),
        ("LEGITIMATE — hard negative", [r for r in records if r.get("hard_negative")]),
        ("LEGITIMATE — plain", [r for r in records
                               if r["category"] == "legitimate" and not r.get("hard_negative")]),
    ]

    if args.json:
        print(json.dumps(
            [{**r, "review_flags": flags(r)} for r in records], indent=2, ensure_ascii=False
        ))
        return 0

    lines: list[str] = [f"# {len(records)} candidate scenarios — {draft}", ""]
    if bank:
        lines.append(f"current bank: {bank.name} v{bank.version}, {len(bank)} scenarios. "
                     f"{len(bank) + len(records)} in total after commit.")
        lines.append("")
    for title, items in groups:
        lines.append(f"## {title} ({len(items)})")
        lines.append("")
        for rec in items:
            tags = []
            if rec.get("late_onset"):
                tags.append("late-onset")
            lines.append(f"### {rec['family']}" + (f"  ({', '.join(tags)})" if tags else ""))
            lines.append(f"- **pretext**: {rec['pretext']}")
            lines.append(f"- **org_kind**: {rec['org_kind']}")
            lines.append(f"- **signs**: {', '.join(rec['expected_signs']) or '—'}")
            if rec.get("disambiguator"):
                lines.append(f"- **when challenged**: {rec['disambiguator']}")
            if rec.get("notes"):
                lines.append(f"- notes: {rec['notes']}")
            for f in flags(rec):
                lines.append(f"- {f}")
            lines.append("")
    lines += [
        "## What to do", "",
        "```bash",
        f"# drop what you don't want + rename, then commit (zero model calls)",
        f"python -m callfactory commit-candidates \\",
        f"    --drop family_a family_b \\",
        f"    --rename old_name=new_name",
        "```", "",
        "The candidate file is plain-text JSONL; you can also edit pretext / signs directly before committing.",
    ]
    text = "\n".join(lines)
    md = draft.with_suffix(".md")
    md.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n(the same content was written to {md}, to view in an editor)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
