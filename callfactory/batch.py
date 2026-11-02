"""Batch generation and writing to disk (C3 / N1 / N3).

Three append-only files written side by side, flushed per record, so an interrupted run can resume in place:

* ``conversations.jsonl`` -- transcripts only; label fields don't exist in its structure
* ``metadata.jsonl``      -- scenario, generation parameters, provenance, prefix labels
* ``failures.jsonl``      -- generation failures and hard-rule rejects, with reasons

Two implementation choices:

* **The semaphore limits conversations in flight, not requests** (N1). The N labelling requests inside one conversation
  are sent concurrently, so peak requests ≈ concurrency × (1 + number of prefixes), far more than concurrency.
  ``estimate_calls`` computes that number, so rate limits aren't sized from concurrency.
* **A lost checkpoint can be rebuilt from metadata** (N3). The checkpoint is only an index, not the truth;
  the truth is the lines already on disk.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .dialogue import DialogueFailure
from .evaluator import prefix_indices
from .pipeline import GenerationSettings, generate_conversation
from .providers import ProviderError, ProviderRoles
from .schema import Conversation, GenerationTask
from .validators import validate_conversation

DEFAULT_CANDIDATE_DIR = "datasets/candidates"


@dataclass
class OutputPaths:
    root: Path
    conversations: Path
    metadata: Path
    failures: Path
    checkpoint: Path

    @classmethod
    def for_run(cls, base: str | Path, run_id: str) -> "OutputPaths":
        return cls.resolve(Path(base) / run_id)

    @classmethod
    def resolve(cls, output: str | Path) -> "OutputPaths":
        """``output`` is a directory; given a ``.jsonl`` path, sibling files are derived from its stem."""
        path = Path(output)
        if path.suffix == ".jsonl":
            root, stem = path.parent, path.stem
            return cls(
                root=root,
                conversations=path,
                metadata=root / f"{stem}.metadata.jsonl",
                failures=root / f"{stem}.failures.jsonl",
                checkpoint=root / f"{stem}._checkpoint.json",
            )
        return cls(
            root=path,
            conversations=path / "conversations.jsonl",
            metadata=path / "metadata.jsonl",
            failures=path / "failures.jsonl",
            checkpoint=path / "_checkpoint.json",
        )

    def mkdirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass
class BatchStats:
    requested: int = 0
    skipped_resumed: int = 0
    written: int = 0
    failed: int = 0
    invalid: int = 0
    scam: int = 0
    #: Scam by intent, but the **scenario premise** declares that no usable evidence is produced (``no_observable_evidence``).
    #:
    #: Note this is a property of the premise, not of the actual dialogue: nothing in the caller prompt requires the caller
    #: to honour it, and in testing 16 of 17 refused a verification channel when pressed, which is exactly the behaviour the rubric's
    #: GROUP B calls "sufficient on its own". So these samples must **not** be removed from the detection-rate denominator
    #: and counted as "correct": that would record an open question awaiting a ruling as a score. They are only reported separately; see
    #: EVAL_SYSTEM §5.7.
    scam_no_evidence: int = 0
    no_evidence_detected: int = 0
    legitimate: int = 0
    hard_negative: int = 0
    late_onset: int = 0
    rounds_total: int = 0
    detections_on_scam: int = 0
    detection_round_total: int = 0
    false_alarms_on_legit: int = 0
    false_alarms_on_hard_negative: int = 0
    blind_disagreements: int = 0
    eval_failures: int = 0
    elapsed_s: float = 0.0
    #: Actual token usage per provider, aggregated by "name/model" (when one provider serves as both
    #: caller and receiver it's counted once). Estimates are in estimate_calls; these are measured.
    usage: dict[str, dict[str, int]] = field(default_factory=dict)

    def totals(self) -> dict[str, int]:
        return {
            "requests": sum(u.get("requests", 0) for u in self.usage.values()),
            "input_tokens": sum(u.get("input_tokens", 0) for u in self.usage.values()),
            "output_tokens": sum(u.get("output_tokens", 0) for u in self.usage.values()),
        }

    def summary(self) -> dict[str, Any]:
        out = asdict(self)
        out["usage_totals"] = self.totals()
        out["mean_rounds"] = round(self.rounds_total / self.written, 2) if self.written else 0.0
        out["scam_detection_rate"] = (
            round(self.detections_on_scam / self.scam, 3) if self.scam else 0.0
        )
        # The slice whose premise declares "no usable evidence" is reported separately, but **stays** in the denominator above; see the field comment.
        out["no_evidence_detection_rate"] = (
            round(self.no_evidence_detected / self.scam_no_evidence, 3)
            if self.scam_no_evidence else None
        )
        out["mean_detection_round_on_detected_scam"] = (
            round(self.detection_round_total / self.detections_on_scam, 2)
            if self.detections_on_scam else None
        )
        # The false-alarm rate must be read alongside latency (R7.3); otherwise "always alarm immediately" looks like the optimal strategy.
        out["legit_false_alarm_rate"] = (
            round(self.false_alarms_on_legit / self.legitimate, 3) if self.legitimate else 0.0
        )
        out["hard_negative_false_alarm_rate"] = (
            round(self.false_alarms_on_hard_negative / self.hard_negative, 3)
            if self.hard_negative else 0.0
        )
        # Label-quality reference (R7.4): the rate at which the evaluator's independent judgement disagrees with ground truth.
        out["blind_disagreement_rate"] = (
            round(self.blind_disagreements / self.written, 3) if self.written else 0.0
        )
        return out


class JSONLWriter:
    """append-only, flushed per record. Concurrent workers share one handle; a lock keeps each line atomic."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None
        self._lock = asyncio.Lock()

    def __enter__(self) -> "JSONLWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None

    async def write(self, record: dict[str, Any]) -> None:
        async with self._lock:
            assert self._handle is not None, "writer not open"
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()


@dataclass
class Checkpoint:
    path: Path
    run_id: str
    config: dict[str, Any] = field(default_factory=dict)
    completed: set[int] = field(default_factory=set)
    rebuilt_from_metadata: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @classmethod
    def load_or_create(
        cls, paths: OutputPaths, run_id: str, config: dict[str, Any], resume: bool
    ) -> "Checkpoint":
        cp = cls(path=paths.checkpoint, run_id=run_id, config=config)
        if not resume:
            return cp
        if paths.checkpoint.exists():
            data = json.loads(paths.checkpoint.read_text(encoding="utf-8"))
            cp.completed = {i for i in data.get("completed", []) if isinstance(i, int)}
            cp.run_id = data.get("run_id", run_id)
            return cp
        if paths.metadata.exists():
            # N3: the checkpoint is only an index; the truth is the lines on disk.
            cp.rebuilt_from_metadata = True
            for line in paths.metadata.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                index = (rec.get("generation") or {}).get("sample_index")
                if isinstance(index, int):
                    cp.completed.add(index)
        return cp

    async def mark(self, index: int) -> None:
        async with self._lock:
            self.completed.add(index)
            self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({
                "run_id": self.run_id,
                "config": self.config,
                "completed": sorted(self.completed),
                "updated_at": time.time(),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)


def estimate_calls(
    tasks: Sequence[GenerationTask],
    settings: GenerationSettings | None = None,
    *,
    concurrency: int = 4,
) -> dict[str, Any]:
    """N6: expected call count and peak in-flight requests, with zero requests."""
    settings = settings or GenerationSettings()
    dialogue = prefix = glob = 0
    peak_prefixes = 0
    for task in tasks:
        rounds = max(settings.min_rounds, min(settings.max_rounds, task.target_rounds))
        n_utt = 2 * rounds
        dialogue += n_utt
        if settings.prefix_eval:
            k = len(prefix_indices(n_utt, settings.eval_stride))
            prefix += k
            peak_prefixes = max(peak_prefixes, k)
        if settings.global_eval:
            glob += 1
    per_conv_peak = (
        min(peak_prefixes, settings.eval_concurrency)
        if settings.eval_concurrency else peak_prefixes
    )
    return {
        "conversations": len(tasks),
        "dialogue_calls": dialogue,
        "prefix_eval_calls": prefix,
        "global_eval_calls": glob,
        "total_calls": dialogue + prefix + glob,
        "concurrency": concurrency,
        # This is the number rate limiting should look at (N1).
        "peak_in_flight_requests": concurrency * max(1, per_conv_peak),
    }


async def run_batch(
    tasks: Iterable[GenerationTask],
    roles: ProviderRoles,
    *,
    output: str | Path,
    run_id: str = "run",
    settings: GenerationSettings | None = None,
    concurrency: int = 4,
    resume: bool = False,
    keep_invalid: bool = False,
    p4_warnings: Sequence[str] | None = None,
    config: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BatchStats:
    settings = settings or GenerationSettings()
    tasks = list(tasks)
    paths = OutputPaths.resolve(output)
    paths.mkdirs()
    checkpoint = Checkpoint.load_or_create(paths, run_id, config or {}, resume)

    stats = BatchStats(requested=len(tasks))
    pending = [t for t in tasks if t.index not in checkpoint.completed]
    stats.skipped_resumed = len(tasks) - len(pending)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    say = progress or (lambda _msg: None)
    started = time.monotonic()
    if stats.skipped_resumed:
        source = "metadata" if checkpoint.rebuilt_from_metadata else "checkpoint"
        say(f"resuming ({source}): {stats.skipped_resumed} already done, skipped")

    with JSONLWriter(paths.conversations) as conv_out, \
         JSONLWriter(paths.metadata) as meta_out, \
         JSONLWriter(paths.failures) as fail_out:

        async def worker(task: GenerationTask) -> None:
            async with semaphore:
                try:
                    conv = await generate_conversation(
                        task, roles,
                        settings=settings,
                        run_id=run_id,
                        p4_warnings=list(p4_warnings or []),
                    )
                except (DialogueFailure, ProviderError, ValueError) as exc:
                    stats.failed += 1
                    await fail_out.write({
                        "task_id": task.task_id,
                        "sample_index": task.index,
                        "scenario_id": task.scenario.scenario_id,
                        "stage": "generation",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    say(f"[{task.index}] FAILED {type(exc).__name__}: {exc}")
                    return

                report = validate_conversation(
                    conv,
                    min_rounds=settings.min_rounds,
                    max_rounds=settings.max_rounds,
                    repetition_threshold=settings.repetition_threshold,
                )
                conv.validation = report.to_list()
                if not report.ok:
                    stats.invalid += 1
                    await fail_out.write({
                        "task_id": task.task_id,
                        "sample_index": task.index,
                        "conversation_id": conv.conversation_id,
                        "scenario_id": task.scenario.scenario_id,
                        "stage": "validation",
                        "errors": [i.to_dict() for i in report.errors],
                    })
                    say(f"[{task.index}] invalid: "
                        + "; ".join(i.code for i in report.errors))
                    if not keep_invalid:
                        return

                await conv_out.write(conv.transcript_record())
                await meta_out.write(conv.metadata_record())
                await checkpoint.mark(task.index)
                _accumulate(stats, conv)
                scn = task.scenario
                say(f"[{task.index}] {scn.category:11s} {scn.family:34s} "
                    f"rounds={conv.rounds:2d} detect@{conv.earliest_detection_index}")

        await asyncio.gather(*(worker(t) for t in pending))

    for provider in roles.all_providers():
        key = f"{provider.config.name}/{provider.model}"
        stats.usage[key] = dict(provider.usage)
    stats.elapsed_s = round(time.monotonic() - started, 2)
    return stats


def _accumulate(stats: BatchStats, conv: Conversation) -> None:
    scn = conv.task.scenario
    stats.written += 1
    stats.rounds_total += conv.rounds
    stats.eval_failures += len(
        (conv.generation.get("evaluation") or {}).get("failures") or []
    )
    detected = conv.earliest_detection_index is not None
    is_scam = scn.category == "scam"
    if is_scam:
        stats.scam += 1
        if conv.task.late_onset:
            stats.late_onset += 1
        if scn.no_observable_evidence:
            stats.scam_no_evidence += 1
            stats.no_evidence_detected += int(detected)
        if detected:
            stats.detections_on_scam += 1
            stats.detection_round_total += (conv.earliest_detection_index // 2) + 1
    else:
        stats.legitimate += 1
        if scn.hard_negative:
            stats.hard_negative += 1
        if detected:
            stats.false_alarms_on_legit += 1
            if scn.hard_negative:
                stats.false_alarms_on_hard_negative += 1
    blind = conv.blind_assessment
    if blind is not None and blind.blind_is_suspicious is not None:
        if blind.blind_is_suspicious != is_scam:
            stats.blind_disagreements += 1
