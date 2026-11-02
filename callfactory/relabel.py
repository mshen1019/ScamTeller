"""Relabel data on disk under a new rubric (R6.7).

**Not one line of dialogue changes**; only the evaluator reruns. It exists because of a methodology problem found in testing:
Claude's Messages API has no seed, so regenerating a batch gives **different dialogues**,
and per-sample comparison across batches can't separate rubric changes from dialogue jitter. In testing one sample
appeared to "lose its detection point" when it had simply become a different conversation.

Relabelling on fixed transcripts removes that noise, and is an order of magnitude cheaper too:

====================  ==========  ========  ======
                      requests    cost      noise
====================  ==========  ========  ======
regenerate 20                424  ~$1.4     all dialogue changes
relabel 20 fixed             151  ~$0.1     none
====================  ==========  ========  ======
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .batch import JSONLWriter, OutputPaths
from .evaluator import DEFAULT_DETECTION_THRESHOLD, EvaluatorAgent, evaluate_conversation
from .metrics import DetectionPositionReport, detection_positions, label_shift
from .providers import LLMProvider, ProviderError
from .schema import BlindAssessment, Conversation, GenerationTask, Utterance
from .validators import annotate_repetition, validate_conversation


@dataclass
class RelabelStats:
    conversations: int = 0
    relabelled: int = 0
    failed: int = 0
    invalid: int = 0
    eval_failures: int = 0
    shift: dict[str, int] = field(default_factory=dict)
    position_before: DetectionPositionReport | None = None
    position_after: DetectionPositionReport | None = None
    from_rubric: list[str] = field(default_factory=list)
    to_rubric: str = ""
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "conversations": self.conversations,
            "relabelled": self.relabelled,
            "failed": self.failed,
            "invalid": self.invalid,
            "eval_failures": self.eval_failures,
            "label_shift": dict(self.shift),
            "from_rubric": sorted(set(self.from_rubric)),
            "to_rubric": self.to_rubric,
            "position_before": self.position_before.summary() if self.position_before else None,
            "position_after": self.position_after.summary() if self.position_after else None,
            "usage": self.usage,
            "elapsed_s": self.elapsed_s,
        }


def read_run(run: str | Path) -> tuple[OutputPaths, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    paths = OutputPaths.resolve(run)
    if not paths.conversations.exists():
        raise FileNotFoundError(f"not found: {paths.conversations}")
    if not paths.metadata.exists():
        raise FileNotFoundError(f"not found: {paths.metadata}")

    def load(path: Path) -> list[dict[str, Any]]:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    convs = load(paths.conversations)
    metas = {m["conversation_id"]: m for m in load(paths.metadata)}
    return paths, convs, metas


async def relabel_run(
    src: str | Path,
    dst: str | Path,
    provider: LLMProvider,
    *,
    stride: int = 1,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    require_persistence: bool = True,
    require_sufficient: bool = False,
    global_eval: bool = True,
    concurrency: int = 4,
    keep_invalid: bool = False,
    progress: Callable[[str], None] | None = None,
) -> RelabelStats:
    src_paths, convs, metas = read_run(src)
    out = OutputPaths.resolve(dst)
    if out.root == src_paths.root:
        # Overwriting the source directory destroys the other half of the comparison, which is this command's whole value.
        raise ValueError("--out cannot be the source directory: the point of relabelling is keeping the old labels for comparison")
    out.mkdirs()

    stats = RelabelStats(conversations=len(convs))
    say = progress or (lambda _m: None)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    started = time.monotonic()
    before: list[int | None] = []
    after: list[int | None] = []
    pos_before: list[tuple[int | None, int, bool]] = []
    pos_after: list[tuple[int | None, int, bool]] = []

    with JSONLWriter(out.conversations) as conv_out, \
         JSONLWriter(out.metadata) as meta_out, \
         JSONLWriter(out.failures) as fail_out:

        async def worker(conv_rec: dict[str, Any]) -> None:
            cid = conv_rec["conversation_id"]
            meta = metas.get(cid)
            if meta is None:
                stats.failed += 1
                await fail_out.write({
                    "conversation_id": cid, "stage": "relabel",
                    "error": "transcript has no matching metadata, can't relabel",
                })
                return
            utterances = [
                Utterance(u["index"], u["speaker"], u["text"]) for u in conv_rec["utterances"]
            ]
            async with semaphore:
                try:
                    evaluator = EvaluatorAgent(
                        provider, seed=(meta.get("generation") or {}).get("derived_seed")
                    )
                    result = await evaluate_conversation(
                        utterances, evaluator,
                        stride=stride, threshold=threshold,
                        require_persistence=require_persistence,
                        require_sufficient=require_sufficient,
                        global_eval=global_eval,
                    )
                except (ProviderError, ValueError) as exc:
                    stats.failed += 1
                    await fail_out.write({
                        "conversation_id": cid, "stage": "relabel",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    say(f"[{cid}] FAILED {type(exc).__name__}: {exc}")
                    return

            old_meta = meta
            old_index = old_meta.get("earliest_detection_index")
            old_eval = (old_meta.get("generation") or {}).get("agents", {}).get("evaluator", {})
            conv = _rebuild(conv_rec, old_meta, utterances, result, evaluator)

            report = validate_conversation(conv, min_rounds=1, max_rounds=10**6)
            conv.validation = report.to_list()
            if not report.ok:
                stats.invalid += 1
                await fail_out.write({
                    "conversation_id": cid, "stage": "validation",
                    "errors": [i.to_dict() for i in report.errors],
                })
                if not keep_invalid:
                    return

            await conv_out.write(conv.transcript_record())
            await meta_out.write(conv.metadata_record())
            stats.relabelled += 1
            stats.eval_failures += len(result.failures)
            stats.from_rubric.append(old_eval.get("rubric_version") or "(not recorded)")
            before.append(old_index)
            after.append(result.earliest_detection_index)
            is_scam = bool((old_meta.get("ground_truth") or {}).get("is_scam"))
            pos_before.append((old_index, len(utterances), is_scam))
            pos_after.append((result.earliest_detection_index, len(utterances), is_scam))
            arrow = (
                "=" if old_index == result.earliest_detection_index
                else f"{old_index} → {result.earliest_detection_index}"
            )
            say(f"[{cid}] detect {arrow}")

        await asyncio.gather(*(worker(c) for c in convs))

    stats.shift = label_shift(before, after)
    stats.position_before = detection_positions(pos_before)
    stats.position_after = detection_positions(pos_after)
    stats.to_rubric = EvaluatorAgent(provider).provenance()["rubric_version"]
    stats.usage = {f"{provider.config.name}/{provider.model}": dict(provider.usage)}
    stats.elapsed_s = round(time.monotonic() - started, 2)
    return stats


def _rebuild(
    conv_rec: dict[str, Any],
    meta: dict[str, Any],
    utterances: list[Utterance],
    result: Any,
    evaluator: EvaluatorAgent,
) -> Conversation:
    """Put the new labels back into a Conversation, keeping the original task and generation provenance."""
    generation = dict(meta.get("generation") or {})
    agents = dict(generation.get("agents") or {})
    agents["evaluator"] = evaluator.provenance()
    generation["agents"] = agents
    generation["evaluation"] = {
        "stride": result.stride,
        "threshold": result.threshold,
        "require_persistence": result.require_persistence,
        "require_sufficient": result.require_sufficient,
        "detection_rule": "detect/v1" if result.require_sufficient else "detect/v2",
        "prefixes_labelled": len(result.prefix_labels),
        "failures": [f.to_dict() for f in result.failures],
    }
    # Keep the other half of the comparison in the data, so there's no need to open two directories to see what changed.
    generation["relabelled_from"] = {
        "run_id": (meta.get("generation") or {}).get("run_id"),
        "rubric_version": ((meta.get("generation") or {}).get("agents", {})
                           .get("evaluator", {}).get("rubric_version")),
        "earliest_detection_index": meta.get("earliest_detection_index"),
        "relabelled_at": time.time(),
    }

    conv = Conversation(
        conversation_id=conv_rec["conversation_id"],
        utterances=utterances,
        task=GenerationTask.from_dict(meta["task"]),
        generation=generation,
        prefix_labels=result.prefix_labels,
        utterance_labels=result.utterance_labels,
        blind_assessment=result.blind_assessment or _blind_from(meta),
        earliest_detection_index=result.earliest_detection_index,
    )
    annotate_repetition(conv)
    return conv


def _blind_from(meta: dict[str, Any]) -> BlindAssessment | None:
    """With --no-global-eval, reuse the old quality evaluation and save a call."""
    data = meta.get("blind_assessment")
    if not isinstance(data, dict):
        return None
    names = {"coherence", "naturalness", "repetition", "role_consistency",
             "blind_is_suspicious", "notes"}
    return BlindAssessment(**{k: v for k, v in data.items() if k in names})
