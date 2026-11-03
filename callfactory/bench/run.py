"""Batch runs + writing to disk (resumable, with a budget cap).

Two constraints that differ from ``batch.py``, both so that prefix caching hits:

* **One call's prefixes must run back to back.** The cache TTL is only minutes; scattering the 912 calls in random order
  means none of them hit. So concurrency is **across conversations**: N calls in flight at once, each strictly in order inside.
* **Requests are sent only on labelled prefixes.** Elsewhere there's no comparable ground truth and nothing to reconcile against; it saves 42%
  of the calls (1586 -> 912). ``ReplaySession`` repeats the previous verdict at the positions in between (R7.6).

One line per call on disk, append-only, resumed by conversation_id after an interruption; the same idea as batch:
the checkpoint is only an index, the truth is the lines already on disk.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..providers.base import ProviderError
from ..schema import Utterance
from . import judge_prompt as JP
from .protocol import ReplayAdapter, replay_call


@dataclass
class BenchRunStats:
    system: str = ""
    calls_done: int = 0
    calls_skipped: int = 0
    failed: int = 0
    requests: int = 0
    elapsed_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    stopped_early: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "system": self.system, "calls_done": self.calls_done,
            "calls_skipped": self.calls_skipped, "failed": self.failed,
            "requests": self.requests, "elapsed_s": round(self.elapsed_s, 1),
            "usage": dict(self.usage), "stopped_early": self.stopped_early,
        }


def load_split(run_dir: str | Path) -> tuple[dict[str, list[Utterance]], dict[str, dict[str, Any]]]:
    """Read a split directory. **Only the transcript is handed to the model under test** (R7.2)."""
    root = Path(run_dir)
    convs: dict[str, list[Utterance]] = {}
    for line in (root / "conversations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        convs[c["conversation_id"]] = [
            Utterance(index=u["index"], speaker=u["speaker"], text=u["text"])
            for u in c["utterances"]
        ]
    metas: dict[str, dict[str, Any]] = {}
    for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        metas[m["conversation_id"]] = m
    return convs, metas


def ask_positions(meta: dict[str, Any]) -> set[int]:
    """Utterance indices at which a request is really sent = the labelled prefixes."""
    return {pl["through_index"] for pl in (meta.get("prefix_labels") or [])}


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.add(json.loads(line)["conversation_id"])
            except (ValueError, KeyError):
                continue
    return out


async def run_bench(
    judge: Any,
    convs: dict[str, list[Utterance]],
    metas: dict[str, dict[str, Any]],
    out_path: str | Path,
    *,
    threshold: float = 0.5,
    concurrency: int = 4,
    limit: int | None = None,
    max_requests: int | None = None,
    resume: bool = True,
    progress: Callable[[str], None] | None = None,
) -> BenchRunStats:
    started = time.monotonic()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = BenchRunStats(system=judge.name)

    already = done_ids(out) if resume else set()
    order = sorted(cid for cid in convs if cid in metas)
    if limit:
        order = order[:limit]
    todo = [cid for cid in order if cid not in already]
    stats.calls_skipped = len(order) - len(todo)

    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    stop = False

    async def one(cid: str) -> None:
        nonlocal stop
        async with sem:
            if stop:
                return
            # One session per call: prefixes advance in order -> each call's input is a strict extension of the previous one
            adapter = ReplayAdapter(judge, ask_at=ask_positions(metas[cid]))
            try:
                rec = await replay_call(adapter, cid, convs[cid], threshold)
            except ProviderError as exc:
                async with lock:
                    stats.failed += 1
                    if progress:
                        progress(f"  ! {cid} failed: {exc}")
                return
            async with lock:
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                stats.calls_done += 1
                stats.requests = judge.usage.requests
                if progress and stats.calls_done % 10 == 0:
                    u = judge.usage.to_dict()
                    progress(f"  {stats.calls_done}/{len(todo)} | requests {u['requests']} | "
                             f"cache hits {u['cache_hit_share']:.0%}")
                if max_requests and judge.usage.requests >= max_requests:
                    stop = True
                    stats.stopped_early = f"hit the request cap {max_requests}"

    await asyncio.gather(*(one(cid) for cid in todo))
    stats.usage = judge.cache_report() if hasattr(judge, 'cache_report') else judge.usage.to_dict()
    stats.elapsed_s = time.monotonic() - started
    return stats


def manifest(systems: Sequence[str], *, threshold: float, split: str,
             previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """Frozen record: prompt fingerprint + threshold + data source. Written down before looking at results.

    ``previous`` is the manifest already in the output directory, if any. One ``bench`` run
    benchmarks **one** system, so writing only that system's name would leave a directory of
    seven run files whose manifest claims one — which is what happened: the committed manifests
    named whichever system ran last. ``systems`` is therefore the **union** with what the
    directory already recorded, and ``created_at`` stays at the first write.
    """
    prior = list((previous or {}).get("systems") or [])
    return {
        "judge_version": JP.JUDGE_VERSION,
        "prompt_fingerprint": JP.fingerprint(),
        "threshold": threshold,
        "split": str(split),
        "systems": sorted(set(prior) | set(systems)),
        "created_at": (previous or {}).get("created_at", time.time()),
        "updated_at": time.time(),
        "note": ("Threshold is fixed and never tuned on eval; prefix caching changes only billing, "
                 "not request content, so comparability is unaffected"),
    }
