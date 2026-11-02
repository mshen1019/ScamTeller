"""Human prefix labelling (the collection end of the R6.5 calibration set).

This is a **measuring instrument**, so its constraints come before convenience:

* **None of our labels are shown**, and neither is ground truth. Seeing `detect@7` and then judging which turn is enough
  measures anchoring, not human judgement.
* **Reveal turn by turn, never the whole conversation at once.** The label is defined as "enough or not, having heard only the first k turns", and knowing
  the ending contaminates that judgement: after reading the full text and pointing back at turn 4, people unconsciously use what came later.
  So this interface only moves forward; there is no looking back.
* **Interruptible and resumable.** Labelling 80 items in one sitting isn't realistic; items already labelled are skipped on the next start.

Output ``human_labels.jsonl`` is stored apart from the machine labels; ``calibrate`` then aligns the two and computes agreement.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

HUMAN_LABEL_FILE = "human_labels.jsonl"

#: How many utterances to reveal at a time. ``round`` = one caller + receiver pair, aligned with the detector's rounds.
REVEAL_MODES = ("round", "utterance")


@dataclass
class HumanLabel:
    conversation_id: str
    is_scam: bool | None                   # None = labeller unsure
    detection_index: int | None            # first utterance where they wanted to alarm; None = never wanted to
    revealed_through: int                  # last utterance actually seen (a label is valid without reaching the end)
    notes: str = ""
    labelled_at: float = field(default_factory=time.time)
    labeller: str = "human"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def detection_round(self) -> int | None:
        if self.detection_index is None:
            return None
        return self.detection_index // 2 + 1


def human_label_path(run: str | Path) -> Path:
    return Path(run) / HUMAN_LABEL_FILE


def load_human_labels(run: str | Path) -> dict[str, HumanLabel]:
    path = human_label_path(run)
    if not path.exists():
        return {}
    out: dict[str, HumanLabel] = {}
    names = {f for f in HumanLabel.__dataclass_fields__}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Later writes override earlier ones: relabelling an item in the append-only file is just appending another line.
        out[data["conversation_id"]] = HumanLabel(
            **{k: v for k, v in data.items() if k in names}
        )
    return out


def append_human_label(run: str | Path, label: HumanLabel) -> None:
    path = human_label_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


# --------------------------------------------------------------------------- #
# Interaction
# --------------------------------------------------------------------------- #

HELP = """\
After each segment, press:
  [Enter]  keep listening to the next round
  !        this is it: there's enough evidence by now, I would raise the alarm
  n        this call shouldn't raise an alarm even heard to the end (skip to the final verdict)
  s        skip this one, don't label it
  q        save and quit (everything labelled is kept)
  ?        show these instructions again
"""


@dataclass
class LabelSessionStats:
    total: int = 0
    already_done: int = 0
    labelled: int = 0
    skipped: int = 0
    quit_early: bool = False

    def summary(self) -> dict[str, Any]:
        return asdict(self)


def _chunks(n_utterances: int, mode: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end_inclusive)`` reveal ranges."""
    if mode not in REVEAL_MODES:
        raise ValueError(f"reveal must be one of {REVEAL_MODES}")
    step = 2 if mode == "round" else 1
    for start in range(0, n_utterances, step):
        yield start, min(start + step - 1, n_utterances - 1)


def label_session(
    run: str | Path,
    conversations: Sequence[dict[str, Any]],
    *,
    read_line: Callable[[str], str],
    write: Callable[[str], None],
    reveal: str = "round",
    only: Iterable[str] | None = None,
    labeller: str = "human",
) -> LabelSessionStats:
    """Run a human labelling session. ``read_line`` / ``write`` are injected for testing."""
    done = load_human_labels(run)
    wanted = set(only) if only is not None else None
    queue = [
        c for c in conversations
        if c["conversation_id"] not in done
        and (wanted is None or c["conversation_id"] in wanted)
    ]
    stats = LabelSessionStats(
        total=len(conversations),
        already_done=sum(
            1 for c in conversations if c["conversation_id"] in done
            and (wanted is None or c["conversation_id"] in wanted)
        ),
    )
    if not queue:
        write("Nothing left to label.\n")
        return stats

    write(HELP)
    for position, conv in enumerate(queue, start=1):
        cid = conv["conversation_id"]
        utterances = conv["utterances"]
        write(f"\n{'=' * 72}\n{cid}   ({position}/{len(queue)}, {stats.total} in the batch)\n")
        write("A call you've just picked up. You don't know who it is or why they're calling.\n")
        write(f"{'=' * 72}\n")

        flagged: int | None = None
        revealed_through = -1
        decided_no_alarm = False
        action = ""

        for start, end in _chunks(len(utterances), reveal):
            for utt in utterances[start:end + 1]:
                write(f"[{utt['index']:>2}] {utt['speaker'].upper():<8} {utt['text']}\n")
            revealed_through = end
            while True:
                action = read_line(f"  ({end + 1}/{len(utterances)}) > ").strip().lower()
                if action == "?":
                    write(HELP)
                    continue
                break
            if action in ("!", "n", "s", "q"):
                break

        if action == "s":
            stats.skipped += 1
            write("  Skipped.\n")
            continue
        if action == "q":
            stats.quit_early = True
            write("  Quitting; everything labelled has been saved.\n")
            break
        if action == "!":
            # The alarm moment is "after reading this segment", so it's recorded on the segment's last utterance.
            flagged = revealed_through
        elif action == "n":
            decided_no_alarm = True

        verdict = ""
        while verdict not in ("y", "n", "?u"):
            raw = read_line("  Whole call: is this a scam? [y]es / [n]o / [u]nsure > ").strip().lower()
            verdict = {"y": "y", "yes": "y", "n": "n", "no": "n",
                       "u": "?u", "unsure": "?u"}.get(raw, "")
            if not verdict:
                write("  Please enter y / n / u.\n")
        notes = read_line("  Notes (optional)> ").strip()

        label = HumanLabel(
            conversation_id=cid,
            is_scam=None if verdict == "?u" else verdict == "y",
            detection_index=flagged,
            revealed_through=revealed_through,
            notes=notes,
            labeller=labeller,
        )
        append_human_label(run, label)
        stats.labelled += 1
        where = (
            f"round {label.detection_round} (utterance {flagged})"
            if flagged is not None else ("no alarm throughout" if decided_no_alarm else "no alarm point marked")
        )
        write(f"  Recorded: {'scam' if label.is_scam else 'not scam' if label.is_scam is False else 'unsure'}, {where}\n")

    return stats
