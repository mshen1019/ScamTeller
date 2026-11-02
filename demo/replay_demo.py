#!/usr/bin/env python3
"""Replay a call utterance by utterance, showing each detector's score at that moment as it goes. For live demos.

**Fully offline, fully deterministic.** It calls no model: the scores are already in the bench run files,
and this just plays them back in utterance order. The last thing you want on stage is a live API call:
it stalls, it varies, and if the network drops the whole demo is gone.

Several systems share one screen because **a single system doesn't show the point**. On the same legitimate call, one line sits at zero throughout
while another wobbles between 0.35 and 0.90; that contrast needs no explanation. With one line it's only "it didn't alarm".

Each system carries its own threshold (``path:threshold``): in this task "score" and "operating point" go together,
and one global threshold would compare systems at different operating points as if they were the same.

    python demo/replay_demo.py                  # the default pair of calls + three lines
    python demo/replay_demo.py --pace 1.2       # autoplay, 1.2 seconds per utterance
    python demo/replay_demo.py --call <id> --runs <path>:<threshold>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The default pair: **same theme, opposite ground truth**. The scam call's scores jump right at the labelled detection point,
#: and the genuine call sits at zero throughout. Both were picked from the 119 calls by the data (jump size / lowest peak), not at random.
DEFAULT_PAIR = ("calib-101-00013", "run-1101-00322")
DEFAULT_SPLIT = "datasets/eval/ds3-rubric-v3-detect07"
DEFAULT_THRESHOLD = 0.50

#: ``path:threshold``. All thresholds were tuned on the train side; background in datasets/bench/README.md.
#: The first is the fine-tuned model, whose predictions are **not in the repo**; if missing it's skipped automatically and only the two closed-source lines play.
DEFAULT_RUNS = [
    "datasets/bench/ds3-eval-c2/scamteller.jsonl:0.50",
    "datasets/bench/ds3-eval-c2/deepseek-flash.c2.jsonl:0.85",
    "datasets/bench/ds3-eval-c2/claude-haiku.c2.jsonl:0.90",
]

BOLD, DIM, RED, GREEN, YELLOW, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m")


def parse_runs(specs):
    """``path:threshold`` -> (display name, path, threshold). Missing files are skipped with a notice."""
    out = []
    for spec in specs:
        path, sep, thr = spec.rpartition(":")
        if not sep or "/" in thr:
            path, thr = spec, str(DEFAULT_THRESHOLD)
        name = Path(path).name
        for suffix in (".c2.jsonl", ".jsonl"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        full = ROOT / path
        if not full.exists():
            print(f"{DIM}(skipping {name}: {path} does not exist){RESET}", file=sys.stderr)
            continue
        out.append((name, full, float(thr)))
    return out


def load_split(split: Path):
    convs, metas = {}, {}
    for line in (split / "conversations.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            convs[c["conversation_id"]] = c["utterances"]
    for line in (split / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            metas[m["conversation_id"]] = m
    return convs, metas


def load_scores(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["conversation_id"]] = {s["index"]: s["risk_score"] for s in r["steps"]}
    return out


def clip(text: str, limit: int) -> str:
    """A stage screen can't fit a whole paragraph. Truncation is for readability, not to save space."""
    t = " ".join(text.split())
    return t if limit <= 0 or len(t) <= limit else t[:limit].rstrip() + " …"


def bar(score: float, threshold: float, width: int) -> str:
    filled = int(round(score * width))
    cut = int(round(threshold * width))
    colour = RED if score >= threshold else GREEN
    cells = []
    for i in range(width):
        cells.append(f"{YELLOW}|{RESET}" if i == cut
                     else (f"{colour}█{RESET}" if i < filled else f"{DIM}·{RESET}"))
    return "".join(cells)


def replay(cid, convs, metas, systems, scores, *, pace, width, limit):
    utts, meta = convs[cid], metas[cid]
    gt = meta.get("ground_truth") or {}
    truth = ("scam" if gt.get("is_scam") else
             "hard negative: a genuine legitimate call that looks like a scam" if gt.get("hard_negative")
             else "ordinary legitimate call")
    onset = meta.get("earliest_detection_index")
    labelled = {p["through_index"] for p in (meta.get("prefix_labels") or [])}
    namew = max(len(n) for n, _, _ in systems)

    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}{cid}{RESET}  —  {truth}")
    print(f"{DIM}scenario family {meta['task']['scenario']['family']} · labelled detection moment "
          f"{'utterance ' + str(onset) if onset is not None else 'none: the evidence never became sufficient'}{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}\n")

    fired = {n: None for n, _, _ in systems}
    for u in utts:
        who = f"{CYAN}caller{RESET}" if u["speaker"] == "caller" else "receiver"
        print(f"  [{u['index']:>2}] {who}: {clip(u['text'], limit)}")
        if u["index"] in labelled:
            if u["index"] == onset:
                print(f"       {YELLOW}↓ label: the evidence is sufficient from this utterance on{RESET}")
            for name, _, thr in systems:
                s = scores[name].get(cid, {}).get(u["index"])
                if s is None:
                    continue
                if s >= thr and fired[name] is None:
                    fired[name] = u["index"]
                flag = f"  {RED}{BOLD}← alarm{RESET}" if fired[name] == u["index"] else ""
                print(f"       {name:<{namew}} {bar(s, thr, width)} {s:.2f}{flag}")
        print()
        if pace is None:
            try:
                input(f"{DIM}       (Enter to continue){RESET}")
                sys.stdout.write("\033[F\033[K")
            except (EOFError, KeyboardInterrupt):
                pace = 0.0
        elif pace:
            time.sleep(pace)

    print(f"  {BOLD}Result{RESET}")
    for name, _, thr in systems:
        f = fired[name]
        if onset is None:
            v = (f"{GREEN}no alarm throughout: correct{RESET}" if f is None
                 else f"{RED}false alarm at utterance {f}{RESET}")
        elif f is None:
            v = f"{RED}missed{RESET}"
        else:
            off = f - onset
            how = "right on the label" if off == 0 else f"{abs(off)} utterances " + ("early" if off < 0 else "late")
            v = f"{GREEN}alarm at utterance {f} ({how}){RESET}"
        print(f"    {name:<{namew}} {v}")
    print()
    return fired


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--runs", action="append", default=None,
                    help="path:threshold, may be given more than once; three lines by default")
    ap.add_argument("--call", action="append", default=None,
                    help="which call to replay, may be given more than once; defaults to the chosen pair")
    ap.add_argument("--pace", type=float, default=None,
                    help="seconds to wait per utterance; if omitted, press Enter to advance manually (manual is advised on stage)")
    ap.add_argument("--width", type=int, default=28)
    ap.add_argument("--truncate", type=int, default=220,
                    help="max characters shown per utterance; 0 = no truncation. 180-240 is advised on stage")
    args = ap.parse_args()

    systems = parse_runs(args.runs or DEFAULT_RUNS)
    if not systems:
        print("error: no run files found at all. Clone the full repo first, or point --runs at existing files.",
              file=sys.stderr)
        return 2

    convs, metas = load_split(ROOT / args.split)
    scores = {n: load_scores(p) for n, p, _ in systems}

    calls = args.call or list(DEFAULT_PAIR)
    missing = [c for c in calls if c not in convs]
    if missing:
        print(f"error: these conversation_ids are not in the split: {missing}", file=sys.stderr)
        return 2

    tracks = []
    for cid in calls:
        tracks.append((cid, replay(cid, convs, metas, systems, scores,
                                   pace=args.pace, width=args.width, limit=args.truncate)))

    # The closing frame: same call, same thresholds, the lines side by side. The demo's "aha" screen
    namew = max(len(n) for n, _, _ in systems)
    print(f"{BOLD}{'─' * 78}{RESET}")
    print(f"{BOLD}Same call, each system at the threshold it tuned on train{RESET}\n")
    for cid, _ in tracks:
        gt = metas[cid].get("ground_truth") or {}
        kind = "scam" if gt.get("is_scam") else "genuine call"
        print(f"  {BOLD}{kind}{RESET}  {cid}")
        labelled = [p["through_index"] for p in (metas[cid].get("prefix_labels") or [])]
        for name, _, thr in systems:
            track = scores[name].get(cid, {})
            cells = " ".join(
                (f"{RED}{track[k]:.2f}{RESET}" if track[k] >= thr else f"{GREEN}{track[k]:.2f}{RESET}")
                for k in labelled if k in track)
            print(f"    {name:<{namew}} @{thr:.2f}  {cells}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
