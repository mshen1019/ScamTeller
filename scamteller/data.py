"""Turn a dataset split into one training / scoring record per decision point.

A decision point is a labelled prefix: for a call of N utterances, the release labels
``utterances[:k+1]`` for every ``through_index`` k in ``prefix_labels`` (the opening utterance and
every receiver turn). The target is that prefix's ``evidence_established``: false before the
detection point, true from it on. The model sees the transcript so far and nothing else.

Each record is a single-turn chat: user = the rubric prompt with the transcript so far in its
``{transcript}`` slot, assistant = ``A`` (scam, evidence established) or ``B`` (not yet, or not a
scam). At serving time the score is P(A) / (P(A) + P(B)) at the first reply token.

Usage:
  python -m scamteller.data datasets/train/ds3-rubric-v3-detect07 --out data/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEAKER = {"caller": "Caller", "receiver": "Receiver"}
PLACEHOLDER = "<<PLACEHOLDER"
DEFAULT_PROMPT = Path(__file__).parent / "prompts" / "rubric_v1.txt"


def load_prompt(path: Path = DEFAULT_PROMPT) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith(PLACEHOLDER) or "{transcript}" not in text:
        raise SystemExit(f"{path}: rubric prompt is missing or has no {{transcript}} slot")
    return text


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_transcript(utterances: list[dict]) -> str:
    return "\n".join(f'{SPEAKER[u["speaker"]]}: {u["text"]}' for u in utterances)


def build_records(split_dir: Path, prompt: str) -> list[dict]:
    convs = {c["conversation_id"]: c for c in read_jsonl(split_dir / "conversations.jsonl")}
    records = []
    for meta in read_jsonl(split_dir / "metadata.jsonl"):
        cid = meta["conversation_id"]
        utts = convs[cid]["utterances"]
        for p in sorted(meta["prefix_labels"], key=lambda p: p["through_index"]):
            k = p["through_index"]
            positive = bool(p["evidence_established"])
            records.append({
                "conversation_id": cid,
                "through_index": k,
                "label": int(positive),
                "messages": [
                    {"role": "user", "content": render_prompt(prompt, utts[:k + 1])},
                    {"role": "assistant", "content": "A" if positive else "B"},
                ],
            })
    return records


def render_prompt(prompt: str, utterances: list[dict]) -> str:
    # str.replace, not str.format: the rubric text may contain literal braces
    return prompt.replace("{transcript}", render_transcript(utterances))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("split_dir", type=Path, help="e.g. datasets/eval/ds3-rubric-v3-detect07")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    args = ap.parse_args()

    records = build_records(args.split_dir, load_prompt(args.prompt))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    calls = len({r["conversation_id"] for r in records})
    pos = sum(r["label"] for r in records)
    print(f"{args.split_dir}: {calls} calls, {len(records)} decision points, {pos} positive -> {args.out}")


if __name__ == "__main__":
    main()
