"""Score every decision point of a split and write a bench run file.

One forward pass per decision point, nothing generated: the prompt ends at the chat template's
generation prompt (enable_thinking=False), so the next token is the verdict letter. P(A) and
P(B) are summed over every single-token spelling of each letter (bare, leading space, newline).

  risk_score = P(A) / (P(A) + P(B))      A = scam (evidence established), B = not yet / not a scam

The output is in the same format as the hosted-detector runs in datasets/bench/, so
`python -m callfactory bench-table` scores it alongside them.

Usage:
  python -m scamteller.score --model runs/scamteller/merged \
      --split datasets/test/ds3-rubric-v3-detect07 --out datasets/bench/ds3-test-c2/scamteller.jsonl
  # untuned base, same prompt:
  python -m scamteller.score --model Qwen/Qwen3-8B --revision b968826 \
      --split datasets/test/ds3-rubric-v3-detect07 --out datasets/bench/ds3-test-c2/qwen3-8b-base.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import DEFAULT_PROMPT, build_records, load_prompt

REF_THRESHOLD = 0.5  # written into alarm_index only; bench-table re-applies each system's threshold


def load_model(name, revision=None, adapter=None, device="cuda"):
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    tok.padding_side = "left"  # the verdict position is then -1 for every row of a batch
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return tok, model.to(device).eval()


def letter_ids(tok):
    ids = {}
    for letter in "AB":
        ids[letter] = sorted({enc[0] for v in (letter, " " + letter, "\n" + letter)
                              if len(enc := tok.encode(v, add_special_tokens=False)) == 1})
    return ids


def last_logits(model, input_ids, attention_mask):
    """Logits at the final position only; full-sequence logits do not fit on a 24 GB GPU."""
    for kw in ("logits_to_keep", "num_logits_to_keep"):
        try:
            return model(input_ids=input_ids, attention_mask=attention_mask, **{kw: 1}).logits[:, -1, :]
        except TypeError:
            continue
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]


def render_prompts(tok, records):
    return [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False) for r in records]


def to_runs(records, scores):
    by_call: dict[str, list[dict]] = {}
    for r, s in zip(records, scores):
        by_call.setdefault(r["conversation_id"], []).append({
            "index": r["through_index"], "risk_score": round(s, 6), "is_alarm": s >= REF_THRESHOLD,
            "signals": [], "rationale": "", "latency_ms": None, "fresh": True,
            "fired": s >= REF_THRESHOLD})
    runs = []
    for cid in sorted(by_call):
        steps = sorted(by_call[cid], key=lambda s: s["index"])
        alarm = next((s["index"] for s in steps if s["fired"]), None)
        runs.append({"conversation_id": cid, "alarm_index": alarm,
                     "alarm_round": None if alarm is None else alarm // 2 + 1, "steps": steps})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local dir (merged model, or base with --adapter)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    records = build_records(args.split, load_prompt(args.prompt))
    tok, model = load_model(args.model, args.revision, args.adapter, args.device)
    ids = letter_ids(tok)
    prompts = render_prompts(tok, records)
    order = sorted(range(len(records)), key=lambda i: len(prompts[i]))  # less padding per batch
    scores, t0 = [0.0] * len(records), time.time()
    with torch.inference_mode():
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(args.device)
            probs = torch.softmax(last_logits(model, enc["input_ids"], enc["attention_mask"]).float(), -1)
            for row, i in enumerate(idx):
                pa, pb = float(probs[row, ids["A"]].sum()), float(probs[row, ids["B"]].sum())
                scores[i] = pa / (pa + pb) if pa + pb > 0 else 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for run in to_runs(records, scores):
            f.write(json.dumps(run) + "\n")
    print(f"[done] {len(records)} decision points in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
