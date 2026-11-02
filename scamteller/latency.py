"""Per-utterance latency: one decision point at a time (batch 1), no KV cache reuse.

Times the full forward pass that produces the verdict logits, from tokenized input on the GPU to
the score on the host, for a sample of decision points from a split. This is the offline
condition reported for ScamTeller and the untuned Qwen3-8B; a streaming server with prefix
caching would only re-encode the new utterance and be faster.

Usage:
  python -m scamteller.latency --model runs/scamteller/merged --split datasets/test/ds3-rubric-v3-detect07
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from .data import DEFAULT_PROMPT, build_records, load_prompt
from .score import last_logits, letter_ids, load_model, render_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--n", type=int, default=200, help="decision points to time")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON summary")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("latency needs a CUDA GPU")

    records = build_records(args.split, load_prompt(args.prompt))
    random.Random(args.seed).shuffle(records)
    tok, model = load_model(args.model, args.revision, args.adapter, "cuda")
    ids = letter_ids(tok)
    prompts = render_prompts(tok, records[:args.warmup + args.n])
    times_ms = []
    with torch.inference_mode():
        for i, p in enumerate(prompts):
            enc = tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            probs = torch.softmax(last_logits(model, enc["input_ids"], enc["attention_mask"]).float(), -1)
            _ = float(probs[0, ids["A"]].sum())  # host sync: the score is ready
            dt = (time.perf_counter() - t0) * 1000
            if i >= args.warmup:
                times_ms.append(dt)
    q = statistics.quantiles(times_ms, n=100)
    summary = {"n": len(times_ms), "p50_ms": round(statistics.median(times_ms), 1),
               "p95_ms": round(q[94], 1), "p99_ms": round(q[98], 1), "max_ms": round(max(times_ms), 1),
               "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(summary))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
