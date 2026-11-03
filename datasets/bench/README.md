# Bench run files — closed-source baselines

Raw per-prefix predictions from the closed-source LLM baselines, kept so that **anyone can
recompute every table in `docs/BENCH_REPORT.md` with no GPU, no API keys and no spend.**

These are *predictions*, not data. `risk_score` and `signals` appear in `LABEL_KEYS`, so a
naive leakage scan flags them — in a dataset file those names mean annotator labels, here they
mean the model's own output. `callfactory validate` deliberately does not scan this directory.

## What is here

| directory | split | systems | purpose |
|---|---|---|---|
| `_tune-train-100/` | 100 calls sampled from **train** | — | the fixed tuning set (a strict subset of train, verified call by call) |
| `tune-c2/` | `_tune-train-100` | 7 | the runs the thresholds were chosen from, plus `_thresholds.json` |
| `ds3-eval-c2/` | `datasets/eval/ds3-rubric-v3-detect07` | 7 + ScamTeller + its base model | eval results |
| `ds3-test-c2/` | `datasets/test/ds3-rubric-v3-detect07` | 7 | test results (unseen families) — **the two local models are missing here**, see below |

`scamteller.jsonl` holds ScamTeller's own predictions: the fine-tuned Qwen3-8B, one forward
pass per decision point, scored at its untuned threshold 0.50 (not in `_thresholds.json`, so
`bench-table --thresholds` falls back to `--threshold 0.5` for it). `qwen3-8b-base.jsonl` is the
**same base model without the adapter, on the same prompt** — the control that says how much of
ScamTeller is fine-tuning rather than Qwen3-8B. `python -m scamteller.score` writes both.

**Both exist on eval only.** Their test predictions were produced on a separate GPU host and are
not committed here, so `ds3-test-c2/` has seven systems and the test rows for these two in the
project README are quoted rather than recomputable. When those two files land in `ds3-test-c2/`,
every table in the report becomes reproducible end to end; until then `BENCH_REPORT.md` §12 says
so at the point where it matters.

Neither local run file carries wall-clock latency (`latency_ms` is `null`): the forward passes
were timed separately by `python -m scamteller.latency`, batch 1 and no cache, which is **not**
the concurrency-4 regime the hosted detectors were measured in. The two are not comparable to
three significant figures.

## Provenance

```
judge version        bench-judge/v1
prompt fingerprint   c74b34d61228
condition            c2 — reasoning minimised wherever the API allows it
requests             one per labelled prefix; 912 on eval, 817 on test, 769 on the tuning set
model input          the transcript of utterances[:k+1] and nothing else (REQUIREMENTS §7.2)
```

Source files the runs are scored against, by SHA-256:

```
eval  conversations.jsonl  834f0a33b18a7482f9d07466bf3556cbf892d5e16018faf506a18b23ee66dddb
      metadata.jsonl       edb9495a4b8c8e917ab6c9e698bf87ff184efe292fbe0fa1283320d212295d3e
test  conversations.jsonl  cc36a597ba1b34e06c4fd5805b115703d5389cc176911751889a1754c2c4cf36
      metadata.jsonl       c3925d869f7a2e3fcc0558f2dfc8149ebf797ff8f9b82c5252202bbade662a35
```

The runs were produced against the same `conversations.jsonl` files. The `metadata.jsonl`
files were later rewritten to translate their free-text `notes`, `provenance.context` and
validation `message` strings into English; no key, id, label or number changed. The
metadata hashes at run time were `8a4f0658b0dc7edfbafb7626b911a1c03ef83f98966096333fc2f9bca8635d0b`
(eval) and `a08094d7f5174bb825ddb8230110c960a5d1f15e70ac28bf60cd4b9edf0098f3` (test).

| system | model id | reasoning in c2 | threshold | on the tuning set |
|---|---|---|---|---|
| `claude-haiku` | `claude-haiku-5-5` | off | 0.90 | detection 1.000, hard-neg 0.079 |
| `claude-opus` | `claude-opus-5-5` | **floor `low`** | 0.85 | detection 1.000, hard-neg 0.053 |
| `gpt-luna` | `gpt-6-luna` | off | 0.95 | detection 1.000, hard-neg 0.105 — budget not met |
| `gpt-sol` | `gpt-6.1-sol` | **floor `low`** | 0.95 | detection 1.000, hard-neg **0.000** |
| `deepseek-flash` | `deepseek-flash` | off | 0.85 | detection 1.000, hard-neg 0.026 |
| `deepseek-pro` | `deepseek-v4-pro` | off | 0.95 | detection 0.846, hard-neg 0.026 |
| `jev` | `jev-1.13.0` | n/a | 0.95 | detection 0.769, hard-neg 0.079 |
| `scamteller` | Qwen3-8B + LoRA (local) | n/a | 0.50 | **never tuned** — the tuning set is inside its training data |
| `qwen3-8b-base` | `Qwen/Qwen3-8B` @ `b968826` | n/a | 0.50 | **never tuned** — reported as the untuned control |

**All seven are tuned on the same train subset and all seven ran both splits.** An earlier
version of this directory held only three tuned systems and three test runs, excluding
`claude-opus`, `gpt-sol` and `deepseek-pro` on latency. That shortcut hid the most interesting
result: at their tuned thresholds `claude-opus` and `gpt-sol` raise **zero** false alarms on
both negative classes on both splits, so what rules them out is latency alone
(`BENCH_REPORT.md` §1, §6). `jev` needs `typesafe-sdk==0.7.4`, which is not in the default
interpreter — see `BENCH_REPORT.md` §8.

One parse failure exists in the whole set: one prefix of `deepseek-pro` on test (1 of 817),
scored 0.0.

**`_manifest.json` used to under-report.** One `bench` invocation benchmarks one system and the
runner wrote the manifest with just that name, so each run clobbered the last and a directory of
seven run files ended up claiming one. The manifests now list the directory's full contents and
the runner unions instead of overwriting (`callfactory/bench/run.py::manifest`, pinned by a
test). `created_at` is the first write; `updated_at` is the most recent.

## A run file

One JSON object per conversation, one `steps` entry per labelled prefix:

```json
{"conversation_id": "...", "alarm_index": 7, "alarm_round": 4,
 "steps": [{"index": 0, "risk_score": 0.1, "is_alarm": false, "signals": [],
            "rationale": "...", "latency_ms": 980.0, "fresh": true, "fired": false}]}
```

`risk_score` is the evidence. `is_alarm` is the **vendor's own** operating point, reported as
its own row and never mixed into ours. `fired` and `alarm_index` were written at the run-time
threshold and are **not** authoritative: judgement is recomputed at scoring time, which is why
re-sweeping a threshold costs nothing. (Conflating the two was a real bug — `docs/BENCH_REPORT.md` §9.)

## Recomputing the tables

```bash
# thresholds, from the tuning runs — zero model calls
python -m callfactory bench-tune datasets/bench/_tune-train-100 --runs datasets/bench/tune-c2

# any table, at any threshold — zero model calls
python -m callfactory bench-table datasets/eval/ds3-rubric-v3-detect07 \
    --runs datasets/bench/ds3-eval-c2 --threshold 0.85
python -m callfactory bench-table datasets/test/ds3-rubric-v3-detect07 \
    --runs datasets/bench/ds3-test-c2 --threshold 0.85
```

`bench-table` takes one threshold for all systems, so the per-system tables in the report are
assembled by scoring each system at its own threshold.
