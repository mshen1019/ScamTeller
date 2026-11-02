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
| `ds3-eval-c2/` | `datasets/eval/ds3-rubric-v3-detect07` | 7 + ScamTeller | eval results |
| `ds3-test-c2/` | `datasets/test/ds3-rubric-v3-detect07` | 7 | test results (unseen families) |

`scamteller.jsonl` holds ScamTeller's own predictions: the fine-tuned Qwen3-8B, one forward
pass per decision point, scored at its untuned threshold 0.50 (not in `_thresholds.json`, so
`bench-table --thresholds` falls back to `--threshold 0.5` for it). `python -m scamteller.score`
writes this format directly.

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

**All seven are tuned on the same train subset and all seven ran both splits.** An earlier
version of this directory held only three tuned systems and three test runs, excluding
`claude-opus`, `gpt-sol` and `deepseek-pro` on latency. That shortcut hid the most interesting
result: at their tuned thresholds `claude-opus` and `gpt-sol` raise **zero** false alarms on
both negative classes on both splits, so what rules them out is latency alone
(`BENCH_REPORT.md` §1, §6). `jev` needs `typesafe-sdk==0.7.4`, which is not in the default
interpreter — see `BENCH_REPORT.md` §8.

One parse failure exists in the whole set: one prefix of `deepseek-pro` on test (1 of 817),
scored 0.0.

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
