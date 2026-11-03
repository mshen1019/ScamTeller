# Benchmark Report: ScamTeller, Its Base Model, and Seven Hosted Detectors

**Data:** `datasets/eval/ds3-rubric-v3-detect07` (119 calls, 912 labelled prefixes) and
`datasets/test/ds3-rubric-v3-detect07` (104 calls, 817 prefixes, six families held out)
**Harness:** `callfactory.bench`, `bench-judge/v1`, prompt fingerprint `c74b34d61228`
**Run:** condition C2 (reasoning minimised), 2026-11-01/02 · **total spend ≈ $30**

Nine systems: **ScamTeller** (the fine-tuned Qwen3-8B this repository is named after), the
same base model **without** fine-tuning and on the same prompt, and seven hosted detectors
including three flagship models. It is not a leaderboard: there is no single "bench score"
anywhere in this document, by design (§2).

**Read this caveat before the tables.** The seven hosted detectors ran on **both** splits.
ScamTeller and its base model have run files in this repository for **eval only** — their test
predictions were produced on a separate GPU host and are not yet committed here, so the test
numbers quoted in the project README cannot be recomputed from this repository. Every number in
*this* document is recomputable from `datasets/bench/` with no GPU, no API keys and no spend
(§11); where a figure is not, it is marked.

---

## 1. The headline

**At the operating point every vendor ships, all seven systems score below the "never alarm"
baseline on prefix accuracy** — 0.371 to 0.773 against 0.837. With one threshold tuned on a
held-out subset of **train**, six of the seven are good, and two of them never raise a single
false alarm on either split.

| | threshold 0.5 (untuned) | tuned on train |
|---|---|---|
| prefix accuracy | 0.371 – 0.773 | **0.892 – 0.939** |
| hard-negative false alarms | 0.263 – 0.974 | **0.000 – 0.171** |
| detection-point pair accuracy | 0.02 – 0.12 | **0.27 – 0.59** |
| median latency vs. the label | 4–6 utterances early | **0 to −2** |

**Same requests, same model outputs, same scores on disk. One number changed.**

Three consequences, in order of how easily they are got wrong:

1. **A detector for this task is a score *plus* an operating point.** "Model X gets 0.50
   prefix accuracy" is not a statement about model X. Quote the threshold with every number.
2. **The threshold is tuned on train, never on eval or test**, under an objective frozen in
   code before any result was read. `bench-tune` exits 2 on a path containing `/eval/` or
   `/test/`; the guard is in the CLI because a document does not stop a command typed in haste.
3. **Cost does not buy quality here.** Within one vendor's family the cheap model beats the
   expensive one by a wide margin: `deepseek-flash` 0.928 balanced accuracy on test against
   `deepseek-pro` 0.724, at a twentieth of the price.

**What separates the systems in the end is latency, not judgement.** `claude-opus` is the best
system on eval (balanced accuracy 0.936) and detects every detectable scam on test with zero
false alarms — and it is unusable for this product at 1995 ms median / 6664 ms p99, because
its reasoning cannot be switched off. That is a far stronger finding than "the flagships were
excluded": they were excluded *despite* being excellent.

---

## 2. Why there is no single score

The task is *when*, not *what*. Two degenerate strategies win any single metric:

* **Alarm on utterance 0 of every call.** Perfect detection rate, zero latency.
* **Never alarm.** Zero false alarms — and **0.837 prefix accuracy** on eval, because only
  149 of 912 prefixes are positive (0.823 on test, 145 of 817).

So every row below carries detection rate *and* false-alarm rate *and* signed latency, and the
`always_no` baseline is printed in each table so you can see at a glance whether a system beat
doing nothing. Any summary of this report that collapses the columns into one number has
thrown away the only thing the eval measures.

---

## 3. What was measured

### 3.1 Systems

| name | model id | reasoning in C2 | threshold | chosen on train |
|---|---|---|---|---|
| `claude-haiku` | `claude-haiku-5-5` | off (`thinking.disabled`) | 0.90 | detection 1.000, hard-neg 0.079 |
| `claude-opus` | `claude-opus-5-5` | **cannot disable** — floor `effort=low` | 0.85 | detection 1.000, hard-neg 0.053 |
| `gpt-luna` | `gpt-6-luna` | off (`reasoning_effort=none`) | 0.95 | detection 1.000, hard-neg 0.105 — **budget missed** |
| `gpt-sol` | `gpt-6.1-sol` | **cannot disable** — floor `reasoning_effort=low` | 0.95 | detection 1.000, hard-neg **0.000** |
| `deepseek-flash` | `deepseek-flash` | off (`thinking.disabled`) | 0.85 | detection 1.000, hard-neg 0.026 |
| `deepseek-pro` | `deepseek-v4-pro` | off (`thinking.disabled`) | 0.95 | detection 0.846, hard-neg 0.026 |
| `jev` | `jev-1.13.0` | n/a (not an LLM) | 0.95 | detection 0.769, hard-neg 0.079 |
| `scamteller` | Qwen3-8B + LoRA, local | n/a (one forward pass) | 0.50 | **not tuned** — see Table E |
| `qwen3-8b-base` | `Qwen/Qwen3-8B` @ `b968826` | n/a (one forward pass) | 0.50 | **not tuned** — see Table E |

**Why the two local models are not tuned.** The tuning set `_tune-train-100` is a subset of
**train**, which is ScamTeller's own training data. Choosing its threshold there would be
choosing a threshold on data it has memorised, so both local rows are reported at a naive 0.50
and the comparison is read against Table A, where the hosted systems are also at 0.50.

**Why C2.** The product is a live call: a verdict that arrives after the victim has already
given out the information is worth nothing, so reasoning is switched off wherever the API
allows it. `claude-opus` and `gpt-sol` have a non-zero floor and were run at that floor.

**All seven are tuned and all seven ran both splits.** An earlier version of this report tuned
only three and ran only three on test, excluding `claude-opus`, `gpt-sol` and `deepseek-pro`
on latency. That shortcut hid the most interesting result — see §1 — and has been removed.

### 3.2 What each system sees

One request per labelled prefix: the system prompt, the transcript of `utterances[:k+1]`, then
the question. Nothing else — **no labels, no rubric, no anchor list, no metadata, and no memory
of the previous prefix of the same call** (`load_split` reads `conversations.jsonl` only;
REQUIREMENTS §7.2). The judge prompt deliberately omits the rubric's anchor list: handing the
model the rubric would measure how well it applies our rules, not whether it can spot a scam.

It returns `{risk_score, alarm, signals, rationale}` (for Jev, a `noul` probability).
912 requests per system on eval, 817 on test, 769 on the tuning set. **Parse failures: zero
everywhere except one prefix of `deepseek-pro` on test** (1 of 817, scored 0.0).

### 3.3 The two splits

| | eval | test |
|---|---|---|
| calls | 119 | 104 |
| labelled prefixes | 912 | 817 |
| positive prefixes | 149 (16.3%) | 145 (17.7%) |
| scam calls | 62 | 52 |
| — with a ground-truth detection point | 41 ← detection denominator | 44 |
| — scam, no detection point | 21 (16 flagged `no_observable_evidence`) | 0 |
| hard negatives | 38 | 35 |
| plain legitimate calls | 19 | 17 |
| scenario families | 51 | **6, disjoint from train** |
| `always_no` prefix accuracy | 0.8366 | 0.8225 |

`test` holds out **whole families**, so it measures generalisation to unseen *kinds* of call.
`eval` holds out conversations from families train also contains — with one exception:
`reconnaissance_detail_confirmation` (17 calls) appears only in eval, and is also the only
slice whose labels nobody has adjudicated. Those two facts land on the same 17 calls, which is
why that slice is reported apart and never folded into a headline (§7).

### 3.4 The tuning set

100 calls sampled from **train** (`datasets/bench/_tune-train-100`), 769 prefixes, 38 hard
negatives, 39 scam calls with a detection point. **Intersection with eval and test: 0 calls**,
verified rather than assumed. Committed to the repo so that everyone tunes on identical data —
thresholds chosen on different subsets are not comparable.

The objective was fixed in code *before* any result was read (`callfactory/bench/tune.py`):

> Maximise detection rate subject to `hard_negative_false_alarm_rate <= 0.10` **and
> `detection_rate > 0`**; ties go to the larger threshold.

The `detection_rate > 0` clause is not decoration. The first version picked threshold 0.95 on
a synthetic case with detection 0 and false alarms 0 — it satisfied the budget by **never
firing**. A test caught it. 0.10 is a chosen product tolerance, not a measurement; changing it
changes the operating point and must be stated.

---

## 4. Results

### Table A — threshold 0.5, untuned (eval)

| system | prefix acc | balanced acc | PR-AUC | detection | hard-neg FA | plain FA | latency | pair |
|---|---|---|---|---|---|---|---|---|
| *(never alarm)* | 0.837 | 0.500 | — | 0.000 | 0.000 | 0.000 | — | — |
| gpt-sol | 0.773 | 0.862 | 0.908 | 1.000 | 0.263 | 0.210 | −4.0 | 0.10 |
| claude-opus | 0.742 | 0.846 | 0.924 | 1.000 | 0.316 | 0.053 | −4.0 | 0.12 |
| gpt-luna | 0.722 | 0.834 | 0.801 | 1.000 | 0.395 | 0.316 | −5.0 | 0.10 |
| deepseek-flash | 0.662 | 0.798 | 0.930 | 1.000 | 0.605 | 0.632 | −4.0 | 0.12 |
| deepseek-pro | 0.581 | 0.750 | 0.763 | 1.000 | 0.921 | 0.789 | −6.0 | 0.05 |
| claude-haiku | 0.503 | 0.703 | 0.887 | 1.000 | 0.895 | 0.789 | −5.0 | 0.05 |
| jev | 0.371 | 0.624 | 0.867 | 1.000 | 0.974 | 0.895 | −6.0 | 0.02 |

Every system: detection 1.000, prefix accuracy **below** the 0.837 do-nothing baseline,
latency 4–6 utterances early. This is the "alarm on everything" corner. Note it is *not* a
failure of the models — it is what a 0.5 cut on a roughly-calibrated risk score does on a task
whose base rate is 16%.

### Table B — eval, threshold tuned on train (applied once)

| system | thr | prefix acc | balanced acc | PR-AUC | detection | hard-neg FA | plain FA | latency | pair |
|---|---|---|---|---|---|---|---|---|---|
| **claude-opus** | 0.85 | 0.916 | **0.936** | 0.924 | 0.951 | **0.000** | **0.000** | −2.0 | 0.29 |
| **deepseek-flash** | 0.85 | 0.934 | 0.931 | **0.930** | **0.976** | 0.026 | 0.053 | 0.0 | **0.49** |
| gpt-luna | 0.95 | 0.883 | 0.924 | 0.801 | 0.976 | 0.053 | 0.105 | −2.0 | 0.41 |
| **gpt-sol** | 0.95 | 0.931 | 0.924 | 0.908 | 0.927 | **0.000** | **0.000** | 0.0 | 0.39 |
| claude-haiku | 0.90 | **0.939** | 0.917 | 0.887 | 0.927 | 0.079 | 0.053 | 0.0 | 0.46 |
| jev | 0.95 | 0.923 | 0.870 | 0.867 | 0.805 | 0.053 | 0.053 | 0.0 | 0.46 |
| deepseek-pro | 0.95 | 0.897 | 0.731 | 0.763 | 0.805 | 0.026 | 0.000 | 2.0 | 0.27 |

### Table C — test, same thresholds (104 calls, six unseen families)

| system | thr | prefix acc | balanced acc | PR-AUC | detection | hard-neg FA | plain FA | latency | pair |
|---|---|---|---|---|---|---|---|---|---|
| *(never alarm)* | — | 0.823 | 0.500 | — | 0.000 | 0.000 | 0.000 | — | — |
| **deepseek-flash** | 0.85 | **0.939** | **0.928** | 0.898 | 0.977 | **0.000** | 0.118 | 0.0 | 0.55 |
| gpt-luna | 0.95 | 0.895 | 0.922 | 0.789 | **1.000** | 0.057 | 0.176 | −1.0 | 0.41 |
| **claude-opus** | 0.85 | 0.909 | 0.921 | 0.893 | **1.000** | **0.000** | **0.000** | −2.0 | 0.34 |
| **gpt-sol** | 0.95 | 0.931 | 0.912 | 0.888 | 0.932 | **0.000** | **0.000** | 0.0 | 0.36 |
| claude-haiku | 0.90 | 0.928 | 0.905 | 0.832 | 0.977 | 0.171 | 0.000 | 0.0 | **0.59** |
| jev | 0.95 | 0.901 | 0.815 | 0.806 | 0.795 | 0.143 | 0.118 | 0.0 | 0.34 |
| deepseek-pro | 0.95 | 0.892 | 0.724 | 0.784 | 0.636 | 0.057 | 0.059 | 1.0 | 0.30 |

**`claude-opus` and `gpt-sol` are the only systems with zero false alarms on both negative
classes, on both splits** — 0/38 and 0/19 on eval, 0/35 and 0/17 on test. `claude-opus`
additionally detects **every** detectable scam on test (44/44).

### Table E — ScamTeller and its base model (eval, both at an untuned 0.50)

Same split, same 912 prefixes, same scoring code as every row above.

| system | thr | prefix acc | balanced acc | PR-AUC | detection | hard-neg FA | plain FA | latency | pair |
|---|---|---|---|---|---|---|---|---|---|
| *(never alarm)* | — | 0.837 | 0.500 | — | 0.000 | 0.000 | 0.000 | — | — |
| **ScamTeller** (Qwen3-8B + LoRA) | 0.50 | **0.962** | **0.950** | **0.971** | 0.976 | **0.000** | **0.000** | 0.0 | **0.63** |
| Qwen3-8B, not fine-tuned, same prompt | 0.50 | 0.868 | 0.886 | 0.826 | 0.976 | 0.342 | 0.105 | 0.0 | 0.44 |

**What fine-tuning bought is not recall.** Detection is **0.976 for both** — 40 of 41. The
difference is entirely on the negative side: hard-negative false alarms **0.342 → 0.000** and
plain false alarms **0.105 → 0.000**. The base model alarms on a third of the legitimate calls
built to sound alarming; the fine-tune alarms on none of them, 0 of 38 and 0 of 19. Anyone
reading only a detection column would conclude fine-tuning did nothing.

Three more things worth separating out:

1. **PR-AUC +0.146** (0.8255 → 0.9713) is threshold-free, so it is not an artefact of both rows
   sitting at 0.50.
2. **`pair` 0.63 is the highest figure in this document**, above the 0.59 bar §11 sets as "the
   result worth writing up", and it is the one column a threshold cannot buy (§11). The base
   model is at 0.44.
3. **0.962 prefix accuracy at a naive 0.50** is above every *tuned* hosted detector in Table B
   (0.883–0.939). Table A shows what the same untuned 0.50 does to those systems: 0.371–0.773,
   all of them below the do-nothing baseline.

**Not in this table, and why.** Wall-clock latency: these run files carry `latency_ms: null`
because the forward passes were timed separately by `python -m scamteller.latency`, not during
scoring. The ~70 ms figure quoted in the README comes from that measurement on the GPU host and
is **not** recomputable here (§12). The `latency` column above is the signed distance in
utterances between the alarm and the labelled detection point, as everywhere else in this
document — both local models land exactly on the label.

### Table D — the vendor's own operating point (the model's `alarm` boolean, eval)

| system | prefix acc | balanced acc | detection | hard-neg FA | plain FA | latency | pair |
|---|---|---|---|---|---|---|---|
| claude-haiku | 0.622 | 0.774 | 1.000 | 0.816 | 0.789 | −4.0 | 0.17 |
| claude-opus | 0.743 | 0.847 | 1.000 | 0.316 | 0.053 | −4.0 | 0.12 |
| deepseek-flash | 0.767 | 0.861 | 1.000 | 0.289 | 0.263 | −4.0 | 0.20 |
| deepseek-pro | 0.589 | 0.754 | 1.000 | 0.895 | 0.789 | −6.0 | 0.05 |
| gpt-luna | 0.722 | 0.834 | 1.000 | 0.395 | 0.316 | −5.0 | 0.10 |
| gpt-sol | 0.773 | 0.862 | 1.000 | 0.263 | 0.210 | −4.0 | 0.10 |

Every model's own "should I alarm?" boolean is far too eager for this product: 26%–82% of hard
negatives trigger it. This is the quantitative answer to "can we just use the model's own alarm
flag?" — no. It is reported as its own row and never mixed into our decision (§9). Jev is not
in this table: its "own boolean" is simply probability >= 0.5, which is its row in Table A.

### Tuning curves on train (why those thresholds)

Detection rate / hard-negative false-alarm rate on the tuning set; `←` marks the chosen point.

```
                     thr 0.50       thr 0.70       thr 0.85       thr 0.90       thr 0.95
claude-haiku      1.000/0.921    1.000/0.658    1.000/0.210    1.000/0.079←   0.564/0.000
claude-opus       1.000/0.289    1.000/0.132    1.000/0.053←   0.949/0.000    0.821/0.000
deepseek-flash    1.000/0.632    1.000/0.289    1.000/0.026←   0.949/0.026    0.590/0.000
deepseek-pro      1.000/0.895    1.000/0.895    1.000/0.342    1.000/0.237    0.846/0.026←
gpt-luna          1.000/0.500    1.000/0.421    1.000/0.237    1.000/0.184    1.000/0.105←
gpt-sol           1.000/0.237    1.000/0.237    1.000/0.079    1.000/0.000    1.000/0.000←
jev               1.000/0.974    1.000/0.868    1.000/0.447    1.000/0.316    0.769/0.079←
```

`gpt-sol` is the only system that holds detection 1.000 all the way to a zero false-alarm rate;
the tuner chose 0.95 between the equally good 0.90 and 0.95 because ties go to the larger threshold.
`claude-opus` and `deepseek-flash` also reach the budget while detection is still 1.000.
`deepseek-pro` and `jev` only reach it by giving up roughly 20% of detection, and `gpt-luna`
never reaches it at all.

---

## 5. Difficulty calibration: eval vs. test

None of these systems trained on anything, so the gap between their eval and test scores is an
estimate of **how much harder one split is than the other** — a control that makes any trained
model's drop interpretable.

| system | eval balanced acc | test balanced acc | Δ | eval detection | test detection | Δ |
|---|---|---|---|---|---|---|
| gpt-luna | 0.924 | 0.922 | −0.002 | 0.976 | 1.000 | +0.024 |
| deepseek-flash | 0.931 | 0.928 | −0.003 | 0.976 | 0.977 | +0.002 |
| deepseek-pro | 0.731 | 0.724 | −0.007 | 0.805 | 0.636 | −0.168 |
| gpt-sol | 0.924 | 0.912 | −0.011 | 0.927 | 0.932 | +0.005 |
| claude-haiku | 0.917 | 0.905 | −0.013 | 0.927 | 0.977 | +0.050 |
| claude-opus | 0.936 | 0.921 | −0.015 | 0.951 | 1.000 | +0.049 |
| **jev** | 0.870 | 0.815 | **−0.055** | 0.805 | 0.795 | −0.009 |

**Six of the seven lose at most 1.5 points of balanced accuracy, and four of them *detect more*
on test.** So the two splits are comparably hard, and a trained model that drops materially
more than ~1.5 points from eval to test is losing it to **family generalisation**, not to test
being harder. That is the main reason this table exists.

**Jev is the outlier at −0.055**, and §8 explains why: its threshold does not transfer.

---

## 6. Latency and cost

| system | median ms | p95 | p99 | max | full eval pass | reasoning off? |
|---|---|---|---|---|---|---|
| **jev** | **90** | 161 | 232 | 464 | (token-metered) | n/a |
| claude-haiku | 1102 | 1397 | 2099 | 3713 | $0.18 | ✅ |
| deepseek-flash | 1109 | 1421 | 1608 | 3143 | $0.19 | ✅ |
| gpt-luna | 1429 | 1802 | 2271 | 4107 | $0.14 | ✅ |
| claude-opus | 1995 | 5526 | 6664 | 8441 | $7.23 | ❌ floor `low` |
| deepseek-pro | 2006 | 2641 | 3064 | 5268 | $0.39 | ✅ |
| gpt-sol | 2396 | 3368 | 4196 | 10450 | $2.79 | ❌ floor `low` |

Measured at concurrency 4, so these are **upper bounds** for a single-call deployment.

**This table, not the accuracy tables, is what rules systems out.** `claude-opus` is the best
judge in Table B and still unusable: a 6.7-second p99 on a live call is not a detector, and its
reasoning floor means that number cannot be brought down. `gpt-sol`'s 10.4-second worst case is
the same problem. `jev` is in a different class entirely — a hosted API at 90 ms median, within
a factor of 1.3 of a local 8B forward pass.

**Cost caveat — O(N²).** A stateless judge wrapped by `ReplayAdapter` re-reads the whole prefix
every turn, so token cost grows quadratically in call length (REQUIREMENTS §7.1). Jev pays this
too: its SDK offers only `system_one(state, questions)`, with no session and no streaming
interface, correcting an earlier claim in REQUIREMENTS §6 that it was natively streaming. Cache
hit rates measured: Anthropic 36%, DeepSeek 43%, **OpenAI 0%** (it races its own cache write at
this request rate), Jev 0% (no prefix cache). A genuinely streaming detector does not pay this.

---

## 7. Where the remaining false positives are

Prefix-level false positives at each system's own threshold, by the kind of call they came from:

**eval**

| system | thr | prefix FP | unadjudicated slice | hard-neg | plain | **early on a real scam** |
|---|---|---|---|---|---|---|
| deepseek-pro | 0.95 | 17 | 4 | 1 | 0 | 12 |
| claude-haiku | 0.90 | 39 | 1 | 5 | 2 | 31 |
| jev | 0.95 | 39 | 17 | 7 | 1 | 14 |
| deepseek-flash | 0.85 | 49 | 13 | 1 | 1 | 34 |
| gpt-sol | 0.95 | 50 | 2 | **0** | **0** | 48 |
| claude-opus | 0.85 | 72 | 7 | **0** | **0** | 65 |
| gpt-luna | 0.95 | 105 | 34 | 6 | 2 | 63 |

**test**

| system | thr | prefix FP | hard-neg | plain | **early on a real scam** |
|---|---|---|---|---|---|
| deepseek-pro | 0.95 | 10 | 2 | 1 | 7 |
| jev | 0.95 | 35 | 12 | 6 | 17 |
| deepseek-flash | 0.85 | 37 | **0** | 3 | 34 |
| gpt-sol | 0.95 | 39 | **0** | **0** | **39 — all of them** |
| claude-haiku | 0.90 | 40 | 9 | 0 | 31 |
| claude-opus | 0.85 | 65 | **0** | **0** | **65 — all of them** |
| gpt-luna | 0.95 | 81 | 3 | 7 | 71 |

Three readings:

**(a) A larger prefix-FP count is not a worse system.** `claude-opus` has the most prefix FPs on
test (65) and **every one of them is an early alarm on a genuine scam**. In product terms it is
error-free on negatives; the prefix metric counts it as wrong only because the label has not
flipped yet. Firing 2–10 utterances early on a real scam is the behaviour we want.

**(b) Two different denominators, deliberately.** The reported `hard-neg FA` column does **not**
include the 21 eval scam calls with no ground-truth detection point — their ground truth is
`None`, so latency is undefined and they do not belong in the hard-negative denominator
(REQUIREMENTS §7.8). They are reported in their own column. They *do* count in prefix accuracy
and balanced accuracy, because every one of their prefixes has `evidence_established = false`.
That is the only channel through which the unadjudicated slice moves Table B. `test` has none
of them, which makes it the cleaner measuring surface.

**(c) Tuning cut false positives everywhere.** For the three systems measured both ways, prefix
FP fell 453→39, 308→49 and 254→105. Nothing went up.

---

## 8. Jev

Jev is not an LLM and behaves unlike one. Its `system_one(state, questions)` returns a `noul`
probability, which goes through the same scoring code as everyone else's `risk_score`, with its
threshold tuned on the same train subset under the same objective.

**It wins decisively on latency:** 90 ms median, 232 ms p99 — twelve times faster than the
fastest LLM here and within 1.3× of a local 8B forward pass. For a live-call product that
matters more than two points of balanced accuracy.

**But its scores do not separate, and its threshold does not transfer.**

```
hard-negative false-alarm rate, at its one usable threshold 0.95
  train  0.079     ← chosen here, inside the 0.10 budget
  eval   0.053     ← fine
  test   0.143     ← budget blown
```

The usable-threshold band tells the same story. Width of the interval where hard-negative false
alarms stay ≤ 0.10 **and** detection stays > 0, measured on eval:

```
scamteller (the fine-tune)     [0.05, 0.95]   width 0.90   detection 1.000 inside
deepseek-flash                  [0.85, 0.95]   width 0.10   detection 0.976
jev                             [0.95, 0.95]   width 0.00   detection 0.805
```

**Jev has exactly one usable threshold, and reaching it costs 20% of detection.** Its PR-AUC
(0.867 eval, 0.806 test) says the same thing from another direction: the curve itself sits
lower, so this is not a mis-chosen operating point.

**A bug worth recording.** `JevJudge` had been written but never executed, so it passed
`questions` as a list while SDK 0.7.4's `normalize_questions` calls `questions.items()` — a
`Mapping[str, Question]`. The first real call raised `AttributeError`. Code that has never run
is not implemented, however finished it looks. The fix also replaced a character-count token
estimate with the server's reported `usage`.

---

## 9. Three bug postmortems

All three are the same family: **a value or a branch that silently stands in for a real one.**

**(i) The threshold short-circuit.** The decision had been written
`fired = verdict.is_alarm or verdict.risk_score >= threshold`. The model's own boolean
short-circuited the threshold: raising it from 0.5 to 0.95 moved the false-alarm rate only
0.92 → 0.82, because everything the model called an alarm bypassed the cut. This was the
**second** instance of the class in this project — the label side had an inert 0.5 threshold
that `sufficient_evidence` short-circuited the same way (`EVAL_SYSTEM.md` §5.9). Fixed in
`protocol.py`, `score.py` and `tune.py`: the decision is score-only and is computed **at
scoring time**, so run files hold raw evidence and re-sweeping a threshold costs nothing.

**(ii) `x or 1.0` on a legitimate zero.** Measuring the usable-threshold band, a fallback wrote
`hard_negative_false_alarm_rate or 1.0`. **`0.0` is falsy**, so every threshold with a *genuine
zero* false-alarm rate was treated as 1.0 and excluded — swallowing the entire 0.10–0.95 plateau
of the best system and leaving a single point. The same idiom was latent in `tune.py`'s fallback
branch and now tests `is None` explicitly.

**(iii) Never-executed code.** §8. Plus, while building the demo replay viewer, a two-level
lookup written as one level (`scores[name]` is keyed by conversation id, not utterance index)
printed no score rows at all, while the summary frame — which had the lookup right — displayed
normally. The contradiction between the two is what exposed it.

The symptom to watch for in your own harness: **a threshold that barely changes the false-alarm
rate is not a robust model, it is an inert knob.**

---

## 10. Slices, and the 16 calls nobody has adjudicated

| system | compliance (n=7) | late-onset (n=11) | alarms on the unadjudicated slice (n=16) | balanced acc excluding that slice |
|---|---|---|---|---|
| claude-opus | 1.000 | 1.000 | 5/16 | 0.936 |
| deepseek-flash | 0.857 | 1.000 | 7/16 | 0.935 |
| gpt-sol | 1.000 | 1.000 | 2/16 | 0.922 |
| gpt-luna | 1.000 | 1.000 | 12/16 | 0.941 |
| claude-haiku | 1.000 | 1.000 | 1/16 | 0.916 |
| jev | 0.714 | 0.909 | 6/16 | 0.881 |
| deepseek-pro | 0.857 | 0.909 | 4/16 | 0.733 |

**Compliance** (the receiver cooperates — no refusal, no suspicion in the transcript) is the
slice the *rubric* is weakest on (0.741 label-side). Five of seven systems are better on it than
the labels are. Small n, so do not over-read it.

**Late onset** (the scam signal appears only in the last third of the call): 11/11 for the five
LLMs that clear the latency bar; `jev` and `deepseek-pro` each miss one.

**The unadjudicated slice spreads the systems from 1/16 to 12/16** — at threshold 0.5 all of
them alarmed on all 16, so tuning turned this slice from uninformative into the widest spread in
the eval. That is exactly why its correct answer matters, and exactly what the human round
failed to settle.

**The unadjudicated slice** is 16 eval calls that are scams by construction but where the
annotator found no observable evidence in the transcript. Nobody has decided whether they
*should* be detectable. They stay in the denominators and are reported apart — the precedent
being avoided is a public dataset retraction where labels turned out not to be derivable from
what the model sees.

**A first human round has now made this slice harder, not easier.** Two annotators labelled
25 eval calls independently, four of them from this family. They disagreed on three of the four
— including two flat contradictions (one said scam, the other said not a scam). On the
verdict overall they agreed on 16 of 21 shared calls, but on the *detection moment* they agreed
on only 3 of 14 (median gap 3 utterances, one-sided: annotator B was never earlier than C). One
annotator's detection points sit a median of 0 utterances from ours; the other's a median of 4
earlier. **The timing dimension — the one this whole dataset exists to provide — is where humans
agree least**, and no written anchor rules were given in that round. Both annotators also called
`genuine_tax_office_letter_followup`, a hard negative, a scam.

So: a model that alarms on this slice is not demonstrably wrong, and a model that stays quiet is
not demonstrably right. Report it, never fold it in.

---

## 11. Running your own model through the same table

Two rules, both there to keep the comparison fair rather than to be bureaucratic:

1. **Tune on train, never on eval or test.** Use `datasets/bench/_tune-train-100`, which is in
   the repo so everyone tunes on the same data.
2. **Use the same objective.** It is in `callfactory/bench/tune.py` and is not a parameter. If
   you change the hard-negative budget, say so next to every number you quote.

```bash
# 1. Add an adapter: subclass BaseJudge in callfactory/bench/adapters.py,
#    implement judge(call_id, prefix) -> Verdict, register it in JUDGES.
#    If your model is genuinely streaming, implement EarlyWarningDetector
#    directly and skip ReplayAdapter — you avoid the O(N²) token cost.

# 2. Pick a threshold on train. --cond c2 = reasoning minimised.
callfactory bench datasets/bench/_tune-train-100 --system my-model --cond c2 \
    --out datasets/bench/tune-mine/my-model.c2.jsonl
callfactory bench-tune datasets/bench/_tune-train-100 --runs datasets/bench/tune-mine

# 3. Run eval, and test ONCE each, at that threshold.
callfactory bench datasets/eval/ds3-rubric-v3-detect07 --system my-model --cond c2 \
    --out datasets/bench/ds3-eval-mine/my-model.c2.jsonl

# 4. Table. Re-sweeping thresholds is free — judgement is computed at scoring
#    time; run files hold raw scores only.
callfactory bench-table datasets/eval/ds3-rubric-v3-detect07 \
    --runs datasets/bench/ds3-eval-mine --threshold 0.85
```

`bench-tune` **refuses a path containing `/eval/` or `/test/`** and exits 2.

**The columns to beat**, in the order they matter for this product:

1. **median latency ≤ 1100 ms and p99 ≤ 1700 ms.** This is what excluded the two best judges,
   so clear it first or nothing else counts.
2. `hard-neg FA` and `plain FA` — **0.000 on both splits** is already achieved by `claude-opus`
   and `gpt-sol`. Zero is the bar, not 0.026.
3. `detection` ≥ 0.977 on test (`claude-opus` reaches 1.000).
4. `pair` > 0.59 — **the one that cannot be bought with a threshold.** The whole table sits at
   0.27–0.59. Tuning lifted it from 0.02–0.12 and then it stalls; roughly half the time these
   systems still cannot separate the prefix *before* the detection point from the prefix *at* it.
   A model clearly above 0.59 here has learned something no threshold will give you, and that is
   the result worth writing up.
5. prefix accuracy ≥ 0.939, and above all ≥ 0.837 / 0.823 (the do-nothing baselines).

---

## 12. Known limitations

* **ScamTeller and its base model have no test run files in this repository.** Table E is eval
  only. Their test predictions exist on a separate GPU host; until
  `datasets/bench/ds3-test-c2/scamteller.jsonl` and `qwen3-8b-base.jsonl` are committed, the
  test figures in the project README are quoted from that host and cannot be recomputed here.
  Treat Table C as the complete, reproducible test picture and Table E as an eval-only result.
* **The two local models' wall-clock latency is not reproducible here** either, for the same
  reason: it was measured by `scamteller.latency` on the GPU host, batch 1, no cache, while the
  hosted detectors were timed at concurrency 4 (§6). The two are not measured the same way and
  should not be compared to three significant figures.
* **ScamTeller's threshold was never tuned.** 0.50 is the naive cut, not a chosen operating
  point, because the only legitimate tuning set is inside its training data (§3.1). Its numbers
  are therefore *not* the best this model can do, and are not directly comparable to the tuned
  hosted rows in Table B — the honest comparison for Table B is Table A.
* **n is small.** 41 scam calls with a detection point in eval and 44 in test, 38 and 35 hard
  negatives, 19 and 17 plain calls, 7 in the compliance slice. A one-call swing moves detection
  rate by about 2.4 points. Treat differences under ~5 points as noise — including the gap
  between the top four systems in Table B.
* **16 eval calls are unadjudicated** (§10), and a first human round found them ambiguous to
  humans too.
* **Human agreement on the detection moment is low** (3 of 14) and was measured without written
  anchor rules, so it bounds how precisely any system can be scored on timing.
* **Latency is an upper bound** — measured at concurrency 4, not in a single-call deployment.
* **Labels come from one annotator model** under `rubric/v3` + `detect/v2`.
* **One parse failure**, on `deepseek-pro`'s test run (1 of 817 prefixes, scored 0.0).
* **`gpt-luna` never meets the false-alarm budget** on train; it is reported at the best point it
  has (0.105 at the top of the grid) rather than at a point where it never fires.
* **Jev pays O(N²)** like the wrapped LLMs, correcting REQUIREMENTS §6.
