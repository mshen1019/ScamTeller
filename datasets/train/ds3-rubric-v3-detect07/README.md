# ds3-rubric-v3-detect07 — train

**Dataset version** `ds3-rubric-v3-detect07` · **Labelling rubric** `prefix-eval/v3`

These are two different version numbers: the dataset version says which release of this data it is, the rubric version says which revision of the labelling rules.
The two are not in step and should not be.

**Training set.**

## What is in here

777 synthetic phone conversations, each labelled with **the round from which the evidence is sufficient to call it a scam**
(`earliest_detection_index`). This is a **moment**, not a class — it is for training and evaluating
early-warning detectors: not "is this a scam?" but "what is the earliest round at which it can be called?".

| | |
|---|---|
| scam | 343 (late-onset 100 / receiver persuaded 81) |
| hard negative (genuine but looks like a scam) | 293 |
| ordinary legitimate call | 141 |
| scenario families | 50 |
| scam detection rate in the labels | 0.892 |

## Two files, joined on conversation_id

* `conversations.jsonl` — **the only input a model may see.** Each utterance has only
  `index` / `speaker` / `text`; no label field exists at any depth.
* `metadata.jsonl` — labels, scenario, ground truth, generation provenance. **Must never enter model input.**

The recommended training target is `prefix_labels[].evidence_established` (equivalently
`utterance_labels[].evidence_established`): a **monotonic** "the evidence is now sufficient" —
false before `earliest_detection_index`, true from it onward, never falling back. The
`sufficient_evidence` in the same record, by contrast, is a per-prefix **in-place** verdict that **may fall back**
(8.3% of conversations across the whole corpus go true and then false), so using it as the target teaches the model to retract warnings.
`evidence_established` is a label derived with hindsight; **do not use it as a feature**.

## What to know before using it

* **Do not mix data from different rubric versions in one training set.** The same scam behaviour would be
  labelled "insufficient evidence" in part of the data and "decidable at round 3" in another. The `split` command rejects such a corpus outright;
  to merge other batches, first bring them to the same rubric with `callfactory relabel`.
* **This version of the labels is conservative.** The corpus-wide detection rate is 0.593, only 0.417 on the "receiver persuaded" samples,
  and not a single false alarm on 366 hard negatives. A model trained on it inherits a bias in this direction:
  **it alarms later than an attentive listener would**, and more so the more cooperative the other side is. This is a known, directional bias, not noise
  — once human calibration produces the next rubric version, the **same transcripts** can be relabelled and the gap measured.
* **Latency and false-alarm rate must be read together.** Looking only at latency, "alarm immediately on every call" is optimal; looking only at false-alarm rate,
  "never alarm" is optimal. Hard negatives exist to measure the latter.
* ⚠️ **The `no_observable_evidence` slice is an open question; do not treat it as settled labels.**
  The flag declares a **scenario premise** (the caller never produces decidable evidence), but nothing in the caller prompt
  requires the caller to honour it: in practice 16 of the 17 such samples refuse when the receiver asks for a callback number —
  exactly the behaviour the rubric's GROUP B lists as "individually sufficient" — while the label still says "insufficient evidence".
  These samples are `eval_only`, so they go to eval only and never to training, and therefore **do not contaminate the training set**; but on eval
  the latency figures for this slice should not be taken at face value until a human has adjudicated them. See `docs/EVAL_SYSTEM.md` §5.7.

The full design notes are in `docs/EVAL_SYSTEM.md`.
