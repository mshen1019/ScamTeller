# How the Evaluation System Was Built

**Audience:** readers with no prior exposure to this project.
**Scope:** what the system is for, how it is put together, and — in detail — every
significant change we made to it and the evidence that prompted each one.

> **Maintenance rule:** this document is updated on every
> significant change to the evaluation system. A change is "significant" if it
> alters what a label means, how a label is produced, what the dataset contains,
> or how we measure any of it.

Last updated: 2026-11-02 · rubric `prefix-eval/v3` · scenario bank v3 (57 families,
author prompt `scenario-author/v2`) · dataset `ds3-rubric-v3-detect07`
(1000 conversations, train/eval/test; detection rule `detect/v2`) · bench harness
`bench-judge/v1` with seven hosted baselines on **both** eval and test
(§5.10, `BENCH_REPORT.md`) · 432 tests, all offline.

---

## 1. What problem this system solves

We are building a detector that spots scam phone calls **while they are still in
progress**. That is a different task from the usual one.

The usual task is classification: here is a transcript, is it fraud? The useful
task is *early warning*: at which point in the call did enough evidence exist for
a careful listener to conclude that this is probably fraud — and can a model reach
that conclusion then, rather than three turns later or three turns too early?

To train and evaluate that, we need conversations where we know **not only whether
the call was a scam, but at which turn it became knowable**. No public dataset
provides that, and real recordings cannot be collected and labelled at this
granularity without serious privacy problems. So we generate the conversations
synthetically, with two independent language models role-playing the two sides of
the call, and a third model annotating the result.

This repository is the factory that produces that data. It is not the detector.

### The label is a *time*, not a class

The central annotation is `earliest_detection_index`: the index of the first turn
such that a careful listener who had heard **only the turns up to that point**
would already have reasonable grounds to treat the call as likely fraudulent.

Everything difficult about this project follows from that one definition.

### Two ways to poison the data, and they are mirror images

| Label error | What it rewards |
|---|---|
| A detection point on a call that is genuinely legitimate | A model that alarms at the first superficially suspicious thing it hears |
| No detection point on a call that genuinely became detectable | A model that stays silent when it should speak |

Both errors invert the grading. This is why we report **detection latency and
false-alarm rate on the same screen, always**: optimise latency alone and
"alarm immediately, every call" wins; optimise false alarms alone and "never
alarm" wins. Neither is a detector.

---

## 2. Architecture

Six components, in a pipeline:

```
human context ──┐
                ├─→ [1 Scenario author]  ─→ versioned scenario bank (JSONL, human-editable)
existing bank ──┘                              │
                                               ↓
                                     [2 Scenario sampler] ← seed
                                               ↓  scenario × caller persona × receiver persona
                             ┌─────────────────┴─────────────────┐
                             ↓                                   ↓
                     [3 Caller agent]  ←── alternating ──→  [4 Receiver agent]
                             └─────────────────┬─────────────────┘
                                               ↓  transcript
                                     [5 Evaluator agent]  annotates each prefix separately
                                               ↓  + prefix labels
                                   programmatic validation + diversity
                                               ↓
                                        candidate dataset
                                               ↓
                                     [6 Dataset gatekeeper]  four-dimension scoring
                                               ↓
                             admitted / rejected / needs-human-review
```

**Status:** components 1, 2, 3, 4, 5 and the programmatic validation are implemented
and exercised end to end against live models. Component 6 is specified but not yet
implemented. See §7.

### The six components

1. **Scenario author** — expands the scenario bank from short human prompts such
   as "bitcoin investment scams" or "impersonating building management" into
   fully structured scenario records. The bank is versioned plain text and
   human-editable; the agent exists to fill gaps a human notices, not to be the
   sole source of scenarios.
2. **Scenario sampler** — a pure function, zero model calls. Turns the bank plus a
   random seed into a list of generation tasks: scenario × caller persona ×
   receiver persona × intended outcome × target length × per-turn direction.
   Stratified, ratio-controlled, and reproducible.
3. **Caller agent** — role-plays one fictional caller. Knows the scenario.
4. **Receiver agent** — role-plays the person who picked up. Knows *only* its own
   persona and what has actually been said aloud.
5. **Evaluator agent** — annotates risk. Called once per prefix, statelessly.
6. **Gatekeeper** — decides whether a finished sample is good enough to enter the
   dataset, scoring realism, consistency, diversity and detection value.

---

## 3. The eight design principles

These were fixed before implementation and each one exists because of a specific
failure mode.

| # | Principle | What goes wrong without it |
|---|---|---|
| **P1** | **Information asymmetry.** The caller knows the scenario; the receiver knows only its persona and what was said aloud; the evaluator sees only a prefix. | If the receiver is told "this is a scam", its suspicion becomes a restatement of the label instead of a reaction to evidence — and the model trained on it learns to read the receiver. |
| **P2** | **Labels and inputs are physically separate files.** The transcript file structurally contains no label field, at any nesting depth. | A hidden label anywhere in the input is a shortcut a downstream model will find and use. |
| **P3** | **Prefix annotation.** The evaluator's *k*-th call receives `utterances[:k+1]` and nothing else, with no memory across calls. | A label produced with knowledge of later turns is not causally available at the time it claims to apply, so early-warning metrics computed against it are fiction. |
| **P4** | **The judge and the judged are different models.** The evaluator must use a different provider from the caller and receiver; the gatekeeper different again from the evaluator. | Otherwise a model grades its own output. |
| **P5** | **Reproducibility.** `(bank_version, seed, index)` determines a generation task uniquely. | Without it a dataset cannot be traced or rebuilt. (Note the real limit in §6.) |
| **P6** | **Diversity is computed, never asked.** Repetition and near-duplication are measured programmatically. | An LLM judge cannot see the other samples, so it cannot assess cross-sample repetition — asking it produces a confident guess. |
| **P7** | **The whole pipeline runs offline.** A deterministic mock provider drives every stage. | Development and CI must not require API keys. |
| **P8** | **Safety boundary.** We produce observable conversational behaviour, never executable fraud procedure. | A transcript that runs the fraud to completion step by step is a usable script. |

P8 in practice: every scam call in this dataset stops before the fraud completes.
Placeholders (`[PERSON_1]`, `[COMPANY_A]`, `[ACCOUNT_REF]`, `[AMOUNT_1]`) stand in
for every identifying noun and number; no digits are ever read out; and when a
receiver agrees to comply, the call ends **at that moment of agreement**, before
any mechanics. A validator re-checks this on the output side.

---

## 4. How the system was actually built: every change and why

This is the part worth reading. Each entry follows the same shape: the symptom we
measured, what it turned out to be, the change, and what the change verifiably
did.

Throughout, "run-7", "run-7-v2" … are successive batches of 20 conversations
generated from the same seed. Dialogue was generated by `claude-sonnet-5-5`;
annotation by `gpt-4.1-mini`, a different provider, as P4 requires.

### 4.1 A perfect-looking offline baseline that meant nothing

**Symptom.** The first end-to-end run used the offline mock provider and reported
a scam detection rate of 1.0, a false-alarm rate of 0.0, and a mean detection
round of 2.0. Every number was ideal.

**Diagnosis.** The mock is not a model. It is a bank of template sentences plus a
keyword table. We measured the overlap: the scam template bank triggers 9 of the
signal groups in the keyword table; the legitimate bank triggers 1. The detector
was matching substrings in sentences written to contain those substrings. The
separation was not learned, it was constructed — and the margin was one word wide
(the "authority claim" cue is `"i'm calling from"`; the legitimate template says
`"this is [PERSON_1] calling from"`, which misses it by a single word).

Worse, the mean detection round of 2.0 meant the dataset was *useless by our own
criterion*: if every scam is detectable at turn 2, no two detectors can be told
apart.

**Change.** None to the code — the mock is doing its job (P7: keeping the pipeline
runnable with no API keys). The change was to the conclusions we were willing to
draw. Mock numbers establish that the plumbing works. They say nothing about data
quality, and we stopped reporting them as if they did.

**Lesson recorded in the spec.** Any configuration in which the annotator and the
generator share a keyword table or a model is self-confirming. This is what P4 is
defending against, and the mock is the clearest illustration of why.

### 4.2 The first live run: 20 of 20 failures in 1.14 seconds

**Symptom.** Every conversation failed with HTTP 400 from the dialogue model:
`` `temperature` is deprecated for this model. ``

**Diagnosis.** Our provider layer was built when the Claude Messages API accepted
`temperature`. It no longer does: non-default sampling parameters are rejected.
Because a 400 is correctly treated as non-retryable, the whole batch died
immediately. We verified the exact boundary with six single-request probes:

| Request body | Result |
|---|---|
| `temperature: 0.9` | **400** |
| `temperature: 1.0` | accepted (it is the default value) |
| no temperature | accepted, returns a `thinking` block, 122 output tokens |
| `thinking: {type: "between_tools"}` | accepted, no thinking block, **63** output tokens |
| `+ output_config: {effort: "low"}` | accepted, **50** output tokens |

**Change.** The Anthropic provider no longer sends `temperature`, `top_p`,
`top_k` or `seed`. It also disables reasoning for dialogue turns, because
role-playing one spoken line is not a reasoning task and leaving reasoning on
roughly doubled output tokens per turn for no gain in dialogue quality.

**Consequence for the design, which matters more than the bug.** Our written
requirement was "generate at high temperature (0.9) for diversity". That is no
longer implementable on this model family. Diversity now comes entirely from the
task cross product — scenario × 12 caller personas × 12 receiver personas ×
outcome × slow-burn flag — plus per-turn direction injection. The requirement is
marked obsolete in the spec with the measurement that killed it, so nobody later
reads it and assumes a temperature knob is doing work.

**Honesty fields.** Each record now carries, per agent:

```json
"caller": {
  "provider": "anthropic", "model": "...",
  "temperature": null, "temperature_requested": 0.9,
  "temperature_honoured_by_provider": false,
  "seed": 7183957, "seed_honoured_by_provider": false
}
```

Recording only the *requested* value would let a reader two weeks later conclude
this data was generated at temperature 0.9 with a fixed seed. Neither is true.

### 4.3 Six of ten scams had no detection point

**Symptom.** First successful batch (run-7): scam detection rate **0.4**. The six
undetected scams all ended with risk scores of 0.60–0.70 and five to eight
accumulated signals.

**Diagnosis.** The score channel was working; the boolean was not. The evaluator
was consistently reporting a high score and `sufficient_evidence: false`. Reading
one case end to end made the cause obvious — the rubric told the model what is
*not* sufficient ("urgency alone is not enough; legitimate organisations are often
urgent") and gave only one vague positive. We had over-corrected against false
alarms and produced a rubric that almost never fires.

**Change.** Rewrote the `sufficient_evidence` instruction as an anchored
definition: an explicit true/false meaning, an explicit list of things that are
*not* sufficient on their own, and an explicit list of caller behaviours that
*are* sufficient the moment they occur (refusing verification through a channel
the listener chooses; demanding details before saying what the call is about;
threatening a consequence; asking for secrecy; contradicting its own stated
purpose; insisting nobody else can help).

This follows the project's own rule for the gatekeeper, applied one component
earlier: when agreement is poor, **fix the rubric, not the threshold**.

**Verified effect.** Detection rate 0.4 → 0.8.

### 4.4 Every conversation collapsed to the minimum length

**Symptom.** In the same batch: all 20 conversations ended because the receiver
hung up, 14 of 20 ran exactly 5 rounds (the configured minimum), and the log
carried 16 notes saying "receiver wanted to hang up earlier". The sampler's
target lengths of 5–10 rounds had no observable effect.

**Diagnosis.** Two causes compounding. First, the rule that ignores a hang-up
request before the minimum length released the receiver at exactly that minimum,
so a receiver that wanted out immediately left at the first legal opportunity.
Second, nothing in the receiver's prompt discouraged ending the call.

This mattered most for slow-burn scams, whose whole definition is that the opening
is unremarkable and pressure appears later. Three slow-burn samples got five to
six rounds each. A slow burn in five rounds is not a slow burn.

**Change.** The hang-up gate moved from `min_rounds` to
`max(min_rounds, target_rounds - 1)`, so the sampled target length now governs,
with one round of slack for a natural ending. The receiver prompt gained a
persona-level framing: people stay on odd calls out of politeness, curiosity or
confusion; end the call only when you have genuinely decided it is finished, and
"this is taking a while" is not a reason. `ended_by`, `target_rounds` and
`hangup_allowed_from_round` are now recorded per sample so a future collapse is
visible immediately.

**Verified effect.** Mean rounds 5.35 → 6.55, distribution 5–9, and the first
conversations that ended by reaching their target rather than being hung up on.

### 4.5 A false alarm on a hard negative — which turned out to be two bugs

**Hard negatives** are the most important samples in the dataset: genuinely
legitimate calls with scam-like surface features. A real bank fraud team ringing
you unprompted about a suspicious transaction. A real IT helpdesk asking you to
install remote-assistance software. What separates them from fraud is not the
surface, it is **how they behave when doubted** — every hard negative scenario
must specify that behaviour, or it is just an ordinary negative.

**Symptom.** After the rubric fix, false alarms on hard negatives went from 0.0 to
**0.25** (one of four). Detection had improved, but at a cost in exactly the place
where cost is least acceptable.

**Diagnosis — defect 1: the evaluator attributed the receiver's behaviour to the
caller.** Its own rationale read: *"Caller requests remote access and offers a
callback via internal portal which **receiver refuses** and plans to verify
independently"* — and it listed `refusal_to_verify` as a caller signal. The refusal
was the receiver's. The caller had offered verification, which is precisely what
that scenario's distinguishing behaviour specifies. And the new anchor text we had
just written made the error easier: the sentence *"refused a direct request to
verify through a channel the receiver chooses"* contains both "refused" and
"receiver", and the model fused them.

**Diagnosis — defect 2: a transient spike permanently set the detection point.**
That sample's final prefix score was 0.20. The evaluator had spiked above the
threshold at one prefix, then revised itself back down. But `earliest_detection_index`
took the *first* qualifying prefix, so a momentary misjudgement the annotator
itself retracted became a permanent ground-truth claim that the call was
detectable from turn 7 onward.

**Changes.**
- Every anchor line now names its subject explicitly ("the caller deflected…"),
  with an explicit counter-example: the receiver declining the caller's suggested
  channel, or deciding to verify independently, is the receiver being careful and
  changes nothing either way.
- A detection point must now also be **not immediately contradicted**: the next
  annotated prefix must still score at or above the threshold. The rule inspects
  only the *score* of the successor, not its boolean, because the score is the
  well-calibrated channel (measured: hard negatives 0.00–0.20, scams 0.60–0.90)
  and the boolean is the noisy one. Genuine scams escalate, so the rule costs them
  almost nothing.

**Verified effect.** Hard-negative false alarms 0.25 → **0.0**, with detection rate
held at 0.8. The 0.25 was not a trade-off; it was two bugs.

### 4.6 A sample labelled "scam" that contained no scam

**Symptom.** One sample ran nine rounds, scored a flat 0.30 throughout, had no
detection point, and the whole-transcript judge independently called it *not*
suspicious.

**Diagnosis.** We read it. The caller never asked for money ("there's no payment
needed on this call as such"), never asked for personal details, accepted the
receiver's right to verify with the building office ("that's absolutely your
right"), then admitted it had mixed up its notes, apologised, and said goodbye. The
scenario's declared signals included `unusual_payment_channel`, which never
appeared at all.

The ground truth said `is_scam: true`, because the scenario's premise is a
fabrication. But nothing in the transcript supports it. That is the mirror of the
hard-negative false alarm: a model that correctly stays silent gets marked as
having missed a scam.

The root cause was a persona–scenario interaction. The caller persona was
`competence: "green"` — an inexperienced caller, valuable because it naturally
leaks `inconsistent_details`. But our prompt told it only that it would make
mistakes and contradict itself; it did not tell it to keep pursuing the call's
purpose. Combined with a sampled outcome of "receiver asks a question the caller
cannot answer and the caller abruptly wraps up", the scam never landed. This also
corrected an earlier assumption of ours, that any persona can carry any scenario:
true for tone, false for competence.

**Changes.**
- The green-competence instruction now states that clumsiness slows the caller
  down and makes it leak inconsistencies but does not make it give up, and that
  the scenario's warning behaviour must actually appear in what it says.
- More importantly, we separated two things we had been conflating. A scam call
  that leaves no detectable evidence is a **real and valuable category** — it would
  test a detector's *restraint*, because alarming on it is a false alarm. It just
  must not arrive by accident. (Whether the family built for this actually achieves
  that is now in doubt — see §5.7. The premise is sound; the generated dialogue does
  not hold to it once the listener pushes back.) So the bank gained an explicit family,
  `reconnaissance_detail_confirmation` (a caller that only ever tries to confirm
  an address and rings off), with two new scenario fields:
  `no_observable_evidence: true` and `eval_only: true` — the latter keeps it out
  of training splits by default.
- Two validators keep that field honest: such a scenario may not declare any of
  the eight **strong signals** (the ones the evaluator rubric treats as
  individually sufficient — declaring one contradicts the claim that no evidence
  exists), and may not have compliance outcomes, because it never asks for
  anything.

Note what this is *not*: it is not "the scam did not complete". **No scam in this
dataset completes** — that is P8, and the scenario bank contains no outcome in
which the receiver pays or hands over a code. Completion is held at zero by
construction. `no_observable_evidence` means something different: the caller never
displayed anything a listener could act on.

### 4.7 The dataset was teaching a shortcut: read the receiver, not the caller

**Symptom.** Found by inspection rather than by a metric. We tabulated the outcome
vocabulary of the scenario bank:

| | Outcomes where the receiver resists | Other |
|---|---|---|
| Scam scenarios (6 outcomes) | **5** | 1 |
| Legitimate scenarios (5 outcomes) | **0** | 5 |

"The receiver resists" almost perfectly separated the two classes. A model could
score well by reading the *receiver* and never looking at the caller. We forbid
exactly this in the evaluator's prompt — and had left it alive in the dataset's
composition.

**Changes.**
- Scam scenarios gained **compliance outcomes**, in which the receiver is talked
  round. Three of them, each ending at the moment of agreement.
- Legitimate scenarios gained **resistance outcomes** — and deliberately, two of
  them describe receiver behaviour **word for word identical** to a scam outcome,
  differing only in how the caller responds:

  | Scam outcome | Legitimate outcome |
  |---|---|
  | receiver insists on calling back on a number they look up themselves | receiver insists on calling back on a number they look up themselves, **and the caller agrees readily** |
  | receiver asks a question the caller cannot answer and the caller **abruptly wraps up** | receiver asks a question the caller cannot answer, **and the caller offers to find out and call back** |

  This forces the discriminative information back onto the caller, which is where
  the rubric says it lives.
- Compliance outcomes are only assigned to receivers whose scepticism is
  `very_low` or `low`. Making a highly sceptical persona be "talked round" fights
  the persona and reads as if the character changed mid-call. When the ratio
  cannot be met, the sampler **warns rather than forcing it**.
- `--compliance-ratio` (default 0.25) controls the proportion, and
  `compliance_intended` is recorded in each record's ground truth so the benchmark
  can slice on it and answer directly: does a model's alarm rate drop when the
  victim is credulous?
- The safety boundary is restated in the prompts for this case: if the other
  person agrees, the call ends on that agreement — no steps, no digits, no
  transfer or install procedure.

**Outcome choice uses a separate random stream** from persona choice, so changing
the compliance ratio does not reshuffle personas. There is a test pinning that.

### 4.8 The *rubric* had the same shortcut

**Symptom.** Of the first two compliance samples, one was annotated normally and
one received no detection point at all — despite being a well-played redelivery-fee
scam. The evaluator's own final rationale read: *"Caller pressures for payment
today over the phone for a parcel with unknown sender; cannot verify sender
identity; insists on payment on call, unusual channel."* It had the facts right and
still answered `sufficient_evidence: false` on all six prefixes.

**Diagnosis.** We checked that the new persistence rule was not the cause (it was
not; the result is identical with the rule disabled). The cause was the shape of
the anchors we wrote in §4.3. Nearly every one of them is **reactive** — it
requires the receiver to have pushed first:

- the caller deflected *when the receiver asked* to verify
- the caller contradicted itself *and let the contradiction stand*
- the caller insisted *nobody else* can deal with it

A cooperative receiver never asks, so the caller never has to do any of those. The
detectability of a scam therefore depended on the receiver's resistance — the same
shortcut as §4.7, one layer up, living in the rubric instead of in the data. We had
fixed the dataset and left the measuring instrument broken in the same way.

A secondary observation pointed the same way: the score dropped from 0.60 to 0.30
on the final prefix, where the last utterance is the receiver saying "I suppose
that's all right then". The receiver's acquiescence was lowering the risk score.

**Changes.**
- The anchors are now split into two labelled groups. **Group A** are things the
  caller does unprompted, which hold even when the listener is perfectly
  cooperative: wanting a payment settled today on a call the listener did not
  initiate, by a route the caller chose; asking for a one-time code, password,
  card detail or remote access *at all*; wanting money for something whose origin
  it cannot state and continuing anyway; asking for secrecy; threatening a
  consequence. **Group B** are the reactive ones, now carrying an explicit
  warning: *absence of these means nothing if the listener never pushed.*
- The prompt's framing gained the mirror of its existing rule. It already said a
  suspicious receiver never *raises* the score; it now says a trusting, cheerful
  or entirely cooperative receiver never *lowers* it, because the risk lives in
  what the caller does and whether this particular listener believes it changes
  nothing.
- A final self-check closes the specific loophole observed: *if your own rationale
  describes the caller pressing for payment, withholding what the call is about,
  or asking for a credential, then `sufficient_evidence` is true — however
  relaxed, friendly or convinced the person on the other end sounds.*

**Verified effect.** See §5.2 — this is the change we measured most carefully,
because it is the one most at risk of being self-deception.

---

## 5. What the numbers actually say

### 5.1 Across generation runs

Twenty conversations each, same seed, same scenario bank (except where noted).

| | run-7 | run-7-v2 | run-7-v3 | run-7-v4 |
|---|---|---|---|---|
| Scam detection rate | 0.40 | 0.80 | 0.80 | 0.80 |
| Hard-negative false-alarm rate | 0.00 | **0.25** | 0.00 | 0.00 |
| Legitimate false-alarm rate | 0.00 | 0.10 | 0.00 | 0.00 |
| Mean rounds | 5.35 | 6.55 | 6.55 | 6.40 |
| Ended by receiver hang-up | 20/20 | 19/20 | 20/20 | 20/20 |
| Annotator/ground-truth disagreement | 0.05 | 0.15 | 0.10 | 0.10 |
| Mean detection round | 4.25 | 4.00 | 4.25 | 3.88 |

**Read this table with care.** Per-sample comparisons across these runs are *not
valid*, for a reason given in §6: the dialogue model has no seed, so every run
produces different conversations. The run-level aggregates are informative; a
single sample appearing to flip between runs is usually dialogue variation. We
made that mistake once and caught it — one sample looked like it had "lost" its
detection point between v3 and v4, and on a fixed transcript it had not moved.

### 5.2 The controlled experiment: same transcripts, new rubric

To isolate the effect of the §4.8 rubric change from dialogue variation, we
re-annotated the **identical 20 transcripts** from run-7-v3 under the new rubric
and compared against their recorded labels.

| | Old rubric | New rubric |
|---|---|---|
| Scam detection rate | 0.80 | **0.90** |
| Legitimate false-alarm rate | 0.00 | 0.00 |
| Hard-negative false-alarm rate | 0.00 | 0.00 |
| Detection points moved earlier | — | **2** |
| Detection points moved later | — | 0 |
| Detection points gained | — | 1 |
| Detection points lost | — | **0** |

The new rubric strictly dominates on fixed input: one more scam annotated, two
annotated earlier, none lost, and not one new false alarm among fourteen
legitimate calls. Specifically:

- the compliance sample from §4.8 went from no detection point to a detection
  point at turn 3 — the proactive anchors fire without waiting for resistance;
- the other compliance sample moved four utterances earlier;
- the explicit no-evidence sample stayed unannotated under **both** rubrics, which we
  read at the time as evidence of a genuine restraint test rather than an artefact of a
  conservative rubric — §5.7 revisits that reading, because the caller in these calls
  does refuse verification when pushed;
- the sample that appeared to regress between generation runs did not move.

### 5.3 Cost, measured

Per record we now record actual token usage per provider. A 20-conversation batch
with per-utterance annotation at stride 2:

| | Requests | Input tokens | Output tokens |
|---|---|---|---|
| Dialogue model | 262 | 502,630 | 35,514 |
| Annotation model | 171 | 262,799 | 12,952 |

At the dialogue model's list price that is roughly **$0.07 per conversation** for
the dialogue side. Re-annotating fixed transcripts costs 151 annotation requests
and no dialogue requests — around **one fourteenth** of a regeneration, with zero
dialogue noise.

---

### 5.4 The first dataset release, and what shipping it early costs

> **Note (added later):** the releases described in this section and in §5.7,
> `ds1-rubric-v3` and `ds2-rubric-v3`, have since been **removed from the working
> tree** — only the current release is kept, so that nobody picking up the data can
> choose the wrong one. Every conversation in both is in `ds3-rubric-v3-detect07`
> (ids and transcript hashes verified), git history still holds them, and ds2's labels
> can be recomputed at zero cost with
> `rederive --threshold 0.5 --require-sufficient`. The numbers below are kept as the
> historical record of what was measured at the time.

**Two version numbers, deliberately not in step.** A *dataset* version says which
release of the data this is (`ds1`); a *rubric* version says which annotation
rulebook produced its labels (`prefix-eval/v3`). The first release is therefore
`ds1-rubric-v3`: dataset 1, labelled under rubric v3, the newest rubric. Each
split directory carries a README naming both, because the two numbers will keep
diverging — the next rubric revision produces `ds2-rubric-v4` over the same
transcripts.

The decision was taken to train on the current rubric's labels now and calibrate
against human labels in parallel, rather than block training on finding
annotators. Two things had to happen first.

**All labels had to come from one rubric.** Five batches were on disk, produced by
three different rubrics, and before rubric stamping their metadata was
indistinguishable. The four older batches were therefore re-annotated under
`prefix-eval/v3` — the dialogues were untouched and perfectly usable; only the
labels were stale. Cost: evaluator calls only. The label shift it produced is
itself informative: the batch generated under the rubric that had the
hard-negative false-alarm bug lost three detection points, which is that bug
being corrected.

**The split had to refuse what had just gone wrong.** `split` now rejects a
corpus that mixes rubric versions outright, rather than warning. It also keeps
near-duplicate transcripts on the same side (lexical similarity within a family,
the same measure Diversity uses), pins `eval_only` scenarios and the calibration
holdout to the eval side, and stratifies on compliance as its own layer — without
that, the fourteen compliance samples scattered at random and the eval side kept
only one, which is the slice most in need of measurement.

| | train | eval |
|---|---|---|
| Conversations | 122 | 38 |
| Scam (of which slow-burn / compliance) | 77 (19 / 11) | 27 (5 / 3) |
| Hard negative | 26 | 6 |
| Plain legitimate | 38 | 10 |
| Families | 24 | 19 |
| Scam detection rate in the labels | 0.71 | 0.59 |

Both sides validate clean: no label key in any transcript, every pair joined, no
label borrowed from the future.

**The known cost of shipping v1 now.** These labels are measurably conservative.
Detection rate is 0.65 over 80 fresh conversations, and compliance samples sit at
0.40 against 0.67 for the rest. A model trained on them will inherit that bias in
a predictable direction: it will alarm later than a careful listener would, and
most of all when the person on the other end is cooperative. That is a known,
directional bias rather than noise, which makes it the right kind of debt — it can
be measured and corrected against v2 labels on the same transcripts. The
rubric-version stamp is what makes that correction measurable later.

---

### 5.5 The scenario pool, and a shortcut hiding in the family names

Shipping `ds1` used 25 hand-written scenario families. The target for the next
release is 1000 conversations, and 1000 over 25 families is 40 conversations per
family — enough repetition that a model could start recognising families instead
of behaviour. So component 1, the scenario-author agent, was built and run.

**What the author is, and what it is not.** It writes *premises*, never dialogue:
a `family`, a one-clause `pretext`, the organisation kind, the signs the call will
show, and — for hard negatives — the `disambiguator` saying how it behaves when
doubted. Three gates stand after it: the schema vocabulary (a signal outside the
closed list discards the whole record, no guessing repairs), semantic
near-duplicate checks against every family and pretext already in the bank, and
one deliberate removal of authority — **the outcome vocabulary is not the
author's to choose.** Compliance and resistance outcomes are attached by the
pipeline, so the property from §4.7 (a scam is not identifiable by a resisting
listener) holds across the whole bank rather than only in the hand-written seed.

**First pass: 30 candidates, and two things wrong with them.** The run was given
the coverage gaps from `review-bank`, which closed all four signals that had no
hard-negative counterpart. It also over-steered: 14 of the 30 were hard negatives
and exactly one was an ordinary legitimate call. `--use-gaps` is the instrument
for targeted gap-filling, not for bulk expansion, and it did what it was told.

The second problem was worth more than the first. Four of the new scam families
were named after nothing but the *signal they demonstrate*:
`caller_swaps_story_under_questioning`, `mystery_official_wont_say_who`,
`do_not_tell_the_branch`, `grandchild_wont_say_name`. A family name is carried in
the record and in the split metadata, so a name that restates `expected_signs` is
a second label attached to the sample — exactly the family→label shortcut that
§4.7's receiver-resistance shortcut was a version of. The author prompt now forbids it
(`scenario-author/v2`): name what the call is *about* and leave the behaviour in
`expected_signs`. The rule has one exception, because two of those four names are
real recognised patterns whose *name is the behaviour* — "don't tell the counter
staff, someone inside is involved", and a caller claiming to be a relative who
makes the listener supply the name. Those were renamed to describe the pretext
(`insider_investigation_secrecy`, `relative_identity_fishing`) and each records in
`notes` why the behaviour stayed in the name. The two whose names were pure
synonyms of their signals were dropped.

A weaker version of the same thing was left standing on purpose: about nine
families carry a pretext root with a signal suffix
(`internet_provider_refuses_any_verification`, `line_keeps_dropping_no_number`,
`tax_refund_figures_dont_add_up`). The root still names the call, so the name is
not the label on its own, but the suffix is redundant with `expected_signs` and
the names should be shortened before the next release.

**Second pass: the plain legitimate calls.** Four more were authored with the
context steered to ordinary business nobody would question — a delivery window, a
reserved library book, a fixture moved by waterlogging, a tradesperson reading
back a quote. All four were accepted, one request, 732 output tokens. These are
the records that stop a detector treating any unexpected call as suspicious, and
the first pass had produced only one.

**Two defects the review surfaced in the tooling itself.** A second authoring
round overwrote the first round's unreviewed candidates, and the first round's
families were invisible to the duplicate check — it compares against the *bank*,
and uncommitted candidates are not in the bank yet, so round two could legally
re-write round one's premises. `--append` now carries the pending candidates into
both the file and the dedup context. Separately, committing the bank was the first
time an authored scenario was read *back*: the author recorded which prompt
version wrote it, the scenario schema had no such field, and the new bank would
not load. The test that should have caught this hand-built the provenance block
instead of taking it from the author; it now goes through the author.

**Bank v3: 57 families** — 26 scam, 21 hard negative, 10 plain legitimate — and
`review-bank` reports no coverage gaps of any kind: every scenario-level signal is
used by some scam, every one has a hard negative that trips it, and no scam family
is without a paired hard negative. At 1000 conversations the pool is no longer the
constraint: all 1000 sampled tasks are distinct in (family, caller persona,
receiver persona), 9 to 30 conversations per family, median 19. At default ratios
that is 500 scam (150 slow-burn, 125 ending in compliance), 200 hard negative,
300 plain legitimate, for 50,844 model calls.

One thing this does *not* fix: component 6, the gatekeeper, is still unbuilt, and
32 of the 57 families are now machine-written premises that no automated judge has
scored for realism or detection value. They were read by hand instead.

---

### 5.6 Sampling for 1000 conversations: even families, and three sides

Two decisions had to be made before generating at scale, and both are about what
the data lets you *measure* afterwards.

**Even families, with existing data counted in.** The default sampler allocates by
category — 50% scam, then 40/60 hard-negative/plain inside the legitimate half —
and round-robins families *within* each layer. That keeps the category mix exactly
on target and lets per-family counts drift with the size of each layer: at 1000
conversations over bank v3 it gave 9 to 30 conversations per family. A family with
30 and a family with 9 are not equally learnable, and the difference is an artifact
of how many families happen to share a layer, which is not something anyone chose.

`--balance-families` inverts the priority: every family gets the same number of
conversations, and the category mix becomes whatever the bank's composition
implies. It is water-filling — each conversation is allocated to the family with
the fewest so far — and `--existing <dir…>` seeds those counts from data already
on disk, so a family that already has 11 conversations receives the difference and
not a full share. That is the whole meaning of "top up to the average". A family
already *above* the average receives nothing and the sampler says so, because
conversations already generated cannot be withdrawn; they can only raise the line.

For the second release that gave: 160 conversations already on disk across 25
families, 840 new ones to reach 1000, and a final count of 17 or 18 per family for
all 57 — a spread of one. The price is stated rather than hidden: the category mix
is no longer a knob. Bank v3 is 26 scam, 21 hard-negative and 10 plain families, so
the corpus comes out **46% scam, 37% hard negative, 18% plain legitimate**, against
65/20/30 in the first release. Hard negatives nearly double their share. For this
dataset that is the right direction — a hard negative is the only thing that
measures false alarms, and §7 used to list four signals that had none — but it is a
deliberate shift in what the data emphasises, not a free improvement, and anyone
training on it should know the plain-legitimate share went down.

**Three sides, because there are two different generalisation questions.** The
release is split with `--by family-holdout`:

* **test** — 10% of families, stratified across scam / hard negative / plain, held
  out *whole*. None of their conversations appear in train or eval. This measures
  generalisation to **call types the model has never seen**.
* **eval** — 10% of the conversations of every *remaining* family. This measures
  something else entirely: on call types the model has seen, **how precisely does
  it get the moment** evidence became sufficient.
* **train** — the rest.

Collapsing these into one eval set averages them, and the average hides exactly the
failure mode this dataset exists to expose: a detector that is sharp on familiar
pretexts and blind the moment the pretext changes scores "fine" on a mixed set. The
stratification of the held-out families is not decoration either — drawing six
families at random from a bank that is 46% scam would often hand back a test side
with no legitimate calls at all, and a test side without legitimate calls cannot
measure false alarms, only misses.

Two existing pins survive into this mode. `eval_only` scenarios (a caller whose
intent is a scam but who never shows actionable evidence) still never enter train,
and the calibration holdout still stays out of train. They are honoured by *taking
up* their family's eval quota rather than adding to it, so a handful of families
contribute three or four conversations to eval instead of two.

---

### 5.7 The second release: 1000 conversations, three sides

`ds2-rubric-v3` — dataset 2, same annotation rubric as the first release. 840 new
conversations generated against bank v3 and merged with the 160 already on disk,
every family at 17 or 18, and **0 generation failures and 0 records rejected by the
programmatic rules** out of 840.

| | train | eval | test |
|---|---|---|---|
| Conversations | 777 | 119 | 104 |
| Scam (slow-burn / compliance) | 343 (100 / 81) | 62 (13 / 10) | 52 (24 / 17) |
| Hard negative | 293 | 38 | 35 |
| Plain legitimate | 141 | 19 | 17 |
| Families | 50 | 51 | 6 (held out whole) |
| Scam detection rate in the labels | 0.627 | 0.435 | 0.558 |
| — of which premise says "no observable evidence" ⚠️ | 0 | 17, none detected | 0 |

The held-out families are three scams, two hard negatives and one ordinary
legitimate call, and none of their conversations appear in train or eval. All three
sides validate clean: no label key at any depth in any transcript, every pair
joined, no label borrowed from the future.

**One slice is an open question, and two wrong readings of it.** eval's detection
rate reads 0.435 against train's 0.627. The whole `no_observable_evidence` family —
`reconnaissance_detail_confirmation`, 17 conversations — is pinned to eval by the
`eval_only` rule, and not one of them has a detection point.

The first reading was that this is a population effect and nothing more: the correct
label for a no-evidence call *is* "no detection point", so eval looks worse only
because it holds a slice whose right answer is silence. On that basis the slice was
taken out of the detection-rate denominator and credited as restraint — 17 of 17
correctly silent — which moved eval to 0.600.

**That was wrong, and reading the transcripts is what showed it.**
`no_observable_evidence` is a property of the *premise*, and nothing in the caller
prompt makes the caller honour it under pressure. In `run-1101-00418` the listener
asks for a callback number at utterance 7, and at utterance 8 the caller answers:
"this line isn't really one that takes calls back, I'm afraid, so I couldn't give you
a number that would reach me." GROUP B of the rubric lists exactly that — refusing to
verify through a channel the listener chose, with a callback number named explicitly —
as individually sufficient. The label says `sufficient_evidence: false` at 0.30, and
its own rationale says "Caller refuses to provide a callback number and won't confirm
the address they have, despite the receiver's requests for verification." 16 of the 17
carry rationales of that shape.

So the slice is neither a clean restraint test nor a plain miss: it is
**unadjudicated**. The denominator now keeps it (train 0.627 / eval 0.435 / test
0.558) and the slice is reported separately as an open question — in the batch
summary, in `inspect`, in the split statistics, and in the README shipped inside each
split. Crediting it as restraint recorded a pending question as a result, which is
worse than the ambiguity it replaced.

The root cause is in generation, not annotation: a caller told to pursue its goal will
deflect when pushed, and the moment it deflects the "no observable evidence" premise
is broken. Either the caller must be told to comply plainly with verification requests
for this family — which is a scammer that does not exist — or the flag should go and
the family be treated as an ordinary weak-evidence scam. That is a decision for the
human labelling round, not for whoever reads the numbers.

**A tempting inference that does not hold.** Across the corpus, 421 of 457 scams (92%)
have some prefix listing a signal that `schema.STRONG_SIGNALS` calls individually
sufficient while `sufficient_evidence` is false; among the 186 scams with no detection
point it is 181 (97%). That looks like the annotator violating its own rubric almost
everywhere. It is not safe to conclude that: the same pattern appears on 70% of hard
negatives and 40% of plain legitimate calls, where "insufficient" is almost certainly
the right answer. What it does establish is narrower and still useful — the `signals`
list is applied far more loosely than the sufficiency decision (any muddle earns
`inconsistent_details`), so the constant's claim that the two must agree does not
describe the labels. Which of the two should move is a question for human
adjudication, not for more inspection of the same labels.

The general lesson survives both wrong readings: a single aggregate moved because the
*population* under it changed, and nothing in the number said so. Any slice whose
correct answer is "stay silent" has to be counted separately from slices scored on how
early they fire — and whether a slice really has that character must be checked
against the transcripts, not assumed from a flag.

**What the corpus measures, over all 1000:**

| | |
|---|---|
| Scam detection rate (457 scams) | 0.593 |
| — the 17 unadjudicated no-evidence scams ⚠️ | none detected; see above |
| — detection rate with that slice set aside | 0.616 |
| False alarms on legitimate calls | **0 / 543** |
| — of which hard negatives | **0 / 366** |
| Compliance samples (listener is talked round) | 0.417 detected |
| Slow-burn scams | 0.642 detected |
| Median detection position | 64% into the call (1 of 271 at utterance 0) |
| Annotator's independent whole-call judgement vs ground truth | 12.0% disagreement |

Two of those rows are the point. **Not one false alarm in 366 hard negatives** —
calls engineered to trip the same signals a scam trips — while **about 40% of scams never earn a
detection point** (0.593 over all 457; 0.616 with the unadjudicated slice set aside). At 20 samples that pattern was a guess; at 1000 it
is the dataset's character: these labels are conservative, and a model trained on
them will alarm later than a careful listener, not earlier. The compliance slice
remains the floor (0.417 against 0.616), exactly as §5.4 predicted from 80 samples,
so the bias is directional and stable rather than noisy — which is what makes it
correctable against a v2 rubric on these same transcripts.

The detection-position guardrail passes: median 64% into the call, and a single
detection at utterance 0 out of 271. The labels are not collapsing toward "alarm
immediately", which is the failure mode that would make the dataset worthless while
every headline number improved.

**Mixed bank versions are fine; mixed rubric versions are not.** This corpus spans
scenario-bank v1, v2 and v3, and each record states which one it came from. That is
a provenance fact, not a contradiction. A mixed *rubric* would be a contradiction —
the same behaviour labelled "insufficient evidence" in one half and "callable at
turn 3" in the other — and `split` still refuses it outright.

**Cost and time, measured.** 18,525 requests, 36.1M input and 2.14M output tokens,
60 minutes of wall time at 10 concurrent conversations. At list prices that is
roughly **$98** for the 840, about $0.117 per conversation — against the $0.0725
measured on the 80-conversation calibration batch. The gap is not a surprise in
hindsight: a dialogue turn resends the entire conversation so far, so input tokens
grow quadratically in call length, and nothing here uses prompt caching. Caching the
system prompt and the transcript prefix is the single largest cost lever left, and
the prefix-annotation calls are all independent, so they could also go through a
batch API at half price. Neither is done.

---

### 5.8 A second label: `evidence_established`

The annotation a prefix carries is `sufficient_evidence` — the annotator's verdict on
*that* prefix, with nothing after it in view. It is deliberately allowed to fall back:
an annotator that becomes suspicious and is then reassured should be able to say so,
and it does, in **83 of the 1000 conversations (8.3%)** the flag goes true and then
false again later in the same call.

That is the right behaviour for an annotation and the wrong shape for a training
target. A deployed detector cannot un-ring an alarm, and no metric defined on this
task rewards retracting one. Worse, latency is scored against
`earliest_detection_index`, so a model trained on the raw per-prefix flag is trained
against a quantity its evaluation does not use.

So every prefix label and every utterance label now also carries
**`evidence_established`**: false before `earliest_detection_index`, true from it
onward, never falling back. It sits next to `is_detection_point`, which marks the
single moment; this one marks everything from that moment on.

| | positive rate over 7,662 prefixes |
|---|---|
| `sufficient_evidence` (per-prefix, non-monotone) | 9.9% (758) |
| `evidence_established` (monotone) | 11.7% (897) |

As a rate that is nearly a rounding difference, and it is worth saying why: **the
annotator is already 98% monotone on its own**, so only 141 of 7,662 prefixes change
(plus 2 the other way, from the persistence rule). The aggregate is the wrong place to
look for the effect, because those 141 are concentrated:

* **81 of the 273 calls with a detection point (30%) contain a retraction.**
* **Within those 81, 46% of the prefixes after the detection point are labelled 0 by
  the raw field** — for nearly a third of the calls that matter, the raw target spends
  about half the post-detection span teaching the model to stand down.
* The worst single call retracts on 5 of its 10 annotated prefixes.

So the change barely moves any headline number and changes the lesson in a third of
the interesting calls. That asymmetry is the reason to materialise it as a field rather
than leave every consumer to derive it, and derive it differently.

One thing the raw field does *not* do: inject false-alarm signal. Exactly **one**
prefix in the release has `sufficient_evidence` true on a legitimate call — another
measurement of the same conservatism as §5.7.

**The caveat, stated in the field's own docstring:** `evidence_established` is derived
*with hindsight*. The persistence rule behind `earliest_detection_index` inspects the
*next* annotated prefix, so the value at the detection point itself used information
from one prefix later. For a label that is legitimate — P3 constrains what the
annotator may *see*, and the annotator still saw only `utterances[:k+1]` — but it does
mean this field is not computable from the first k utterances alone, unlike
`risk_score` and `sufficient_evidence`. It is a target, never a feature.

The field was backfilled into every metadata file on disk (1000 released conversations
plus the candidate runs). No split assignment changed, and all three sides still
validate clean.

---

### 5.9 Changing the detection rule: `detect/v2`

§5.7 shipped a corpus whose labels detect 59.3% of scams and raise **zero** false
alarms on 543 legitimate calls. §3's measurement of the label fields explains why
both halves of that sentence are true at once, and the explanation turned out to be
a defect in the rule, not in the annotator.

**The diagnosis.** Of the 169 genuine misses (scams with no detection point, setting
aside the 17 unadjudicated ones), **136 — 81% — peak at exactly `risk_score` 0.7 with
`sufficient_evidence` false.** The annotator had seen enough to score them 0.7 and
declined to flip the boolean. Meanwhile the score channel separates the classes far
better than the boolean does: hard negatives peak at 0.3 for 180 of 366, and only 5
of the 366 ever reach 0.7. The misses are also spread over 23 of the 26 scam
families, so this was systemic — a bar question, not a few broken families.

The old rule required **both** `sufficient_evidence` and `risk_score >= 0.5`. The
threshold was inert (no prefix in the corpus is sufficient below 0.5), so in practice
the rule *was* the boolean, and the boolean is the lossiest step in the chain. The
persistence clause had already been built on the opposite judgement — it checks the
next prefix's *score*, not its boolean, with the note "the score is the well-calibrated
channel, the boolean is the noisy one". `detect/v2` applies that same conclusion to
the gate itself: **first prefix with `risk_score >= 0.7`, persistence unchanged.**

**Measured on the same 1000 conversations, zero model calls:**

| | detect/v1 | detect/v2 |
|---|---|---|
| Scam detection rate (all 457) | 0.593 | **0.856** |
| — detectable scams only (440) | 0.616 | 0.886 |
| Listener-complies slice (108) | 0.417 | **0.741** |
| Slow-burn scams (137) | 0.642 | 0.920 |
| **False alarms on hard negatives (366)** | **0** | **0** |
| **False alarms on plain legitimate (177)** | **0** | **0** |
| Unadjudicated no-evidence slice (17) | 0 fire | 1 fires |
| Median detection position | 64% | 56% |
| Detections at utterance 0 | 1 / 271 | 2 / 391 |
| `evidence_established` positive rate | 11.7% | 18.4% |

+0.26 detection at no measured false-alarm cost, and the weakest slice — the listener
who is talked round — gains the most. The position guardrail still passes with room:
median 56% into the call against a 25% alarm threshold, and 0.5% of detections at
utterance 0 against a 30% threshold.

**The persistence rule is what makes it free, and that is measured, not assumed.**
Re-deriving the same batch with persistence switched off: detection 0.910 instead of
0.862, and **5 false alarms appear** (4 hard negatives, 1 plain). So the rule costs
0.048 of detection and buys away every false alarm the threshold move would otherwise
have caused. A hard negative that spikes to 0.7 does not stay there; a real scam does.

**What this change is, and is not.** It is a move of the **operating point**, using
scores the annotator already produced. It does not make the annotator better, and it
is justified by correlation with `is_scam`, **not** by human judgement that 0.7 is
enough evidence to warrant warning a real person. Detecting a scam at a point where a
careful listener could not yet justify it is a lucky guess, not a detection, and this
table cannot tell the two apart. The 0.7 band remains the question the human labelling
round has to answer, and `sufficient_evidence` stays in every record untouched so that
a human verdict can overturn this without re-annotating anything.

**One conversation lost its detection point**, which is worth naming because it shows
the threshold is used in two places. `run-7-v4-00005` scored `0.7, 0.6, 0.8, 0.6` on
its last four annotated prefixes; the 0.8 passed persistence at the old threshold
(the next prefix's 0.6 ≥ 0.5) and fails it at the new one (0.6 < 0.7). Raising the bar
also tightens the persistence test. One sample in a thousand, and the coupling is
deliberate: a detection should be followed by a prefix still above the bar.

**Mechanics.** A new command, `rederive`, recomputes detection points from labels
already on disk — zero model calls, zero dialogue noise. It is not `relabel`:
`relabel` re-runs the annotator and changes `risk_score` / `signals` /
`sufficient_evidence` (and costs money); `rederive` does not touch a byte of the
annotations, only `earliest_detection_index` and the two fields derived from it. Every
record keeps a `rederived_from` block with its previous rule and detection point, and
`generation.evaluation.detection_rule` names the rule that produced it. Like `relabel`,
it refuses to write over its source.

The release is `ds3-rubric-v3-detect07`: same transcripts, same annotations, same
split assignment as `ds2-rubric-v3` (verified id by id), new detection points.

### 5.10 Baselines: seven hosted detectors, on eval and on test

Full numbers and the workflow for adding a model are in `BENCH_REPORT.md` (bilingual). What
belongs here is what the exercise said about the *eval system*, not about the models.

**At the operating point every vendor ships, all seven score below the "never alarm" baseline**
on prefix accuracy (0.371–0.773 against 0.837), with hard-negative false-alarm rates of
0.263–0.974 and alarms 4–6 utterances before the label. At a threshold tuned on a disjoint
100-call subset of **train**, six of the seven are good: prefix accuracy 0.892–0.939,
hard-negative false alarms 0.000–0.171. **Same requests, same model outputs, same scores on
disk; one number changed.** So:

* **A detector for this task is a score plus an operating point**, and the two must be quoted
  together. "Model X gets 0.50 prefix accuracy" is not a statement about model X.
* **The threshold is tuned on train, never on eval or test**, under an objective frozen in code
  before any result was read (`bench/tune.py`). `bench-tune` exits 2 on a path containing
  `/eval/` or `/test/` — the guard is in the CLI because a document does not stop a command
  typed in haste.
* **The vendors' own `alarm` boolean is not usable here**: 26%–82% of hard negatives trigger it.
  It is reported as its own row, never mixed into our decision.

**The eval/test pair now has a difficulty control.** None of these systems trained on anything,
so their eval→test gap estimates how much harder one split is than the other. Six of the seven
lose at most 1.5 points of balanced accuracy and four *detect more* on test. That means a
trained model that drops materially more than ~1.5 points is losing it to **family
generalisation**, not to test being harder — which is the whole reason the test side holds out
whole families. The one outlier is Jev at −0.055, whose tuned threshold does not transfer
(hard-negative false alarms 0.079 on train → 0.053 on eval → **0.143 on test**, outside the
budget it was chosen to meet).

**Two findings about the data rather than the models.** First, `claude-opus` and `gpt-sol` raise
**zero** false alarms on both negative classes on both splits, and every one of their
prefix-level false positives is an *early* alarm on a genuine scam — so the prefix metric
penalises exactly the behaviour the product wants, and a raw false-positive count is not a
ranking. Second, the `compliance` slice (receiver cooperates) is the rubric's weakest at 0.741
label-side, and five of seven systems beat the labels on it.

**The bug that made the first version of this table worse than reality** is why the decision is
now score-only. It had been written `fired = is_alarm or risk_score >= threshold`, and the
model's boolean short-circuited the threshold: 0.5 → 0.95 moved the false-alarm rate only
0.92 → 0.82. **This was the second instance of this bug class in this project** — §5.9 above is
the first, where `sufficient_evidence` short-circuited the label side's 0.5 threshold into a
no-op. A third followed: `hard_negative_false_alarm_rate or 1.0` treated a *genuine zero* as 1.0,
because `0.0` is falsy, and silently discarded the entire usable-threshold plateau of the best
system. All three share one symptom: *a threshold that barely changes the outcome is not
robustness, it is an inert knob.* The fix also moved judgement out of the run files: they now
store `risk_score`, the model's `is_alarm`, signals and rationale as raw evidence, and
`bench-table` computes the decision at scoring time, so re-sweeping a threshold is free.

---

## 6. How we measure changes to the evaluation itself

Four pieces of machinery exist purely to keep us honest.

### Re-annotation on fixed transcripts

The dialogue model accepts no seed, so regenerating a batch produces different
conversations and a per-sample before/after comparison conflates the rubric change
with dialogue variation. The `relabel` command therefore re-runs **only** the
annotator over transcripts already on disk, leaving the conversations untouched,
and reports the label shift (gained / lost / earlier / later / unchanged) against
the previous labels. It refuses to write over its source directory, because the
old labels are the other half of the comparison. Every re-annotated record keeps
a `relabelled_from` block naming the source run, the previous rubric version and
the previous detection point.

### Rubric versioning

The annotation rubric now carries a hand-maintained version
(`prefix-eval/v3`) and a content fingerprint of the prompt text, both recorded in
every record. The version states intent; the fingerprint catches the mistake of
editing the prompt without bumping the version. This matters concretely: four
batches on disk were produced by three different rubrics, and before this change
their metadata was indistinguishable — identical model, identical temperature,
identical threshold. Assembling a training set from two of them would have
produced systematically inconsistent labels with nothing to warn anyone.

### Human prefix labelling

The calibration set is collected through a dedicated interface
(`python -m callfactory label <run>`) built as a measuring instrument rather than
for convenience, with three constraints:

- **It shows no machine label and no ground truth.** Seeing `detect@7` before
  judging would measure an anchoring effect, not a human judgement. There is a
  test asserting that none of `risk_score`, `sufficient_evidence`, `is_scam`,
  `hard negative` or the family name can appear in its output.
- **It reveals one round at a time and cannot go back.** The label means "was
  there enough evidence having heard only the first *k* turns"; reading the whole
  transcript first contaminates exactly that judgement, because a reader who
  knows how the call ends cannot un-know it when pointing at turn 4.
- **It is resumable.** Labelling 80 conversations is several sittings; already
  labelled items are skipped on the next run, and the file is append-only so a
  revised label is auditable against the original.

Human labels land in `human_labels.jsonl`, kept separate from the machine labels;
the agreement computation aligns the two afterwards.

### A guardrail against drifting earlier

"Earlier" is not monotonically better. A rubric that fires at the first turn of
every call would score a perfect detection rate and produce a dataset in which no
two detectors can be distinguished. So every batch and every re-annotation reports
the distribution of detection position as a fraction of call length, and warns
when the median drops below 25% of the call or when more than 30% of detections
land on the very first utterance.

Measured on the three most recent batches, the median detection point sits at
**56%, 58% and 67%** of the call, with no detections at all on the first utterance
and none within the first 20%. So the current risk is not collapse toward the
start — if anything these detection points are on the late side, which is the
other failure direction and worth watching as the rubric evolves.

---

## 7. Status and known limitations

Stated plainly, because several of these bound how much the numbers above can be
trusted.

**Implemented and exercised against live models:** scenario bank (v3: 57 families,
25 hand-written and 32 authored, versioned), scenario-author agent with its three
acceptance gates and a review/commit workflow, sampler, caller and receiver agents
with 12 + 12 hand-written personas, prefix annotator, programmatic validation,
batch generation with checkpointing and resumption, re-annotation, human-labelling
interface (terminal and a browser console), calibration split and agreement,
family-balanced sampling, three-way train/eval/test split with whole families held
out, detection-rule re-derivation at zero cost, cost accounting, and the benchmark
harness: the streaming detector interface, the replay adapter for stateless judges,
adapters for Anthropic / OpenAI-compatible / Gemini / Jev, resumable runs with a
request cap, train-side threshold tuning under a frozen objective, and scoring with
slices and degenerate baselines. 432 tests, all runnable offline.

**Specified but not implemented:** the dataset gatekeeper (component 6). That gap
now matters more than it did: 32 of the 57 families are machine-written premises
admitted on a human read, with no automated realism or detection-value score behind
them. The harness now produces the gatekeeper's input — `discrimination_feedback`
returns the conversations no system discriminates on (9 of 119 on eval) and the ones
where every system disagrees with the label (0) — but nothing consumes it yet.

**Baselines exist for seven hosted detectors, on both eval and test** (§5.10), each at a
threshold tuned on the same train subset. Jev is among them. The test side has now been
spent for these seven: it was run once, at thresholds fixed beforehand, and the run files
are in the repo so nobody needs to spend it again. A candidate model should still be final
before it is run on test — the protective rule has not changed, the baselines simply no
longer need it.

**The rubric is still not validated against human judgement, but the gap is now measured.**
A first human round has landed: two of three annotators labelled 25 eval calls independently.
They agreed on the *verdict* for 16 of 21 shared calls, but on the *detection moment* for only
3 of 14 — a median gap of 3 utterances, one-sided, with one annotator never earlier than the
other. One annotator's detection points sit a median of 0 utterances from ours, the other's a
median of 4 earlier. **The timing dimension, the only thing this dataset adds over a
conversation-level corpus, is where humans agree least** — and that round gave the annotators
no written anchor rules, so it measured intuition rather than the rubric. Both annotators also
called a hard negative (`genuine_tax_office_letter_followup`) a scam, which says the hard
negatives are hard rather than that the annotators were careless. The third annotator's 25 are
not collected, so there is no 2-of-3 majority and no way yet to say which annotator is the
outlier. Until there is, the rubric's remaining support is indirect: false alarms on legitimate
calls stayed at zero while detection improved, the annotator's independent whole-call judgement
disagrees with ground truth only 10% of the time, and detection position has not drifted toward
the start of the call.

**The rubric was tuned on the same twenty samples used to diagnose it.** Three
rubric revisions, each justified by inspecting the same batch, with no held-out
data. The defence is that each change was derived from a *named mechanism* — a
misattributed subject, a reactive-only anchor set, a self-retracted spike — rather
than from score-chasing, and mechanism-driven fixes generalise better than
score-driven ones. It is still unvalidated. The planned correction is to split the
human-labelled calibration set, tune against the larger part and check once
against a held-out remainder.

**Batch size.** The current release is 1000 conversations (840 in one run), so the
aggregates in §5.7 are stable. Everything in §4, by contrast, was diagnosed on runs
of twenty: those findings are mechanisms observed on small batches, not measured
effect sizes. Nothing at single-sample granularity is significant at either scale.

**Reproducibility is partial.** Generation *tasks* are fully reproducible from
`(bank_version, seed, index)`. The *conversations* are not, because the dialogue
model accepts no seed. Each record states this rather than implying bit-level
reproducibility.

**The four signals that had no hard-negative counterpart now have one.**
`refusal_to_verify`, `callback_evasion`, `secrecy_request` and `inconsistent_details`
were uncovered in bank v2; bank v3 closes all four (§5.5), and the premises are
plausible — a fraud team that puts a call on hold and rings back on the listed
number, a training provider calling from a withheld number but confirming by email,
a genuine investigator asking for discretion while welcoming independent checking,
a real ticket office whose junior staff member gets a detail wrong. They are
premises, not evidence: no conversation has yet been generated from them, so
whether the annotator actually holds its nerve on them is unmeasured.

---

## 8. Reproducing this

```bash
pip install -e ".[dev]"
pytest -q                                    # 404 tests, no API keys needed

python -m callfactory init-bank              # write the hand-written seed bank
python -m callfactory review-bank            # coverage gaps: what the bank is missing

# expand the bank (component 1). --dry-run holds the candidates for review;
# --append keeps an earlier round's candidates and dedups against them.
python -m callfactory author-scenarios --num 10 --dry-run --use-gaps \
    --provider anthropic --model <author-model>
python -m callfactory review-candidates      # per-candidate review + a readable .md, zero calls
python -m callfactory commit-candidates --drop <family> --rename <old>=<new>

python -m callfactory plan --num-samples 20 --seed 7      # what would be generated, zero requests
python -m callfactory estimate --num-samples 20 --seed 7   # predicted call count and peak concurrency

# fully offline, no keys, mock provider end to end
python examples/run_dialogue.py --index 1 --label

# live generation; the annotator must use a different provider (P4)
python -m callfactory generate --num-samples 20 --seed 7 --max-rounds 10 \
    --provider anthropic --model <dialogue-model> \
    --eval-provider openai --eval-model <annotation-model> \
    --eval-stride 2 --concurrency 4

python -m callfactory validate datasets/candidates/run-7   # leakage, pairing, label structure
python -m callfactory inspect  datasets/candidates/run-7 --index 0

# re-annotate fixed transcripts under the current rubric
python -m callfactory relabel datasets/candidates/run-7 --eval-provider openai

# collect human prefix labels for calibration (shows no machine label, round by round)
python -m callfactory label  datasets/candidates/run-7
python -m callfactory label  datasets/candidates/run-7 --status
python -m callfactory calibrate datasets/candidates/run-7 --make-split 20   # before any analysis
python -m callfactory calibrate datasets/candidates/run-7                   # agreement, tune only

# train / eval split; refuses a corpus that mixes rubric versions
python -m callfactory split datasets/candidates/calib-101 --name ds1-rubric-v3 --ratio 0.8 --seed 11
```

The exact two commands that produced `ds2-rubric-v3` (§5.6, §5.7). Families are
balanced with the 160 conversations already on disk counted in, and the split is
three-way with whole families held out:

```bash
python -m callfactory generate --num-samples 840 --seed 1101 --run-id run-1101 \
    --balance-families \
    --existing datasets/train/ds1-rubric-v3 datasets/eval/ds1-rubric-v3 \
    --provider anthropic --model claude-sonnet-5-5 \
    --eval-provider openai --eval-model gpt-4.1-mini \
    --min-rounds 5 --max-rounds 10 --eval-stride 2 --concurrency 10

python -m callfactory split \
    datasets/candidates/run-1101 \
    datasets/train/ds1-rubric-v3 datasets/eval/ds1-rubric-v3 \
    --name ds2-rubric-v3 --by family-holdout \
    --holdout-family-ratio 0.1 --eval-ratio 0.1 --seed 1101
```

The dialogue model accepts no seed, so re-running the first command produces
different conversations with the same task list; `(bank_version, seed, index)`
reproduces the *tasks* exactly, which is the part that is reproducible at all.

Running a detector against the result, and the three commands behind §5.10. Step 2
exits 2 if the split path contains `/eval/` or `/test/`:

```bash
# 1. run a system on a split; one request per labelled prefix, resumable,
#    --cond c2 minimises reasoning, --max-requests caps the spend
python -m callfactory bench datasets/bench/_tune-train-100 \
    --system deepseek-flash --cond c2 \
    --out datasets/bench/tune-c2/deepseek-flash.c2.jsonl --concurrency 4

# 2. pick a threshold on the TRAIN side, under the frozen objective (zero calls)
python -m callfactory bench-tune datasets/bench/_tune-train-100 \
    --runs datasets/bench/tune-c2

# 3. run eval once, then score at that threshold (re-sweeping is free)
python -m callfactory bench datasets/eval/ds3-rubric-v3-detect07 \
    --system deepseek-flash --cond c2 \
    --out datasets/bench/ds3-eval-c2/deepseek-flash.c2.jsonl --concurrency 4
python -m callfactory bench-table datasets/eval/ds3-rubric-v3-detect07 \
    --runs datasets/bench/ds3-eval-c2 --threshold 0.85
```

The full requirements specification, including every numbered requirement
referenced above, is in [REQUIREMENTS.md](REQUIREMENTS.md).
