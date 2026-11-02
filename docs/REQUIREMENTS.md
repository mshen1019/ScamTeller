# ScamTeller call factory — Requirements v0.1

A synthetic phone-conversation factory that produces training / eval data for early-warning
scam detection.

This document is the basis for implementation and the acceptance criteria; it contains no
implementation. All code and identifiers are in English, and the generated data is in **English**.

The predecessor `scam-convo-gen` is kept as a prototype and no longer evolves; its provider layer,
validators and schema are carried over and refactored.

---

## 0. Design principles (every component must obey them)

| # | Principle | Why |
|---|---|---|
| P1 | **Information asymmetry**: the caller knows the scenario, the receiver knows only its own persona and what has been said aloud, the evaluator knows only the prefix | The receiver's wariness must be evidence, not a restatement of the label |
| P2 | **Labels physically separated from inputs**: the transcript file structurally contains no label fields | Prevents downstream models from reading hidden labels |
| P3 | **Prefix labelling**: the evaluator's k-th call receives only `utterances[:k+1]`, with no memory across calls | Early-warning labels must be causally available |
| P4 | **Judge and judged come from different sources**: the evaluator uses a different provider from the caller/receiver; the gatekeeper is different again from the evaluator | Avoids grading your own work |
| P5 | **Reproducible**: `(bank_version, seed, index)` uniquely determines one generation task | Datasets are traceable and rebuildable |
| P6 | **Diversity is computed by the program, not delegated to an LLM** | An LLM cannot see the already-admitted samples, so it cannot judge cross-sample repetition |
| P7 | **The whole pipeline runs offline**: a mock provider drives the full chain | Development without keys; CI can test it |
| P8 | **Safety boundary**: produce only observable conversational behaviour, never an executable scam procedure | The data is for defence and must not become a script |

## 1. Data flow

```
human context ──┐
                ├─→ [1 Scenario author agent] ─→ datasets/scenario_banks/<name>.v<N>.jsonl
existing bank ──┘                                  │  (human-editable)
                                                   ↓
                                         [2 Scenario sampler] ← seed
                                                   ↓  scenario × caller persona × receiver persona = task
                                   ┌───────────────┴───────────────┐
                                   ↓                               ↓
                           [3 Caller agent] ←─ alternate ─→ [4 Receiver agent]
                                   └───────────────┬───────────────┘
                                                   ↓  transcript
                                         [5 Evaluator agent]  labels each prefix
                                                   ↓  + prefix labels
                                         program checks + Diversity computation
                                                   ↓
                                         datasets/candidates/
                                                   ↓
                                         [6 Gatekeeper agent]  four-dimension scoring
                                                   ↓
                                   admitted/  |  rejected/  |  review/
                                                   ↓
                                         (separate command) train / eval split
```

## 2. Component requirements

### Component 1 — Scenario author agent

Uses an LLM to grow the scenario bank, replacing the prototype's fixed enumeration.

**Inputs**
- `--context`: optional free-text guidance from a human. Examples: "supplement repurchase scams targeting elderly people who live alone", "add an impersonating-the-property-manager type". `--context-file` reads longer text.
- `--num N`: number of new scenarios in this run
- `--bank <path>`: the target scenario bank. **Existing scenarios must be passed in as deduplication context**, otherwise the same types are generated over and over
- Intended category mix (scam / legitimate / hard negative / late-onset)

**Output**: appended to the scenario bank JSONL. Every entry must be a structured record, not free text:

| Field | Requirement |
|---|---|
| `scenario_id` | Stable and unique, of the form `scn-<bank>-<seq>` |
| `category` | `scam` \| `legitimate` |
| `family` | Short snake_case name, semantically unique within the bank |
| `pretext` | One sentence: the reason given for the call |
| `org_kind` | The kind of organisation the caller claims to be (fictional) |
| `expected_signs` | **Must fall entirely inside the closed vocabulary**; entries with out-of-vocabulary terms are rejected |
| `hard_negative` | bool; when true, `expected_signs` must not be empty and `disambiguator` must be filled in |
| `disambiguator` | Hard negatives only: how it behaves like a genuine organisation when challenged (e.g. quotes a ticket number only the real organisation would know, welcomes a call back) |
| `late_onset` | bool, may be true only for scams |
| `compliant_outcomes` | Subset of `plausible_outcomes`: endings where the receiver **is persuaded**. Scams only. See R2.8 |
| `no_observable_evidence` | bool, scams only: the intent is fraud but no decidable evidence ever surfaces. Must not declare strong signals. See R7.8 |
| `eval_only` | bool; kept out of the training set by default and goes to eval only (enforced by the split command) |
| `plausible_outcomes` | 2–4 candidate endings; the sampler picks one |
| `provenance` | `{author_model, context, created_at, bank_version}` |

**Requirements**
- R1.1 Read the existing scenario bank before generating; the prompt lists the existing families and **explicitly forbids repeating them**
- R1.2 Before a new scenario enters the bank, run schema validation plus a semantic near-duplicate check against existing scenarios (similarity of family name + pretext); above the threshold, reject and retry
- R1.3 Entries whose `expected_signs` fall outside the closed vocabulary are **dropped whole**, with no guessed mapping
- R1.4 **A hard negative must carry a `disambiguator`.** A hard negative with no "how it behaves when challenged" is just an ordinary negative and has no training value
- R1.5 The scenario bank is **versioned**: each write produces a new version file, never an in-place rewrite; `bank_version` goes into the provenance of every data item
- R1.6 Support `--dry-run`, which prints candidate scenarios without adding them, for human review
- R1.7 The scenario bank is **human-editable plain text**; after a manual edit, bumping the version number is enough for the sampler to use it
- R1.9 **The granularity of `--context` is a key behaviour keyword**, not a structured template. Examples: `--context "bitcoin / crypto investment scam"`,
  `--context "impersonating the property manager to demand arrears"`. The agent expands the keyword into complete structured scenario records
- R1.10 Must support an **iterative gap-filling workflow** (the main way it is used):
  1. `author-scenarios --num 20` lets the agent generate a batch freely
  2. `review-bank`: a human reviews what the existing families cover and what is missing
  3. `author-scenarios --context "<the missing type>"` fills the gap in a targeted way
  4. Back to 2 until coverage is satisfactory
  This makes R1.1 (passing the existing bank in as deduplication context) a **hard requirement**; otherwise step 3 generates duplicate types
- R1.8 Safety: never generate a pretext containing an executable payment channel or credential-theft steps

**Companion: the scenario bank review view (`review-bank`)**
- R1.11 List every family currently in the bank, grouped by `category` / `hard_negative` / `late_onset`, with counts for each
- R1.12 Flag **coverage gaps**: which closed-vocabulary signals no scenario covers yet; how hard negatives pair with scams
  (which scam families have no matching hard negative yet to guard against false alarms)
- R1.13 This is what a human uses to decide what to add next; it must make zero model calls and return immediately

**Acceptance**
- Given `--context "impersonating the property manager"`, the new scenarios include a property-management family that does not duplicate any existing family
- A model output with an out-of-vocabulary term in `expected_signs` → that entry is dropped and the rest are added normally
- A hard_negative with no `disambiguator` → rejected

---

### Component 2 — Scenario sampler

A pure function; zero model calls.

**Inputs**: scenario bank, `seed`, `num_samples`, mix constraints, persona banks

**Output**: a list of `GenerationTask` = scenario × caller persona × receiver persona × outcome × target number of rounds × per-round directives

**Requirements**
- R2.1 `(bank_version, seed, index)` uniquely determines a task, via mixed derivation rather than `seed + index`
- R2.2 **Stratified sampling**, not pure random: stratify by `category` → `hard_negative`/`late_onset` → `family`, so small families are not swamped by large ones
- R2.3 Controllable mix: `--scam-ratio`, `--hard-negative-ratio`, `--late-onset-ratio`; warn when the actual mix deviates from the target by more than the tolerance
- R2.4 Support `--include-family` / `--exclude-family` for targeted generation
- R2.5 **The same scenario may be sampled many times, but the persona combination must differ** — this is the main source of diversity, so the bank does not need to be large
- R2.6 The `plan` subcommand prints what would be generated without sending any request
- R2.7 Each task records the triple `(scenario_id, caller_persona_id, receiver_persona_id)`, used later for Diversity and leakage-proof splitting
- R2.8 **Compliant outcomes must make up a share of the data** (`--compliance-ratio`, default 0.25).
  Measured motivation: in seed bank v1, 5 of the 6 scam outcomes were "the receiver resists" and none
  of the 5 legitimate outcomes were — "the receiver resists" separated the two classes almost perfectly, so a model could
  score by **reading the receiver** without looking at the caller at all,
  which directly contradicts the premise of R5.9. Both sides need filling: add compliant outcomes to scams and resisting outcomes to legitimate calls,
  and make both sides contain **word-for-word identical** receiver behaviour (e.g. "the receiver insists on looking up the number and calling back"),
  so the only difference is how the caller reacts.
  Compliant outcomes are dealt only to receivers with `skepticism ∈ {very_low, low}` — having a highly sceptical persona "be persuaded"
  fights the persona; when not enough can be matched, warn rather than force it.
  The safety boundary is unchanged (P8): the call ends the moment the receiver agrees, and never plays out the action itself

**Acceptance**: two samplings with the same seed are identical field by field; over 500 samples each family's share deviates from the stratification target by less than the tolerance; when a scenario appears several times, the persona triples do not repeat

---

### Component 3 — Caller agent

**The persona bank is separate** (not shared with the receiver). Each caller persona contains at least:

| Dimension | Examples |
|---|---|
| `role` | account services specialist / security operations / case handler / collections |
| `speech_style` | over-rehearsed, reading off a screen, very fast talker, softly apologetic |
| `pressure_profile` | how pressure is applied: direct threat / manufactured time window / emotional leverage / authority |
| `challenge_response` | **behaviour when challenged** — the most important dimension and the real boundary between a scam and a hard negative |
| `competence` | practised / clumsy (the clumsy ones contradict themselves) |

**Requirements**
- R3.1 The system prompt contains all the scenario information + its own persona + the intended outcome + the target number of rounds
- R3.2 The conversation history is rolled up from **its own point of view**: its own lines are `assistant`, the other side's are `user`
- R3.3 When `late_onset=true`, the first few rounds must read as an entirely ordinary call
- R3.4 Placeholders are mandatory: all proper nouns and identifying numbers use `[PERSON_1]`, `[COMPANY_A]`, `[ACCOUNT_REF]`, `[AMOUNT_1]` and so on
- R3.5 The safety boundary is written explicitly into the prompt and re-checked on the output side by validators
- R3.6 Output is pure speech: no stage directions, no speaker prefixes, no quotation marks; a cleaning layer strips any that slip through
- R3.7 Each round injects a sampled directive (interrupt / stammer / mishear / go back and ask again / contradict yourself / emotion flattens out); about 1/4 of rounds get no directive, so it does not become a verbal tic
- R3.8 The last round injects a wrap-up prompt

### Component 4 — Receiver agent

**The persona bank is completely separate from the caller's.** Each entry contains at least:

| Dimension | Examples |
|---|---|
| `skepticism` | very_low / low / medium / high |
| `tech_familiarity` | affects whether technical claims are understood |
| `emotional_state` | calm / anxious / tired / irritable |
| `situation` | what they are doing when they answer (cooking, driving, minding children, queuing) |
| `speech_style` | short sentences, goes off topic, repeats back to confirm, talks over, nervously over-explains |
| `prior_exposure` | whether they have been scammed or heard of scams — affects whether the source of their wariness is plausible |

**Requirements**
- R4.1 **None of `category`, `family`, `pretext` or `expected_signs` may appear in the prompt.** An automated test must assert that none of them appear in the receiver prompt
- R4.2 React only to what is heard; the prompt says explicitly "do not presume good or bad intent"
- R4.3 May mishear, be confused, go off topic, give in, deflect or lose their temper
- R4.4 Outputs `[END_CALL]` to hang up; **before `min_rounds` is reached the request is ignored** and the conversation continues
- R4.5 Never reveals real personal data; always placeholders
- R4.6 Uses a different seed offset from the caller, to avoid correlated sampling

**Requirements for both**
- R4.11 **The first version of the persona banks is hand-written**, 8–12 each for caller and receiver, with no separate persona-generation agent. Diversity comes from
  the cross product scenario × caller persona × receiver persona (12 × 12 = 144 combinations is enough); persona quality matters more than quantity
- R4.7 At least 5 rounds, configurable up to 20; strict alternation starting with the caller
- R4.8 An empty reply is backfilled with one line; past the consecutive limit, the task fails and is recorded in failures
- R4.9 Adjacent messages from the same role are merged (some backends reject consecutive messages with the same role)
- R4.10 ~~High conversation temperature (default 0.9) for diversity~~ — **no longer valid, measured 2026-10**: Claude's Messages API
  returns 400 for any non-default `temperature` / `top_p` / `top_k` (`temperature is deprecated for this model`).
  Diversity therefore can come **only** from the task cross product (scenario × 12 caller personas × 12 receiver personas × outcome × late_onset)
  and the per-round directives; there is no knob to turn at the sampler level. OpenAI-compatible endpoints still accept temperature,
  so the evaluator side (temperature 0) is unaffected. Every data item records `temperature_honoured_by_provider`

---

### Component 5 — Evaluator agent (prefix-level labelling)

**Requirements**
- R5.1 **One independent, stateless call per prefix**; the k-th call receives only `utterances[:k+1]`; conversational accumulation is forbidden
- R5.2 There must be a test asserting that the content of future rounds appears in no evaluator request
- R5.3 Temperature 0, for reproducible labels — **this holds only when the provider accepts temperature**. When the evaluator runs
  on Claude there is neither temperature nor seed, so labelling the same prefix twice may differ; this fact is recorded in every data item via
  `temperature_honoured_by_provider` / `seed_honoured_by_provider`, not just written in the docs
- R5.4 **Must use a different provider from the caller/receiver**; on the same source, warn at startup (`--allow-same-provider` overrides explicitly and is recorded in metadata)
- R5.5 **Blind labelling**: must not see `category` / `family` / `hard_negative` / `late_onset`
- R5.6 Per-call output: `risk_score` ∈ [0,1], `risk_tier`, `signals` (everything evidenced within the prefix), `new_signals` (added in the last round), `sufficient_evidence`, `rationale` (≤240 characters, must point at evidence)
- R5.7 `signals` is a strict closed vocabulary; out-of-vocabulary terms are **dropped**, not mapped by guessing
- R5.8 `signals` is forced to be monotonic (once evidenced it never disappears); `risk_score` keeps its raw value and may go down
- R5.9 Judge only the caller's behaviour. **Close both directions**: the receiver's suspicion is not itself a risk signal (does not raise the score),
  and the receiver's cooperation, trust or agreement **does not lower the score** either.
  The second half was added after measurement: a fully played-out pickup-fee scam paired with a receiver who cooperated throughout,
  where the evaluator listed every fact correctly yet judged `sufficient_evidence=false` on all six prefixes,
  and the score dropped from 0.60 to 0.30 the moment the receiver gave in.
  The root cause was that almost all anchors were "reactive" (they required the receiver to push back first), so **whether a scam was decidable depended on
  the receiver's resistance** — the same shortcut R2.8 cuts, only living in the rubric instead of the data.
  The anchors are therefore split into two groups: **proactive** (things the caller does without needing the other side's cooperation) and **reactive**
  (things that surface only when the other side pushes back), with an explicit statement that "the absence of a reactive anchor means nothing when the other side never pushed back"
- R5.10 `earliest_detection_index` = the first prefix that satisfies both `sufficient_evidence` and `risk_score >= threshold`,
  **and is not overturned by the next labelled prefix** (the next prefix's `risk_score` is still >= threshold).
  Measured motivation: a hard negative spiked on one prefix and set `sufficient_evidence`, after which
  the evaluator itself brought the score back to 0.20, but "take the first prefix that satisfies the condition" turned that self-overturned misjudgement
  into the ground-truth detection point for good. Only the next prefix's **score** is checked, not the boolean: the score is the well-calibrated
  channel (measured hard negatives 0.00–0.20 vs scams 0.60–0.90), while the boolean is noisy.
  Real scams' scores rise all the way, so this rule costs them almost nothing
- R5.11 Unparseable output degrades to an empty label and is recorded in failures; it must not abort the whole batch
- R5.12 Prefixes are independent of one another and are **sent concurrently**
- R5.13 A separate whole-conversation quality call produces `blind_is_suspicious` (its own independent judgement), used to cross-check label quality against ground truth; this result is **never used as an early-warning label**
- R5.14 Cost: `--eval-stride` downsamples, `--no-prefix-eval` turns it off

**Acceptance**: build a conversation and assert that the request body of the 3rd call does not contain the text of round 4; output with an out-of-vocabulary signal is dropped; a sequence in which a signal disappears is corrected to be monotonic

---

### Component 6 — Dataset gatekeeper

Decides whether a candidate data item enters the final dataset. **It judges admission only and does no train/eval split** (splitting is a separate command).

**Four-dimension scoring, each dimension 1–5 and each with anchor descriptions.** Bare numbers are not comparable, so anchors are a hard requirement:

| Dimension | How it is scored | Anchor examples (all 5 levels must be given in full in the prompt) |
|---|---|---|
| **Realism** | LLM | 2 = both sides speak in complete sentences with no interruptions or self-corrections, like two chatbots; 4 = natural slips and interruptions, but the receiver's reactions are idealised; 5 = reads like a transcript of a real call recording |
| **Consistency** | LLM + program cross-check | Whether identity, reason, details and emotional reactions stay consistent; **specific contradictions must be listed** (which round conflicts with which), not just a score |
| **Diversity** | **Computed purely by the program; the LLM is not asked** | Maximum similarity to the admitted pool; see below |
| **Detection Value** | LLM, needs the prefix labels | Whether it contains decidable risk signals; **too obvious (decidable at T0) scores as low as having no signal at all** |

**How Diversity is computed (R6.1)**
- Two levels: ① task-fingerprint repetition — how many times the `(family, caller_persona, receiver_persona, outcome)` combination already appears in the admitted pool; ② text similarity — maximum shingle similarity to the admitted pool
- The index over the admitted pool is maintained incrementally; no full pairwise comparison
- Above the threshold, reject directly, **without spending an LLM call**

**Other requirements**
- R6.2 **Must use a different provider from the evaluator**, so labels and scores do not share a source (P4)
- R6.3 **Staged masking**: Realism / Consistency are scored without ground truth or prefix labels (otherwise the labels lead the judgement); prefix labels are given only for Detection Value. In practice this is two separate calls
- R6.4 Output `decision` ∈ `admit` / `reject` / `review`, with `reasons: list[str]` — every rejection must be auditable down to a specific dimension and specific rounds
- R6.5 Thresholds are configurable, **but the first version does not pick them out of thin air**: first run a calibration batch, have a human independently score **50–100 items**, compute
  agreement with the gatekeeper (a correlation coefficient for each of the four dimensions + kappa on the decision), then set the thresholds. The calibration script and calibration set go into the repo.
  A dimension with too little agreement means its rubric anchors are not clear enough: change the rubric, not the threshold
- R6.6 Verdicts are written to a separate `decisions.jsonl`, apart from the candidate data, keeping a fully traceable chain: `(conversation_id, scores, decision, reasons, judge_model, rubric_version)`
- R6.7 When `rubric_version` changes, past verdicts are not invalidated automatically but must be identifiable, and re-judging under the new rubric is supported
- R6.8 Program-side hard rules (schema / role alternation / number of rounds / label structure / leakage / placeholders) run **before** the gatekeeper; items that fail them do not waste LLM calls
- R6.9 The `review` tier goes into its own directory for human handling and is not counted in any set by default
- R6.10 **Companion human review view (the `review` subcommand)**; a human will actually go through items one by one:
  - Show each item's transcript, the gatekeeper's four scores and `reasons`, the prefix label trend, and the ground truth
  - Single-key decisions: admit / reject / skip, with resume after interruption (it remembers where you got to)
  - Human decisions are written to `decisions.jsonl` with `decided_by: human`; they **override** the gatekeeper verdict while keeping the original for comparison
  - Disagreements between human and gatekeeper accumulate automatically and feed directly into the R6.5 calibration set

**Acceptance**: re-judging the same item under the same rubric keeps score fluctuation within tolerance; an item that heavily duplicates an admitted sample → rejected without calling the LLM; an item decidable at T0 → low Detection Value

---

## 3. Data contract

```
datasets/
  scenario_banks/<name>.v<N>.jsonl      scenario banks (human-editable, versioned)
  candidates/<run_id>/
    conversations.jsonl                 transcripts, no label fields at all
    metadata.jsonl                      scenario / generation parameters / prefix labels / diversity
    failures.jsonl                      generation failures and hard-rule failures
    _checkpoint.json                    completed indices
  admitted/<run_id>/                    passed the gatekeeper (same two-file separation)
  rejected/<run_id>/
  review/<run_id>/
  decisions.jsonl                       every verdict, auditable
```

- C1 Each utterance in `conversations.jsonl` has only `index` / `speaker` / `text`; a label key appearing at **any depth** fails validation
- C2 The two files are joined on `conversation_id`
- C3 All files are append-only, flushed record by record, and support resuming
- C4 `metadata` must record `bank_version`, `seed`, `derived_seed`, each agent's provider/model, `seed_honoured_by_provider` and `rubric_version`

## 4. Non-functional requirements

- N1 Concurrency: a semaphore limits the number of **conversations** in flight; note that the N labelling calls inside one conversation are sent concurrently, so peak request count is far above the concurrency setting — the docs must say so
- N2 Retries: exponential backoff with jitter on 429/5xx/timeouts; no retry on 4xx
- N3 Resuming: if the checkpoint is lost, the completed set can be rebuilt from metadata
- N4 Providers: OpenAI / Anthropic / OpenAI-compatible (Fireworks, vLLM, local) + offline mock
- N5 Anthropic has no seed parameter and no longer accepts temperature, so every data item must record
  `seed_honoured_by_provider` and `temperature_honoured_by_provider`; never claim bit-for-bit reproducibility across the whole pipeline
- N6 Cost: the `plan` and `estimate` subcommands give the expected number of calls and order of magnitude
- N7 All tests run offline; HTTP paths are tested for request shape and retries against a local stub server
- N8 CLI: `author-scenarios` / `review-bank` / `plan` / `generate` / `judge` / `review` / `split` /
  `inspect` / `validate` / `estimate` / `calibrate`

## 5. Settled decisions (confirmed in review)

| Question | Decision |
|---|---|
| `review` tier | **Kept**, with a human item-by-item review view (R6.10). Human verdicts override the gatekeeper and feed back into the calibration set automatically |
| Calibration size | A human labels **50–100 items** to set thresholds (R6.5) |
| Granularity of human context | **Key behaviour keywords** (e.g. "bitcoin investment scam"), paired with the `review-bank` view for iterative gap filling (R1.9–R1.13) |
| Persona banks | **Hand-written first version**, 8–12 each, no separate persona-generation agent (R4.11) |
| Detector protocol | **Fixed now**; see the next section |

## 6. EarlyWarningDetector Protocol (reserved for the bench harness)

The bench harness **is implemented** (`callfactory/bench/`, CLI `bench` / `bench-table`). The interface is below.

The system under test only needs to satisfy one Protocol (structural typing: **having these methods is enough — no inheritance, no need to import this package**).

**Streaming is the reference**, because early warning is inherently a streaming task and real-time detectors are stateful:

```python
class DetectorSession(Protocol):
    async def push(self, utterance: Utterance) -> Verdict: ...   # push one turn, return the verdict at this moment
    async def close(self) -> None: ...

class EarlyWarningDetector(Protocol):
    name: str
    def open_call(self, call_id: str) -> DetectorSession: ...
```

```python
@dataclass
class Verdict:
    risk_score: float          # [0,1]
    is_alarm: bool             # whether to alarm at this moment
    signals: list[str] = ...   # optional; must fall inside the closed vocabulary
    latency_ms: float | None   # optional; for real-time systems the latency is itself a metric
```

**Requirements**
- R7.1 Provide a `ReplayAdapter` that automatically wraps a stateless model (a prompted LLM) as a streaming one: each `push` replays the full prefix.
  Token cost is O(N²) and must be flagged in the report.
  **~~Jev is natively streaming, uses the native Protocol and bypasses the adapter~~ — corrected to match reality:** Jev's SDK
  (`typesafe-sdk`) has only `system_one(state, questions)`, **with no session and no streaming interface**,
  so "one score per round" is likewise implemented as one call per round carrying the accumulated state. **Jev goes through the same
  `ReplayAdapter` as the LLMs, at the same O(N²) cost.**
  Measured mitigation: 84% of this data's input tokens can hit the prefix cache (the system prompt alone
  is a third of it and is resent 912 times); with caching on, the bill roughly halves. Caching changes only billing, not request content,
  so it does not affect P3 or comparability
- R7.6 **Granularity alignment**: `push` always works at **utterance** granularity (the finest). A model that updates its judgement only once per round
  (such as Jev, one score per round) just returns the same `Verdict` as last time on the intermediate `push`.
  The report gives **both utterance-level and round-level detection latency**, and comparisons across models of mixed granularity must state which one is used,
  because an error of 1 round equals 2 utterances
- R7.7 `risk_score` is continuous (confirmed as supported on the Jev side), so prefix-level AUC / PR curves are available;
  if a model with only binary output is added later, the metrics degrade automatically to latency + false-alarm rate, and the report says so
- R7.2 The model under test **sees only the transcript** (`index` / `speaker` / `text`) and must never touch metadata
- R7.3 Core metrics:
  - **detection latency** = the turn of the model's first `is_alarm` − ground truth `earliest_detection_index`
    (positive means late, negative means early)
  - prefix-level AUC / precision-recall
  - **false-alarm rate on hard negatives** — must be reported together with latency, otherwise the degenerate strategy "always alarm immediately" is rewarded
  - sliced by `family` / `late_onset` / `hard_negative`
- R7.4 Ground truth is produced by the evaluator and is a model judgement, not the true value; the report must also give the
  disagreement rate between `blind_is_suspicious` and `ground_truth.is_scam` as a reference for label quality
- R7.8 **Samples with `is_scam=true` and `earliest_detection_index=None` form their own slice.**
  Note that this is **not** "the scam was not completed" — no scam in this dataset is completed (P8: none of the candidate outcomes in the scenario bank
  has the receiver paying or handing over a verification code). What it means is: the caller never revealed decidable evidence,
  and the person on the phone genuinely had nothing to go on.
  - Latency **cannot be computed** (the ground truth is None) and must not be included in latency statistics
  - It is a **restraint test**: alarming on these samples counts towards the false-alarm rate
  - There are two sources, which must be kept apart: explicit scenarios (`no_observable_evidence: true`, generated on purpose,
    `eval_only` by default) and **generation defects** (the scenario declared `expected_signs` but none were played out).
    The latter are detected on the program side and handed to the gatekeeper; they must not be left in silently — otherwise the batch's class composition goes out of control
- R7.5 **Discrimination feedback**: samples on which every model under test alarms at T0 are marked "no discriminative power"; samples on which no model alarms
  while the ground truth says decidable are marked "suspected label error"; both are fed back to the gatekeeper as a reference for later admission

## 7. Confirmed on the model-under-test side (Jev)

| Item | Decision | Impact |
|---|---|---|
| Streaming / one-shot | One-shot. `system_one(state, questions)`, **no session / no streaming interface** | Goes through `ReplayAdapter` like the LLMs, **at the same O(N²) cost** (previously written as "natively streaming, no O(N²)"; corrected to match reality) |
| Output form | **Continuous risk score** | Prefix-level AUC / PR available; metrics need not degrade |
| Granularity | Round level (= 2 utterances) | See R7.6: the interface advances per utterance, and the report uses both measures |

The requirements are now complete and implementation can begin. Suggested order:

1. Provider layer + schema carried over and refactored (mechanical, no design decisions)
2. Component 2, scenario sampler (pure function, easiest to test)
3. Components 3 / 4, caller / receiver + the two hand-written persona banks
4. Component 5, evaluator
5. Component 1, scenario author agent + `review-bank`
6. Component 6, gatekeeper + Diversity index + the `review` human review view
7. (Later) the bench harness, implemented per §6

The gatekeeper comes last because its thresholds have to be calibrated on real data produced by the earlier stages (R6.5).
