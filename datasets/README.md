# datasets/

Where generated data lands. Everything outside `scenario_banks/` is a build artefact and should not be edited by hand.

| Directory | Contents | Written by |
|---|---|---|
| `scenario_banks/` | Versioned scenario banks, **human-editable** | scenario author agent + humans |
| `candidates/<run_id>/` | Generated, passed the program's hard rules, awaiting the gatekeeper's verdict | generate |
| `admitted/<run_id>/` | Passed the gatekeeper; usable for the train/eval split | judge |
| `rejected/<run_id>/` | Rejected by the gatekeeper; kept to analyse the distribution of rejection reasons | judge |
| `review/<run_id>/` | Flagged by the gatekeeper for human review | judge |
| `decisions.jsonl` | Every verdict, auditable | judge |
| `train/<name>/` | Training set | split |
| `eval/<name>/` | Development set: a share of every family, measuring timing accuracy on **call types already seen** | split |
| `test/<name>/` | Held-out set: whole families absent from the other two, measuring generalisation to **unseen call types** | split |

In every run directory, `conversations.jsonl` (no labels) and `metadata.jsonl` (labels) are joined on
`conversation_id`. Splitting is a separate command and not the gatekeeper's job.

`<name>` looks like `ds2-rubric-v3`: the **dataset version** (which release of this data) + the **rubric version**
(which revision of the labelling rules). The two are not in step and should not be — relabelling the same transcripts
under a new rubric gives new labels for the same dataset version. Each split directory carries its own `README.md`
explaining both numbers.
