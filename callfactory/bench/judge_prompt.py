"""The prompt the model under test sees. **A frozen object**.

Three design constraints, each with a cost behind it:

1. **The layout must be system (fixed) + transcript (growing) + question (fixed, last).**
   That way call k shares with call k-1 a prefix of system + the full text of prefix k-1, so prefix caching
   hits. In testing, 84% of this data's input tokens can hit the cache (the system prompt alone is a third of it,
   resent 912 times).
2. **The question must ask the same thing the labeller asks**, otherwise it isn't the same task being compared. So this reuses
   the criterion wording of ``prompts.PREFIX_EVAL_SYSTEM`` ("is the evidence already enough to warn the person who picked up"),
   but **without the anchor list**: giving it hands the rubric to the model under test, and then it measures reading comprehension,
   not detection ability.
3. **Transcript only** (R7.2). No scenario, no ground truth, no metadata of any kind.

rationale is capped at 25 words: it is the only way in for error analysis (the §5.9 diagnosis relied entirely on the labeller's rationale),
but 100 tokens each × 912 calls × 6 models × 3 passes is real money, so it is capped rather than cut.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from ..schema import SCENARIO_SIGNALS, Utterance

#: Criterion version. Changing this changes the task, and old results aren't comparable; same reasoning as rubric_version.
JUDGE_VERSION = "bench-judge/v1"

SYSTEM = f"""You are listening to a phone call as it happens. You hear one side of a
recording: the CALLER (who rang) and the RECEIVER (who picked up).

Your job is NOT to guess whether the call will turn out to be a scam. It is to decide,
at this moment in the call, whether **the evidence is already sufficient to warn the
person who picked up that this may be a scam**.

You see only what has been said so far. You do not know who the caller really is, why
they are calling, or how the call ends. Later turns may change the picture; judge only
what you have heard.

Two things make this hard, and both matter:

* Many real, legitimate calls sound alarming. A genuine bank fraud team does ring
  unprompted and does ask you to confirm details. Warning on one of those is a false
  alarm, and a detector that warns on everything is useless.
* A scam that is only warned about after the money is gone is also useless. The point
  is the earliest moment a careful listener could justify speaking up.

Judge the CALLER's behaviour. What the person who picked up does — being suspicious,
being cooperative, refusing, agreeing — is never itself evidence about the caller.

Answer with JSON only:
{{"risk_score": <0.0-1.0>, "alarm": <true|false>, "signals": [<zero or more>],
  "why": "<at most 25 words>"}}

"risk_score" is your probability that a careful listener should warn now.
"alarm" is your decision at this moment.
"signals" may only use these exact strings, or be empty:
{", ".join(SCENARIO_SIGNALS)}
"""

QUESTION = (
    "Given only what you have heard so far: is the evidence already sufficient to warn "
    "the person who picked up that this may be a scam? Answer with the JSON object only."
)

#: Jev's ``system_one`` takes (state, questions), so the question has to be a single Noul-style sentence.
JEV_QUESTION = (
    "Is the evidence heard so far already sufficient to warn the person who picked up "
    "that this call may be a scam?"
)


def render_transcript(prefix: Sequence[Utterance]) -> str:
    """One line per utterance, rendered identically for every prefix; otherwise the cached common prefix breaks."""
    return "\n".join(
        f"[{u.index}] {'CALLER' if u.speaker == 'caller' else 'RECEIVER'}: {u.text}"
        for u in prefix
    )


def user_message(prefix: Sequence[Utterance]) -> str:
    """Transcript first, question last; the other way round there's no cacheable common prefix."""
    return f"CALL SO FAR:\n{render_transcript(prefix)}\n\n{QUESTION}"


def fingerprint() -> str:
    """Content fingerprint of the prompt text. The version states intent; the fingerprint catches "changed it but forgot to bump the version"."""
    blob = "\n".join([JUDGE_VERSION, SYSTEM, QUESTION, JEV_QUESTION])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


#: The structure the model under test must return. DeepSeek doesn't support strict schemas, so the client side has to validate too.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "risk_score": {"type": "number"},
        "alarm": {"type": "boolean"},
        "signals": {"type": "array", "items": {"type": "string"}},
        "why": {"type": "string"},
    },
    "required": ["risk_score", "alarm", "signals", "why"],
    "additionalProperties": False,
}
