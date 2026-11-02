"""bench harness (REQUIREMENTS §6/§7): run detectors over data on disk and score them.

Layout:

* ``protocol``     -- ``EarlyWarningDetector`` / ``Verdict`` / ``ReplayAdapter``
* ``judge_prompt`` -- the prompt the model under test sees (a frozen object with a content fingerprint)
* ``score``        -- metrics. **Latency and false-alarm rate must be reported together**, so there's no single overall score
* ``adapters``     -- per-provider judges (with prefix caching and a hit self-check)
* ``run``          -- batch runs: in conversation order, resumable, with a budget cap
"""

from .protocol import (
    DetectorSession,
    EarlyWarningDetector,
    ReplayAdapter,
    ReplaySession,
    StatelessJudge,
    Verdict,
    replay_call,
)
from .judge_prompt import JUDGE_VERSION, fingerprint, render_transcript, user_message
from .score import (
    average_precision,
    discrimination_feedback,
    score_system,
    slice_scores,
    trivial_baselines,
)

__all__ = [
    "DetectorSession", "EarlyWarningDetector", "ReplayAdapter", "ReplaySession",
    "StatelessJudge", "Verdict", "replay_call",
    "JUDGE_VERSION", "fingerprint", "render_transcript", "user_message",
    "average_precision", "discrimination_feedback", "score_system", "slice_scores",
    "trivial_baselines",
]
