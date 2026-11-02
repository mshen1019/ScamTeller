"""Scenario sampler (component 2). Pure functions, zero model calls.

Key points:
* ``(bank_version, seed, index)`` uniquely determines a task (P5 / R2.1), by mixing rather than seed+index
* **Stratified** sampling, not pure random (R2.2): quotas are first allocated by category, legitimate is split again into hard negative /
  plain, and families are round-robined within each stratum; otherwise the stratum with many families swamps the small ones
* A scenario may be reused, but the ``(scenario, caller persona, receiver persona)`` triple never repeats (R2.5);
  this is the main source of diversity
* When ratios miss their targets it **warns instead of staying silent** (R2.3); the usual cause is a stratum with too small a scenario pool
* ``balance_families=True`` switches the quota logic: instead of category ratios, it makes **each family's
  final count as equal as possible**, counting existing data and only filling the gap. The cost is that class ratios are no longer controllable: they're
  set by the bank's make-up, so parameters such as ``--scam-ratio`` no longer take effect
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..personas import CALLER_PERSONAS, RECEIVER_PERSONAS
from ..schema import CallerPersona, GenerationTask, ReceiverPersona, Scenario
from .bank import ScenarioBank

#: Per-turn conversational colour. About 1/4 are None, so no gimmick becomes a verbal tic (R3.7).
TURN_DIRECTIVES: list[str | None] = [
    "Cut in before the other person has finished their sentence.",
    "Hesitate -- start a sentence, break off, restart it differently.",
    "Mishear one specific word and respond to the wrong thing.",
    "Ask for a clarification of something said two turns ago.",
    "Repeat yourself slightly, the way people do when they are not being heard.",
    "Let something in your surroundings distract you for a beat.",
    "Say something that lightly contradicts a detail from earlier in the call.",
    "Lower the energy -- a flat, tired, one-line turn.",
    "Talk over a brief silence, filling it with a filler phrase.",
    "Answer at more length than the question needed.",
    None, None, None,
]


#: Only receivers with these skepticism levels are given a "compliant" outcome.
#: Having a skepticism=high persona "won over" fights the persona itself, and the dialogue comes out unnatural:
#: compliance must come from the persona, not from an instruction.
COMPLIANCE_SKEPTICISM = ("very_low", "low")


@dataclass
class SamplingWarning:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def derive_seed(seed: int, index: int, bank_version: int = 1) -> int:
    """Stable per-sample seed. Mixed, not added: adjacent seeds must not produce overlapping batches."""
    return (seed * 1_000_003 + index * 7919 + bank_version * 104_729 + 17) % (2**31 - 1)


def _round_robin(pool: Sequence[Scenario], count: int, rng: random.Random) -> list[Scenario]:
    """Shuffle once, then cycle, so every family is used evenly."""
    if not pool or count <= 0:
        return []
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return [shuffled[i % len(shuffled)] for i in range(count)]


def _layer_of(scn: Scenario) -> str:
    if scn.category == "scam":
        return "scam"
    return "hard_negative" if scn.hard_negative else "plain"


def family_quotas(
    families: Sequence[str],
    total: int,
    existing: Mapping[str, int] | None = None,
    rng: random.Random | None = None,
) -> dict[str, int]:
    """How many more items each family needs so that each family's **final** count is as equal as possible.

    Water-filling: each item goes to the family with the lowest current final count. A family that already has ``existing[family]``
    items starts from that height; that's exactly what "already-sampled families are only topped up to the mean" means. A family
    already above the mean gets 0: the extra items can't be deleted, so they can only raise the mean.
    """
    if not families:
        return {}
    existing = existing or {}
    order = list(families)
    if rng is not None:
        # Who gets the remainder (the extra few when total isn't divisible by the number of families) is decided by the seed,
        # not by alphabetical order; otherwise the same few families get one extra every time.
        rng.shuffle(order)
    quota = {f: 0 for f in order}
    heap = [(int(existing.get(f, 0)), i, f) for i, f in enumerate(order)]
    heapq.heapify(heap)
    for _ in range(max(0, total)):
        count, i, family = heapq.heappop(heap)
        quota[family] += 1
        heapq.heappush(heap, (count + 1, i, family))
    return {f: quota[f] for f in sorted(quota)}


def _pick_balanced(
    bank: ScenarioBank,
    num_samples: int,
    existing: Mapping[str, int],
    rng: random.Random,
    warnings: list[SamplingWarning],
) -> list[tuple[str, Scenario]]:
    by_family: dict[str, list[Scenario]] = {}
    for scn in bank.scenarios:
        by_family.setdefault(scn.family, []).append(scn)

    quotas = family_quotas(sorted(by_family), num_samples, existing, rng)
    picked: list[tuple[str, Scenario]] = []
    for family in sorted(quotas):
        for scn in _round_robin(by_family[family], quotas[family], rng):
            picked.append((_layer_of(scn), scn))
    # Shuffle: otherwise items of the same family sit together, and batch progress and resuming would both advance family by family
    rng.shuffle(picked)

    full = sorted(f for f in quotas if quotas[f] == 0)
    if full:
        warnings.append(SamplingWarning(
            "balance.already_full",
            f"{len(full)} families already have at least the mean count, so this batch gives them no quota: "
            + ", ".join(full[:8]) + ("…" if len(full) > 8 else ""),
        ))
    final = {f: quotas[f] + int(existing.get(f, 0)) for f in quotas}
    spread = max(final.values()) - min(final.values()) if final else 0
    if spread > 1:
        warnings.append(SamplingWarning(
            "balance.uneven",
            f"final family counts still differ by {spread} ({min(final.values())}-"
            f"{max(final.values())}), because the existing data is itself uneven and generated items can't be withdrawn",
        ))
    return picked


def _allocate(
    num_samples: int, scam_ratio: float, hard_negative_ratio: float
) -> tuple[int, int, int]:
    n_scam = round(num_samples * scam_ratio)
    n_legit = num_samples - n_scam
    n_hard = round(n_legit * hard_negative_ratio)
    return n_scam, n_hard, n_legit - n_hard


def _pick_by_category_ratio(
    bank: ScenarioBank,
    num_samples: int,
    scam_ratio: float,
    hard_negative_ratio: float,
    rng: random.Random,
    warnings: list[SamplingWarning],
) -> list[tuple[str, Scenario]]:
    """Default quota logic: stratify by category ratio first, then round-robin families within each stratum (R2.2)."""
    n_scam, n_hard, n_plain = _allocate(num_samples, scam_ratio, hard_negative_ratio)

    # When a stratum's pool is empty, hand its quota to the other strata of the same class and warn; swallowing it silently would quietly skew the ratios
    layers: list[tuple[str, list[Scenario], int]] = []
    pools = {"scam": bank.scams, "hard_negative": bank.hard_negatives, "plain": bank.plain_negatives}
    wanted = {"scam": n_scam, "hard_negative": n_hard, "plain": n_plain}
    for key in ("scam", "hard_negative", "plain"):
        if wanted[key] and not pools[key]:
            fallback = "plain" if key == "hard_negative" else ("hard_negative" if key == "plain" else None)
            if fallback and pools[fallback]:
                wanted[fallback] += wanted[key]
                warnings.append(SamplingWarning(
                    "layer.empty",
                    f"no {key} scenarios in the bank; {wanted[key]} quota moved to {fallback}",
                ))
            else:
                warnings.append(SamplingWarning(
                    "layer.empty", f"no {key} scenarios in the bank; that stratum's {wanted[key]} quota can't be met"
                ))
            wanted[key] = 0
    for key in ("scam", "hard_negative", "plain"):
        layers.append((key, pools[key], wanted[key]))

    picked: list[tuple[str, Scenario]] = []
    for key, pool, count in layers:
        for scn in _round_robin(pool, count, rng):
            picked.append((key, scn))
    return picked


def sample_tasks(
    bank: ScenarioBank,
    num_samples: int,
    seed: int,
    *,
    min_rounds: int = 5,
    max_rounds: int = 20,
    scam_ratio: float = 0.5,
    hard_negative_ratio: float = 0.4,
    late_onset_ratio: float = 0.3,
    compliance_ratio: float = 0.25,
    include_family: Iterable[str] | None = None,
    exclude_family: Iterable[str] | None = None,
    balance_families: bool = False,
    existing_counts: Mapping[str, int] | None = None,
    callers: Sequence[CallerPersona] = tuple(CALLER_PERSONAS),
    receivers: Sequence[ReceiverPersona] = tuple(RECEIVER_PERSONAS),
    tolerance: float = 0.08,
) -> tuple[list[GenerationTask], list[SamplingWarning]]:
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if min_rounds < 1 or max_rounds < min_rounds:
        raise ValueError("requires 1 <= min_rounds <= max_rounds")
    if not callers or not receivers:
        raise ValueError("persona banks must not be empty")

    if include_family or exclude_family:
        bank = bank.filter_families(include_family, exclude_family)

    warnings: list[SamplingWarning] = []
    rng = random.Random(derive_seed(seed, -1, bank.version))

    if balance_families:
        picked = _pick_balanced(bank, num_samples, existing_counts or {}, rng, warnings)
    else:
        picked = _pick_by_category_ratio(
            bank, num_samples, scam_ratio, hard_negative_ratio, rng, warnings
        )
    if len(picked) < num_samples:
        warnings.append(SamplingWarning(
            "allocation.short",
            f"only {len(picked)}/{num_samples} tasks assembled; the scenario bank is too small or the family filter too strict",
        ))

    # late_onset: only allocated to scenarios that "suit slow burn" (the division of meaning between R2.3 and Scenario.late_onset)
    scam_positions = [i for i, (key, _) in enumerate(picked) if key == "scam"]
    capable = [i for i in scam_positions if picked[i][1].late_onset]
    want_late = round(len(scam_positions) * late_onset_ratio)
    late_set: set[int] = set()
    if want_late:
        if len(capable) < want_late:
            warnings.append(SamplingWarning(
                "late_onset.insufficient",
                f"{want_late} slow-burn samples needed, but only {len(capable)} sampled scenarios suit slow burn; "
                "mark more scams as late_onset in the scenario bank",
            ))
        chosen = capable if len(capable) <= want_late else rng.sample(capable, want_late)
        late_set = set(chosen)

    combos = [(c, r) for c in callers for r in receivers]
    used_triples: set[tuple[str, str, str]] = set()

    # Pass 1: personas, round counts, directives. Outcomes wait for pass 2: compliant outcomes are allocated by the receiver's skepticism,
    # and skepticism is only known once the persona is fixed.
    drafts: list[dict[str, Any]] = []
    for index, (_key, scn) in enumerate(picked):
        task_rng = random.Random(derive_seed(seed, index, bank.version))
        caller, receiver = _pick_personas(scn, combos, used_triples, task_rng, warnings)
        used_triples.add((scn.scenario_id, caller.persona_id, receiver.persona_id))
        target_rounds = task_rng.randint(min_rounds, max_rounds)
        drafts.append({
            "scenario": scn,
            "caller": caller,
            "receiver": receiver,
            "target_rounds": target_rounds,
            "directives": [
                task_rng.choice(TURN_DIRECTIVES) for _ in range(2 * target_rounds + 4)
            ],
        })

    compliance_set = _assign_compliance(drafts, compliance_ratio, rng, warnings)

    tasks: list[GenerationTask] = []
    for index, d in enumerate(drafts):
        scn = d["scenario"]
        # A separate rng: drawing outcomes must not disturb pass 1's sequence, otherwise changing a ratio would change the personas too.
        outcome_rng = random.Random(derive_seed(seed, index + 1_000_003, bank.version))
        pool = _outcome_pool(scn, index in compliance_set)
        tasks.append(GenerationTask(
            task_id=f"task-{seed}-{index:05d}",
            index=index,
            seed=seed,
            derived_seed=derive_seed(seed, index, bank.version),
            bank_version=bank.version,
            scenario=scn,
            caller_persona=d["caller"],
            receiver_persona=d["receiver"],
            outcome=outcome_rng.choice(pool),
            target_rounds=d["target_rounds"],
            late_onset=index in late_set,
            turn_directives=d["directives"],
        ))

    warnings.extend(_ratio_warnings(
        tasks, scam_ratio, hard_negative_ratio, late_onset_ratio, compliance_ratio, tolerance,
        check_category_ratios=not balance_families,
    ))
    return tasks, warnings


def _outcome_pool(scn: Scenario, compliant: bool) -> list[str]:
    if compliant and scn.compliant_outcomes:
        return list(scn.compliant_outcomes)
    rest = [o for o in scn.plausible_outcomes if o not in scn.compliant_outcomes]
    return rest or list(scn.plausible_outcomes)


def _assign_compliance(
    drafts: list[dict[str, Any]],
    ratio: float,
    rng: random.Random,
    warnings: list[SamplingWarning],
) -> set[int]:
    """Choose which scam samples have the receiver won over.

    Two conditions: the scenario has compliant outcomes, and the receiver's skepticism is low enough that being won over isn't awkward. When the target ratio can't be reached
    it warns rather than forcing it: forced "credulity" reads as if the persona suddenly became someone else.
    """
    scam_positions = [
        i for i, d in enumerate(drafts) if d["scenario"].category == "scam"
    ]
    eligible = [
        i for i in scam_positions
        if d_ok(drafts[i])
    ]
    want = round(len(scam_positions) * ratio)
    if not want:
        return set()
    if len(eligible) < want:
        warnings.append(SamplingWarning(
            "compliance.insufficient",
            f"{want} compliant-outcome samples needed, but only {len(eligible)} tasks have a receiver whose "
            f"skepticism is within {COMPLIANCE_SKEPTICISM} and a scenario with compliant outcomes",
        ))
        return set(eligible)
    return set(rng.sample(eligible, want))


def d_ok(draft: dict[str, Any]) -> bool:
    return bool(
        draft["scenario"].compliant_outcomes
        and draft["receiver"].skepticism in COMPLIANCE_SKEPTICISM
    )


def _pick_personas(
    scn: Scenario,
    combos: Sequence[tuple[CallerPersona, ReceiverPersona]],
    used: set[tuple[str, str, str]],
    rng: random.Random,
    warnings: list[SamplingWarning],
) -> tuple[CallerPersona, ReceiverPersona]:
    """Linear scan from a random start for a persona combination this scenario hasn't used yet (R2.5)."""
    start = rng.randrange(len(combos))
    for offset in range(len(combos)):
        caller, receiver = combos[(start + offset) % len(combos)]
        if (scn.scenario_id, caller.persona_id, receiver.persona_id) not in used:
            return caller, receiver
    warnings.append(SamplingWarning(
        "personas.exhausted",
        f"all {len(combos)} persona combinations for scenario {scn.family} are used up; combinations will repeat",
    ))
    return combos[start]


def _ratio_warnings(
    tasks: Sequence[GenerationTask],
    scam_ratio: float,
    hard_negative_ratio: float,
    late_onset_ratio: float,
    compliance_ratio: float,
    tolerance: float,
    check_category_ratios: bool = True,
) -> list[SamplingWarning]:
    if not tasks:
        return []
    out: list[SamplingWarning] = []
    total = len(tasks)
    scams = [t for t in tasks if t.scenario.category == "scam"]
    legits = [t for t in tasks if t.scenario.category == "legitimate"]

    def check(code: str, label: str, hits: int, size: int, target: float) -> None:
        """The tolerance must at least absorb one sample's worth of rounding error.

        A fixed tolerance warns constantly on small batches (e.g. 5 scams at 30% slow burn, round() can only give
        2/5 = 0.40), and that noise teaches people to ignore warnings, real problems included.
        """
        if size == 0:
            return
        effective = max(tolerance, 1.0 / size)
        actual = hits / size
        if abs(actual - target) > effective:
            out.append(SamplingWarning(
                code,
                f"{label} actual {actual:.2f} ({hits}/{size}) deviates from target {target:.2f} "
                f"by more than the tolerance {effective:.2f}",
            ))

    if check_category_ratios:
        check("ratio.scam", "scam share", len(scams), total, scam_ratio)
        check(
            "ratio.hard_negative", "hard negative share of legitimate",
            sum(1 for t in legits if t.scenario.hard_negative), len(legits), hard_negative_ratio,
        )
    else:
        # In family-balance mode the ratios are an **outcome**, not an input: report them, but not as deviations
        hard = sum(1 for t in legits if t.scenario.hard_negative)
        out.append(SamplingWarning(
            "balance.composition",
            f"family-balance mode: class ratios are set by the bank's make-up; scam {len(scams)/total:.0%}"
            f" ({len(scams)}), hard negative {hard/total:.0%} ({hard}), "
            f"ordinary legitimate {(len(legits)-hard)/total:.0%} ({len(legits)-hard})",
        ))
    check(
        "ratio.late_onset", "late-onset share of scam",
        sum(1 for t in scams if t.late_onset), len(scams), late_onset_ratio,
    )
    check(
        "ratio.compliance", "receiver-won-over share of scam",
        sum(1 for t in scams if t.outcome in t.scenario.compliant_outcomes),
        len(scams), compliance_ratio,
    )
    return out
