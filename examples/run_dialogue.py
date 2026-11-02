"""Run one dialogue, label it along the way, and print it: the manual inspection entry point for components 3 / 4 / 5.

The proper `generate` command waits on the batch-writing step; until then use this script to look at dialogues and labels themselves.

    python examples/run_dialogue.py                       # offline mock, dialogue only
    python examples/run_dialogue.py --index 3 --label      # with the prefix labels too
    python examples/run_dialogue.py --provider anthropic --model claude-sonnet-5-5 \
        --label --eval-provider openai --eval-model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callfactory.agents import CallerAgent, ReceiverAgent          # noqa: E402
from callfactory.dialogue import run_dialogue                      # noqa: E402
from callfactory.evaluator import (                                # noqa: E402
    DEFAULT_DETECTION_THRESHOLD,
    EvaluatorAgent,
    evaluate_conversation,
)
from callfactory.providers import ProviderRoles, build_provider    # noqa: E402
from callfactory.scenarios import latest_bank, sample_tasks        # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", default="datasets/scenario_banks")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--index", type=int, default=0, help="which sampled task to take")
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-rounds", type=int, default=5)
    ap.add_argument("--max-rounds", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--show-prompts", action="store_true")
    ap.add_argument("--label", action="store_true", help="run the evaluator and print the prefix-label trend")
    ap.add_argument("--eval-provider", default="mock", help="must come from a different source than the dialogue side (P4)")
    ap.add_argument("--eval-model", default=None)
    ap.add_argument("--eval-stride", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD)
    ap.add_argument("--allow-same-provider", action="store_true")
    args = ap.parse_args()

    bank = latest_bank(args.bank_dir)
    tasks, _ = sample_tasks(
        bank, args.num_samples, args.seed,
        min_rounds=args.min_rounds, max_rounds=args.max_rounds,
    )
    if not (0 <= args.index < len(tasks)):
        print(f"index out of range: only {len(tasks)} tasks sampled")
        return 2
    task = tasks[args.index]
    scn = task.scenario

    print(f"task        {task.task_id}  (derived_seed {task.derived_seed})")
    print(f"scenario    {scn.family}  [{scn.category}"
          f"{', hard negative' if scn.hard_negative else ''}"
          f"{', late onset' if task.late_onset else ''}]")
    print(f"pretext     {scn.pretext}")
    print(f"signs       {', '.join(scn.expected_signs) or '(none)'}")
    print(f"caller      {task.caller_persona.persona_id}  {task.caller_persona.speech_style}")
    print(f"receiver    {task.receiver_persona.persona_id}  "
          f"skepticism={task.receiver_persona.skepticism}, "
          f"{task.receiver_persona.situation}")
    print(f"outcome     {task.outcome}")
    print(f"target      {task.target_rounds} rounds\n")

    provider = build_provider(args.provider, model=args.model, temperature=args.temperature)
    evaluator_provider = (
        build_provider(args.eval_provider, model=args.eval_model, temperature=0.0)
        if args.label else None
    )
    if evaluator_provider is not None:
        roles = ProviderRoles(
            caller=provider, receiver=provider, evaluator=evaluator_provider
        )
        for issue in roles.enforce(allow_same_provider=args.allow_same_provider):
            print(f"!! P4 warning (explicitly allowed): {issue}\n")
    caller = CallerAgent(provider, task, seed=task.derived_seed, temperature=args.temperature)
    receiver = ReceiverAgent(provider, task, seed=task.derived_seed, temperature=args.temperature)

    if args.show_prompts:
        print("=" * 72, "\nCALLER SYSTEM PROMPT\n" + "=" * 72)
        print(caller.system)
        print("=" * 72, "\nRECEIVER SYSTEM PROMPT (deliberately contains no scenario information)\n" + "=" * 72)
        print(receiver.system, "\n")

    evaluation = None
    try:
        result = await run_dialogue(
            task, caller, receiver,
            min_rounds=args.min_rounds, max_rounds=args.max_rounds,
        )
        if evaluator_provider is not None:
            evaluation = await evaluate_conversation(
                result.utterances,
                EvaluatorAgent(evaluator_provider, seed=task.derived_seed),
                stride=args.eval_stride,
                threshold=args.threshold,
            )
    finally:
        await provider.aclose()
        if evaluator_provider is not None:
            await evaluator_provider.aclose()

    labels = {u.index: u for u in (evaluation.utterance_labels if evaluation else [])}
    for u in result.utterances:
        line = f"[{u.index:>2}] {u.speaker.upper():<8} {u.text}"
        label = labels.get(u.index)
        if label:
            mark = " <== DETECTION POINT" if label.is_detection_point else ""
            new = f" +{','.join(label.new_signals)}" if label.new_signals else ""
            carried = "" if label.label_from_index == u.index else f" (from {label.label_from_index})"
            line = (f"{label.risk_score:>5.2f} {label.risk_tier:<6}{new}{carried}{mark}\n"
                    f"      {line}")
        print(line)
    print(f"\nended_by={result.ended_by}  rounds={result.rounds}  "
          f"fallbacks={result.empty_fallbacks}")
    for note in result.notes:
        print(f"  note: {note}")

    if evaluation:
        gt = "scam" if scn.category == "scam" else "legitimate"
        print(f"\nearliest_detection  index={evaluation.earliest_detection_index} "
              f"round={None if evaluation.earliest_detection_index is None else evaluation.earliest_detection_index // 2 + 1}")
        print(f"signals at the end  {', '.join(evaluation.prefix_labels[-1].signals) or '(none)'}")
        if evaluation.blind_assessment:
            ba = evaluation.blind_assessment
            agree = "agree" if ba.blind_is_suspicious == (scn.category == "scam") else "disagree"
            print(f"blind_is_suspicious {ba.blind_is_suspicious}  (ground truth {gt} → {agree})")
            print(f"craft               coherence={ba.coherence} naturalness={ba.naturalness} "
                  f"repetition={ba.repetition}")
        for f in evaluation.failures:
            print(f"  failure: {f.to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
