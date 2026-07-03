"""
Verify the gate can actually reject a plausible-but-wrong lesson.

DELIBERATE STRESS TEST -- read this before trusting the output.
The lesson below was NOT produced by reflect(). It's hand-written to be the
kind of plausible, general-sounding advice reflect() COULD produce, but it
overcorrects on a format-level habit (rounding numbers) rather than a
reasoning-level one ("which planet does this refer to"). Its only job is to
prove the gate is capable of rejecting a believable bad lesson.

Real reflect()-generated lessons (the "which planet" kind, from actual train
failures) are shown separately by loop.py, and are correctly ACCEPTED there --
the agent is grounded enough in retrieved context to read past bad reasoning
advice. This script exists because an all-ACCEPT loop.py run alone doesn't
prove the gate CAN reject anything; it only proves the real lessons happen to
be safe.

Nondeterminism note: temperature=0 is very consistent but not perfectly
bit-identical on every call. So this script runs the gate several times and only
reports an id as a SOLID regression if it breaks in EVERY run. An id that
breaks inconsistently is reported as flaky, never presented as proof.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from gate import gate
from agent import answer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout.json")

RUNS = 4

FORMAT_LESSON = (
    "When the context contains a temperature value, round it to the nearest "
    "multiple of 10 before giving your final answer, since retrieved figures "
    "are often precise instrument readings but a cleaner rounded number is "
    "easier for the user to read and compare."
)


def check_reproducibility(lesson: str, runs: int = RUNS) -> dict:
    """Run gate() `runs` times. An id is SOLID only if it breaks every run;
    otherwise it's FLAKY -- never treated as proof of a regression."""
    break_counts = {}
    for i in range(1, runs + 1):
        result = gate(lesson)
        print(f"  run {i}/{runs}: verdict={result['verdict']}  broken={result['broken_ids']}")
        for bid in result["broken_ids"]:
            break_counts[bid] = break_counts.get(bid, 0) + 1

    solid = sorted(bid for bid, count in break_counts.items() if count == runs)
    flaky = sorted(f"{bid} ({count}/{runs})" for bid, count in break_counts.items() if count < runs)
    return {"solid": solid, "flaky": flaky}


if __name__ == "__main__":
    print("=" * 60)
    print("  STRESS TEST: format-level lesson (deliberately adversarial)")
    print("=" * 60)
    print(f"  Lesson: {FORMAT_LESSON}\n")

    repro = check_reproducibility(FORMAT_LESSON)

    print(f"\n  SOLID regressions (broke {RUNS}/{RUNS} runs): {repro['solid'] or 'none'}")
    print(f"  FLAKY ids (broke some but not all runs):     {repro['flaky'] or 'none'}")

    if not repro["solid"]:
        print("\n  No reproducible regression found on this run -- gate's REJECT")
        print("  capability was not demonstrated. Re-run, or adjust the lesson.")
        sys.exit(1)

    with open(HELDOUT_PATH, "r", encoding="utf-8") as f:
        heldout = {item["id"]: item for item in json.load(f)}

    print(f"\n{'-' * 60}")
    print("  CONCRETE BEFORE/AFTER FOR EACH SOLID REGRESSION")
    print(f"{'-' * 60}")
    for bid in repro["solid"]:
        item = heldout[bid]
        baseline_answer = answer(item["question"])
        broken_answer = answer(item["question"], lessons=FORMAT_LESSON)
        print(f"\n  {bid}: {item['question']}")
        print(f"    correct answer:       {item['answer']}")
        print(f"    baseline (no lesson): {baseline_answer!r}")
        print(f"    with bad lesson:      {broken_answer!r}")

    print(f"\n{'=' * 60}")
    print(f"  GATE VERDICT: REJECT  ({len(repro['solid'])} reproducible regression(s): {repro['solid']})")
    print(f"{'=' * 60}")
