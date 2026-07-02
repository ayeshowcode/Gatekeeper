"""
Gate: the metacognitive evaluation step. Before a candidate lesson is kept, gate()
re-runs the entire sealed held-out set WITH the lesson injected and checks whether any
question that passed at baseline now fails. If nothing regresses, the lesson is
ACCEPTED; if anything breaks, it's REJECTED and the broken ids are reported.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from harness import evaluate

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout.json")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")


def gate(candidate_lesson: str) -> dict:
    """
    Test a candidate lesson against the sealed held-out set.
    1. run all of heldout.json through answer() with the lesson injected
    2. compute the new pass-set
    3. compare against the frozen baseline pass-set
    4. ACCEPT if nothing that used to pass now fails, else REJECT with the broken ids
    """
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_passed_ids = set(json.load(f))

    result = evaluate(HELDOUT_PATH, lessons=candidate_lesson)
    new_passed_ids = set(result["passed_ids"])

    broken_ids = sorted(baseline_passed_ids - new_passed_ids)
    verdict = "REJECT" if broken_ids else "ACCEPT"

    return {
        "verdict": verdict,
        "broken_ids": broken_ids,
        "new_passed_ids": sorted(new_passed_ids),
    }


if __name__ == "__main__":
    test_lesson = (
        "When the context contains information about multiple entities, such as "
        "different planets, and the question refers to one of them indirectly, ensure "
        "to first identify the specific entity the question pertains to before "
        "extracting any values."
    )
    result = gate(test_lesson)
    print(f"\nGate verdict: {result['verdict']}")
    if result["broken_ids"]:
        print(f"Would break: {result['broken_ids']}")
