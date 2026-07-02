"""
Ablation: proves the gate matters by running the SAME candidate lessons
through two pipelines -- gate ON (what this project actually does) and gate
OFF (every candidate installed unconditionally, no regression check) -- and
comparing held-out accuracy after each one.

The candidates are the 3 real ACCEPTed lessons from data/candidates.json
(loop.py's actual output) plus the deliberately adversarial format-level
lesson from verify_gate.py -- the one already proven, across 4 independent
runs, to reproducibly break h01 and h13 when it's active. Gate ON rejects
that lesson and never installs it; gate OFF has no such check.

Every score below comes from a real evaluate() call -- nothing here is
simulated or hand-typed. Results are appended to data/log_nogate.jsonl.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from harness import evaluate
from gate import gate
from verify_gate import FORMAT_LESSON
from logger import log_event

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.json")
LOG_NOGATE_PATH = os.path.join(DATA_DIR, "log_nogate.jsonl")


def _format_lessons(lessons: list) -> str:
    return "\n".join(f"- {lesson}" for lesson in lessons)


def run_ablation() -> list:
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    real_lessons = [c for c in candidates if c["gate_verdict"] == "ACCEPT"]

    steps = [{"label": c["source_id"], "lesson": c["lesson"]} for c in real_lessons]
    steps.append({"label": "adversarial (format-level)", "lesson": FORMAT_LESSON})

    gate_on_installed = []
    gate_off_installed = []

    baseline = evaluate(HELDOUT_PATH, lessons="")
    on_result = baseline
    results = [{
        "step": 0,
        "label": "baseline (no lessons)",
        "gate_verdict": None,
        "gate_on_score": baseline["score"], "gate_on_total": baseline["total"],
        "gate_off_score": baseline["score"], "gate_off_total": baseline["total"],
    }]
    log_event("ablation_step", path=LOG_NOGATE_PATH, **results[-1])
    print(f"  step 0 [baseline]: ON={baseline['score']}/{baseline['total']}  OFF={baseline['score']}/{baseline['total']}")

    for i, step in enumerate(steps, start=1):
        verdict = gate(step["lesson"])
        gate_off_installed.append(step["lesson"])
        off_result = evaluate(HELDOUT_PATH, lessons=_format_lessons(gate_off_installed))

        if verdict["verdict"] == "ACCEPT":
            gate_on_installed.append(step["lesson"])
            on_result = evaluate(HELDOUT_PATH, lessons=_format_lessons(gate_on_installed))
        # else: gate_on's installed set is unchanged -- on_result stays what it was

        record = {
            "step": i,
            "label": f"+{step['label']}",
            "gate_verdict": verdict["verdict"],
            "gate_on_score": on_result["score"], "gate_on_total": on_result["total"],
            "gate_off_score": off_result["score"], "gate_off_total": off_result["total"],
        }
        results.append(record)
        log_event("ablation_step", path=LOG_NOGATE_PATH, **record)
        print(f"  step {i} [{step['label']}]: gate={verdict['verdict']}  "
              f"ON={on_result['score']}/{on_result['total']}  OFF={off_result['score']}/{off_result['total']}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  ABLATION: gate ON vs gate OFF, same candidate lessons")
    print("=" * 60)
    run_ablation()
