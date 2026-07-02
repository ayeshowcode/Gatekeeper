"""
Reflexion loop: runs the agent on the train set, calls reflect() for every
failure to generate a candidate lesson, then immediately gates that lesson
through gate(). ACCEPTED lessons are kept as safe; REJECTED lessons are kept
too (for the record) but flagged with the held-out ids they would have broken.
At the end of the run, promote() merges every ACCEPTED lesson from this run
into the insight store -- nothing reaches data/insights.json without having
passed the gate first.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from agent import answer, reflect
from retriever import retrieve
from harness import matches, evaluate
from gate import gate
from insights import promote
from logger import log_event, LOG_PATH

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.json")
TRAIN_BASELINE_PATH = os.path.join(DATA_DIR, "train_baseline.json")
BASELINE_PATH = os.path.join(DATA_DIR, "baseline.json")
INSIGHTS_PATH = os.path.join(DATA_DIR, "insights.json")


def _count_insights_in_store() -> int:
    if not os.path.exists(INSIGHTS_PATH):
        return 0
    with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
        return len(json.load(f))


def _count_iteration_snapshots() -> int:
    if not os.path.exists(LOG_PATH):
        return 0
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return sum(1 for line in f if json.loads(line).get("event") == "iteration_snapshot")


def bootstrap_baseline_snapshot() -> None:
    """
    Log iteration 0 exactly once: the frozen pre-history state captured on Day 2,
    before any lesson existed. Uses the frozen train_baseline.json / baseline.json
    pass-id counts directly (no re-run, no new API calls) so this point reflects
    the actual original baseline rather than a re-measurement.
    """
    if _count_iteration_snapshots() > 0:
        return

    with open(TRAIN_BASELINE_PATH, "r", encoding="utf-8") as f:
        train_baseline_ids = json.load(f)
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        heldout_baseline_ids = json.load(f)
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_total = len(json.load(f))
    with open(HELDOUT_PATH, "r", encoding="utf-8") as f:
        heldout_total = len(json.load(f))

    log_event(
        "iteration_snapshot",
        iteration=0,
        label="before any lessons",
        train_score=len(train_baseline_ids),
        train_total=train_total,
        heldout_score=len(heldout_baseline_ids),
        heldout_total=heldout_total,
        lessons_accepted=0,
        lessons_rejected=0,
        insights_in_store=0,
    )


def log_snapshot(train_score: int, train_total: int, accepted_count: int, rejected_count: int) -> None:
    """
    Log the next iteration point for the hero graph: the train score this run
    already measured (no need to re-run train.json a second time), plus a
    fresh held-out evaluate() with the CURRENT insight store (whatever is in
    data/insights.json right now, after this run's promotions). Real
    evaluate() call, not fabricated.
    """
    iteration = _count_iteration_snapshots()
    insights_in_store = _count_insights_in_store()

    heldout_result = evaluate(HELDOUT_PATH, lessons=None)

    log_event(
        "iteration_snapshot",
        iteration=iteration,
        label=f"after {insights_in_store} lesson(s) stored + injected",
        train_score=train_score,
        train_total=train_total,
        heldout_score=heldout_result["score"],
        heldout_total=heldout_result["total"],
        lessons_accepted=accepted_count,
        lessons_rejected=rejected_count,
        insights_in_store=insights_in_store,
    )


def run_loop() -> dict:
    bootstrap_baseline_snapshot()

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    candidates = []
    accepted = []
    rejected = []
    train_passed = 0

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        correct = item["answer"]

        got = answer(question)
        if matches(correct, got):
            print(f"  [PASS] {qid}")
            train_passed += 1
            continue

        print(f"  [FAIL] {qid}: {question}")
        print(f"         expected: {correct!r}")
        print(f"         got:      {got!r}")

        context = "\n\n".join(retrieve(question, k=3))
        lesson = reflect(question, got, context, correct)
        print(f"         lesson:   {lesson}")

        candidate = {
            "source_id": qid,
            "question": question,
            "wrong_answer": got,
            "correct_answer": correct,
            "lesson": lesson,
        }

        verdict = gate(lesson)
        candidate["gate_verdict"] = verdict["verdict"]
        if verdict["verdict"] == "ACCEPT":
            print(f"         gate:     ACCEPT\n")
            accepted.append(candidate)
        else:
            candidate["broken_ids"] = verdict["broken_ids"]
            print(f"         gate:     REJECT -- would break held-out {verdict['broken_ids']}\n")
            rejected.append(candidate)

        candidates.append(candidate)

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"\n{len(candidates)} candidate lesson(s) saved -> {CANDIDATES_PATH}")
    print(f"{len(accepted)} ACCEPTED, {len(rejected)} REJECTED.")

    promoted = promote()
    if promoted:
        print(f"{len(promoted)} new insight(s) promoted -> {os.path.join(DATA_DIR, 'insights.json')}")
    else:
        print("No new insights to promote (already in the store, or nothing ACCEPTED).")

    log_snapshot(train_passed, len(dataset), len(accepted), len(rejected))
    print(f"iteration snapshot logged -> {LOG_PATH}")

    return {"accepted": accepted, "rejected": rejected, "promoted": promoted}


if __name__ == "__main__":
    run_loop()
