"""
Reflexion loop: runs the agent on the train set, calls reflect() for every
failure to generate a candidate lesson, then immediately gates that lesson
through gate(). ACCEPTED lessons are kept as safe; REJECTED lessons are kept
too (for the record) but flagged with the held-out ids they would have broken.
Nothing gets promoted to the insight store (Day 4) without passing the gate.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from agent import answer, reflect
from retriever import retrieve
from harness import matches
from gate import gate

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.json")


def run_loop() -> dict:
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    candidates = []
    accepted = []
    rejected = []

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        correct = item["answer"]

        got = answer(question)
        if matches(correct, got):
            print(f"  [PASS] {qid}")
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
    return {"accepted": accepted, "rejected": rejected}


if __name__ == "__main__":
    run_loop()
