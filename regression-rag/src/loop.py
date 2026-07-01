"""
Reflexion loop: runs the agent on the train set, calls reflect() for every
failure to generate a candidate lesson, and writes all candidates to
data/candidates.json. No gate yet — every lesson is saved. The gate (Day 3)
will filter this list before anything gets promoted to the insight store.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from agent import answer, reflect
from retriever import retrieve
from harness import matches

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.json")


def run_loop() -> list[dict]:
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    candidates = []
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
        print(f"         lesson:   {lesson}\n")

        candidates.append({
            "source_id": qid,
            "question": question,
            "wrong_answer": got,
            "correct_answer": correct,
            "lesson": lesson,
        })

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"\n{len(candidates)} candidate lesson(s) saved -> {CANDIDATES_PATH}")
    return candidates


if __name__ == "__main__":
    run_loop()
