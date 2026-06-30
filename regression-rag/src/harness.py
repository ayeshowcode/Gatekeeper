"""
Harness: loads a dataset file (train.json or heldout.json), runs answer() on every
question, compares the model's answer to the known correct answer using normalized
exact-match (lowercase, stripped whitespace and punctuation), and prints PASS/FAIL
per question plus an overall score like "train: 16/20".
"""

import sys
import os
import json
import argparse
import re

sys.path.insert(0, os.path.dirname(__file__))
from agent import answer


def normalize(text: str) -> str:
    """Lowercase, strip whitespace, remove trailing punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[.!?,;:]+$", "", text).strip()
    return text


def evaluate(dataset_path: str, lessons: str = "") -> dict:
    """
    Run answer() over every question in dataset_path.
    Returns {score, total, passed_ids, failed_ids}.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    label = os.path.basename(dataset_path).replace(".json", "")
    passed, failed = [], []

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected = normalize(item["answer"])

        got = normalize(answer(question, lessons=lessons))
        ok = got == expected

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {qid}: {question}")
        if not ok:
            print(f"         expected: {expected!r}")
            print(f"         got:      {got!r}")

        if ok:
            passed.append(qid)
        else:
            failed.append(qid)

    total = len(dataset)
    score = len(passed)
    print(f"\n{label}: {score}/{total}")
    return {"score": score, "total": total, "passed_ids": passed, "failed_ids": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "train.json"),
        help="Path to train.json or heldout.json",
    )
    args = parser.parse_args()
    evaluate(args.dataset)
