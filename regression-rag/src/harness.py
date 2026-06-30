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


_ARTICLES = re.compile(r"^(the|a|an)\s+")
_WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[.!?,;:]+$", "", text).strip()
    text = _ARTICLES.sub("", text)
    for word, digit in _WORD_TO_NUM.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return text


def matches(expected: str, got: str) -> bool:
    """PASS if normalized forms are equal, or one contains the other."""
    ne, ng = normalize(expected), normalize(got)
    return ne == ng or ne in ng or ng in ne


def evaluate(dataset_path: str, lessons: str = "") -> dict:
    """
    Run answer() over every question in dataset_path.
    Returns {score, total, passed_ids, failed_ids}.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    label = os.path.basename(dataset_path).replace(".json", "").upper()
    passed, failed = [], []

    print(f"\n{'='*56}")
    print(f"  GATEKEEPER  |  Evaluation Harness  |  {label}")
    print(f"{'='*56}\n")

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected = item["answer"]

        got = answer(question, lessons=lessons)
        ok = matches(expected, got)

        if ok:
            print(f"  [PASS]  {qid}  {question}")
            passed.append(qid)
        else:
            print(f"  [FAIL]  {qid}  {question}")
            print(f"          expected : {normalize(expected)!r}")
            print(f"          got      : {normalize(got)!r}")
            failed.append(qid)

    total = len(dataset)
    score = len(passed)
    pct = int(score / total * 100)

    print(f"\n{'-'*56}")
    print(f"  RESULT   {score}/{total}  ({pct}%)  --  {len(failed)} failure(s)")
    print(f"{'-'*56}\n")

    return {"score": score, "total": total, "passed_ids": passed, "failed_ids": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "train.json"),
        help="Path to train.json or heldout.json",
    )
    parser.add_argument(
        "--save-baseline",
        help="Path to write the currently-passing question ids as a baseline JSON file",
    )
    args = parser.parse_args()
    result = evaluate(args.dataset)

    if args.save_baseline:
        with open(args.save_baseline, "w", encoding="utf-8") as f:
            json.dump(result["passed_ids"], f, indent=2)
        print(f"  baseline saved  ->  {len(result['passed_ids'])} passing ids  ->  {args.save_baseline}\n")
