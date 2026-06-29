"""
Harness: loads a dataset file (train.json or heldout.json), runs answer() on every
question, compares the model's answer to the known correct answer using normalized
exact-match (lowercase, stripped whitespace and punctuation), and prints PASS/FAIL
per question plus an overall score like "train: 16/20".
"""
