"""
Plot: reads every iteration_snapshot record from data/log.jsonl and draws the
hero graph -- train accuracy and held-out accuracy across iterations of
loop.py. Saved to docs/hero_graph.png.
"""

import os
import json

import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
LOG_PATH = os.path.join(DATA_DIR, "log.jsonl")
LOG_NOGATE_PATH = os.path.join(DATA_DIR, "log_nogate.jsonl")
OUT_PATH = os.path.join(DOCS_DIR, "hero_graph.png")
ABLATION_OUT_PATH = os.path.join(DOCS_DIR, "ablation_graph.png")


def load_snapshots() -> list:
    snapshots = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("event") == "iteration_snapshot":
                snapshots.append(record)
    return sorted(snapshots, key=lambda r: r["iteration"])


def plot_hero_graph() -> str:
    snapshots = load_snapshots()
    if not snapshots:
        raise RuntimeError("No iteration_snapshot records in log.jsonl -- run src/loop.py first.")

    iterations = [s["iteration"] for s in snapshots]
    train_pct = [100 * s["train_score"] / s["train_total"] for s in snapshots]
    heldout_pct = [100 * s["heldout_score"] / s["heldout_total"] for s in snapshots]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, train_pct, marker="o", label="Train accuracy", color="#2563eb", linewidth=2)
    ax.plot(iterations, heldout_pct, marker="o", label="Held-out accuracy", color="#16a34a", linewidth=2)

    for x, y, s in zip(iterations, train_pct, snapshots):
        ax.annotate(f"{s['train_score']}/{s['train_total']}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", color="#2563eb")
    for x, y, s in zip(iterations, heldout_pct, snapshots):
        ax.annotate(f"{s['heldout_score']}/{s['heldout_total']}", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", color="#16a34a")

    labels = [s.get("label", f"iter {s['iteration']}") for s in snapshots]
    ax.set_xticks(iterations)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Gatekeeper: train vs held-out accuracy across loop.py iterations")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.5, 0.04,
        "Both lines are flat: 3 lessons were gate-ACCEPTED and promoted, but changed neither\n"
        "train nor held-out accuracy. Storage ≠ effectiveness -- see JOURNAL.md, Day 4.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#555555",
        style="italic",
    )

    fig.tight_layout()
    os.makedirs(DOCS_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"hero graph saved -> {OUT_PATH}")
    return OUT_PATH


def load_ablation_steps() -> list:
    steps = []
    with open(LOG_NOGATE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("event") == "ablation_step":
                steps.append(record)
    return sorted(steps, key=lambda r: r["step"])


def plot_ablation_graph() -> str:
    steps = load_ablation_steps()
    if not steps:
        raise RuntimeError("No ablation_step records in log_nogate.jsonl -- run src/ablation.py first.")

    x = [s["step"] for s in steps]
    on_pct = [100 * s["gate_on_score"] / s["gate_on_total"] for s in steps]
    off_pct = [100 * s["gate_off_score"] / s["gate_off_total"] for s in steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, on_pct, marker="o", label="Gate ON (this system)", color="#16a34a", linewidth=2, zorder=3)
    ax.plot(x, off_pct, marker="o", label="Gate OFF (no regression check)", color="#dc2626", linewidth=2, linestyle="--", zorder=2)

    for xi, y, s in zip(x, on_pct, steps):
        ax.annotate(f"{s['gate_on_score']}/{s['gate_on_total']}", (xi, y), textcoords="offset points", xytext=(0, 10), ha="center", color="#16a34a")
    for xi, y_on, y_off, s in zip(x, on_pct, off_pct, steps):
        if y_off != y_on:
            ax.annotate(f"{s['gate_off_score']}/{s['gate_off_total']}", (xi, y_off), textcoords="offset points", xytext=(0, -16), ha="center", color="#dc2626")

    reject_step = next((s for s in steps if s["gate_verdict"] == "REJECT"), None)
    if reject_step:
        ax.axvline(reject_step["step"], color="#dc2626", alpha=0.15, linewidth=12, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in steps], rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_title("Ablation: held-out accuracy as the same candidate lessons are installed")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)

    if reject_step:
        fig.text(
            0.5, 0.02,
            f"At '{reject_step['label']}': the gate REJECTS it, so gate-ON never installs it and stays "
            f"{reject_step['gate_on_score']}/{reject_step['gate_on_total']}.\n"
            f"Gate-OFF installs it unconditionally and drops to {reject_step['gate_off_score']}/{reject_step['gate_off_total']} "
            "-- the same lesson, the only difference is the gate.",
            ha="center", va="bottom", fontsize=9, color="#555555", style="italic",
        )

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    os.makedirs(DOCS_DIR, exist_ok=True)
    fig.savefig(ABLATION_OUT_PATH, dpi=150)
    print(f"ablation graph saved -> {ABLATION_OUT_PATH}")
    return ABLATION_OUT_PATH


if __name__ == "__main__":
    plot_hero_graph()
    plot_ablation_graph()
