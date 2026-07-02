"""
MCP server: exposes the regression-proof self-improving loop as three tools --
ask (answer a question), improve (run one Reflexion+gate loop pass and report
verdicts), and report (current train/held-out scores and insight store size).
This is the delivery layer: it lets any MCP host call into the gate-protected
agent without touching the Python modules directly.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

from agent import answer as agent_answer
from harness import evaluate
from loop import run_loop
from insights import INSIGHTS_PATH

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout.json")

mcp = FastMCP("gatekeeper")


@mcp.tool()
def ask(question: str) -> str:
    """Answer a question with the RAG agent, using every lesson currently in the insight store."""
    return agent_answer(question)


@mcp.tool()
def improve() -> dict:
    """
    Run one full Reflexion + gate pass over train.json: draft a lesson for
    every training failure, gate each one against the sealed held-out set,
    and promote whatever gets ACCEPTed. Returns accepted lessons, rejected
    lessons (with the held-out ids each would have broken), and how many
    new insights were promoted into the store.
    """
    result = run_loop()

    return {
        "accepted": [
            {"source_id": c["source_id"], "lesson": c["lesson"]}
            for c in result["accepted"]
        ],
        "rejected": [
            {
                "source_id": c["source_id"],
                "lesson": c["lesson"],
                "would_break_heldout_ids": c["broken_ids"],
            }
            for c in result["rejected"]
        ],
        "promoted_count": len(result["promoted"]),
    }


@mcp.tool()
def report() -> dict:
    """Current train/held-out accuracy (using the live insight store) and how many lessons are stored."""
    train = evaluate(TRAIN_PATH, lessons=None)
    heldout = evaluate(HELDOUT_PATH, lessons=None)

    insights_count = 0
    if os.path.exists(INSIGHTS_PATH):
        with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
            insights_count = len(json.load(f))

    return {
        "train_score": train["score"],
        "train_total": train["total"],
        "heldout_score": heldout["score"],
        "heldout_total": heldout["total"],
        "insights_in_store": insights_count,
    }


if __name__ == "__main__":
    mcp.run()
