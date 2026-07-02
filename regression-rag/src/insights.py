"""
Insight store: the durable home for lessons that survived the gate. promote()
reads every candidate in data/candidates.json, keeps only the ones with
gate_verdict == "ACCEPT", and merges them into data/insights.json (deduped by
source_id, so re-running loop.py doesn't create duplicate entries).

load_insights() reads that store back out as a single formatted string --
this is what agent.answer() loads automatically whenever a caller doesn't
pass its own lessons argument.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from logger import log_event

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidates.json")
INSIGHTS_PATH = os.path.join(DATA_DIR, "insights.json")


def promote() -> list[dict]:
    """
    Merge every ACCEPTED candidate into the insight store. Existing insights
    are kept as-is; a candidate whose source_id is already stored is skipped
    rather than duplicated or overwritten.
    """
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    if os.path.exists(INSIGHTS_PATH):
        with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
            insights = json.load(f)
    else:
        insights = []

    known_ids = {insight["source_id"] for insight in insights}
    added = []

    for candidate in candidates:
        if candidate.get("gate_verdict") != "ACCEPT":
            continue
        if candidate["source_id"] in known_ids:
            continue
        insight = {
            "source_id": candidate["source_id"],
            "lesson": candidate["lesson"],
        }
        insights.append(insight)
        added.append(insight)
        known_ids.add(candidate["source_id"])
        log_event("insight_promoted", source_id=insight["source_id"], lesson=insight["lesson"])

    with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2)

    return added


def load_insights() -> str:
    """
    Read data/insights.json and format it as a single lessons string ready to
    drop into the agent's prompt. Returns "" if the store doesn't exist yet
    or is empty, so callers can treat "no insights" the same as "no lessons".
    """
    if not os.path.exists(INSIGHTS_PATH):
        return ""

    with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
        insights = json.load(f)

    if not insights:
        return ""

    return "\n".join(f"- {insight['lesson']}" for insight in insights)


if __name__ == "__main__":
    added = promote()
    if added:
        print(f"{len(added)} new insight(s) promoted -> {INSIGHTS_PATH}")
        for insight in added:
            print(f"  [{insight['source_id']}] {insight['lesson']}")
    else:
        print(f"No new insights to promote. {INSIGHTS_PATH} unchanged.")
