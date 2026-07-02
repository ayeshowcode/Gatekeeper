"""
Logger: appends one JSON line per event to data/log.jsonl -- a permanent
audit trail. Unlike candidates.json (overwritten every loop.py run) or
insights.json (deduplicated, current-state-only), this file only ever grows:
every gate verdict and every promotion gets its own timestamped record here,
even ones that get superseded later.
"""

import os
import json
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
LOG_PATH = os.path.join(DATA_DIR, "log.jsonl")


def log_event(event_type: str, **fields) -> None:
    """Append one timestamped JSON line to data/log.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **fields,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
