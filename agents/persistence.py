import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "runs.jsonl"
LOG_DIR.mkdir(exist_ok=True)


def _safe_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:

    # store citations + score
    notes = state.get("notes", []) or []
    if not isinstance(notes, list):
        notes = []

    # compact notes for logs
    notes_compact = []

    for i, n in enumerate(notes, start=1):
        if not isinstance(n, dict):
            continue

        c = n.get("citation") or {}

        notes_compact.append({
            "citation": c,
            "score": n.get("score"),
        })



    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": state.get("task"),
        "retrieval_query": state.get("retrieval_query"),
        "final": state.get("final") or state.get("draft"),
        "trace": state.get("trace", []) or [],      
        "notes": notes_compact,
        "retried": bool(state.get("retried", False)),
        "latency_ms": state.get("latency_ms"),
    }


def save_run(state: Dict[str, Any]) -> str:
    record = _safe_state_snapshot(state)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(LOG_FILE)
