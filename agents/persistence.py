import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "runs.jsonl"
LOG_DIR.mkdir(exist_ok=True)


def _safe_state_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persist only what you need for demos/eval.
    Avoid storing secrets or huge payloads.
    """
    notes = state.get("notes", []) or []
    # store only citations + score (not full chunk text)
    notes_compact = [
        {
            "citation": n.get("citation", {}),
            "score": n.get("score"),
        }
        for n in notes
    ]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": state.get("task"),
        "retrieval_query": state.get("retrieval_query"),
        "final": state.get("final") or state.get("draft"),
        "trace": state.get("trace", []),
        "sources": state.get("source_refs", []),
        "notes": notes_compact,
        "retried": state.get("retried", False),
    }


def save_run(state: Dict[str, Any]) -> str:
    record = _safe_state_snapshot(state)
    line = json.dumps(record, ensure_ascii=False)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return str(LOG_FILE)
