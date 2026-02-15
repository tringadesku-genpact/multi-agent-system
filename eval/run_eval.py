import os
os.environ["EVAL_MODE"] = "1"


import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

from agents.graph import run as run_graph


def normalize(s: str) -> str:
    return (s or "").strip().lower()


def count_citations(text: str) -> int:
    return len(re.findall(r"\[(\d+)\]", text or ""))


def count_list_items(text: str) -> int:
    lines = (text or "").splitlines()
    return sum(1 for ln in lines if re.match(r"^\s*(-|\u2022|\d+\.)\s+", ln))


def format_check(fmt: str, final: str) -> Tuple[bool, str]:
    f = normalize(final)

    if fmt == "list":
        if count_list_items(final) < 3:
            return False, "format=list check failed (expected >= 3 list items)"
        return True, ""

    if fmt == "ranking":
        if count_list_items(final) < 3:
            return False, "format=ranking check failed (expected >= 3 ranked items)"
        return True, ""

    if fmt == "where":
        if not any(k in f for k in ["stage", "production", "transport", "storage", "retail", "warehouse", "farm", "cold chain", "where"]):
            return False, "format=where check failed (no stage/category signal)"
        return True, ""

    if fmt == "why":
        if not any(k in f for k in ["because", "reason", "due to", "helps", "enables"]):
            return False, "format=why check failed (no rationale signal)"
        return True, ""

    if fmt == "how":
        if "step" not in f and count_list_items(final) < 2:
            return False, "format=how check failed (no steps/bullets)"
        return True, ""

    if fmt == "compare":
        if not any(k in f for k in ["compare", "vs", "versus", "difference", "differences"]):
            return False, "format=compare check failed (no comparison signal)"
        return True, ""

    if fmt == "definition":
        if not any(k in f for k in ["is", "refers to", "means", "definition"]):
            return False, "format=definition check failed (no definitional signal)"
        return True, ""

    if fmt == "risks":
        if not any(k in f for k in ["risk", "risks", "challenge", "threat", "mitigat"]):
            return False, "format=risks check failed (no risk/mitigation signal)"
        return True, ""

    # Unknown format - do nothing
    return True, ""



def run_one(case: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    inp = case["input"]
    exp = case.get("expect", {})

    state = run_graph(task=inp, top_k=case.get("top_k", 5))

    final = (state.get("final") or state.get("draft") or "").strip()
    stop = bool(state.get("stop", False))

    errors: List[str] = []

    if "should_stop" in exp:
        expected_stop = bool(exp["should_stop"])
        if stop != expected_stop:
            errors.append(f"should_stop expected {expected_stop} got {stop}")

    if "final_contains" in exp:
        needle = normalize(exp["final_contains"])
        if needle not in normalize(final):
            errors.append(f"final missing substring: {exp['final_contains']}")

    if "min_citations" in exp:
        c = count_citations(final)
        if c < int(exp["min_citations"]):
            errors.append(f"min_citations expected {exp['min_citations']} got {c}")

    if "must_include_any" in exp:
        options = exp["must_include_any"]
        if not any(normalize(s) in normalize(final) for s in options):
            errors.append(f"must_include_any not satisfied: {options}")

    if "must_not_include_any" in exp:
        bad = [s for s in exp["must_not_include_any"] if normalize(s) in normalize(final)]
        if bad:
            errors.append(f"must_not_include_any violated: {bad}")

    if "min_items" in exp:
        n = count_list_items(final)
        if n < int(exp["min_items"]):
            errors.append(f"min_items expected {exp['min_items']} got {n}")

    fmt = exp.get("format")
    if fmt:
        ok, msg = format_check(fmt, final)
        if not ok:
            errors.append(msg)

    passed = len(errors) == 0
    details = {
        "stop": stop,
        "latency_ms": state.get("latency_ms"),
        "final_preview": final[:900],
    }
    return passed, errors, details



def main() -> None:
    questions_path = Path(__file__).parent / "questions.json"
    if not questions_path.exists():
        print(f"ERROR: questions.json not found at {questions_path}")
        sys.exit(2)

    cases = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        print("ERROR: questions.json must be a non-empty list of cases")
        sys.exit(2)

    total = len(cases)
    passed_n = 0
    results_cases = []

    print(f"Running {total} evaluation cases from questions.json...\n")

    for case in cases:
        case_id = case.get("id", "(no id)")
        ok, errors, details = run_one(case)

        if ok:
            passed_n += 1
            print(f"PASS  {case_id}  |  latency={details.get('latency_ms')} ms")
        else:
            print(f"FAIL  {case_id}")
            for e in errors:
                print(f"  - {e}")
            print(f"  stop={details['stop']} latency_ms={details['latency_ms']}")
            print(f"  final:\n{details['final_preview']}\n")

        results_cases.append({
            "id": case_id,
            "passed": ok,
            "errors": errors,
            "stop": details["stop"],
            "latency_ms": details.get("latency_ms")
        })

    print(f"\nResult: {passed_n}/{total} passed")

    results = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "total": total,
        "passed": passed_n,
        "failed": total - passed_n,
        "cases": results_cases
    }

    results_path = Path(__file__).parent / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nEval results saved to {results_path}")

    sys.exit(0 if passed_n == total else 1)



if __name__ == "__main__":
    main()
