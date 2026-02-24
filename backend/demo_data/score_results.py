#!/usr/bin/env python3
"""Cross-patient metrics for SymptomPal simulation results.

Computes per-patient and aggregate quality metrics:
  - cheerleader_rate: % of responses with cheerleader phrases
  - pattern_echo_rate: % of responses parroting pattern/trend text
  - diagnosis_violation_rate: % of responses with diagnostic language
  - protocol_match_rate: % of logs where expected == actual protocol
  - clinician_alert_rate: % of logs that fired a clinician alert
  - degraded_rate: % of logs in degraded mode

Usage:
    python score_results.py
    python score_results.py --verbose   # Show flagged text per violation
"""

import argparse
import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# --- Detection regexes ---

CHEERLEADER_RE = re.compile(
    r"(?i)(?:that'?s\s+(?:fantastic|wonderful|amazing|incredible|excellent|great)\b"
    r"|keep\s+up\s+the\s+(?:great|good|excellent)\s+work"
    r"|which\s+is\s+(?:fantastic|wonderful|amazing|excellent)\s+news)"
)

PATTERN_ECHO_RE = re.compile(
    r"(?i)(?:symptoms?\s+(?:often|frequently)\s+occur"
    r"|(?:precede|coincide)\s+with"
    r"|rescue\s+inhaler\s+use\s+increases"
    r"|you'?ve\s+logged\s+\d+\s+times)"
)

DIAGNOSIS_RE = re.compile(
    r"(?i)(?:you\s+(?:might|may|could)\s+have\s+\w"
    r"|this\s+(?:could\s+be|is\s+likely|is\s+probably)\s+\w"
    r"|I\s+think\s+you\s+have"
    r"|sounds\s+like\s+you\s+have)"
)


def collect_patient_facing_texts(log: dict) -> list[str]:
    """Collect all patient-facing text from a single log entry."""
    texts = []
    resp = log.get("actual_response", {}).get("agent_response", {})

    ack = resp.get("acknowledgment")
    if ack:
        texts.append(ack)

    nudge = resp.get("insight_nudge")
    if nudge:
        texts.append(nudge)

    for ex in log.get("followup_exchanges") or []:
        agent_resp = ex.get("agent_response")
        if agent_resp:
            texts.append(agent_resp)

    return texts


def score_patient(data: dict, verbose: bool = False) -> dict:
    """Score a single patient's simulation results."""
    logs = data.get("log_responses", [])
    patient = data.get("patient", "unknown")

    total_responses = 0
    cheerleader_hits = 0
    pattern_echo_hits = 0
    diagnosis_hits = 0

    total_logs = len(logs)
    protocol_expected = 0
    protocol_matched = 0
    clinician_alerts = 0
    degraded = 0

    for log in logs:
        day = log.get("day", "?")
        texts = collect_patient_facing_texts(log)
        total_responses += len(texts)

        for text in texts:
            if CHEERLEADER_RE.search(text):
                cheerleader_hits += 1
                if verbose:
                    match = CHEERLEADER_RE.search(text)
                    print(f"  CHEER Day {day}: ...{match.group()}...")

            if PATTERN_ECHO_RE.search(text):
                pattern_echo_hits += 1
                if verbose:
                    match = PATTERN_ECHO_RE.search(text)
                    print(f"  ECHO  Day {day}: ...{match.group()}...")

            if DIAGNOSIS_RE.search(text):
                diagnosis_hits += 1
                if verbose:
                    match = DIAGNOSIS_RE.search(text)
                    print(f"  DIAG  Day {day}: ...{match.group()}...")

        # Protocol matching
        expected = log.get("expected_protocol")
        actual = (log.get("actual_response", {})
                  .get("agent_response", {})
                  .get("protocol_id"))
        if expected is not None:
            protocol_expected += 1
            if expected == actual:
                protocol_matched += 1

        # Clinician alerts
        if log.get("actual_response", {}).get("contact_clinician_note"):
            clinician_alerts += 1

        # Degraded mode
        if log.get("degraded_mode"):
            degraded += 1

    return {
        "patient": patient,
        "total_logs": total_logs,
        "total_responses": total_responses,
        "cheerleader_rate": cheerleader_hits / total_responses if total_responses else 0,
        "cheerleader_count": cheerleader_hits,
        "pattern_echo_rate": pattern_echo_hits / total_responses if total_responses else 0,
        "pattern_echo_count": pattern_echo_hits,
        "diagnosis_violation_rate": diagnosis_hits / total_responses if total_responses else 0,
        "diagnosis_count": diagnosis_hits,
        "protocol_match_rate": protocol_matched / protocol_expected if protocol_expected else 0,
        "protocol_expected": protocol_expected,
        "protocol_matched": protocol_matched,
        "clinician_alert_rate": clinician_alerts / total_logs if total_logs else 0,
        "clinician_alert_count": clinician_alerts,
        "degraded_rate": degraded / total_logs if total_logs else 0,
        "degraded_count": degraded,
    }


def print_table(scores: list[dict]) -> None:
    """Print a formatted metrics table."""
    print("\n" + "=" * 90)
    print(f"{'Patient':<20} {'Logs':>5} {'Resp':>5} "
          f"{'Cheer%':>7} {'Echo%':>7} {'Diag%':>7} "
          f"{'Proto%':>7} {'Alert%':>7} {'Degr%':>7}")
    print("-" * 90)

    for s in scores:
        print(f"{s['patient']:<20} {s['total_logs']:>5} {s['total_responses']:>5} "
              f"{s['cheerleader_rate']:>6.1%} {s['pattern_echo_rate']:>6.1%} "
              f"{s['diagnosis_violation_rate']:>6.1%} "
              f"{s['protocol_match_rate']:>6.1%} {s['clinician_alert_rate']:>6.1%} "
              f"{s['degraded_rate']:>6.1%}")

    # Aggregate
    total_resp = sum(s["total_responses"] for s in scores)
    total_logs = sum(s["total_logs"] for s in scores)
    total_cheer = sum(s["cheerleader_count"] for s in scores)
    total_echo = sum(s["pattern_echo_count"] for s in scores)
    total_diag = sum(s["diagnosis_count"] for s in scores)
    total_proto_exp = sum(s["protocol_expected"] for s in scores)
    total_proto_match = sum(s["protocol_matched"] for s in scores)
    total_alerts = sum(s["clinician_alert_count"] for s in scores)
    total_degraded = sum(s["degraded_count"] for s in scores)

    print("-" * 90)
    print(f"{'AGGREGATE':<20} {total_logs:>5} {total_resp:>5} "
          f"{total_cheer / total_resp if total_resp else 0:>6.1%} "
          f"{total_echo / total_resp if total_resp else 0:>6.1%} "
          f"{total_diag / total_resp if total_resp else 0:>6.1%} "
          f"{total_proto_match / total_proto_exp if total_proto_exp else 0:>6.1%} "
          f"{total_alerts / total_logs if total_logs else 0:>6.1%} "
          f"{total_degraded / total_logs if total_logs else 0:>6.1%}")
    print("=" * 90)

    # Detail counts
    print(f"\nDetail: {total_cheer} cheerleader, {total_echo} pattern echo, "
          f"{total_diag} diagnosis violations across {total_resp} responses")
    print(f"Protocol: {total_proto_match}/{total_proto_exp} matched")
    print(f"Clinician alerts: {total_alerts}/{total_logs} logs")
    print(f"Degraded: {total_degraded}/{total_logs} logs")


def main():
    parser = argparse.ArgumentParser(description="Score SymptomPal simulation results")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show flagged text per violation")
    parser.add_argument("files", nargs="*",
                        help="Specific result files (default: all in results/)")
    args = parser.parse_args()

    if args.files:
        result_files = [Path(f) for f in args.files]
    else:
        result_files = sorted(RESULTS_DIR.glob("*_responses.json"))

    if not result_files:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    scores = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        print(f"\nScoring: {data.get('patient', path.stem)}")
        score = score_patient(data, verbose=args.verbose)
        scores.append(score)

    print_table(scores)


if __name__ == "__main__":
    main()
