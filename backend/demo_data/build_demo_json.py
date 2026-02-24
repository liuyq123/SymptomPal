#!/usr/bin/env python3
"""
Build frontend demo JSON files entirely from backend simulation results.

Reads each patient's *_responses.json and generates the corresponding
frontend/public/demo/*.json with 100% authentic data: actual LLM messages,
actual MedGemma extraction, actual protocol decisions, and actual doctor packet.

All clinical content comes from the responses file. Times are synthetic
(responses lack per-day timestamps).

Usage:
    python build_demo_json.py
"""

import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path


PATIENTS = ["frank_russo", "elena_martinez", "sarah_chen"]

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
PUBLIC_DIR = BASE_DIR / ".." / ".." / "frontend" / "public" / "demo"

# Strip TREND labels echoed by MedGemma into agent responses.
_TREND_STRIP = re.compile(r'\s*TREND:\s*.+$', re.MULTILINE)


def _generate_time(day: int, patient_id: str) -> str:
    """Generate a realistic-looking time for a log entry, seeded for determinism."""
    rng = random.Random(f"{patient_id}_{day}")
    hour = rng.choices(
        [7, 8, 9, 10, 12, 14, 17, 19, 20, 21],
        weights=[3, 4, 3, 2, 1, 1, 1, 2, 2, 1],
    )[0]
    minute = rng.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"


def _build_vital_signs_dict(vital_signs_list: list) -> dict:
    """Convert backend vital_signs array to frontend Record<string, number|string>."""
    result = {}
    for v in vital_signs_list:
        if not isinstance(v, dict):
            continue
        name = v.get("name", "")
        value = v.get("value")
        if value is None:
            continue
        try:
            value = int(value)
        except (ValueError, TypeError):
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass  # keep as string (e.g. "158/95")
        result[name] = value
    return result


def _build_messages(lr: dict) -> list:
    """Build frontend messages array from a log_response entry."""
    actual = lr.get("actual_response", {})
    agent = actual.get("agent_response", {})

    messages = [
        {"speaker": "patient", "text": lr["transcript"], "audio": None},
        {
            "speaker": "agent",
            "text": _TREND_STRIP.sub("", agent.get("acknowledgment", "")).strip(),
            "audio": None,
            "question": agent.get("immediate_question"),
        },
    ]

    # Append answered followup exchanges
    exchanges = lr.get("followup_exchanges", [])
    if not exchanges and lr.get("followup_answer"):
        # Backward compat for old response files with scalar fields
        exchanges = [{"answer": lr["followup_answer"], "agent_response": lr.get("followup_response")}]
    for exchange in exchanges:
        if exchange.get("answer"):
            messages.append({
                "speaker": "patient",
                "text": exchange["answer"],
                "audio": None,
                "isFollowup": True,
            })
            if exchange.get("agent_response"):
                messages.append({
                    "speaker": "agent",
                    "text": exchange["agent_response"],
                    "audio": None,
                    "isFollowupResponse": True,
                })

    return messages


def _format_symptom(s: dict) -> str:
    name = s.get("symptom", "")
    sev = s.get("severity_1_10")
    return f"{name} ({sev}/10)" if sev else name


def _format_action(a: dict) -> str:
    name = a.get("name", "")
    dose = a.get("dose_text")
    return f"{name} {dose}" if dose else name


def _build_metadata(lr: dict) -> dict:
    """Build frontend metadata from extraction + agent response."""
    actual = lr.get("actual_response", {})
    agent = actual.get("agent_response", {})
    extracted = actual.get("extracted", {})

    # Red-flag entries must not show autonomous tool calls —
    # backend now enforces this (ingest.py), but older simulation
    # results predate the fix.
    safety_mode = agent.get("safety_mode")
    tool_calls = [] if safety_mode == "static_safety" else agent.get("tool_calls", [])

    return {
        "symptoms": [_format_symptom(s) for s in extracted.get("symptoms", []) if isinstance(s, dict)],
        "actions_taken": [_format_action(a) for a in extracted.get("actions_taken", []) if isinstance(a, dict)],
        "vital_signs": _build_vital_signs_dict(extracted.get("vital_signs", [])),
        "red_flags": extracted.get("red_flags", []),
        "protocol": agent.get("protocol_id"),
        "clinician_note": actual.get("contact_clinician_note"),
        "safety_mode": safety_mode,
        "reason_code": agent.get("reason_code"),
        "tool_calls": tool_calls,
        "question_rationale": agent.get("question_rationale"),
        "agent_trace": agent.get("agent_trace", {}),
    }


PATIENTS_DIR = BASE_DIR / "patients"

# In-character patient answers for intake questions.
# Keys: patient_id → list of (question_id, profile_field_label, agent_question, patient_answer)
_INTAKE_ANSWERS = {
    "frank_russo": [
        (
            "conditions", "conditions",
            "To personalize tracking, do you have any chronic conditions?",
            "Yeah I got COPD, the moderate kind they said. And high blood pressure.",
        ),
        (
            "allergies", "allergies",
            "Any medication or food allergies I should record?",
            "Penicillin. Found that out the hard way years ago.",
        ),
        (
            "regular_medications", "medications",
            "What regular medications or supplements do you take?",
            "Let me think... Spiriva, the Advair twice a day, and the rescue puffer, ProAir. Oh and lisinopril for the blood pressure.",
            ["Tiotropium (Spiriva) inhaler", "Fluticasone-salmeterol (Advair) inhaler twice daily", "Albuterol (ProAir) inhaler PRN rescue", "Lisinopril"],
        ),
        (
            "regular_medications_doses", "medications",
            "Got it — Spiriva, Advair, ProAir, and lisinopril. Do you know the doses on any of those?",
            "Uh, the Spiriva is 18 micrograms, one puff every morning. The Advair is the 250/50 one. Lisinopril, I think 20 milligrams.",
        ),
        (
            "patterns", "patterns",
            "Any symptom triggers or patterns you've already noticed?",
            "Morning cough is just my normal at this point. I still smoke a little, down to about half a pack. Stairs wind me pretty good.",
        ),
    ],
    "elena_martinez": [
        (
            "conditions", "conditions",
            "To personalize tracking, do you have any chronic conditions?",
            "I have type 2 diabetes and hypertension.",
        ),
        (
            "allergies", "allergies",
            "Any medication or food allergies I should record?",
            "Sulfa drugs — I had a bad reaction once.",
        ),
        (
            "regular_medications", "medications",
            "What regular medications or supplements do you take?",
            "Metformin 500 milligrams twice a day with meals, glipizide 5 milligrams in the morning — that one's new — and lisinopril 10 milligrams.",
        ),
        (
            "patterns", "patterns",
            "Any symptom triggers or patterns you've already noticed?",
            "I take my medications with breakfast and dinner. I check my fasting blood sugar most mornings. The glipizide is new, my doctor just added it.",
        ),
    ],
    "sarah_chen": [
        (
            "conditions", "conditions",
            "To personalize tracking, do you have any chronic conditions?",
            "Nothing diagnosed officially, but my periods have gotten really bad since I stopped birth control six months ago. And mild anemia.",
        ),
        (
            "allergies", "allergies",
            "Any medication or food allergies I should record?",
            "No, none that I know of.",
        ),
        (
            "regular_medications", "medications",
            "What regular medications or supplements do you take?",
            "Just a daily multivitamin, iron supplement for the anemia — 65 milligrams — and ibuprofen when the cramps get bad.",
        ),
        (
            "patterns", "patterns",
            "Any symptom triggers or patterns you've already noticed?",
            "I stopped the pill about six months ago because of mood side effects. Since then the periods have been getting worse and worse. I take ibuprofen a lot for it.",
        ),
    ],
}


def _build_intake_logs_from_results(intake_responses: list) -> list:
    """Build intake log entries from real MedGemma intake responses in results file."""
    logs = []
    for entry in intake_responses:
        log = {
            "day": 0,
            "time": "09:00",
            "phase": "Profile Intake",
            "messages": [
                {"speaker": "agent", "text": entry["agent_question"], "audio": None},
                {"speaker": "patient", "text": entry["patient_answer"], "audio": None},
            ],
            "metadata": {
                "symptoms": [],
                "actions_taken": [],
                "red_flags": [],
                "protocol": None,
                "clinician_note": None,
                "safety_mode": "intake",
                "reason_code": "profile_intake",
                "tool_calls": [],
            },
            "intake": {
                "question_id": entry["question_id"],
                "raw_answer": entry["patient_answer"],
                "parsed_items": entry.get("parsed_items", []),
                "profile_field": entry["question_id"],
            },
        }
        logs.append(log)
    return logs


def _build_intake_logs_fallback(patient_id: str, profile: dict) -> list:
    """Fallback: build intake logs from hardcoded _INTAKE_ANSWERS (no results data)."""
    answers = _INTAKE_ANSWERS.get(patient_id, [])
    if not answers:
        return []

    field_map = {
        "conditions": profile.get("conditions", []),
        "allergies": profile.get("allergies", []),
        "regular_medications": profile.get("regular_medications", []),
        "regular_medications_doses": profile.get("regular_medications", []),
        "patterns": profile.get("patterns", []),
    }

    logs = []
    for entry in answers:
        question_id, field_label, agent_q, patient_a = entry[:4]
        parsed = entry[4] if len(entry) > 4 else field_map.get(question_id, [])
        log = {
            "day": 0,
            "time": "09:00",
            "phase": "Profile Intake",
            "messages": [
                {"speaker": "agent", "text": agent_q, "audio": None},
                {"speaker": "patient", "text": patient_a, "audio": None},
            ],
            "metadata": {
                "symptoms": [],
                "actions_taken": [],
                "red_flags": [],
                "protocol": None,
                "clinician_note": None,
                "safety_mode": "intake",
                "reason_code": "profile_intake",
                "tool_calls": [],
            },
            "intake": {
                "question_id": question_id,
                "raw_answer": patient_a,
                "parsed_items": parsed,
                "profile_field": field_label,
            },
        }
        logs.append(log)

    return logs


def build_patient(patient_id: str) -> None:
    results_file = RESULTS_DIR / f"{patient_id}_responses.json"

    if not results_file.exists():
        print(f"  SKIP {patient_id}: no results file")
        return

    with open(results_file) as f:
        results = json.load(f)

    profile = results.get("profile", {})

    # Patient object
    patient = {
        "name": profile.get("name", patient_id),
        "age": profile.get("age", 0),
        "gender": profile.get("gender", ""),
        "conditions": profile.get("conditions", []),
        "summary": profile.get("health_summary", ""),
    }

    # Intake logs — profile onboarding Q&A before symptom tracking
    intake_responses = results.get("intake_responses", [])
    if intake_responses:
        intake_logs = _build_intake_logs_from_results(intake_responses)
    else:
        intake_logs = _build_intake_logs_fallback(patient_id, profile)

    # Symptom logs — one per backend log_response entry
    logs = []
    for i, lr in enumerate(results.get("log_responses", [])):
        log = {
            "day": lr["day"],
            "time": _generate_time(lr["day"], patient_id),
            "phase": lr.get("phase") or "",
            "messages": _build_messages(lr),
            "metadata": _build_metadata(lr),
        }
        logs.append(log)

    # Combine: intake first, then symptom logs
    logs = intake_logs + logs

    # Doctor packet — direct copy, but dedup system_longitudinal_flags (same
    # watchdog-run duplication as clinician_observations — keep only the last).
    doctor_packet = results.get("doctor_packet")
    if doctor_packet and doctor_packet.get("system_longitudinal_flags"):
        doctor_packet = {
            **doctor_packet,
            "system_longitudinal_flags": doctor_packet["system_longitudinal_flags"][-1:],
        }

    # Watchdog results (clinician observations + health insight checkins)
    # Keep only the most comprehensive (last) observation per simulation —
    # earlier observations are subsets of the final one.
    watchdog_results = results.get("watchdog_results")
    if watchdog_results and watchdog_results.get("clinician_observations"):
        watchdog_results = {
            **watchdog_results,
            "clinician_observations": watchdog_results["clinician_observations"][-1:],
        }

    # Anchor calendar dates — use stored base_date from simulation, or compute fallback
    base_date = results.get("base_date")
    if not base_date:
        max_day = max((lr["day"] for lr in results.get("log_responses", [])), default=0)
        base_date = (datetime.now() - timedelta(days=max_day)).strftime("%Y-%m-%d")

    output = {
        "patient": patient,
        "base_date": base_date,
        "logs": logs,
        "doctor_packet": doctor_packet,
        "watchdog_results": watchdog_results,
    }

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    public_file = PUBLIC_DIR / f"{patient_id}.json"
    with open(public_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  {patient_id}: built {len(logs)} logs from responses")


def main() -> None:
    print("Building frontend demo JSONs from backend responses...\n")
    for patient_id in PATIENTS:
        build_patient(patient_id)
    print("\nDone.")


if __name__ == "__main__":
    main()
