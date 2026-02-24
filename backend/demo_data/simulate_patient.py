#!/usr/bin/env python3
"""
Simulate a full patient journey for SymptomPal demo.

Usage:
    python simulate_patient.py maria_chen
    python simulate_patient.py elena_martinez --save-responses
    python simulate_patient.py maria_chen --with-audio  # Include ambient sessions
"""

import argparse
import json
import sys
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_KEY = "dev_local_key"


def reset_patient_data(user_id: str) -> None:
    """Clear all existing data for a user before simulation."""
    backend_dir = str(Path(__file__).parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from app.services.storage import reset_user_data
    counts = reset_user_data(user_id)
    total = sum(counts.values())
    if total > 0:
        print(f"\n🧹 Reset: cleared {total} existing records for {user_id}")
        for table, count in counts.items():
            if count > 0:
                print(f"   - {table}: {count} rows deleted")
    else:
        print(f"\n✨ No existing data for {user_id} (clean start)")


def get_headers(user_id: str) -> dict:
    return {
        "X-API-Key": API_KEY,
        "X-User-Id": user_id,
        "Content-Type": "application/json"
    }

def create_profile(patient_data: dict) -> bool:
    """Create or update the user profile."""
    profile = patient_data["profile"]
    user_id = profile["user_id"]

    print(f"\n{'='*60}")
    print(f"PATIENT: {profile['name']}")
    print(f"{'='*60}")
    print(f"Age: {profile['age']}, Gender: {profile['gender']}")
    print(f"Conditions: {', '.join(profile['conditions'])}")
    print(f"Allergies: {', '.join(profile['allergies'])}")
    print(f"Medications: {len(profile['regular_medications'])} regular")
    print(f"{'='*60}")

    # Update profile via API (PATCH to /api/profile)
    resp = requests.patch(
        f"{BASE_URL}/api/profile",
        json={
            "user_id": user_id,
            "add_conditions": profile["conditions"],
            "add_allergies": profile["allergies"],
            "add_regular_medications": profile["regular_medications"],
            "add_patterns": profile["patterns"],
            "health_summary": profile["health_summary"]
        },
        headers=get_headers(user_id)
    )

    if resp.status_code == 200:
        print("✅ Profile created/updated")
        return True
    else:
        print(f"⚠️  Profile update failed: {resp.status_code} - {resp.text[:100]}")
        return False


def _compute_intake_diff(before, after, question_id: str) -> list:
    """Diff profile before/after intake response to extract parsed items."""
    field_map = {
        "conditions": "conditions",
        "allergies": "allergies",
        "regular_medications": "regular_medications",
        "medication_doses": "regular_medications",
        "surgeries": "surgeries",
        "family_history": "family_history",
        "social_history": "social_history",
        "patterns": "patterns",
    }
    field = field_map.get(question_id)
    if not field:
        # Scalar demographic fields
        if question_id == "name" and after.name != before.name:
            return [after.name] if after.name else []
        if question_id == "age_sex":
            items = []
            if after.age and after.age != before.age:
                items.append(f"{after.age} years old")
            if after.gender and after.gender != before.gender:
                items.append(after.gender)
            return items
        # health_summary — return as single-item list if changed
        if question_id == "health_summary" and after.health_summary != before.health_summary:
            return [after.health_summary] if after.health_summary else []
        return []

    before_set = set(getattr(before, field, []) or [])
    after_set = set(getattr(after, field, []) or [])
    return list(after_set - before_set)


def run_intake(patient_data: dict, save_responses: bool = False) -> list:
    """Run profile intake flow through the actual backend API.

    Returns list of intake response dicts if save_responses=True.
    """
    profile = patient_data["profile"]
    user_id = profile["user_id"]
    intake_answers = patient_data.get("intake_answers", [])

    if not intake_answers:
        return []

    print(f"\n📋 Running profile intake ({len(intake_answers)} questions)...")

    # Answer queue keyed by question_id
    answer_map = {a["question_id"]: a["answer"] for a in intake_answers}
    intake_results = []

    # Ensure storage imports are available
    backend_dir = str(Path(__file__).parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from app.services.storage import get_or_create_user_profile

    # Demographics are now collected through intake questions (name, age_sex)
    # so no direct update_user_profile call needed here.

    # Poll for intake checkins and respond
    max_rounds = len(intake_answers) + 2  # safety limit
    for round_num in range(max_rounds):
        time.sleep(0.5)

        # Get pending checkins
        resp = requests.get(
            f"{BASE_URL}/api/checkins/pending",
            params={"user_id": user_id},
            headers=get_headers(user_id)
        )
        if resp.status_code != 200:
            print(f"  ⚠️  Failed to poll checkins: {resp.status_code}")
            break

        pending = resp.json()

        # Find intake checkin
        intake_checkin = None
        for c in pending:
            if c.get("checkin_type") == "profile_intake":
                intake_checkin = c
                break

        if not intake_checkin:
            break  # No more intake questions

        question_id = (intake_checkin.get("context") or {}).get("question_id")
        agent_question = intake_checkin["message"]
        checkin_id = intake_checkin["id"]

        if not question_id:
            print(f"  ⚠️  Intake checkin missing question_id, skipping")
            break

        patient_answer = answer_map.get(question_id)
        if not patient_answer:
            print(f"  ⚠️  No answer defined for question '{question_id}', stopping intake")
            break

        # Capture profile state BEFORE responding (for parsed items diff)
        profile_before = get_or_create_user_profile(user_id)

        # Respond to checkin
        resp = requests.post(
            f"{BASE_URL}/api/checkins/{checkin_id}/respond",
            json={"response": patient_answer},
            headers=get_headers(user_id)
        )

        if resp.status_code != 200:
            print(f"  ⚠️  Checkin respond failed: {resp.status_code} - {resp.text[:100]}")
            break

        # Capture profile state AFTER responding
        profile_after = get_or_create_user_profile(user_id)

        # Compute parsed items from profile diff
        parsed_items = _compute_intake_diff(profile_before, profile_after, question_id)

        q_preview = agent_question[:80]
        a_preview = patient_answer[:80]
        print(f"  [{round_num + 1}] Q: \"{q_preview}{'...' if len(agent_question) > 80 else ''}\"")
        print(f"      A: \"{a_preview}{'...' if len(patient_answer) > 80 else ''}\"")
        if parsed_items:
            print(f"      → Parsed: {parsed_items}")
        else:
            print(f"      → Parsed: (none/empty)")

        if save_responses:
            intake_results.append({
                "question_id": question_id,
                "agent_question": agent_question,
                "patient_answer": patient_answer,
                "parsed_items": parsed_items,
            })

    print(f"  ✅ Intake complete: {len(intake_results)} questions answered")
    return intake_results


def submit_cycle_data(patient_data: dict, base_date: datetime) -> int:
    """Submit cycle day logs (period tracking data) before symptom logs.

    This establishes cycle boundaries so that symptom logs get tagged
    with cycle day/phase during ingest.  In a real-world scenario, users
    would mark period days via the Cycle tab *and* mention their period
    in voice logs (which triggers auto-detection via _auto_log_period).
    The simulation pre-populates the full ground truth here because the
    patient doesn't record a voice log on every period day — roughly
    half the period days have no corresponding symptom transcript.

    Returns:
        Number of cycle days submitted.
    """
    profile = patient_data["profile"]
    user_id = profile["user_id"]
    cycle_logs = patient_data.get("cycle_day_logs", [])

    if not cycle_logs:
        return 0

    print(f"\n🔴 Submitting {len(cycle_logs)} cycle day logs...")

    submitted = 0
    for entry in cycle_logs:
        day_offset = entry["day"] - 1
        log_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        resp = requests.post(
            f"{BASE_URL}/api/cycle/day",
            json={
                "user_id": user_id,
                "date": log_date,
                "flow_level": entry["flow_level"],
                "notes": entry.get("notes"),
            },
            headers=get_headers(user_id),
        )

        if resp.status_code == 200:
            submitted += 1
        else:
            print(f"  ⚠️  Cycle day {log_date} failed: {resp.status_code}")

    # Find period start dates for display
    period_starts = []
    prev_day = None
    for entry in sorted(cycle_logs, key=lambda e: e["day"]):
        if prev_day is None or entry["day"] - prev_day > 5:
            start_date = (base_date + timedelta(days=entry["day"] - 1)).strftime("%Y-%m-%d")
            period_starts.append(start_date)
        prev_day = entry["day"]

    print(f"  ✅ {submitted}/{len(cycle_logs)} cycle days logged")
    print(f"  📅 Period starts: {', '.join(period_starts)}")
    print(f"  🔄 {len(period_starts)} cycles established")

    return submitted


def submit_symptom_logs(patient_data: dict, delay_seconds: float = 1.0, save_responses: bool = False, base_date: datetime = None, log_range: tuple = None) -> tuple:
    """Submit symptom logs for the patient.

    Args:
        log_range: Optional (start, end) tuple to submit a slice of logs.
                   Defaults to all logs.

    Returns:
        (log_ids, full_responses) where full_responses is a list of dicts if save_responses=True
    """
    profile = patient_data["profile"]
    user_id = profile["user_id"]
    all_logs = patient_data["symptom_logs"]
    start, end = log_range if log_range else (0, len(all_logs))
    logs = all_logs[start:end]

    print(f"\n📝 Submitting {len(logs)} symptom logs{f' (range {start}-{end})' if log_range else ''}...")

    if base_date is None:
        base_date = datetime.now() - timedelta(days=max(log["day"] for log in logs))
    log_ids = []
    full_responses = []

    for i, log in enumerate(logs):
        day_offset = log.get("day", i + 1) - 1
        time_str = log.get("time", "09:00")
        hour, minute = map(int, time_str.split(":"))

        recorded_at = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)

        # Show phase header if present
        phase = log.get("phase")
        if phase:
            print(f"\n  {'─'*50}")
            print(f"  📌 {phase}")
            print(f"  {'─'*50}")

        print(f"\n  Day {log['day']} ({recorded_at.strftime('%Y-%m-%d %H:%M')}):")
        transcript = log['transcript']
        print(f"  Patient: \"{transcript[:100]}...\"" if len(transcript) > 100 else f"  Patient: \"{transcript}\"")

        resp = requests.post(
            f"{BASE_URL}/api/ingest/voice",
            json={
                "user_id": user_id,
                "description_text": log["transcript"],
                "recorded_at": recorded_at.isoformat()
            },
            headers=get_headers(user_id)
        )

        if resp.status_code == 200:
            data = resp.json()
            log_entry = data.get("log", {})
            extracted = log_entry.get("extracted", {})
            agent = data.get("agent_response", {})

            symptoms = [s.get("symptom", "unknown") for s in extracted.get("symptoms", [])]
            actions = [a.get("name", "unknown") for a in extracted.get("actions_taken", [])]
            red_flags = extracted.get("red_flags", [])

            print(f"  ✅ Symptoms: {symptoms}")
            if actions:
                print(f"     Actions: {actions}")
            if red_flags:
                print(f"  🚨 RED FLAGS: {red_flags}")

            # Show agent response
            if agent.get("acknowledgment"):
                ack = agent["acknowledgment"]
                print(f"  💬 Agent: \"{ack[:120]}...\"" if len(ack) > 120 else f"  💬 Agent: \"{ack}\"")

            if agent.get("immediate_question"):
                print(f"  ❓ Follow-up: {agent['immediate_question']}")

            if agent.get("scheduled_checkin"):
                checkin = agent["scheduled_checkin"]
                print(f"  ⏰ Check-in: {checkin.get('message', 'scheduled')} (in {checkin.get('delay_minutes', '?')}min)")

            if agent.get("tool_calls"):
                print(f"     Tool calls: {agent['tool_calls']}")
            if agent.get("protocol_id"):
                print(f"     Protocol: {agent['protocol_id']} ({agent.get('reason_code', '')})")

            # Contact clinician note
            if log_entry.get("contact_clinician_note"):
                print(f"  ⚠️  CLINICIAN NOTE: {log_entry['contact_clinician_note']}")

            # Degraded mode warning
            if data.get("degraded_mode"):
                warnings = data.get("warnings", [])
                print(f"  ⚡ Degraded mode: {', '.join(warnings)}")

            # Show expected vs actual if expected_protocol is present
            expected = log.get("expected_protocol")
            actual_protocol = agent.get("protocol_id")
            if expected is not None:
                match = "✅" if expected == actual_protocol else "❌"
                print(f"     Expected protocol: {expected} | Actual: {actual_protocol} {match}")

            log_ids.append(log_entry.get("id"))

            # Submit followup answer if defined in patient data
            followup_answer = log.get("followup_answer")
            followup_response_text = None
            followup_exchanges_data = None
            log_id = log_entry.get("id")
            followup_q = log_entry.get("followup_question") or agent.get("immediate_question")
            if followup_answer and log_id:
                if not followup_q:
                    print(f"  💬 (Unsolicited followup — agent didn't ask a question)")
                followup_resp = requests.post(
                    f"{BASE_URL}/api/logs/{log_id}/followup",
                    json={"answer": followup_answer},
                    headers=get_headers(user_id)
                )
                followup_exchanges_data = None
                if followup_resp.status_code == 200:
                    resp_json = followup_resp.json()
                    followup_response_text = resp_json.get("followup_response")
                    followup_exchanges_data = resp_json.get("followup_exchanges", [])
                    ans_preview = followup_answer[:100]
                    first_name = profile["name"].split()[0]
                    print(f"  💬 {first_name}: \"{ans_preview}...\"" if len(followup_answer) > 100 else f"  💬 {first_name}: \"{ans_preview}\"")
                    if followup_response_text:
                        print(f"  🤖 Agent: \"{followup_response_text[:100]}\"")
                else:
                    print(f"  ⚠️  Followup answer failed: {followup_resp.status_code} - {followup_resp.text[:100]}")

            if save_responses:
                full_responses.append({
                    "day": log["day"],
                    "week": log.get("week"),
                    "phase": log.get("phase"),
                    "transcript": log["transcript"],
                    "expected_protocol": log.get("expected_protocol"),
                    "expected_action": log.get("expected_action"),
                    "actual_response": {
                        "extracted": extracted,
                        "agent_response": agent,
                        "contact_clinician_note": log_entry.get("contact_clinician_note"),
                        "contact_clinician_reason": log_entry.get("contact_clinician_reason"),
                        "image_analysis": log_entry.get("image_analysis"),
                        "log_id": log_entry.get("id")
                    },
                    "followup_answer": log.get("followup_answer"),
                    "followup_response": followup_response_text if followup_answer else None,
                    "followup_exchanges": followup_exchanges_data if followup_answer else None,
                    "degraded_mode": data.get("degraded_mode", False),
                    "warnings": data.get("warnings", [])
                })
        else:
            print(f"  ❌ Failed: {resp.status_code} - {resp.text[:200]}")

        time.sleep(delay_seconds)

    return log_ids, full_responses


def run_ambient_sessions(patient_data: dict) -> list:
    """Run ambient monitoring sessions."""
    profile = patient_data["profile"]
    user_id = profile["user_id"]
    sessions = patient_data.get("ambient_sessions", [])

    if not sessions:
        print("\n📊 No ambient sessions defined")
        return []

    print(f"\n🎙️ Running {len(sessions)} ambient sessions...")
    session_ids = []

    for session in sessions:
        session_type = session["type"]
        print(f"\n  {session_type.upper()}: {session['description']}")

        # Start session
        resp = requests.post(
            f"{BASE_URL}/api/ambient/sessions/start",
            json={
                "user_id": user_id,
                "session_type": session_type,
                "label": session["description"]
            },
            headers=get_headers(user_id)
        )

        if resp.status_code != 200:
            print(f"  ❌ Failed to start: {resp.status_code}")
            continue

        session_id = resp.json()["session"]["id"]

        # Upload a fake chunk (real audio would go here)
        import base64
        import numpy as np

        # Generate 10 seconds of silence
        audio = np.zeros(16000 * 10, dtype=np.float32)
        audio_b64 = base64.b64encode((audio * 32767).astype(np.int16).tobytes()).decode()

        resp = requests.post(
            f"{BASE_URL}/api/ambient/sessions/upload",
            json={
                "session_id": session_id,
                "user_id": user_id,
                "chunk_index": 0,
                "audio_b64": audio_b64,
                "duration_seconds": 10.0
            },
            headers=get_headers(user_id)
        )

        # End session
        resp = requests.post(
            f"{BASE_URL}/api/ambient/sessions/end",
            json={
                "session_id": session_id,
                "user_id": user_id
            },
            headers=get_headers(user_id)
        )

        if resp.status_code == 200:
            result = resp.json()["result"]
            print(f"  ✅ Completed: {result['summary'][:60]}...")
            session_ids.append(session_id)
        else:
            print(f"  ❌ Failed to end: {resp.status_code}")

    return session_ids


def generate_doctor_packet(patient_data: dict) -> dict:
    """Generate the doctor packet summary."""
    profile = patient_data["profile"]
    user_id = profile["user_id"]

    # Use days=0 for full history (map-reduce chunked summarization)
    print(f"\n📋 Generating Doctor Packet (full history)...")

    resp = requests.post(
        f"{BASE_URL}/api/summarize/doctor-packet",
        json={
            "user_id": user_id,
            "days": 0
        },
        headers=get_headers(user_id)
    )

    if resp.status_code == 200:
        packet = resp.json()
        print("\n" + "="*60)
        print("DOCTOR PACKET")
        print("="*60)
        print(f"\n📝 HPI:\n{packet.get('hpi', 'N/A')}")

        print(f"\n✅ Pertinent Positives:")
        for pp in packet.get("pertinent_positives", []):
            print(f"   • {pp}")

        print(f"\n❌ Pertinent Negatives:")
        for pn in packet.get("pertinent_negatives", []):
            print(f"   • {pn}")

        print(f"\n📅 Timeline:")
        for tb in packet.get("timeline_bullets", []):
            print(f"   • {tb}")

        print(f"\n❓ Questions for Clinician:")
        for q in packet.get("questions_for_clinician", []):
            print(f"   • {q}")

        return packet
    else:
        print(f"❌ Failed: {resp.status_code} - {resp.text[:100]}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Simulate a patient journey")
    parser.add_argument("patient", help="Patient file name (without .json)")
    parser.add_argument("--with-audio", action="store_true", help="Include ambient sessions")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between logs (seconds)")
    parser.add_argument("--save-responses", action="store_true", help="Save full responses to JSON file")
    parser.add_argument("--no-reset", action="store_true", help="Skip clearing previous data")
    args = parser.parse_args()

    # Load patient data
    patient_file = Path(__file__).parent / "patients" / f"{args.patient}.json"
    if not patient_file.exists():
        print(f"❌ Patient file not found: {patient_file}")
        print(f"   Available patients:")
        for p in (Path(__file__).parent / "patients").glob("*.json"):
            print(f"   - {p.stem}")
        return

    with open(patient_file) as f:
        patient_data = json.load(f)

    # Check server health
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=30)
        if resp.status_code != 200:
            print(f"❌ Server not healthy: {resp.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("   Start the server with: python -m uvicorn app.main:app --port 8000")
        return

    print("\n" + "="*60)
    print("SYMPTOMPAL - PATIENT SIMULATION")
    print("="*60)

    # Reset stale data from previous runs
    if not args.no_reset:
        user_id = patient_data["profile"]["user_id"]
        reset_patient_data(user_id)

    # Compute shared base_date from the max day offset across all log types
    all_days = [log["day"] for log in patient_data.get("symptom_logs", [])]
    all_days += [log["day"] for log in patient_data.get("cycle_day_logs", [])]
    base_date = (datetime.now() - timedelta(days=max(all_days))).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) if all_days else datetime.now()

    # Run simulation — with or without intake
    intake_results = []
    has_intake = bool(patient_data.get("intake_answers"))

    if has_intake:
        # Onboarding-first: profile setup → intake → cycle data → all symptom logs
        profile = patient_data["profile"]
        user_id = profile["user_id"]

        print(f"\n{'='*60}")
        print(f"PATIENT: {profile['name']} (onboarding mode)")
        print(f"{'='*60}")

        # Create minimal profile (just user_id, triggers get_or_create)
        requests.patch(
            f"{BASE_URL}/api/profile",
            json={"user_id": user_id},
            headers=get_headers(user_id)
        )
        print("✅ Minimal profile created (onboarding will populate)")

        # Start onboarding — creates first intake checkin without needing a symptom log
        resp = requests.post(
            f"{BASE_URL}/api/profile/onboarding/start",
            params={"user_id": user_id},
            headers=get_headers(user_id)
        )
        if resp.status_code == 200:
            print(f"✅ Onboarding started: {resp.json().get('status')}")
        else:
            print(f"⚠️  Onboarding start failed: {resp.status_code} - {resp.text[:100]}")

        # Run intake through real backend API (MedGemma-powered)
        intake_results = run_intake(patient_data, save_responses=args.save_responses)

        # Submit cycle data (establishes cycle boundaries for symptom tagging)
        submit_cycle_data(patient_data, base_date)

        # Submit ALL symptom logs (profile now fully populated by intake)
        _, full_responses = submit_symptom_logs(
            patient_data,
            delay_seconds=args.delay,
            save_responses=args.save_responses,
            base_date=base_date,
        )
    else:
        # Legacy mode: full profile upfront, no intake
        create_profile(patient_data)

        # Submit cycle data first (establishes cycle boundaries for symptom tagging)
        submit_cycle_data(patient_data, base_date)

        _, full_responses = submit_symptom_logs(
            patient_data,
            delay_seconds=args.delay,
            save_responses=args.save_responses,
            base_date=base_date,
        )

    if args.with_audio:
        run_ambient_sessions(patient_data)

    # Wait for Watchdog background tasks to complete, then capture results
    watchdog_results = {}
    if args.save_responses:
        user_id = patient_data["profile"]["user_id"]
        print("\n⏳ Waiting for Watchdog background tasks to finish...")
        time.sleep(5)

        try:
            from app.services.storage import get_pending_checkins, get_watchdog_observations
            checkins = get_pending_checkins(user_id)
            observations = get_watchdog_observations(user_id, limit=50)
            watchdog_results = {
                "pending_checkins": [
                    {
                        "id": c.id,
                        "checkin_type": c.checkin_type,
                        "scheduled_for": c.scheduled_for.isoformat(),
                        "message": c.message,
                        "context": c.context,
                    }
                    for c in checkins
                ],
                "clinician_observations": observations,
            }
            print(f"   Found {len(checkins)} pending check-ins, {len(observations)} clinician observations")
        except Exception as e:
            print(f"   ⚠️ Could not capture Watchdog results: {e}")

    doctor_packet = generate_doctor_packet(patient_data)

    # Save full responses if requested
    if args.save_responses:
        output_file = Path(__file__).parent / "results" / f"{args.patient}_responses.json"
        output_file.parent.mkdir(exist_ok=True)

        # Fetch actual profile from storage (reflects intake updates)
        from app.services.storage import get_or_create_user_profile as _get_profile
        actual_profile = _get_profile(user_id)
        profile_dict = {
            "user_id": actual_profile.user_id,
            "name": patient_data["profile"].get("name"),
            "age": patient_data["profile"].get("age"),
            "gender": patient_data["profile"].get("gender"),
            "conditions": actual_profile.conditions or [],
            "allergies": actual_profile.allergies or [],
            "regular_medications": actual_profile.regular_medications or [],
            "surgeries": actual_profile.surgeries or [],
            "family_history": actual_profile.family_history or [],
            "social_history": actual_profile.social_history or [],
            "patterns": actual_profile.patterns or [],
            "health_summary": actual_profile.health_summary or "",
            "intake_completed": actual_profile.intake_completed,
        }

        output = {
            "patient": args.patient,
            "profile": profile_dict,
            "simulation_timestamp": datetime.now().isoformat(),
            "base_date": base_date.strftime("%Y-%m-%d"),
            "intake_responses": intake_results,
            "log_responses": full_responses,
            "doctor_packet": doctor_packet,
            "watchdog_results": watchdog_results,
            "total_logs": len(full_responses),
            "degraded_logs": sum(1 for r in full_responses if r.get("degraded_mode"))
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n💾 Full responses saved to: {output_file}")

    print("\n" + "="*60)
    print("✅ SIMULATION COMPLETE")
    print("="*60)
    print(f"\nView patient data at: http://localhost:5173")
    print(f"User ID: {patient_data['profile']['user_id']}")


if __name__ == "__main__":
    main()
