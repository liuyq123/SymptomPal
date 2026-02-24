#!/usr/bin/env python3
"""
Regenerate patient concerns (questions_for_clinician) in demo data.

Calls the backend /api/summarize/doctor-packet endpoint with the updated
prompt that generates patient-perspective concerns instead of clinician
diagnostic questions. Patches only the questions_for_clinician field in
both the results JSON and the frontend demo JSON.

Requires the backend to be running with MedGemma configured.

Usage:
    python regen_patient_concerns.py                  # All patients
    python regen_patient_concerns.py frank_russo       # Single patient
"""

import argparse
import json
import sys
import requests
from pathlib import Path

PATIENTS = ["frank_russo", "elena_martinez", "sarah_chen"]

BASE_URL = "http://localhost:8000"
API_KEY = "dev_local_key"

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
PUBLIC_DIR = BASE_DIR / ".." / ".." / "frontend" / "public" / "demo"


def get_headers(user_id: str) -> dict:
    return {
        "X-API-Key": API_KEY,
        "X-User-Id": user_id,
        "Content-Type": "application/json",
    }


def regen_patient(patient_id: str) -> bool:
    """Regenerate patient concerns for a single patient."""
    results_file = RESULTS_DIR / f"{patient_id}_responses.json"
    public_file = PUBLIC_DIR / f"{patient_id}.json"

    if not results_file.exists():
        print(f"  SKIP {patient_id}: no results file at {results_file}")
        return False

    with open(results_file) as f:
        results = json.load(f)

    profile = results.get("profile", {})
    user_id = profile.get("user_id")

    if not user_id:
        print(f"  SKIP {patient_id}: no user_id in profile")
        return False

    # First, we need logs in the backend for this patient.
    # The doctor-packet endpoint reads logs from storage.
    # Check if the backend has logs for this user.
    print(f"\n  Generating patient concerns for {patient_id} (user: {user_id})...")

    resp = requests.post(
        f"{BASE_URL}/api/summarize/doctor-packet",
        json={"user_id": user_id, "days": 0},
        headers=get_headers(user_id),
    )

    if resp.status_code != 200:
        print(f"  ERROR: doctor-packet returned {resp.status_code}: {resp.text[:200]}")
        print(f"  Hint: Make sure the backend is running and has logs for {user_id}.")
        print(f"  You may need to run: python simulate_patient.py {patient_id} --save-responses")
        return False

    packet = resp.json()
    new_concerns = packet.get("questions_for_clinician", [])

    if not new_concerns:
        print(f"  WARNING: got empty patient concerns from MedGemma")
        return False

    print(f"  Got {len(new_concerns)} patient concerns:")
    for c in new_concerns:
        print(f"    - {c}")

    # Patch results file
    old_results_concerns = (
        results.get("doctor_packet", {}).get("questions_for_clinician", [])
    )
    if results.get("doctor_packet"):
        results["doctor_packet"]["questions_for_clinician"] = new_concerns
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Patched {results_file.name} ({len(old_results_concerns)} -> {len(new_concerns)} concerns)")

    # Patch public demo file
    if public_file.exists():
        with open(public_file) as f:
            public_data = json.load(f)

        old_public_concerns = (
            public_data.get("doctor_packet", {}).get("questions_for_clinician", [])
        )
        if public_data.get("doctor_packet"):
            public_data["doctor_packet"]["questions_for_clinician"] = new_concerns
            with open(public_file, "w") as f:
                json.dump(public_data, f, indent=2)
            print(f"  Patched {public_file.name} ({len(old_public_concerns)} -> {len(new_concerns)} concerns)")
    else:
        print(f"  NOTE: No public demo file at {public_file} — run build_demo_json.py to create it")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate patient concerns in demo data via MedGemma"
    )
    parser.add_argument(
        "patient",
        nargs="?",
        help="Patient ID (e.g. frank_russo). Omit for all patients.",
    )
    args = parser.parse_args()

    patients = [args.patient] if args.patient else PATIENTS

    print("Regenerating patient concerns via MedGemma...\n")
    print(f"Backend: {BASE_URL}")
    print(f"Patients: {', '.join(patients)}")

    success = 0
    for pid in patients:
        if regen_patient(pid):
            success += 1

    print(f"\nDone. {success}/{len(patients)} patients updated.")
    if success < len(patients):
        print("For failed patients, ensure the backend is running with MedGemma")
        print("and that logs exist (run simulate_patient.py first if needed).")


if __name__ == "__main__":
    main()
