#!/usr/bin/env python3
"""
Test the ambient monitor with real audio files.

Usage:
    python test_audio_monitor.py cough demo_data/audio/synthetic_cough_session.wav
    python test_audio_monitor.py sleep path/to/sleep_recording.wav
    python test_audio_monitor.py voice path/to/voice_recording.wav
"""

import argparse
import base64
import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_KEY = "dev_local_key"
USER_ID = "audio-test-user"

def get_headers():
    return {
        "X-API-Key": API_KEY,
        "X-User-Id": USER_ID,
        "Content-Type": "application/json"
    }

def load_audio_file(path: str) -> tuple:
    """Load audio file and return base64 + duration."""
    import wave

    path = Path(path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    if path.suffix.lower() == '.wav':
        # Get duration first
        with wave.open(str(path), 'rb') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            duration = n_frames / sample_rate

        # Read the entire file (including headers) for proper decoding
        with open(path, 'rb') as f:
            audio_bytes = f.read()
    else:
        print(f"❌ Unsupported format: {path.suffix}")
        print("   Supported: .wav")
        sys.exit(1)

    audio_b64 = base64.b64encode(audio_bytes).decode()
    return audio_b64, duration

def chunk_audio(audio_b64: str, duration: float, chunk_duration: float = 10.0) -> list:
    """Split audio into chunks."""
    import base64

    audio_bytes = base64.b64decode(audio_b64)
    bytes_per_second = len(audio_bytes) / duration

    chunks = []
    offset = 0
    chunk_bytes = int(bytes_per_second * chunk_duration)

    while offset < len(audio_bytes):
        end = min(offset + chunk_bytes, len(audio_bytes))
        chunk = audio_bytes[offset:end]
        chunk_b64 = base64.b64encode(chunk).decode()
        chunk_dur = (end - offset) / bytes_per_second
        chunks.append((chunk_b64, chunk_dur))
        offset = end

    return chunks

def run_monitor_session(session_type: str, audio_path: str):
    """Run a complete monitor session with the audio file."""
    print(f"\n{'='*60}")
    print(f"AMBIENT MONITOR TEST - {session_type.upper()}")
    print(f"{'='*60}")
    print(f"Audio: {audio_path}")

    # Load audio
    print("\n1. Loading audio file...")
    audio_b64, duration = load_audio_file(audio_path)
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Size: {len(audio_b64) / 1024:.1f} KB (base64)")

    # Split into chunks
    chunks = chunk_audio(audio_b64, duration, chunk_duration=10.0)
    print(f"   Chunks: {len(chunks)}")

    # Start session
    print("\n2. Starting session...")
    resp = requests.post(
        f"{BASE_URL}/api/ambient/sessions/start",
        json={
            "user_id": USER_ID,
            "session_type": session_type,
            "label": f"Test from {Path(audio_path).name}"
        },
        headers=get_headers()
    )

    if resp.status_code != 200:
        print(f"   ❌ Failed: {resp.status_code} - {resp.text}")
        return

    session_id = resp.json()["session"]["id"]
    print(f"   ✅ Session: {session_id}")

    # Upload chunks
    print("\n3. Uploading audio chunks...")
    total_events = []

    for i, (chunk_b64, chunk_dur) in enumerate(chunks):
        resp = requests.post(
            f"{BASE_URL}/api/ambient/sessions/upload",
            json={
                "session_id": session_id,
                "user_id": USER_ID,
                "chunk_index": i,
                "audio_b64": chunk_b64,
                "duration_seconds": chunk_dur
            },
            headers=get_headers()
        )

        if resp.status_code == 200:
            events = resp.json().get("events_detected", [])
            total_events.extend(events)
            event_types = [e["event_type"] for e in events]
            print(f"   Chunk {i}: {len(events)} events {event_types if events else ''}")
        else:
            print(f"   Chunk {i}: ❌ {resp.status_code}")

    # End session
    print("\n4. Ending session...")
    resp = requests.post(
        f"{BASE_URL}/api/ambient/sessions/end",
        json={"session_id": session_id, "user_id": USER_ID},
        headers=get_headers()
    )

    if resp.status_code == 200:
        result = resp.json()["result"]

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"\n📊 Summary: {result['summary']}")
        print(f"⏱️  Duration: {result['duration_minutes']:.1f} minutes")

        if result.get("cough_metrics"):
            cm = result["cough_metrics"]
            print(f"\n🫁 Cough Metrics:")
            print(f"   Total coughs: {cm['total_coughs']}")
            print(f"   Coughs/hour: {cm['coughs_per_hour']:.1f}")
            if cm.get("peak_cough_period"):
                print(f"   Peak period: {cm['peak_cough_period']}")

        if result.get("sleep_quality"):
            sq = result["sleep_quality"]
            print(f"\n😴 Sleep Quality:")
            print(f"   Rating: {sq['quality_rating']}")
            print(f"   Breathing score: {sq['breathing_regularity_score']:.0f}%")
            print(f"   Apnea events: {sq['apnea_events']}")
            print(f"   Snoring: {sq['snoring_minutes']:.1f} min")

        if result.get("voice_biomarkers"):
            vb = result["voice_biomarkers"]
            print(f"\n🎤 Voice Biomarkers:")
            print(f"   Stress: {vb['stress_level']:.0f}%")
            print(f"   Fatigue: {vb['fatigue_level']:.0f}%")
            print(f"   Congestion: {'Yes' if vb['congestion_detected'] else 'No'}")
            print(f"   Clarity: {vb['voice_clarity_score']:.0f}%")

        if result.get("events_timeline"):
            print(f"\n📋 Events ({len(result['events_timeline'])} total):")
            for e in result["events_timeline"][:10]:
                print(f"   - {e['event_type']}: {e['confidence']*100:.0f}%")
    else:
        print(f"   ❌ Failed: {resp.status_code} - {resp.text}")

    print(f"\n{'='*60}")
    print("✅ TEST COMPLETE")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test ambient monitor with audio files")
    parser.add_argument("type", choices=["cough", "sleep", "voice", "general"],
                       help="Session type")
    parser.add_argument("audio_file", help="Path to WAV audio file")
    args = parser.parse_args()

    # Map friendly names to API session types
    type_map = {
        "cough": "cough_monitor",
        "sleep": "sleep",
        "voice": "voice_biomarker",
        "general": "general"
    }

    run_monitor_session(type_map[args.type], args.audio_file)


if __name__ == "__main__":
    main()
