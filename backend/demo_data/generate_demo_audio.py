#!/usr/bin/env python3
"""
Generate demo data with Gemini TTS audio for the frontend demo player.

This script reads saved simulation responses (from simulate_patient.py --save-responses)
and the original patient JSON, then generates TTS audio for each message using Gemini TTS.
The output is a single JSON file that the frontend DemoPlayer component can replay.

Usage:
    # Prerequisites: run simulation first (requires backend + MedGemma API)
    python simulate_patient.py frank_russo --save-responses

    # Then generate TTS audio (requires GEMINI_API_KEY)
    export GEMINI_API_KEY=your_key_here
    python generate_demo_audio.py frank_russo

    # Generate without TTS (audio fields will be null, uses timer-based playback)
    python generate_demo_audio.py frank_russo --no-audio

    # Resume: skip clips with matching audio, re-generate only failed/changed
    python generate_demo_audio.py frank_russo --resume

    # Output: ../../frontend/public/demo/frank_russo.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import base64
import struct
import io
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Voice configuration ---
AGENT_VOICE = "Aoede"
PATIENT_VOICES = {
    "frank_russo": "Orus",
    "elena_martinez": "Leda",
    "sarah_chen": "Zephyr",
}
DEFAULT_PATIENT_VOICE = "Puck"

# --- Per-patient personality profiles for emotional TTS ---
PATIENT_PROFILES = {
    "frank_russo": {
        "base": "Frank, a 61-year-old retired construction foreman with COPD. He's gruff, blue-collar, and tends to minimize his symptoms.",
        "calm": "casual and matter-of-fact, like he's just checking in",
        "worried": "trying to sound tough but clearly worried",
        "distressed": "breathless and struggling, short gasping phrases",
        "scared": "genuinely frightened, voice shaking, barely able to speak",
        "recovering": "relieved but tired, slower and quieter than usual",
    },
    "elena_martinez": {
        "base": "Elena, a 62-year-old retired teacher managing diabetes and hypertension. She's warm, compliant, but frustrated by side effects.",
        "calm": "gentle and conversational, reporting her routine",
        "worried": "anxious and hesitant, voice tight with concern",
        "distressed": "frustrated and defeated, close to tears",
        "scared": "panicking, voice trembling, speaking rapidly",
        "recovering": "cautiously hopeful, warmer and more relaxed",
    },
    "sarah_chen": {
        "base": "Sarah, a 29-year-old software engineer. She's casual, tech-savvy, and initially dismissive of her symptoms.",
        "calm": "casual millennial tone, almost breezy",
        "worried": "frustrated and annoyed, starting to take it seriously",
        "distressed": "exhausted and drained, speaking slowly like every word costs energy",
        "scared": "in significant pain, voice tight and strained",
        "recovering": "emotional and relieved, finally feeling heard",
    },
}
DEFAULT_PROFILE = {
    "base": "a patient describing their symptoms",
    "calm": "conversational", "worried": "concerned", "distressed": "upset",
    "scared": "frightened", "recovering": "relieved",
}

# --- Emotional tone helpers ---

def _patient_emotional_state(phase: str, red_flags: list, transcript: str) -> str:
    """Determine emotional state from clinical context."""
    phase_lower = (phase or "").lower()
    transcript_lower = transcript.lower()

    # Red flags = most urgent
    if red_flags:
        return "scared"

    # Phase-based detection
    if any(w in phase_lower for w in ["crisis", "wrong", "hits hard", "not just"]):
        return "distressed"
    if any(w in phase_lower for w in ["recovery", "new normal", "armed with data", "stabiliz"]):
        return "recovering"
    if any(w in phase_lower for w in ["escalat", "frustration", "heavier", "repeats"]):
        return "worried"

    # Transcript-based fallback
    if any(w in transcript_lower for w in ["can't breathe", "passed out", "emergency", "er ", "hospital"]):
        return "scared"
    if any(w in transcript_lower for w in ["worse", "barely", "killing me", "can't sleep", "terrible"]):
        return "distressed"
    if any(w in transcript_lower for w in ["worried", "concerned", "frustrated", "sick of"]):
        return "worried"

    return "calm"


def _build_patient_tone(patient_id: str, phase: str, red_flags: list, transcript: str) -> str:
    """Build per-message tone prompt for patient TTS."""
    profile = PATIENT_PROFILES.get(patient_id, DEFAULT_PROFILE)
    state = _patient_emotional_state(phase, red_flags, transcript)
    mood = profile.get(state, profile.get("calm", "conversational"))
    return f"Say this as {profile['base']} Right now they are {mood}."


def _build_agent_tone(safety_mode: str | None, red_flags: list) -> str:
    """Build per-message tone prompt for agent TTS."""
    if safety_mode == "static_safety" or red_flags:
        return "Speak as a healthcare assistant in a calm but urgent tone. This is a safety-critical moment — be steady, clear, and reassuring."
    if safety_mode == "safety_override":
        return "Speak as a healthcare assistant with a firm, professional tone. This requires clinical attention."
    return "Speak as a warm, professional healthcare assistant in a slightly upbeat and brisk manner."


# --- TTS functions (adapted from appoint-ready/gemini_tts.py) ---

def _convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Convert raw audio data to WAV format."""
    bits_per_sample = 16
    sample_rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip().lower()
        if param.startswith("rate="):
            try:
                sample_rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError):
                pass

    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ",
        16, 1, num_channels, sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", data_size
    )
    return header + audio_data


def generate_tts(text: str, voice: str) -> str | None:
    """
    Generate TTS audio using Gemini TTS API (google-genai SDK).
    Returns base64 data URL string or None on failure.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        logging.error(f"google-genai import failed: {e}. Run: pip install google-genai")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY not set")
        return None

    client = genai.Client(api_key=api_key)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=os.environ.get("TTS_MODEL", "gemini-2.5-flash-preview-tts"),
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )

            # Guard against empty responses (API returns 200 but no audio)
            try:
                part = response.candidates[0].content.parts[0]
                audio_data = part.inline_data.data
                mime_type = part.inline_data.mime_type
            except (IndexError, AttributeError, TypeError):
                if attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    logging.warning(f"Empty response from API, waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait)
                    continue
                logging.error("Empty response from API after all retries")
                return None

            if not audio_data:
                logging.warning("No audio data returned")
                return None

            # Convert to WAV
            mime_lower = (mime_type or "").lower()
            needs_wav = any(p in mime_lower for p in ("audio/l16", "audio/l24", "audio/l8"))
            if needs_wav or not mime_lower.startswith(("audio/wav", "audio/mpeg", "audio/ogg")):
                wav_data = _convert_to_wav(audio_data, mime_type or "audio/L16;rate=24000")
                final_mime = "audio/wav"
            else:
                wav_data = audio_data
                final_mime = mime_type

            # Try MP3 compression
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(io.BytesIO(wav_data), format="wav")
                mp3_buffer = io.BytesIO()
                audio_segment.export(mp3_buffer, format="mp3", bitrate="64k")
                mp3_bytes = mp3_buffer.getvalue()
                return f"data:audio/mpeg;base64,{base64.b64encode(mp3_bytes).decode('utf-8')}"
            except Exception as e:
                logging.warning(f"MP3 compression failed ({e}), using WAV")
                return f"data:{final_mime};base64,{base64.b64encode(wav_data).decode('utf-8')}"

        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "500" in err_str) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                logging.warning(f"Error ({err_str[:80]}...), waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
                continue
            logging.error(f"TTS generation failed: {e}")
            return None

    return None


def _save_json(path: Path, data: dict) -> None:
    """Write JSON atomically (write tmp then rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def _load_existing_audio(output_file: Path | None) -> dict:
    """Load existing output JSON and build a text→audio lookup.

    Returns a dict: { (key, msg_idx): {"text": str, "audio": str|None} }
    Key is "intake" for intake log entries or the integer day for symptom logs.
    msg_idx is the message index within that log entry's messages array.
    For intake, a separate running counter across all intake log entries is used.
    """
    if not output_file or not output_file.exists():
        return {}
    try:
        with open(output_file) as f:
            data = json.load(f)
        lookup = {}
        intake_msg_idx = 0
        for log_entry in data.get("logs", []):
            if log_entry.get("intake"):
                # Intake log entry — use running "intake" counter
                for msg in log_entry.get("messages", []):
                    lookup[("intake", intake_msg_idx)] = {
                        "text": msg.get("text", ""),
                        "audio": msg.get("audio"),
                    }
                    intake_msg_idx += 1
            else:
                # Symptom log entry — key by day
                day = log_entry["day"]
                for idx, msg in enumerate(log_entry.get("messages", [])):
                    lookup[(day, idx)] = {
                        "text": msg.get("text", ""),
                        "audio": msg.get("audio"),
                    }
        # Backward compat: also check old "intake" top-level key
        for idx, msg in enumerate(data.get("intake", [])):
            key = ("intake", idx)
            if key not in lookup:
                lookup[key] = {
                    "text": msg.get("text", ""),
                    "audio": msg.get("audio"),
                }
        return lookup
    except Exception as e:
        logging.warning(f"Could not load existing output for resume: {e}")
        return {}


def build_demo_data(patient_id: str, generate_audio: bool = True,
                    output_file: Path | None = None, resume: bool = False) -> dict:
    """Build the demo data JSON from saved responses and patient data.

    Args:
        resume: If True, load existing output and skip messages where the text
                matches and audio already exists.  Re-generates only null audio
                (failed clips) or changed text (updated followups).
    """
    base_dir = Path(__file__).parent

    # Load patient JSON for profile and phase info
    patient_file = base_dir / "patients" / f"{patient_id}.json"
    if not patient_file.exists():
        print(f"Patient file not found: {patient_file}")
        sys.exit(1)

    with open(patient_file) as f:
        patient_data = json.load(f)

    # Load saved responses
    responses_file = base_dir / "results" / f"{patient_id}_responses.json"
    if not responses_file.exists():
        print(f"Saved responses not found: {responses_file}")
        print(f"Run first: python simulate_patient.py {patient_id} --save-responses")
        sys.exit(1)

    with open(responses_file) as f:
        responses_data = json.load(f)

    profile = patient_data["profile"]
    patient_voice = PATIENT_VOICES.get(patient_id, DEFAULT_PATIENT_VOICE)

    # Build patient info
    demo_patient = {
        "name": profile["name"],
        "age": profile["age"],
        "gender": profile.get("gender", "Unknown"),
        "conditions": profile["conditions"],
        "summary": profile.get("health_summary", ""),
    }

    # Base date for calendar anchoring
    base_date = responses_data.get("base_date")

    # Doctor packet + watchdog — fetched early for incremental saves
    # Dedup: keep only the last (most comprehensive) longitudinal flag and
    # clinician observation — earlier ones are subsets of the final.
    doctor_packet = responses_data.get("doctor_packet")
    if doctor_packet and doctor_packet.get("system_longitudinal_flags"):
        doctor_packet = {
            **doctor_packet,
            "system_longitudinal_flags": doctor_packet["system_longitudinal_flags"][-1:],
        }
    watchdog_results = responses_data.get("watchdog_results")
    if watchdog_results and watchdog_results.get("clinician_observations"):
        watchdog_results = {
            **watchdog_results,
            "clinician_observations": watchdog_results["clinician_observations"][-1:],
        }

    # Build authoritative followup lookup from patient JSON (may be newer than results)
    followup_by_day = {}
    for log in patient_data.get("symptom_logs", []):
        if log.get("followup_answer"):
            followup_by_day[log["day"]] = log["followup_answer"]

    # Resume support: load existing audio to skip already-generated clips
    existing = _load_existing_audio(output_file) if resume else {}
    if existing:
        n_with_audio = sum(1 for v in existing.values() if v["audio"])
        print(f"  Resume: loaded {len(existing)} existing messages ({n_with_audio} with audio)")

    # Build intake log entries (one per intake Q&A, matching build_demo_json.py structure)
    intake_logs = []
    total_audio_clips = 0
    generated_clips = 0
    skipped_clips = 0
    reused_clips = 0
    # Global message index for intake resume cache (flat across all intake messages)
    intake_msg_idx = 0

    for intake_resp in responses_data.get("intake_responses", []):
        agent_q = intake_resp.get("agent_question", "")
        patient_a = intake_resp.get("patient_answer", "")
        qid = intake_resp.get("question_id", "")

        messages = []

        # Agent intake question
        if agent_q:
            total_audio_clips += 1
            agent_audio = None
            if generate_audio:
                cached = existing.get(("intake", intake_msg_idx))
                if cached and cached["text"] == agent_q and cached["audio"]:
                    agent_audio = cached["audio"]
                    reused_clips += 1
                    print(f"  [reuse] intake agent TTS ({qid})")
                else:
                    print(f"  Generating intake agent TTS ({qid})...")
                    tone = "Speak as a warm, professional healthcare assistant. This is a friendly intake question."
                    agent_audio = generate_tts(f"{tone}: {agent_q}", AGENT_VOICE)
                    if agent_audio:
                        generated_clips += 1
                    time.sleep(4)
            messages.append({
                "speaker": "agent",
                "text": agent_q,
                "audio": agent_audio,
            })
            intake_msg_idx += 1

        # Patient intake answer
        if patient_a:
            total_audio_clips += 1
            patient_audio = None
            if generate_audio:
                cached = existing.get(("intake", intake_msg_idx))
                if cached and cached["text"] == patient_a and cached["audio"]:
                    patient_audio = cached["audio"]
                    reused_clips += 1
                    print(f"  [reuse] intake patient TTS ({qid})")
                else:
                    print(f"  Generating intake patient TTS ({qid})...")
                    patient_profile = PATIENT_PROFILES.get(patient_id, DEFAULT_PROFILE)
                    tone = f"Say this as {patient_profile['base']} Right now they are {patient_profile.get('calm', 'conversational')}."
                    patient_audio = generate_tts(f"{tone}: {patient_a}", patient_voice)
                    if patient_audio:
                        generated_clips += 1
                    time.sleep(4)
            messages.append({
                "speaker": "patient",
                "text": patient_a,
                "audio": patient_audio,
            })
            intake_msg_idx += 1

        # Build the log entry matching frontend DemoLogEntry shape
        intake_logs.append({
            "day": 0,
            "time": "09:00",
            "phase": "Profile Intake",
            "messages": messages,
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
                "question_id": qid,
                "raw_answer": patient_a,
                "parsed_items": intake_resp.get("parsed_items", []),
                "profile_field": qid,
            },
        })

        # Incremental save after each intake pair
        if output_file and generate_audio:
            _save_json(output_file, {
                "patient": demo_patient,
                "base_date": base_date,
                "logs": intake_logs,
                "doctor_packet": doctor_packet,
                "watchdog_results": watchdog_results,
            })

    if intake_logs:
        print(f"  Intake: {len(intake_logs)} log entries processed")

    # Build log entries
    demo_logs = []

    # Map from patient JSON to get phase info
    phase_by_day = {}
    for log in patient_data.get("symptom_logs", []):
        if log.get("phase"):
            phase_by_day[log["day"]] = log["phase"]

    for log_resp in responses_data.get("log_responses", []):
        day = log_resp["day"]
        phase = log_resp.get("phase") or phase_by_day.get(day, "")

        # Find time from patient data
        time_str = "09:00"
        for orig_log in patient_data.get("symptom_logs", []):
            if orig_log["day"] == day:
                time_str = orig_log.get("time", "09:00")
                break

        messages = []
        msg_idx = 0  # track message index within this day for resume lookup
        actual = log_resp.get("actual_response", {})
        agent = actual.get("agent_response", {})
        extracted = actual.get("extracted", {})

        # Extract metadata early — needed for emotional tone in TTS
        symptoms = [s.get("symptom", "unknown") for s in extracted.get("symptoms", [])]
        actions = [a.get("name", "unknown") for a in extracted.get("actions_taken", [])]
        red_flags = extracted.get("red_flags", [])
        protocol = agent.get("protocol_id")
        clinician_note = actual.get("contact_clinician_note")
        safety_mode = agent.get("safety_mode")
        reason_code = agent.get("reason_code")
        # Red-flag entries must not show autonomous tool calls —
        # backend now enforces this (ingest.py), but older simulation
        # results predate the fix.
        tool_calls = [] if safety_mode == "static_safety" else agent.get("tool_calls", [])

        # Patient transcript message
        transcript = log_resp.get("transcript", "")
        if transcript:
            total_audio_clips += 1
            patient_audio = None
            if generate_audio:
                # Check resume cache
                cached = existing.get((day, msg_idx))
                if cached and cached["text"] == transcript and cached["audio"]:
                    patient_audio = cached["audio"]
                    reused_clips += 1
                    print(f"  [reuse] patient TTS Day {day}")
                else:
                    print(f"  Generating patient TTS for Day {day}...")
                    tone = _build_patient_tone(patient_id, phase, red_flags, transcript)
                    patient_audio = generate_tts(f"{tone}: {transcript}", patient_voice)
                    if patient_audio:
                        generated_clips += 1
                    time.sleep(4)  # Rate limit: stay under 15 RPM

            messages.append({
                "speaker": "patient",
                "text": transcript,
                "audio": patient_audio,
            })
            msg_idx += 1

        # Agent response message
        ack = agent.get("acknowledgment", "")
        question = agent.get("immediate_question")
        agent_text = ack
        if question and question not in ack:
            agent_text = f"{ack} {question}" if ack else question

        if agent_text:
            total_audio_clips += 1
            agent_audio = None
            if generate_audio:
                cached = existing.get((day, msg_idx))
                if cached and cached["text"] == agent_text and cached["audio"]:
                    agent_audio = cached["audio"]
                    reused_clips += 1
                    print(f"  [reuse] agent TTS Day {day}")
                else:
                    print(f"  Generating agent TTS for Day {day}...")
                    agent_tone = _build_agent_tone(safety_mode, red_flags)
                    agent_audio = generate_tts(f"{agent_tone}: {agent_text}", AGENT_VOICE)
                    if agent_audio:
                        generated_clips += 1
                    time.sleep(4)  # Rate limit: stay under 15 RPM

            messages.append({
                "speaker": "agent",
                "text": agent_text,
                "audio": agent_audio,
                "question": question,
            })
            msg_idx += 1

        # Patient followup answer — prefer patient JSON (authoritative) over results
        followup = followup_by_day.get(day) or log_resp.get("followup_answer")
        if followup:
            total_audio_clips += 1
            followup_audio = None
            if generate_audio:
                cached = existing.get((day, msg_idx))
                if cached and cached["text"] == followup and cached["audio"]:
                    followup_audio = cached["audio"]
                    reused_clips += 1
                    print(f"  [reuse] followup TTS Day {day}")
                else:
                    reason = "changed" if cached and cached["text"] != followup else "new"
                    print(f"  Generating followup TTS for Day {day} ({reason})...")
                    tone = _build_patient_tone(patient_id, phase, red_flags, followup)
                    followup_audio = generate_tts(f"{tone}: {followup}", patient_voice)
                    if followup_audio:
                        generated_clips += 1
                    time.sleep(4)  # Rate limit: stay under 15 RPM

            messages.append({
                "speaker": "patient",
                "text": followup,
                "audio": followup_audio,
                "isFollowup": True,
            })
            msg_idx += 1

        demo_logs.append({
            "day": day,
            "time": time_str,
            "phase": phase,
            "messages": messages,
            "metadata": {
                "symptoms": symptoms,
                "actions_taken": actions,
                "red_flags": red_flags,
                "protocol": protocol,
                "clinician_note": clinician_note,
                "safety_mode": safety_mode,
                "reason_code": reason_code,
                "tool_calls": tool_calls,
                "question_rationale": agent.get("question_rationale"),
            },
        })

        # Incremental save after each day — protects against crashes/rate limits
        if output_file and generate_audio:
            _save_json(output_file, {
                "patient": demo_patient,
                "base_date": base_date,
                "logs": intake_logs + demo_logs,
                "doctor_packet": doctor_packet,
                "watchdog_results": watchdog_results,
            })
            print(f"    Saved progress: {len(demo_logs)} days, {generated_clips} new + {reused_clips} reused / {total_audio_clips} total")

    print(f"\n  Audio: {generated_clips} generated, {reused_clips} reused, {total_audio_clips - generated_clips - reused_clips} missing")

    return {
        "patient": demo_patient,
        "base_date": base_date,
        "logs": intake_logs + demo_logs,
        "doctor_packet": doctor_packet,
        "watchdog_results": watchdog_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate demo data with TTS audio")
    parser.add_argument("patient", help="Patient file name (without .json)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Skip TTS generation (audio fields will be null)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip clips that already have audio and matching text. "
                             "Re-generates only failed (null) or changed clips.")
    args = parser.parse_args()

    print(f"\nGenerating demo data for: {args.patient}")
    print(f"Audio generation: {'disabled' if args.no_audio else 'enabled'}")
    if args.resume:
        print(f"Resume mode: will reuse existing audio where text matches")

    # Prepare output path early for incremental saving
    output_dir = Path(__file__).parent / ".." / ".." / "frontend" / "public" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.patient}.json"

    demo_data = build_demo_data(
        args.patient,
        generate_audio=not args.no_audio,
        output_file=output_file,
        resume=args.resume,
    )

    # Final write
    _save_json(output_file, demo_data)

    file_size = output_file.stat().st_size
    size_str = f"{file_size / 1024 / 1024:.1f} MB" if file_size > 1024 * 1024 else f"{file_size / 1024:.0f} KB"

    print(f"\nOutput: {output_file}")
    print(f"Size: {size_str}")
    print(f"Logs: {len(demo_data['logs'])}")
    print(f"Doctor packet: {'included' if demo_data['doctor_packet'] else 'not available'}")
    print(f"\nDone! Start the frontend with: cd frontend && npm run dev")
    print(f"Then visit: http://localhost:5173?demo=true")


if __name__ == "__main__":
    main()
