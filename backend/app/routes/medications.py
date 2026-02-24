import re
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends

from ..models import (
    MedicationEntry,
    MedicationLogEntry,
    MedicationCreateRequest,
    MedicationLogRequest,
    MedicationVoiceRequest,
    MedicationVoiceResponse,
    PendingMedicationReminder,
    ReminderActionRequest,
    MedicationUpdateRequest,
)
from ..services import storage
from ..services.auth import get_request_user_id, enforce_user_match

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/medications", tags=["medications"])


@router.post("", response_model=MedicationEntry)
async def create_medication(
    request: MedicationCreateRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """Create a new saved medication."""
    enforce_user_match(request_user_id, request.user_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    entry = MedicationEntry(
        id=str(uuid.uuid4()),
        user_id=request.user_id,
        name=request.name,
        dose=request.dose,
        frequency=request.frequency,
        reason=request.reason,
        start_date=request.start_date,
        end_date=None,
        notes=request.notes,
        is_active=True,
        reminder_enabled=request.reminder_enabled,
        reminder_times=request.reminder_times,
        created_at=now,
        updated_at=now,
    )
    storage.create_medication(entry)
    return entry


# === Voice-based medication extraction ===

_FREQUENCY_PATTERNS = [
    (r'\b(once\s+(?:a\s+)?dai?ly|once\s+a\s+day|qd)\b', 'once daily'),
    (r'\b(twice\s+(?:a\s+)?dai?ly|twice\s+a\s+day|bid|two\s+times\s+a\s+day)\b', 'twice daily'),
    (r'\b(three\s+times\s+(?:a\s+)?day|tid)\b', 'three times daily'),
    (r'\b(four\s+times\s+(?:a\s+)?day|qid)\b', 'four times daily'),
    (r'\b(every\s+\d+\s+hours?)\b', None),
    (r'\b(as\s+needed|prn)\b', 'as needed'),
    (r'\b(at\s+(?:bed\s*time|night))\b', 'at bedtime'),
    (r'\b(every\s+(?:morning|evening|night))\b', None),
    (r'\b(weekly|once\s+a\s+week)\b', 'weekly'),
]

_REASON_STOP_WORDS = {'me', 'you', 'that', 'this', 'it', 'a while', 'now', 'today', 'example'}


def _extract_frequency(transcript: str) -> Optional[str]:
    lower = transcript.lower()
    for pattern, canonical in _FREQUENCY_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            return canonical if canonical else m.group(1)
    return None


def _extract_reason(transcript: str) -> Optional[str]:
    lower = transcript.lower()
    m = re.search(r'\bfor\s+([\w\s]{2,30}?)(?:\.|,|$)', lower)
    if m:
        reason = m.group(1).strip()
        if reason not in _REASON_STOP_WORDS and len(reason) > 2:
            return reason
    return None


@router.post("/voice-extract", response_model=MedicationVoiceResponse)
async def extract_medication_from_voice(
    request: MedicationVoiceRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """
    Transcribe audio and extract medication fields (name, dose, frequency, reason).
    Does NOT create any log entry or medication. Returns structured data for form pre-fill.
    """
    enforce_user_match(request_user_id, request.user_id)

    from ..services.medasr_client import get_medasr_client
    from ..services.medgemma_client import get_medgemma_client
    from ..models import ExtractionResult

    # Step 1: Transcribe
    asr_client = get_medasr_client()
    transcript = await asr_client.transcribe(request.audio_b64)
    if not transcript or not transcript.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe audio. Please try again.")

    # Step 2: Extract medications via MedGemma
    gemma_client = get_medgemma_client()
    extraction_failed = False
    try:
        extraction = await gemma_client.extract(transcript)
    except Exception as e:
        logger.warning("MedGemma extraction failed — returning empty extraction: %s", e)
        extraction = ExtractionResult(transcript=transcript)
        extraction_failed = True

    medications = extraction.actions_taken

    # Step 3: Extract frequency and reason from transcript
    frequency = _extract_frequency(transcript)
    reason = _extract_reason(transcript)

    return MedicationVoiceResponse(
        transcript=transcript,
        medications=medications,
        frequency=frequency,
        reason=reason,
        extraction_failed=extraction_failed,
    )


@router.get("", response_model=List[MedicationEntry])
async def list_medications(
    user_id: str = Query(...),
    active_only: bool = Query(True),
    request_user_id: str = Depends(get_request_user_id),
):
    """List medications for a user."""
    enforce_user_match(request_user_id, user_id)
    return storage.list_medications(user_id, active_only=active_only)


@router.get("/{med_id}", response_model=MedicationEntry)
async def get_medication(
    med_id: str,
    request_user_id: str = Depends(get_request_user_id),
):
    """Get a medication by ID."""
    med = storage.get_medication(med_id)
    if med is None:
        raise HTTPException(status_code=404, detail="Medication not found")
    enforce_user_match(request_user_id, med.user_id)
    return med


@router.patch("/{med_id}", response_model=MedicationEntry)
async def update_medication(
    med_id: str,
    updates: MedicationUpdateRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """Update a medication."""
    existing = storage.get_medication(med_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Medication not found")
    enforce_user_match(request_user_id, existing.user_id)
    med = storage.update_medication(med_id, updates.model_dump(exclude_unset=True))
    return med


@router.delete("/{med_id}")
async def deactivate_medication(
    med_id: str,
    request_user_id: str = Depends(get_request_user_id),
):
    """Deactivate (soft delete) a medication."""
    existing = storage.get_medication(med_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Medication not found")
    enforce_user_match(request_user_id, existing.user_id)
    storage.update_medication(med_id, {"is_active": False})
    return {"status": "deactivated", "id": med_id}


# === Medication Logs (dose tracking) ===

@router.post("/log", response_model=MedicationLogEntry)
async def log_medication(
    request: MedicationLogRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """Log taking a medication dose."""
    enforce_user_match(request_user_id, request.user_id)
    # Resolve medication name
    med_name = request.medication_name
    if request.medication_id:
        med = storage.get_medication(request.medication_id)
        if med:
            med_name = med.name
        elif not med_name:
            raise HTTPException(status_code=404, detail="Medication not found")

    if not med_name:
        raise HTTPException(status_code=400, detail="Either medication_id or medication_name required")

    entry = MedicationLogEntry(
        id=str(uuid.uuid4()),
        user_id=request.user_id,
        medication_id=request.medication_id,
        medication_name=med_name,
        dose_taken=request.dose_taken,
        taken_at=request.taken_at or datetime.now(timezone.utc).replace(tzinfo=None),
        notes=request.notes,
        symptom_log_id=request.symptom_log_id,
    )
    storage.create_medication_log(entry)
    return entry


@router.get("/log/history", response_model=List[MedicationLogEntry])
async def list_medication_logs(
    user_id: str = Query(...),
    limit: int = Query(50, le=200),
    days: Optional[int] = Query(None),
    request_user_id: str = Depends(get_request_user_id),
):
    """List medication log history for a user."""
    enforce_user_match(request_user_id, user_id)
    return storage.list_medication_logs(user_id, limit=limit, days=days)


# === Medication Reminders ===

@router.get("/reminders/pending", response_model=List[PendingMedicationReminder])
async def get_pending_reminders(
    user_id: str = Query(..., description="User ID to get reminders for"),
    request_user_id: str = Depends(get_request_user_id),
):
    """
    Get pending medication reminders for a user.

    Returns medications with reminders enabled that are due now.
    Frontend should poll this endpoint regularly (e.g., every 5 minutes).
    """
    enforce_user_match(request_user_id, user_id)
    reminders = storage.get_pending_medication_reminders(user_id)
    return reminders


@router.post("/reminders/take")
async def take_medication_from_reminder(
    request: ReminderActionRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """
    Mark a medication reminder as taken.

    Automatically creates a medication log entry and records the action.
    """
    # Verify medication exists
    enforce_user_match(request_user_id, request.user_id)
    med = storage.get_medication(request.medication_id)
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found")
    enforce_user_match(request_user_id, med.user_id)

    # Log the medication as taken
    log_entry = MedicationLogEntry(
        id=str(uuid.uuid4()),
        user_id=request.user_id,
        medication_id=request.medication_id,
        medication_name=med.name,
        dose_taken=med.dose,
        taken_at=datetime.now(timezone.utc).replace(tzinfo=None),
        notes=f"Taken from reminder (scheduled: {request.due_at.strftime('%H:%M')})",
        symptom_log_id=None,
    )
    storage.create_medication_log(log_entry)

    # Record reminder action
    storage.record_reminder_action(
        user_id=request.user_id,
        medication_id=request.medication_id,
        due_at=request.due_at,
        action="taken"
    )

    return {
        "status": "taken",
        "medication_log_id": log_entry.id,
        "message": f"{med.name} logged successfully"
    }


@router.post("/reminders/dismiss")
async def dismiss_medication_reminder(
    request: ReminderActionRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """
    Dismiss a medication reminder (skip this dose).

    Records that the user chose not to take this dose.
    """
    # Verify medication exists
    enforce_user_match(request_user_id, request.user_id)
    med = storage.get_medication(request.medication_id)
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found")
    enforce_user_match(request_user_id, med.user_id)

    # Record reminder action
    storage.record_reminder_action(
        user_id=request.user_id,
        medication_id=request.medication_id,
        due_at=request.due_at,
        action="dismissed"
    )

    return {
        "status": "dismissed",
        "message": f"{med.name} reminder dismissed"
    }


@router.post("/reminders/snooze")
async def snooze_medication_reminder(
    request: ReminderActionRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """
    Snooze a medication reminder.

    Reminds again after the specified number of minutes (5-240).
    """
    if not request.snooze_minutes:
        raise HTTPException(status_code=400, detail="snooze_minutes is required")

    # Verify medication exists
    enforce_user_match(request_user_id, request.user_id)
    med = storage.get_medication(request.medication_id)
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found")
    enforce_user_match(request_user_id, med.user_id)

    # Calculate snooze until time
    from datetime import timedelta
    snoozed_until = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=request.snooze_minutes)

    # Record reminder action
    storage.record_reminder_action(
        user_id=request.user_id,
        medication_id=request.medication_id,
        due_at=request.due_at,
        action="snoozed",
        snoozed_until=snoozed_until
    )

    return {
        "status": "snoozed",
        "snoozed_until": snoozed_until,
        "message": f"{med.name} reminder snoozed for {request.snooze_minutes} minutes"
    }
