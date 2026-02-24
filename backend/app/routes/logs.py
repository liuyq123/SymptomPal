import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends

from ..models import LogEntry, FollowupExchange, FollowupRequest
from ..services.storage import get_log, list_logs, list_logs_in_date_range, update_followup_exchanges, update_extraction, delete_log, permanent_delete_log, get_or_create_user_profile
from ..services.medgemma_client import get_medgemma_client
from ..services.clinician_alerts import get_red_flag_note
from ..services.response_generator import clean_patient_text
from ..services.auth import get_request_user_id, enforce_user_match

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/logs", tags=["logs"])

UNSOLICITED_FOLLOWUP_QUESTION = (
    "Is there anything else you'd like to share about how you're feeling?"
)


@router.get("", response_model=list[LogEntry])
async def get_logs(
    user_id: str = Query(..., description="User ID to fetch logs for"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of logs to return"),
    start_date: Optional[str] = Query(None, description="ISO 8601 start date (inclusive)"),
    end_date: Optional[str] = Query(None, description="ISO 8601 end date (exclusive)"),
    request_user_id: str = Depends(get_request_user_id),
) -> list[LogEntry]:
    """Return recent logs for a user, optionally filtered by date range."""
    enforce_user_match(request_user_id, user_id)
    if start_date and end_date:
        return list_logs_in_date_range(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    return list_logs(user_id=user_id, limit=limit)


@router.get("/{log_id}", response_model=LogEntry)
async def get_single_log(
    log_id: str,
    request_user_id: str = Depends(get_request_user_id),
) -> LogEntry:
    """Get a single log by ID."""
    log = get_log(log_id)
    if log is None:
        raise HTTPException(status_code=404, detail="Log not found")
    enforce_user_match(request_user_id, log.user_id)
    return log


@router.post("/{log_id}/followup", response_model=LogEntry)
async def submit_followup(
    log_id: str,
    request: FollowupRequest,
    request_user_id: str = Depends(get_request_user_id),
) -> LogEntry:
    """
    Answer the current pending follow-up question.

    Supports multi-turn: after answering, if fields are still missing
    a new follow-up question is automatically appended.
    """
    existing_log = get_log(log_id)
    if existing_log is None:
        raise HTTPException(status_code=404, detail="Log not found")
    enforce_user_match(request_user_id, existing_log.user_id)

    # Find the first unanswered exchange
    exchanges = [FollowupExchange(**e.model_dump()) for e in existing_log.followup_exchanges]
    unanswered = next((e for e in exchanges if e.answer is None), None)

    # No pending question — create a synthetic exchange for unsolicited patient input
    if not unanswered:
        unanswered = FollowupExchange(question=UNSOLICITED_FOLLOWUP_QUESTION)
        exchanges.append(unanswered)

    # Store the answer
    unanswered.answer = request.answer
    unanswered.answered_at = datetime.now(timezone.utc)

    # Re-extract from combined transcript (all exchanges) to fill missing fields
    # Then generate an agent response to the followup answer
    try:
        combined = existing_log.transcript
        for ex in exchanges:
            if ex.answer:
                combined += f". When asked '{ex.question}', patient said: '{ex.answer}'"

        gemma_client = get_medgemma_client()
        new_extraction = await gemma_client.extract(combined)
        update_extraction(log_id, new_extraction)

        # Red-flag guard: if the original log had red flags, return a brief
        # static acknowledgment instead of generating an LLM response.
        if existing_log.extracted.red_flags:
            unanswered.agent_response = get_red_flag_note(existing_log.extracted.red_flags)
        else:
            # Generate agent acknowledgment of the followup answer
            profile = get_or_create_user_profile(existing_log.user_id)
            followup_ack = await gemma_client.respond_to_followup(
                original_transcript=existing_log.transcript,
                followup_question=unanswered.question,
                followup_answer=request.answer,
                patient_name=getattr(profile, 'name', None),
                user_profile=profile,
            )
            unanswered.agent_response = clean_patient_text(followup_ack)

        # Check if more fields are still missing — generate next question if so
        from ..services.followup import choose_followup
        next_q = choose_followup(new_extraction)
        if next_q:
            exchanges.append(FollowupExchange(question=next_q))
    except Exception as e:
        logger.warning("Followup re-extraction/response failed for log %s: %s", log_id, e)

    update_followup_exchanges(log_id, exchanges)
    return get_log(log_id)


@router.delete("/{log_id}")
async def delete_symptom_log(
    log_id: str,
    permanent: bool = Query(False, description="If true, permanently delete. If false (default), soft delete."),
    request_user_id: str = Depends(get_request_user_id),
) -> dict:
    """
    Delete a symptom log entry.

    By default, performs a soft delete (marks as deleted but keeps in database).
    Use ?permanent=true to permanently remove from database.

    Soft delete is recommended - allows recovery if deleted by mistake.
    """
    log = get_log(log_id)
    if log is None:
        raise HTTPException(status_code=404, detail="Log not found")
    enforce_user_match(request_user_id, log.user_id)

    if permanent:
        success = permanent_delete_log(log_id)
        if not success:
            raise HTTPException(status_code=404, detail="Log not found")
        return {"status": "permanently_deleted", "log_id": log_id}
    else:
        success = delete_log(log_id)
        if not success:
            raise HTTPException(status_code=404, detail="Log not found")
        return {"status": "deleted", "log_id": log_id}
