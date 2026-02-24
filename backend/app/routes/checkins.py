"""Routes for proactive check-in system."""

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Query, Depends

from ..models import ScheduledCheckin, CheckinRespondRequest, CheckinType
from ..services.storage import (
    get_pending_checkins,
    get_scheduled_checkin,
    mark_checkin_triggered,
    respond_to_checkin,
    dismiss_checkin,
    create_scheduled_checkin,
    get_or_create_user_profile,
    update_user_profile,
)
from ..services.medasr_client import get_medasr_client
from ..services.profile_intake import (
    build_intake_profile_patch,
    create_intake_checkin,
    get_next_intake_question,
    parse_answer_and_generate_next_question,
    _QUESTION_FLOW,
)
from ..services.logging import log_request, log_response, log_error
from ..services.auth import get_request_user_id, enforce_user_match

router = APIRouter(prefix="/api/checkins", tags=["checkins"])


@router.get("/pending", response_model=List[ScheduledCheckin])
async def get_due_checkins(
    user_id: str = Query(..., description="User ID"),
    request_user_id: str = Depends(get_request_user_id),
) -> List[ScheduledCheckin]:
    """
    Get check-ins that are due now.

    Frontend should poll this periodically (e.g., every 30 seconds)
    and display any returned check-ins.
    """
    enforce_user_match(request_user_id, user_id)
    log_request("/checkins/pending", user_id)
    checkins = get_pending_checkins(user_id, as_of=datetime.now(timezone.utc).replace(tzinfo=None))
    log_response("/checkins/pending", user_id, count=len(checkins))
    return checkins


@router.post("/{checkin_id}/respond", response_model=ScheduledCheckin)
async def submit_checkin_response(
    checkin_id: str,
    request: CheckinRespondRequest,
    request_user_id: str = Depends(get_request_user_id),
) -> ScheduledCheckin:
    """Record user's response to a check-in."""
    checkin = get_scheduled_checkin(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")
    enforce_user_match(request_user_id, checkin.user_id)

    response_text = (request.response or "").strip()
    if not response_text and request.response_audio_b64:
        try:
            asr_client = get_medasr_client()
            response_text = (await asr_client.transcribe(request.response_audio_b64)).strip()
        except Exception as e:
            log_error("/checkins/respond", f"Failed to transcribe check-in response: {e}", checkin.user_id)
            raise HTTPException(status_code=500, detail="Failed to transcribe check-in response")

    if not response_text:
        raise HTTPException(status_code=400, detail="Check-in response is empty")

    log_request("/checkins/respond", checkin.user_id, checkin_id=checkin_id)
    result = respond_to_checkin(checkin_id, response_text)
    if not result:
        log_error("/checkins/respond", "Failed to update check-in", checkin.user_id)
        raise HTTPException(status_code=500, detail="Failed to update check-in")

    # Advance profile-intake flow after each intake response.
    if checkin.checkin_type == CheckinType.PROFILE_INTAKE:
        try:
            profile = get_or_create_user_profile(checkin.user_id)
            question_id = (checkin.context or {}).get("question_id") if checkin.context else None
            if not question_id:
                question_id = profile.intake_last_question_id or "health_summary"

            # Pre-compute the next question_id so LLM can generate both
            # the parsed answer and contextual next question in one call.
            temp_answered = set(profile.intake_answered_question_ids or [])
            temp_answered.add(question_id)
            peek_next_id = None
            for qid, _ in _QUESTION_FLOW:
                if qid not in temp_answered:
                    peek_next_id = qid
                    break

            # LLM-powered parsing + contextual question generation
            parsed_items = None
            custom_next_text = None
            try:
                from ..services.medgemma import get_medgemma_client
                client = get_medgemma_client()
                parsed_items, custom_next_text = await parse_answer_and_generate_next_question(
                    client, profile, question_id, checkin.message,
                    response_text, peek_next_id,
                )
            except Exception as e:
                log_error("/checkins/respond", f"LLM intake enhancement failed: {e}", checkin.user_id)

            patch = build_intake_profile_patch(profile, question_id, response_text, parsed_items=parsed_items)
            # Merge any pending raw responses into profile
            pending_add = patch.pop("intake_pending_raw_add", None)
            if pending_add:
                existing_pending = dict(profile.intake_pending_raw or {})
                existing_pending.update(pending_add)
                patch["intake_pending_raw"] = existing_pending
            updated_profile = update_user_profile(checkin.user_id, **patch)

            # Drain pending raw intake answers now that MedGemma is available
            if parsed_items is not None and updated_profile.intake_pending_raw:
                try:
                    from ..services.profile_intake import drain_pending_intake_raw
                    drain_patch = await drain_pending_intake_raw(client, updated_profile)
                    if drain_patch:
                        updated_profile = update_user_profile(checkin.user_id, **drain_patch)
                except Exception as e:
                    log_error("/checkins/respond", f"Drain pending intake failed: {e}", checkin.user_id)

            next_question = get_next_intake_question(updated_profile)
            if next_question:
                next_question_id, default_next_message = next_question
                # Only use LLM's custom text if the actual next question matches what was peeked
                next_message = custom_next_text if (custom_next_text and next_question_id == peek_next_id) else default_next_message
                create_scheduled_checkin(
                    create_intake_checkin(checkin.user_id, next_question_id, next_message)
                )
        except Exception as e:
            log_error("/checkins/respond", f"Profile intake continuation failed: {e}", checkin.user_id)

    log_response("/checkins/respond", checkin.user_id, checkin_id=checkin_id)
    return result


@router.post("/{checkin_id}/dismiss")
async def dismiss_checkin_endpoint(
    checkin_id: str,
    request_user_id: str = Depends(get_request_user_id),
) -> dict:
    """User dismissed check-in without responding."""
    checkin = get_scheduled_checkin(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")
    enforce_user_match(request_user_id, checkin.user_id)

    log_request("/checkins/dismiss", checkin.user_id, checkin_id=checkin_id)
    dismiss_checkin(checkin_id)
    log_response("/checkins/dismiss", checkin.user_id, checkin_id=checkin_id)
    return {"status": "dismissed", "checkin_id": checkin_id}


@router.post("/{checkin_id}/trigger")
async def mark_as_triggered(
    checkin_id: str,
    request_user_id: str = Depends(get_request_user_id),
) -> dict:
    """Mark check-in as shown to user (prevents re-showing)."""
    checkin = get_scheduled_checkin(checkin_id)
    if not checkin:
        raise HTTPException(status_code=404, detail="Check-in not found")
    enforce_user_match(request_user_id, checkin.user_id)

    mark_checkin_triggered(checkin_id)
    return {"status": "triggered", "checkin_id": checkin_id}
