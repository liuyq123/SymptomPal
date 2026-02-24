"""
Ambient audio monitoring routes.
Handles session lifecycle and chunk uploads for HeAR-based analysis.
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends

from ..models import (
    AmbientSession,
    AmbientChunk,
    SessionStatus,
    StartSessionRequest,
    StartSessionResponse,
    UploadChunkRequest,
    UploadChunkResponse,
    EndSessionRequest,
    EndSessionResponse,
    AmbientSessionResult,
)
from ..services import storage
from ..services.storage import ActiveSessionExistsError
from ..services.audio_classifier import get_audio_classifier_client
from ..services.logging import log_request, log_response, log_error, log_warning
from ..services.auth import get_request_user_id, enforce_user_match

router = APIRouter(prefix="/api/ambient", tags=["ambient"])

UPLOAD_INTERVAL_SECONDS = 30
IDLE_TIMEOUT_SECONDS = UPLOAD_INTERVAL_SECONDS * 3


def _is_session_stale(session: AmbientSession, now: datetime) -> bool:
    last_activity = session.updated_at or session.started_at
    if last_activity.tzinfo is not None:
        last_activity = last_activity.replace(tzinfo=None)
    return (now - last_activity) > timedelta(seconds=IDLE_TIMEOUT_SECONDS)


@router.post("/sessions/start", response_model=StartSessionResponse)
async def start_session(
    request: StartSessionRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """Start a new ambient monitoring session."""
    enforce_user_match(request_user_id, request.user_id)
    log_request(
        "/ambient/sessions/start",
        request.user_id,
        session_type=request.session_type.value,
    )

    # Check if user already has an active session
    active = storage.get_active_session(request.user_id)
    if active:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if _is_session_stale(active, now):
            log_warning(
                f"Cancelling stale session {active.id}",
                user_id=request.user_id,
            )
            storage.update_ambient_session(
                active.id,
                {
                    "status": SessionStatus.CANCELLED.value,
                    "ended_at": now,
                },
            )
        else:
            log_error(
                "/ambient/sessions/start",
                f"Active session exists: {active.id}",
                request.user_id,
            )
            raise HTTPException(
                status_code=409,
                detail=f"User already has an active session: {active.id}"
            )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    session = AmbientSession(
        id=str(uuid.uuid4()),
        user_id=request.user_id,
        session_type=request.session_type,
        status=SessionStatus.ACTIVE,
        label=request.label,
        started_at=now,
        ended_at=None,
        chunk_count=0,
        total_duration_seconds=0.0,
        created_at=now,
        updated_at=now,
    )

    try:
        storage.create_ambient_session(session)
    except ActiveSessionExistsError:
        # Race condition: another session was created between our check and insert
        log_error(
            "/ambient/sessions/start",
            "Race condition: active session created by concurrent request",
            request.user_id,
        )
        raise HTTPException(
            status_code=409,
            detail="User already has an active session (concurrent request)"
        )

    log_response(
        "/ambient/sessions/start",
        request.user_id,
        session_id=session.id,
    )
    return StartSessionResponse(
        session=session,
        upload_interval_seconds=UPLOAD_INTERVAL_SECONDS,
    )


@router.post("/sessions/upload", response_model=UploadChunkResponse)
async def upload_chunk(
    request: UploadChunkRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """Upload an audio chunk and get detected events."""
    enforce_user_match(request_user_id, request.user_id)
    log_request(
        "/ambient/sessions/upload",
        request.user_id,
        session_id=request.session_id,
        chunk_index=request.chunk_index,
        duration_seconds=request.duration_seconds,
    )

    # Verify session exists and is active
    session = storage.get_ambient_session(request.session_id)
    if session is None:
        log_error("/ambient/sessions/upload", "Session not found", request.user_id, session_id=request.session_id)
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status != SessionStatus.ACTIVE:
        log_error("/ambient/sessions/upload", f"Session not active: {session.status}", request.user_id, session_id=request.session_id)
        raise HTTPException(status_code=400, detail=f"Session is not active (status: {session.status})")

    if session.user_id != request.user_id:
        log_error("/ambient/sessions/upload", "User does not own session", request.user_id, session_id=request.session_id)
        raise HTTPException(status_code=403, detail="User does not own this session")

    # Idempotency: if this chunk index was already processed, return stored events.
    existing_chunk = storage.get_ambient_chunk_by_index(request.session_id, request.chunk_index)
    if existing_chunk:
        existing_events = storage.list_session_events_for_chunk(request.session_id, request.chunk_index)
        log_response(
            "/ambient/sessions/upload",
            request.user_id,
            chunk_id=existing_chunk.id,
            events_count=len(existing_events),
            deduped=True,
        )
        return UploadChunkResponse(
            chunk_id=existing_chunk.id,
            events_detected=existing_events,
        )

    # Analyze the audio chunk
    classifier = get_audio_classifier_client()
    chunk_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    try:
        # Use async version if available (runs TF in thread pool to avoid asyncio issues)
        if hasattr(classifier, 'analyze_chunk_async'):
            events = await classifier.analyze_chunk_async(
                audio_b64=request.audio_b64,
                session_type=session.session_type,
                session_id=request.session_id,
                user_id=request.user_id,
                chunk_index=request.chunk_index,
                chunk_timestamp=chunk_timestamp,
            )
        else:
            events = classifier.analyze_chunk(
                audio_b64=request.audio_b64,
                session_type=session.session_type,
                session_id=request.session_id,
                user_id=request.user_id,
                chunk_index=request.chunk_index,
                chunk_timestamp=chunk_timestamp,
            )
    except Exception as e:
        import traceback
        log_error("/ambient/sessions/upload", f"HeAR analysis failed: {e}\n{traceback.format_exc()}", request.user_id)
        raise HTTPException(status_code=500, detail="Audio analysis failed")

    # Store the chunk metadata
    chunk = AmbientChunk(
        id=str(uuid.uuid4()),
        session_id=request.session_id,
        user_id=request.user_id,
        chunk_index=request.chunk_index,
        duration_seconds=request.duration_seconds,
        uploaded_at=chunk_timestamp,
        processed=True,
        events_detected=len(events),
    )
    inserted = storage.create_ambient_chunk(chunk)
    if not inserted:
        # Concurrent duplicate upload (same chunk index): return existing state.
        existing_chunk = storage.get_ambient_chunk_by_index(request.session_id, request.chunk_index)
        existing_events = storage.list_session_events_for_chunk(request.session_id, request.chunk_index)
        log_response(
            "/ambient/sessions/upload",
            request.user_id,
            chunk_id=existing_chunk.id if existing_chunk else chunk.id,
            events_count=len(existing_events),
            deduped=True,
        )
        return UploadChunkResponse(
            chunk_id=existing_chunk.id if existing_chunk else chunk.id,
            events_detected=existing_events,
        )

    # Store detected events (batch insert for efficiency)
    storage.create_ambient_events_batch(events)

    # Update session stats
    storage.increment_ambient_session_stats(
        request.session_id,
        request.duration_seconds,
    )

    log_response(
        "/ambient/sessions/upload",
        request.user_id,
        chunk_id=chunk.id,
        events_count=len(events),
    )
    return UploadChunkResponse(
        chunk_id=chunk.id,
        events_detected=events,
    )


@router.post("/sessions/end", response_model=EndSessionResponse)
async def end_session(
    request: EndSessionRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """End a session and compute final metrics."""
    enforce_user_match(request_user_id, request.user_id)
    log_request("/ambient/sessions/end", request.user_id, session_id=request.session_id)

    session = storage.get_ambient_session(request.session_id)
    if session is None:
        log_error("/ambient/sessions/end", "Session not found", request.user_id, session_id=request.session_id)
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != request.user_id:
        log_error("/ambient/sessions/end", "User does not own session", request.user_id, session_id=request.session_id)
        raise HTTPException(status_code=403, detail="User does not own this session")

    if session.status != SessionStatus.ACTIVE:
        log_error("/ambient/sessions/end", f"Session not active: {session.status}", request.user_id, session_id=request.session_id)
        raise HTTPException(status_code=400, detail=f"Session is not active (status: {session.status})")

    # Mark as processing
    storage.update_ambient_session(request.session_id, {"status": SessionStatus.PROCESSING.value})

    # Get all events for the session
    events = storage.list_session_events(request.session_id)

    # Compute metrics
    classifier = get_audio_classifier_client()
    result = classifier.compute_session_metrics(
        session_type=session.session_type,
        events=events,
        total_duration_seconds=session.total_duration_seconds,
    )
    result.session_id = request.session_id

    # Mark as completed and save result
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    updated_session = storage.update_ambient_session(
        request.session_id,
        {
            "status": SessionStatus.COMPLETED.value,
            "ended_at": now.isoformat(),
            "result_json": result.model_dump_json(),
        }
    )

    log_response(
        "/ambient/sessions/end",
        request.user_id,
        session_id=request.session_id,
        duration_minutes=result.duration_minutes,
        events_count=len(events),
    )
    return EndSessionResponse(
        session=updated_session,
        result=result,
    )


@router.get("/sessions", response_model=List[AmbientSession])
async def list_sessions(
    user_id: str = Query(...),
    limit: int = Query(20, le=100),
    request_user_id: str = Depends(get_request_user_id),
):
    """List ambient sessions for a user."""
    enforce_user_match(request_user_id, user_id)
    return storage.list_ambient_sessions(user_id, limit=limit)


@router.get("/sessions/active", response_model=Optional[AmbientSession])
async def get_active_session(
    user_id: str = Query(...),
    request_user_id: str = Depends(get_request_user_id),
):
    """Get the user's active session, if any."""
    enforce_user_match(request_user_id, user_id)
    return storage.get_active_session(user_id)


@router.get("/sessions/{session_id}/result", response_model=Optional[AmbientSessionResult])
async def get_session_result(
    session_id: str,
    user_id: str = Query(...),
    request_user_id: str = Depends(get_request_user_id),
):
    """Get the result for a completed session."""
    enforce_user_match(request_user_id, user_id)

    # Verify session exists and belongs to user
    session = storage.get_ambient_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="User does not own this session")

    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Session is not completed (status: {session.status})")

    # Get stored result
    result_json = storage.get_session_result_json(session_id)
    if result_json is None:
        # No stored result - try to recompute from events
        events = storage.list_session_events(session_id)
        if not events:
            raise HTTPException(status_code=404, detail="No result available for this session")

        # Recompute metrics
        classifier = get_audio_classifier_client()
        result = classifier.compute_session_metrics(
            session_type=session.session_type,
            events=events,
            total_duration_seconds=session.total_duration_seconds,
        )
        result.session_id = session_id

        # Save for future requests
        storage.update_ambient_session(session_id, {"result_json": result.model_dump_json()})
        return result

    import json
    return AmbientSessionResult.model_validate(json.loads(result_json))


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(
    session_id: str,
    user_id: str = Query(...),
    request_user_id: str = Depends(get_request_user_id),
):
    """Cancel a session without computing results."""
    enforce_user_match(request_user_id, user_id)
    session = storage.get_ambient_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="User does not own this session")

    if session.status not in [SessionStatus.ACTIVE, SessionStatus.PROCESSING]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel session with status: {session.status}")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    storage.update_ambient_session(
        session_id,
        {
            "status": SessionStatus.CANCELLED.value,
            "ended_at": now.isoformat(),
        }
    )

    return {"status": "cancelled", "session_id": session_id}
