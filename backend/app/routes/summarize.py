from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    SummarizeRequest,
    DoctorPacket,
    TimelineReveal,
)
from ..services.storage import list_logs_in_days, get_or_create_user_profile
from ..services.medgemma_client import get_medgemma_client
from ..services.auth import get_request_user_id, enforce_user_match
from ..services.logging import log_warning

router = APIRouter(prefix="/api/summarize", tags=["summarize"])


@router.post("/doctor-packet", response_model=DoctorPacket)
async def generate_doctor_packet(
    request: SummarizeRequest,
    request_user_id: str = Depends(get_request_user_id),
) -> DoctorPacket:
    """
    Generate clinician-facing packet (HPI + bullets).

    The packet includes:
    - HPI (History of Present Illness) - 3 sentences max
    - Pertinent positives
    - Pertinent negatives
    - Timeline bullets (chronological)
    - Questions for the clinician
    """
    enforce_user_match(request_user_id, request.user_id)
    # Cap at 90 days to bound map-reduce cost. In production, logs older than
    # 90 days would be permanently compressed into a rolling baseline summary
    # (patient_baseline_summary) that the map-reduce incorporates as its first
    # chunk — giving the packet full longitudinal context without unbounded
    # token cost. For now, 90 days captures 3 months of raw clinical detail,
    # which is sufficient for most specialist referral packets.
    max_days = min(request.days, 90) if request.days > 0 else 90
    logs = list_logs_in_days(request.user_id, max_days)
    gemma_client = get_medgemma_client()
    try:
        user_profile = get_or_create_user_profile(request.user_id)
        return await gemma_client.doctor_packet(logs, request.days, user_id=request.user_id, user_profile=user_profile)
    except Exception as exc:
        log_warning("doctor_packet_failed", user_id=request.user_id, reason=str(exc))
        raise HTTPException(status_code=503, detail=f"MedGemma model unavailable: {exc}")


@router.post("/timeline", response_model=TimelineReveal)
async def generate_timeline(
    request: SummarizeRequest,
    request_user_id: str = Depends(get_request_user_id),
) -> TimelineReveal:
    """
    Generate Timeline Reveal story points.

    The timeline shows:
    - Onset → Progression → Trigger → Current status
    - Each point has timestamp, label, and details
    """
    enforce_user_match(request_user_id, request.user_id)
    logs = list_logs_in_days(request.user_id, request.days)
    gemma_client = get_medgemma_client()
    try:
        return await gemma_client.timeline(logs, request.days)
    except Exception as exc:
        log_warning("timeline_failed", user_id=request.user_id, reason=str(exc))
        raise HTTPException(status_code=503, detail=f"MedGemma model unavailable: {exc}")
