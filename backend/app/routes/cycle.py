"""Menstrual cycle tracking routes."""

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Depends, Query

from ..models import (
    CycleDayLog, CycleDayLogRequest, PeriodStartRequest,
    CycleInfo, CyclePatternReport, FlowLevel,
)
from ..services.storage import (
    upsert_cycle_day_log, list_cycle_day_logs,
    get_cycle_day_log, delete_cycle_day_log, list_all_logs,
)
from ..services.cycle_engine import compute_cycles, detect_correlations
from ..services.auth import get_request_user_id, enforce_user_match
from ..services.logging import log_request, log_response

router = APIRouter(prefix="/api/cycle", tags=["cycle"])


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


@router.post("/day", response_model=CycleDayLog)
async def log_cycle_day(
    request: CycleDayLogRequest,
    request_user_id: str = Depends(get_request_user_id),
) -> CycleDayLog:
    """Log or update a single cycle day (flow level). Upserts by user+date."""
    enforce_user_match(request_user_id, request.user_id)
    log_request("/cycle/day", request.user_id)
    now = _utcnow()
    entry = CycleDayLog(
        id=f"cycle_{uuid.uuid4().hex[:12]}",
        user_id=request.user_id,
        date=request.date,
        flow_level=request.flow_level,
        is_period_day=request.flow_level != FlowLevel.NONE,
        notes=request.notes,
        created_at=now,
        updated_at=now,
    )
    upsert_cycle_day_log(entry)
    log_response("/cycle/day", request.user_id)
    return entry


@router.post("/period-start")
async def mark_period_start(
    request: PeriodStartRequest,
    request_user_id: str = Depends(get_request_user_id),
):
    """Quick action: mark period start with medium flow."""
    enforce_user_match(request_user_id, request.user_id)
    log_request("/cycle/period-start", request.user_id)
    now = _utcnow()
    entry = CycleDayLog(
        id=f"cycle_{uuid.uuid4().hex[:12]}",
        user_id=request.user_id,
        date=request.date,
        flow_level=FlowLevel.MEDIUM,
        is_period_day=True,
        created_at=now,
        updated_at=now,
    )
    upsert_cycle_day_log(entry)
    log_response("/cycle/period-start", request.user_id)
    return {"status": "ok", "date": request.date}


@router.get("/days", response_model=List[CycleDayLog])
async def get_cycle_days(
    user_id: str = Query(...),
    limit: int = Query(default=365),
    request_user_id: str = Depends(get_request_user_id),
):
    """List cycle day logs."""
    enforce_user_match(request_user_id, user_id)
    return list_cycle_day_logs(user_id, limit)


@router.get("/cycles", response_model=List[CycleInfo])
async def get_cycles(
    user_id: str = Query(...),
    request_user_id: str = Depends(get_request_user_id),
):
    """Get computed cycle boundaries."""
    enforce_user_match(request_user_id, user_id)
    cycle_logs = list_cycle_day_logs(user_id, limit=730)
    return compute_cycles(cycle_logs)


@router.get("/correlations", response_model=CyclePatternReport)
async def get_correlations(
    user_id: str = Query(...),
    request_user_id: str = Depends(get_request_user_id),
):
    """Run the symptom-cycle correlation engine."""
    enforce_user_match(request_user_id, user_id)
    cycle_logs = list_cycle_day_logs(user_id, limit=730)
    cycles = compute_cycles(cycle_logs)
    symptom_logs = list_all_logs(user_id)
    correlations = detect_correlations(symptom_logs, cycles)

    completed = [c for c in cycles if c.length_days]
    avg_length = (
        sum(c.length_days for c in completed) / len(completed)
        if completed else None
    )
    avg_period = (
        sum(c.period_length_days for c in completed) / len(completed)
        if completed else None
    )

    return CyclePatternReport(
        user_id=user_id,
        analysis_window_cycles=len(cycles),
        average_cycle_length=avg_length,
        average_period_length=avg_period,
        correlations=correlations,
        generated_at=_utcnow(),
    )


@router.delete("/day/{date_str}")
async def remove_cycle_day(
    date_str: str,
    user_id: str = Query(...),
    request_user_id: str = Depends(get_request_user_id),
):
    """Delete a cycle day entry."""
    enforce_user_match(request_user_id, user_id)
    deleted = delete_cycle_day_log(user_id, date_str)
    if not deleted:
        raise HTTPException(status_code=404, detail="Cycle day not found")
    return {"status": "deleted", "date": date_str}
