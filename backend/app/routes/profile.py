"""
User Profile Routes - Long-term health memory for personalized interactions.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends

from ..models import CheckinType, UserProfile, ProfileUpdateRequest
from ..services.storage import (
    get_or_create_user_profile,
    update_user_profile,
    list_open_checkins,
    create_scheduled_checkin,
)
from ..services.profile_intake import get_next_intake_question, create_intake_checkin
from ..services.logging import log_request, log_response, log_error
from ..services.auth import get_request_user_id, enforce_user_match

router = APIRouter(prefix="/api/profile", tags=["profile"])


@router.get("", response_model=UserProfile)
async def get_profile(
    user_id: str,
    request_user_id: str = Depends(get_request_user_id),
) -> UserProfile:
    """
    Get a user's health profile (long-term memory).

    Returns the profile with conditions, allergies, patterns, etc.
    Creates an empty profile if one doesn't exist.
    """
    enforce_user_match(request_user_id, user_id)
    log_request("/profile", user_id)

    try:
        profile = get_or_create_user_profile(user_id)
        log_response("/profile", user_id, conditions_count=len(profile.conditions))
        return profile
    except Exception as e:
        log_error("/profile", str(e), user_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve profile")


@router.patch("", response_model=UserProfile)
async def patch_profile(
    request: ProfileUpdateRequest,
    request_user_id: str = Depends(get_request_user_id),
) -> UserProfile:
    """
    Update a user's health profile with incremental changes.

    Supports adding/removing conditions, allergies, patterns, etc.
    Items are deduplicated automatically.
    """
    enforce_user_match(request_user_id, request.user_id)
    log_request("/profile PATCH", request.user_id)

    try:
        profile = update_user_profile(
            request.user_id,
            add_conditions=request.add_conditions if request.add_conditions else None,
            remove_conditions=request.remove_conditions if request.remove_conditions else None,
            add_allergies=request.add_allergies if request.add_allergies else None,
            remove_allergies=request.remove_allergies if request.remove_allergies else None,
            add_regular_medications=request.add_regular_medications if request.add_regular_medications else None,
            remove_regular_medications=request.remove_regular_medications if request.remove_regular_medications else None,
            add_surgeries=request.add_surgeries if request.add_surgeries else None,
            remove_surgeries=request.remove_surgeries if request.remove_surgeries else None,
            add_family_history=request.add_family_history if request.add_family_history else None,
            remove_family_history=request.remove_family_history if request.remove_family_history else None,
            add_social_history=request.add_social_history if request.add_social_history else None,
            remove_social_history=request.remove_social_history if request.remove_social_history else None,
            add_patterns=request.add_patterns if request.add_patterns else None,
            remove_patterns=request.remove_patterns if request.remove_patterns else None,
            health_summary=request.health_summary,
        )
        log_response("/profile PATCH", request.user_id, conditions_count=len(profile.conditions))
        return profile
    except Exception as e:
        log_error("/profile PATCH", str(e), request.user_id)
        raise HTTPException(status_code=500, detail="Failed to update profile")


@router.post("/onboarding/start")
async def start_onboarding(
    user_id: str,
    request_user_id: str = Depends(get_request_user_id),
) -> dict:
    """
    Start the onboarding intake flow for a new user.

    Creates the first profile intake checkin if intake hasn't started yet.
    Idempotent — safe to call multiple times.
    """
    enforce_user_match(request_user_id, user_id)
    log_request("/profile/onboarding/start", user_id)

    try:
        profile = get_or_create_user_profile(user_id)

        if profile.intake_completed:
            return {"status": "completed"}

        # Check for an existing pending intake checkin (idempotent)
        existing = list_open_checkins(
            user_id, checkin_type=CheckinType.PROFILE_INTAKE, limit=1
        )
        if existing:
            return {"status": "in_progress", "checkin_id": existing[0].id}

        next_q = get_next_intake_question(profile)
        if not next_q:
            return {"status": "completed"}

        question_id, message = next_q
        checkin = create_intake_checkin(user_id, question_id, message)
        create_scheduled_checkin(checkin)
        update_user_profile(
            user_id,
            intake_last_question_id=question_id,
            intake_started_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        log_response("/profile/onboarding/start", user_id, question_id=question_id)
        return {"status": "started", "checkin_id": checkin.id}

    except Exception as e:
        log_error("/profile/onboarding/start", str(e), user_id)
        raise HTTPException(status_code=500, detail="Failed to start onboarding")
