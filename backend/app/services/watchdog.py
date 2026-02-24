"""Clinical Watchdog Agent — background diagnostic analysis.

After each log submission, this runs asynchronously to analyze the patient's
longitudinal symptom history. If it detects a concerning pattern, it stores
a safe, non-diagnostic nudge as a ScheduledCheckin. The internal diagnostic
rationale is logged to the terminal only — never stored in user-facing DB.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone

from ..models import CheckinType, ScheduledCheckin
from ..services.medgemma_client import get_medgemma_client
from ..services.response_generator import llm_safety_check
from ..services.storage import (
    list_logs_in_days, create_scheduled_checkin,
    get_last_watchdog_run, get_last_force_watchdog_run, record_watchdog_run,
    store_watchdog_observation,
)

logger = logging.getLogger(__name__)

WATCHDOG_COOLDOWN_HOURS = 24
WATCHDOG_FORCE_COOLDOWN_HOURS = 4
WATCHDOG_LOOKBACK_DAYS = 30
WATCHDOG_MIN_LOGS = 5


async def run_watchdog(user_id: str, force: bool = False) -> None:
    """Background task: analyze longitudinal history for concerning patterns."""
    try:
        # Throttle: force runs only check last *force* run (4h);
        # normal runs check any last run (24h)
        if force:
            last_force = get_last_force_watchdog_run(user_id)
            if last_force:
                if last_force.tzinfo is None:
                    last_force = last_force.replace(tzinfo=timezone.utc)
                if last_force > datetime.now(timezone.utc) - timedelta(hours=WATCHDOG_FORCE_COOLDOWN_HOURS):
                    logger.debug("[WATCHDOG] Skipping force run — last force run %s (within %dh cooldown)",
                                 last_force, WATCHDOG_FORCE_COOLDOWN_HOURS)
                    return
            logger.info("[WATCHDOG] Force-triggered by MedGemma tool_call for user=%s", user_id)
        else:
            last_run = get_last_watchdog_run(user_id)
            if last_run:
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=timezone.utc)
                if last_run > datetime.now(timezone.utc) - timedelta(hours=WATCHDOG_COOLDOWN_HOURS):
                    logger.debug("[WATCHDOG] Skipping — last run %s (within %dh cooldown)",
                                 last_run, WATCHDOG_COOLDOWN_HOURS)
                    return

        # Need enough history to reason over
        logs = list_logs_in_days(user_id, WATCHDOG_LOOKBACK_DAYS)
        if len(logs) < WATCHDOG_MIN_LOGS:
            return

        # Build condensed history (uses cached map-reduce)
        client = get_medgemma_client()
        history_context = await client.build_full_history_context(logs, user_id=user_id)

        # Run watchdog analysis
        result = await client.watchdog_analysis(history_context)

        # Log internal rationale to terminal (never stored in user-facing DB)
        if result.concerning_pattern_detected:
            logger.info(
                "[WATCHDOG] user=%s | Rationale: %s | Nudge: %s",
                user_id,
                result.internal_clinical_rationale,
                result.safe_patient_nudge,
            )

            # Safety check: ensure LLM-generated nudge doesn't contain diagnoses or medical advice
            if not await llm_safety_check(result.safe_patient_nudge):
                logger.warning(
                    "[WATCHDOG] Safety check blocked nudge for user=%s: %s",
                    user_id,
                    result.safe_patient_nudge[:200],
                )
                record_watchdog_run(user_id, force=force)
                return

            checkin = ScheduledCheckin(
                id=f"watchdog_{uuid.uuid4().hex[:12]}",
                user_id=user_id,
                checkin_type=CheckinType.HEALTH_INSIGHT,
                scheduled_for=datetime.now(timezone.utc),
                message=result.safe_patient_nudge,
                context={"source": "watchdog", "rationale_logged": True},
                created_at=datetime.now(timezone.utc),
            )
            create_scheduled_checkin(checkin)

            # Persist clinician-facing observation for Doctor Packet injection
            if result.clinician_facing_observation:
                store_watchdog_observation(user_id, result.clinician_facing_observation)
        else:
            logger.debug("[WATCHDOG] user=%s | No concerning patterns detected.", user_id)

        # Record run time for cooldown (regardless of outcome)
        record_watchdog_run(user_id, force=force)

        # Compress historical logs into rolling baseline (piggybacks on watchdog cooldown)
        try:
            await client.compress_history_if_needed(user_id)
        except Exception as comp_exc:
            logger.warning("[BASELINE] Compression failed for %s: %s", user_id, comp_exc)

    except Exception as exc:
        logger.warning("[WATCHDOG] Background analysis failed for %s: %s", user_id, exc)
