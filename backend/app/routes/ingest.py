import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, Response

from ..models import (
    VoiceIngestRequest,
    LogEntry,
    FollowupExchange,
    EnhancedIngestResponse,
    ScheduledCheckin,
    CheckinType,
    ImageIngestRequest,
    ImageIngestResponse,
    ImageAnalysisResult,
    AgentResponse,
    ExtractionResult,
    CycleDayLog,
    FlowLevel,
)
from ..services.storage import (
    create_log,
    list_logs_in_days, list_medication_logs, create_scheduled_checkin, list_open_checkins,
    get_or_create_user_profile, update_user_profile,
    list_logs_with_image_analysis,
    list_cycle_day_logs, upsert_cycle_day_log,
)
from ..services.medasr_client import get_medasr_client, MedASRClient
from ..services.medgemma_client import get_medgemma_client, MedGemmaClient
from ..services.image_analyzer import get_image_analyzer_client
from ..services.response_generator import get_response_generator, ResponseContext, parse_known_medication_doses
from ..services.profile_intake import (
    should_start_intake,
    get_next_intake_question,
    create_intake_checkin,
)
from ..services.safety import detect_red_flags
from ..services.protocols import ProtocolContext, get_protocol_registry
from ..services.watchdog import run_watchdog
from ..services.clinician_alerts import clinician_note_for_log
from ..services.logging import log_request, log_response, log_error, log_warning
from ..services.auth import get_request_user_id, enforce_user_match
from ..services.cycle_engine import compute_cycles, tag_log_with_cycle_day, detect_correlations

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


# ---------------------------------------------------------------------------
# Helper functions for ingest_voice pipeline
# ---------------------------------------------------------------------------


def _is_duplicate_checkin(candidate: ScheduledCheckin, existing_checkins: list[ScheduledCheckin]) -> bool:
    """Prevent check-in spam from repeated near-identical prompts."""
    candidate_msg = candidate.message.strip().lower()
    candidate_med = (candidate.context or {}).get("medication_name", "").strip().lower()

    for existing in existing_checkins:
        if existing.checkin_type != candidate.checkin_type:
            continue

        delta_seconds = abs((candidate.scheduled_for - existing.scheduled_for).total_seconds())
        existing_msg = existing.message.strip().lower()
        existing_med = (existing.context or {}).get("medication_name", "").strip().lower()

        if candidate.checkin_type == CheckinType.MEDICATION_FOLLOWUP:
            if candidate_med and existing_med and candidate_med == existing_med and delta_seconds <= 6 * 3600:
                return True

        if candidate_msg and existing_msg and candidate_msg == existing_msg and delta_seconds <= 2 * 3600:
            return True

    return False


async def _transcribe_audio(
    asr_client: MedASRClient,
    request: VoiceIngestRequest,
) -> tuple[list[str], Optional[str]]:
    """Transcribe main audio and description inputs, returning (transcripts, description)."""
    transcripts: list[str] = []
    description: Optional[str] = None

    # Step 1: Transcribe main audio if provided
    if request.audio_b64:
        main_transcript = await asr_client.transcribe(request.audio_b64)
        transcripts.append(main_transcript)

    # Step 2: Handle description (text or voice)
    if request.description_text:
        description = request.description_text
        transcripts.append(request.description_text)
    elif request.description_audio_b64:
        description = await asr_client.transcribe(request.description_audio_b64)
        transcripts.append(description)

    return transcripts, description


def _validate_transcripts(transcripts: list[str], user_id: str) -> str:
    """Validate that transcripts are non-empty and return the combined transcript."""
    if not transcripts:
        log_error("/ingest/voice", "No input provided", user_id)
        raise HTTPException(
            status_code=400,
            detail="At least one of audio_b64, description_text, or description_audio_b64 must be provided"
        )

    combined_transcript = " ".join(transcripts).strip()

    if not combined_transcript:
        log_error("/ingest/voice", "Empty transcript after processing", user_id)
        raise HTTPException(
            status_code=400,
            detail="Provided audio/text could not be transcribed. Please try again."
        )

    return combined_transcript


async def _analyze_image(
    photo_b64: str,
    user_id: str,
    recorded_at: datetime,
) -> Optional[ImageAnalysisResult]:
    """Analyze a photo via MedSigLIP and compare against prior images for progression."""
    try:
        analyzer = get_image_analyzer_client()
        image_analysis_result = await analyzer.analyze_image(photo_b64)

        # Compare against previous photos for progression tracking
        if image_analysis_result.lesion_detected:
            try:
                previous_image_logs = list_logs_with_image_analysis(user_id, limit=5)
                if previous_image_logs and previous_image_logs[0].image_analysis:
                    prev = previous_image_logs[0]
                    note = analyzer.compare_progression(
                        image_analysis_result, prev.image_analysis,
                        recorded_at, prev.recorded_at,
                    )
                    if note:
                        image_analysis_result = ImageAnalysisResult(
                            clinical_description=f"{image_analysis_result.clinical_description} {note}",
                            confidence=image_analysis_result.confidence,
                            lesion_detected=image_analysis_result.lesion_detected,
                            skin_lesion=image_analysis_result.skin_lesion,
                            raw_classifications=image_analysis_result.raw_classifications,
                        )
            except Exception as e:
                log_error("/ingest/voice", f"Progression comparison failed: {e}", user_id)

        return image_analysis_result
    except Exception as e:
        log_error("/ingest/voice", f"Image analysis failed: {e}", user_id)
        return None


def _prepend_image_tag(
    combined_transcript: str,
    image_analysis_result: Optional[ImageAnalysisResult],
) -> str:
    """Prepend image findings to the transcript so downstream extraction sees them."""
    if image_analysis_result is None:
        return combined_transcript

    if image_analysis_result.lesion_detected:
        image_tag = f"[Image shows: {image_analysis_result.clinical_description}]"
    else:
        image_tag = "[Image analyzed: No visible abnormality detected]"
    return f"{image_tag} {combined_transcript}"


async def _run_extraction(
    gemma_client: MedGemmaClient,
    combined_transcript: str,
    user_id: str,
):
    """Run MedGemma structured extraction on the combined transcript."""
    try:
        return await gemma_client.extract(combined_transcript)
    except Exception as e:
        log_error("/ingest/voice", f"MedGemma extraction failed: {e}", user_id)
        raise HTTPException(status_code=503, detail=f"MedGemma model unavailable: {e}")


def _detect_safety_flags(combined_transcript: str, extraction) -> list[str]:
    """Detect red flags in the transcript and attach them to the extraction."""
    red_flags = detect_red_flags(combined_transcript)
    extraction.red_flags = red_flags
    return red_flags


_SEEKING_CARE_SIGNALS = [
    "going to the er", "going to the hospital", "taking me to",
    "driving me to", "on the way to", "called 911", "ambulance",
    "heading to the er", "heading to the hospital", "at the er",
    "at the hospital", "emergency room",
]


def _static_safety_response(extraction, user_profile) -> str:
    """Build a context-aware static safety response for red flag events."""
    from ..services.clinician_alerts import get_red_flag_note

    transcript_lower = (extraction.transcript or "").lower()
    full_name = getattr(user_profile, 'name', None)
    first_name = full_name.split()[0] if full_name else None
    greeting = f"I hear you, {first_name}." if first_name else "I hear you."

    already_seeking = any(signal in transcript_lower for signal in _SEEKING_CARE_SIGNALS)

    if already_seeking:
        return f"{greeting} You're doing the right thing getting to care. The medical team will take it from here."
    else:
        flag_note = get_red_flag_note(extraction.red_flags)
        return f"{greeting} {flag_note}"


def _auto_log_period(extraction: ExtractionResult, user_id: str, recorded_at: datetime) -> None:
    """If extraction detected a period day, auto-create a cycle entry."""
    ms = extraction.menstrual_status
    if not ms or not ms.is_period_day:
        return

    flow_str = (ms.flow_level or "medium").lower()
    flow_map = {
        "spotting": FlowLevel.SPOTTING,
        "light": FlowLevel.LIGHT,
        "medium": FlowLevel.MEDIUM,
        "heavy": FlowLevel.HEAVY,
    }
    flow = flow_map.get(flow_str, FlowLevel.MEDIUM)
    date_str = recorded_at.strftime("%Y-%m-%d") if hasattr(recorded_at, 'strftime') else str(recorded_at)[:10]
    now = datetime.now(timezone.utc)

    entry = CycleDayLog(
        id=f"auto_{user_id}_{date_str}",
        user_id=user_id,
        date=date_str,
        flow_level=flow,
        is_period_day=True,
        notes="Auto-detected from symptom log",
        created_at=now,
        updated_at=now,
    )
    try:
        upsert_cycle_day_log(entry)
        log_warning("period_auto_detected", user_id=user_id, reason=f"flow={flow_str} date={date_str}")
    except Exception as e:
        log_warning("period_auto_detect_failed", user_id=user_id, reason=str(e))


def _compute_cycle_context(
    user_id: str,
    recorded_at: datetime,
    extraction,
) -> tuple[Optional[str], bool]:
    """Compute menstrual-cycle context for symptom-cycle correlation."""
    cycle_tag: Optional[str] = None
    has_cycle_correlation: bool = False
    try:
        cycle_logs = list_cycle_day_logs(user_id, limit=365)
        if cycle_logs:
            cycles = compute_cycles(cycle_logs)
            cycle_tag = tag_log_with_cycle_day(recorded_at, cycles)
            if cycle_tag:
                recent_for_correlation = list_logs_in_days(user_id, days=365, limit=500)
                if len(cycles) >= 2:
                    correlations = detect_correlations(recent_for_correlation, cycles, min_cycles=2)
                    current_symptoms = {s.symptom.lower() for s in extraction.symptoms}
                    has_cycle_correlation = any(
                        c.symptom in current_symptoms for c in correlations
                    )
    except Exception as e:
        log_warning("cycle_context_failed", user_id=user_id, reason=str(e))

    return cycle_tag, has_cycle_correlation


def _protocol_only_fallback(
    extraction,
    user_id: str,
    user_profile,
    recent_symptom_logs: list,
    image_analysis_result=None,
    cycle_tag=None,
    has_cycle_correlation: bool = False,
) -> AgentResponse:
    """Generate a meaningful fallback response using protocol evaluation when LLM fails."""
    from collections import Counter
    from datetime import timezone as tz

    # Build symptom history for protocol context
    # Use most recent log's recorded_at as reference (handles simulation data)
    if recent_symptom_logs:
        ref = recent_symptom_logs[0].recorded_at
        if isinstance(ref, str):
            ref = datetime.fromisoformat(ref.replace('Z', '+00:00'))
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=tz.utc)
        now = ref
    else:
        now = datetime.now(tz.utc)
    symptom_history: dict = {
        "symptom_counts_7d": Counter(),
        "symptom_counts_24h": Counter(),
        "symptom_counts_yesterday": Counter(),
    }
    for log in recent_symptom_logs:
        try:
            recorded = log.recorded_at
            if isinstance(recorded, str):
                recorded = datetime.fromisoformat(recorded.replace('Z', '+00:00'))
            if recorded.tzinfo is None:
                recorded = recorded.replace(tzinfo=tz.utc)
            days_ago = (now - recorded).days
            hours_ago = (now - recorded).total_seconds() / 3600
            for symptom in log.extracted.symptoms:
                name = symptom.symptom.lower()
                if days_ago < 7:
                    symptom_history["symptom_counts_7d"][name] += 1
                if hours_ago < 24:
                    symptom_history["symptom_counts_24h"][name] += 1
                if 24 <= hours_ago < 48:
                    symptom_history["symptom_counts_yesterday"][name] += 1
        except Exception:
            continue

    # Inject cycle context if available
    if cycle_tag:
        symptom_history["cycle_context"] = {
            "cycle_day": cycle_tag.cycle_day,
            "cycle_phase": cycle_tag.cycle_phase,
            "cycle_number": cycle_tag.cycle_number,
            "has_prior_correlation": has_cycle_correlation,
        }

    # Build known medication doses from profile (shared utility handles
    # both standard "MedName dose" and parenthetical "Generic (Brand) dose")
    known_doses = parse_known_medication_doses(user_profile)

    # Scan recent logs for protocol cooldown (mirrors _build_protocol_context logic)
    recent_protocol_ids: list[str] = []
    for log in recent_symptom_logs[:14]:
        q = (log.followup_question or "").lower()
        if "cycle" in q and "phase" in q:
            recent_protocol_ids.append("menstrual_cycle_protocol")

    protocol_context = ProtocolContext(
        extraction=extraction,
        user_id=user_id,
        user_profile=user_profile,
        symptom_history=symptom_history,
        image_analysis=image_analysis_result,
        known_medication_doses=known_doses,
        recent_protocol_ids=recent_protocol_ids,
    )
    registry = get_protocol_registry()
    decision = registry.evaluate(protocol_context)

    # Build acknowledgment
    symptoms = extraction.symptoms
    if symptoms:
        symptom_names = [s.symptom for s in symptoms[:3]]
        # Check for high severity or emotional distress
        max_severity = max((s.severity_1_10 or 0) for s in symptoms)
        transcript_lower = (extraction.transcript or "").lower()
        has_distress = any(w in transcript_lower for w in
                          ["not normal", "frustrated", "scared", "nobody", "terrible"])
        if has_distress:
            ack = f"I hear you — logged your {', '.join(symptom_names)}. Your experience matters."
        elif max_severity >= 7:
            ack = f"That's significant pain. Noted your {', '.join(symptom_names)}."
        else:
            ack = f"Thanks for logging that — noted your {', '.join(symptom_names)}."
    else:
        ack = "Thanks for logging that."

    # Check for concerning signals (Part 9: defense-in-depth)
    concerning = False
    if len(symptoms) >= 3:
        concerning = True
    for s in symptoms:
        if s.severity_1_10 is not None and s.severity_1_10 >= 6:
            concerning = True
    for vs in extraction.vital_signs:
        if vs.name and "glucose" in vs.name.lower() or "sugar" in vs.name.lower():
            try:
                if float(vs.value) > 200:
                    concerning = True
            except (ValueError, TypeError):
                pass
    if concerning:
        ack += " You're reporting several symptoms today. If these feel concerning, consider contacting your doctor."

    # Use protocol question if available, otherwise build a basic follow-up
    question = decision.immediate_question
    if not question and not decision.schedule_checkin:
        # Ask for the highest-priority missing field
        if extraction.missing_fields:
            if "severity" in extraction.missing_fields and symptoms:
                symptom_label = symptoms[0].symptom
                question = f"On a scale of 1-10, how bad is the {symptom_label}?"
            elif extraction.actions_taken:
                for action in extraction.actions_taken:
                    if action.name and not action.dose_text:
                        question = f"What dose of {action.name} did you take?"
                        break

    # Build scheduled check-in from protocol
    scheduled_checkin = None
    if decision.schedule_checkin:
        hours = decision.checkin_hours or 2
        checkin_type = decision.checkin_type or CheckinType.SYMPTOM_PROGRESSION
        scheduled_checkin = ScheduledCheckin(
            id=f"checkin_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            checkin_type=checkin_type,
            scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None)
                + timedelta(hours=hours),
            message=decision.checkin_message or "How are you feeling now?",
            context={"protocol_id": decision.protocol_id, "reason_code": decision.reason_code},
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    return AgentResponse(
        acknowledgment=ack,
        immediate_question=question,
        scheduled_checkin=scheduled_checkin,
        protocol_id=decision.protocol_id,
        reason_code=decision.reason_code,
        safety_mode="protocol_fallback",
        tool_calls=[],
    )


async def _build_response(
    extraction,
    user_id: str,
    user_profile,
    image_analysis_result: Optional[ImageAnalysisResult],
    cycle_tag: Optional[str],
    has_cycle_correlation: bool,
    warnings: list[str],
    recorded_at: Optional[datetime] = None,
) -> tuple[AgentResponse, list["LogEntry"], "LLMResponseGenerator", "ResponseContext"]:
    """Generate the proactive agent response and return it with recent logs, generator, and context."""
    recent_symptom_logs = list_logs_in_days(user_id, days=30, limit=200, reference_date=recorded_at)
    response_gen = get_response_generator()

    # Static safety response: bypass LLM entirely when red flags are detected
    if extraction.red_flags:
        static_ack = _static_safety_response(extraction, user_profile)
        agent_response = AgentResponse(
            acknowledgment=static_ack,
            immediate_question=None,
            scheduled_checkin=None,
            protocol_id="red_flag_static",
            reason_code="red_flags_detected",
            safety_mode="static_safety",
            tool_calls=[],
        )
        # No replan context needed for static safety
        dummy_context = ResponseContext(
            extraction=extraction,
            recent_med_logs=[],
            recent_symptom_logs=recent_symptom_logs,
            user_id=user_id,
            user_profile=user_profile,
        )
        return agent_response, recent_symptom_logs, response_gen, dummy_context

    # Query recent ambient sessions for HeAR context
    ambient_summary = _build_ambient_summary(user_id)

    context = ResponseContext(
        extraction=extraction,
        recent_med_logs=list_medication_logs(user_id, limit=20, days=30, reference_date=recorded_at),
        recent_symptom_logs=recent_symptom_logs,
        user_id=user_id,
        user_profile=user_profile,
        image_analysis=image_analysis_result,
        cycle_tag=cycle_tag,
        has_cycle_correlation=has_cycle_correlation,
        ambient_summary=ambient_summary,
    )
    try:
        agent_response, gen_meta = await response_gen.generate(context)
        if gen_meta.get("fallback_reason"):
            warnings.append(f"Agent response: {gen_meta['fallback_reason']}")
    except Exception as e:
        log_error("/ingest/voice", f"Agent response generation failed: {e}", user_id)
        warnings.append(f"Agent response unavailable: {e}")
        agent_response = _protocol_only_fallback(
            extraction=extraction,
            user_id=user_id,
            user_profile=user_profile,
            recent_symptom_logs=recent_symptom_logs,
            image_analysis_result=image_analysis_result,
            cycle_tag=cycle_tag,
            has_cycle_correlation=has_cycle_correlation,
        )

    return agent_response, recent_symptom_logs, response_gen, context


def _build_ambient_summary(user_id: str) -> Optional[str]:
    """Build a summary of recent ambient monitoring sessions for LLM context."""
    try:
        from ..services.storage import list_ambient_sessions
        sessions = list_ambient_sessions(user_id, limit=5)
        if not sessions:
            return None

        summaries = []
        for session in sessions:
            if session.status != "completed" or not session.result_json:
                continue
            import json
            try:
                result = json.loads(session.result_json) if isinstance(session.result_json, str) else session.result_json
            except (json.JSONDecodeError, TypeError):
                continue
            summary = result.get("summary", "")
            if summary:
                label = session.label or session.session_type or "session"
                summaries.append(f"- {label}: {summary}")

        return "\n".join(summaries) if summaries else None
    except Exception:
        return None


def _dedup_checkin(agent_response: AgentResponse, user_id: str) -> None:
    """De-duplicate scheduled check-ins against existing open ones (mutates agent_response)."""
    if not agent_response.scheduled_checkin:
        return

    open_checkins = list_open_checkins(
        user_id,
        checkin_type=agent_response.scheduled_checkin.checkin_type,
        limit=50,
    )
    if _is_duplicate_checkin(agent_response.scheduled_checkin, open_checkins):
        # Avoid telling the UI a check-in is scheduled when it was deduped.
        agent_response.scheduled_checkin = None
    else:
        inserted = create_scheduled_checkin(agent_response.scheduled_checkin)
        if not inserted:
            agent_response.scheduled_checkin = None


def _dispatch_agent_tools(
    tool_calls: list[str],
    user_id: str,
    log_entry: LogEntry,
    background_tasks: BackgroundTasks,
) -> tuple[dict[str, str], list[ScheduledCheckin]]:
    """Execute agent tool calls. Returns (results, pending_checkins).

    Checkins are collected but NOT persisted — the caller must persist them
    after the parent log entry is saved, to avoid orphan side effects.
    """
    from ..services.response_generator import ESCALATION_NOTES
    results: dict[str, str] = {}
    pending_checkins: list[ScheduledCheckin] = []
    for tc in tool_calls:
        if tc == "run_watchdog_now":
            background_tasks.add_task(run_watchdog, user_id, force=True)
            results[tc] = "scheduled (force=True)"

        elif tc.startswith("schedule_checkin:"):
            try:
                parts = tc.split(":", 2)
                hours = int(parts[1])
                message = parts[2] if len(parts) > 2 else "How are you feeling?"
            except (IndexError, ValueError) as e:
                logger.warning("Malformed schedule_checkin tool call %r: %s", tc, e)
                results[tc] = "error: malformed"
                continue
            checkin = ScheduledCheckin(
                id=f"checkin_{uuid.uuid4().hex[:12]}",
                user_id=user_id,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None)
                    + timedelta(hours=hours),
                message=message,
                context={"source": "agent_tool", "log_id": log_entry.id},
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            pending_checkins.append(checkin)
            results[tc] = f"created {checkin.id}"

        elif tc.startswith("escalate_clinician_alert:"):
            reason = tc.split(":", 1)[1]
            if not log_entry.contact_clinician_note:
                log_entry.contact_clinician_note = ESCALATION_NOTES.get(reason, "")
                log_entry.contact_clinician_reason = reason
                results[tc] = "clinician note added"
            else:
                results[tc] = "already escalated by protocol"

        # invoke_protocol:* is a routing label, not an executed tool — skip
    return results, pending_checkins


def _create_log_entry(
    request: VoiceIngestRequest,
    combined_transcript: str,
    description: Optional[str],
    extraction,
    image_analysis_result: Optional[ImageAnalysisResult],
    agent_response: AgentResponse,
) -> LogEntry:
    """Build a LogEntry object (does not persist)."""
    contact_clinician_note, contact_clinician_reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id=agent_response.protocol_id,
        reason_code=agent_response.reason_code,
        image_analysis=image_analysis_result,
    )
    followup_exchanges = []
    if agent_response.immediate_question:
        followup_exchanges = [FollowupExchange(question=agent_response.immediate_question)]
    log_id = f"log_{uuid.uuid4().hex[:12]}"
    return LogEntry(
        id=log_id,
        user_id=request.user_id,
        recorded_at=request.recorded_at,
        transcript=combined_transcript,
        description=description,
        photo_b64=request.photo_b64,
        extracted=extraction,
        image_analysis=image_analysis_result,
        contact_clinician_note=contact_clinician_note,
        contact_clinician_reason=contact_clinician_reason,
        followup_exchanges=followup_exchanges,
    )


def _bootstrap_profile_intake(
    user_id: str,
    user_profile,
    total_recent_logs: int,
) -> None:
    """Ask one intake question after the very first symptom log if needed."""
    existing_intake_checkins = list_open_checkins(
        user_id,
        checkin_type=CheckinType.PROFILE_INTAKE,
        limit=5,
    )
    if not existing_intake_checkins and should_start_intake(user_profile, total_recent_logs):
        next_question = get_next_intake_question(user_profile)
        if next_question:
            question_id, question_message = next_question
            inserted = create_scheduled_checkin(
                create_intake_checkin(user_id, question_id, question_message)
            )
            if inserted:
                update_user_profile(
                    user_id,
                    intake_last_question_id=question_id,
                    intake_started_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )


_SPECULATIVE_PREFIXES = ("suspected", "possible", "probable", "likely", "rule out", "r/o", "query")


def _filter_speculative_conditions(conditions: list[str]) -> list[str]:
    """Remove speculative/diagnostic labels from LLM-suggested conditions."""
    return [c for c in conditions if not c.lower().strip().startswith(_SPECULATIVE_PREFIXES)]


async def _evolve_user_profile(
    gemma_client: MedGemmaClient,
    user_id: str,
    user_profile,
    log_entry: LogEntry,
    recent_symptom_logs: list[LogEntry],
    total_recent_logs: int,
) -> None:
    """Periodically update user profile (~every 10th log) based on recent history."""
    if total_recent_logs % 10 != 0:
        return

    try:
        profile_logs = [log_entry] + recent_symptom_logs
        profile_update = await gemma_client.generate_profile_update(
            profile_logs[:200],
            {
                "conditions": user_profile.conditions,
                "patterns": user_profile.patterns,
                "health_summary": user_profile.health_summary,
            }
        )
        add_conditions = profile_update.get("add_conditions")
        if add_conditions:
            add_conditions = _filter_speculative_conditions(add_conditions)
        if add_conditions or profile_update.get("add_patterns") or profile_update.get("health_summary"):
            update_user_profile(
                user_id,
                add_conditions=add_conditions,
                add_patterns=profile_update.get("add_patterns"),
                health_summary=profile_update.get("health_summary"),
            )
    except Exception as e:
        log_error("/ingest/voice", f"Profile update failed: {e}", user_id)


def _finalize_response(
    http_response: Response,
    log_entry: LogEntry,
    agent_response: AgentResponse,
    extraction,
    red_flags: list[str],
    image_analysis_result: Optional[ImageAnalysisResult],
    warnings: list[str],
) -> EnhancedIngestResponse:
    """Log the response, set degraded-mode headers, and build the final response model."""
    unique_warnings = list(dict.fromkeys(warnings))
    followup_question = agent_response.immediate_question

    log_response(
        "/ingest/voice",
        log_entry.user_id,
        log_id=log_entry.id,
        symptoms_count=len(extraction.symptoms),
        red_flags_count=len(red_flags),
        has_checkin=bool(agent_response.scheduled_checkin),
        protocol_id=agent_response.protocol_id,
        reason_code=agent_response.reason_code,
        safety_mode=agent_response.safety_mode,
        llm_fallback_used=agent_response.safety_mode == "llm_fallback",
        degraded_mode=bool(unique_warnings),
        warnings_count=len(unique_warnings),
    )
    if unique_warnings:
        http_response.headers["X-Degraded-Mode"] = "true"
        http_response.headers["X-Degraded-Reasons"] = ";".join(unique_warnings[:3])
        log_warning("ingest_degraded_mode", user_id=log_entry.user_id, reasons=";".join(unique_warnings))

    return EnhancedIngestResponse(
        log=log_entry,
        agent_response=agent_response,
        followup_question=followup_question,
        image_analysis=image_analysis_result,
        degraded_mode=bool(unique_warnings),
        warnings=unique_warnings,
    )


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------


@router.post("/voice", response_model=EnhancedIngestResponse)
async def ingest_voice(
    request: VoiceIngestRequest,
    response: Response,
    background_tasks: BackgroundTasks,
    request_user_id: str = Depends(get_request_user_id),
) -> EnhancedIngestResponse:
    """
    Audio/Text/Photo -> ASR -> Extraction -> Persist Log -> Return Log + optional follow-up.

    This endpoint supports multiple input methods:
    - audio_b64: Main voice recording (transcribed via ASR)
    - description_text: Text description (used directly)
    - description_audio_b64: Voice description (transcribed via ASR)
    - photo_b64: Photo attachment (stored with log)

    At least one of audio_b64, description_text, or description_audio_b64 must be provided.
    """
    enforce_user_match(request_user_id, request.user_id)

    log_request(
        "/ingest/voice",
        request.user_id,
        has_audio=bool(request.audio_b64),
        has_text=bool(request.description_text),
        has_photo=bool(request.photo_b64),
    )

    asr_client = get_medasr_client()
    warnings: list[str] = []

    try:
        # Step 1-2: Transcribe audio / collect text inputs
        transcripts, description = await _transcribe_audio(asr_client, request)
        combined_transcript = _validate_transcripts(transcripts, request.user_id)

        # Step 2.5: Analyze photo if provided (MedSigLIP + progression)
        image_analysis_result: Optional[ImageAnalysisResult] = None
        if request.photo_b64:
            image_analysis_result = await _analyze_image(
                request.photo_b64, request.user_id, request.recorded_at,
            )
        combined_transcript = _prepend_image_tag(combined_transcript, image_analysis_result)

        # Step 3: Extract structured data via MedGemma
        gemma_client = get_medgemma_client()
        extraction = await _run_extraction(gemma_client, combined_transcript, request.user_id)

        # Step 4: Check for red flags
        red_flags = _detect_safety_flags(combined_transcript, extraction)

        # Step 4b: Auto-log period if mentioned in transcript
        _auto_log_period(extraction, request.user_id, request.recorded_at)

        # Step 5: Fetch user profile (long-term memory)
        user_profile = get_or_create_user_profile(request.user_id)

        # Step 5b: Compute cycle context for symptom-cycle correlation
        cycle_tag, has_cycle_correlation = _compute_cycle_context(
            request.user_id, request.recorded_at, extraction,
        )

        # Step 6: Generate proactive agent response
        agent_response, recent_symptom_logs, response_gen, resp_context = await _build_response(
            extraction, request.user_id, user_profile,
            image_analysis_result, cycle_tag, has_cycle_correlation, warnings,
            recorded_at=request.recorded_at,
        )

        # Step 7: Persist scheduled check-in (with dedup)
        _dedup_checkin(agent_response, request.user_id)

        # Step 8: Build log entry (not yet persisted)
        log_entry = _create_log_entry(
            request, combined_transcript, description, extraction,
            image_analysis_result, agent_response,
        )

        # Step 9: Execute agent tools (may modify log_entry, collects pending checkins)
        tool_results, pending_checkins = _dispatch_agent_tools(
            agent_response.tool_calls, request.user_id, log_entry, background_tasks,
        )
        agent_response.agent_trace["tool_results"] = tool_results

        # Step 10: Replan — refine acknowledgment based on tool execution results
        agent_response = await response_gen.replan(agent_response, tool_results, resp_context)

        # Step 11: Persist log first, then flush pending checkins (avoids orphan side effects)
        create_log(log_entry)
        for checkin in pending_checkins:
            create_scheduled_checkin(checkin)
        background_tasks.add_task(run_watchdog, request.user_id, force=False)

        total_recent_logs = len(recent_symptom_logs) + 1

        # Step 8c: First-run profile intake bootstrap
        _bootstrap_profile_intake(request.user_id, user_profile, total_recent_logs)

        # Step 8b: Periodically evolve user profile
        await _evolve_user_profile(
            gemma_client, request.user_id, user_profile,
            log_entry, recent_symptom_logs, total_recent_logs,
        )

        # Finalize: log, set headers, return response
        return _finalize_response(
            response, log_entry, agent_response, extraction,
            red_flags, image_analysis_result, warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("/ingest/voice", str(e), request.user_id)
        error_msg = "Failed to process voice input"
        # Provide more specific error messages for common issues
        if "transcribe" in str(e).lower():
            error_msg = "Failed to transcribe audio - please try recording again"
        elif "extract" in str(e).lower():
            error_msg = "Failed to analyze symptoms - please try again"
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/image", response_model=ImageIngestResponse)
async def ingest_image(
    request: ImageIngestRequest,
    request_user_id: str = Depends(get_request_user_id),
) -> ImageIngestResponse:
    """
    Analyze a medical image (rash, swelling, etc.) using MedSigLIP.

    Returns a clinical description of the image that can be used to supplement
    symptom logging. For example: "Erythematous circular lesion, approx 3cm."

    Input:
    - image_b64: Base64-encoded image (PNG/JPEG, max 5MB)
    - context: Optional context about what to look for (e.g., "rash on arm")

    Output:
    - clinical_description: Machine-generated description of what's visible
    - confidence: Model's confidence in the analysis
    - lesion_detected: Whether a skin lesion/abnormality was detected
    """
    enforce_user_match(request_user_id, request.user_id)

    log_request(
        "/ingest/image",
        request.user_id,
        has_context=bool(request.context),
    )

    try:
        # Analyze image using MedSigLIP
        analyzer = get_image_analyzer_client()
        analysis = await analyzer.analyze_image(request.image_b64, request.context)

        # Generate transcript addition
        if analysis.lesion_detected:
            transcript_addition = f"[Image shows: {analysis.clinical_description}]"
        else:
            transcript_addition = "[Image analyzed: No visible abnormality detected]"

        log_response(
            "/ingest/image",
            request.user_id,
            lesion_detected=analysis.lesion_detected,
            confidence=analysis.confidence,
        )

        return ImageIngestResponse(
            analysis=analysis,
            transcript_addition=transcript_addition,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("/ingest/image", str(e), request.user_id)
        error_msg = "Failed to analyze image"
        if "decode" in str(e).lower() or "invalid" in str(e).lower():
            error_msg = "Invalid image format - please use PNG or JPEG"
        elif "size" in str(e).lower():
            error_msg = "Image too large - please use an image under 5MB"
        raise HTTPException(status_code=500, detail=error_msg)
