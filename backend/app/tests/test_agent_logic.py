"""Tests for proactive agent logic and check-in dedupe safeguards."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from ..models import (
    AgentResponse,
    CheckinType,
    ExtractionResult,
    LogEntry,
    ScheduledCheckin,
    SymptomEntity,
    UserProfile,
    VitalSignEntry,
)
from ..routes.ingest import _dispatch_agent_tools, _is_duplicate_checkin
from ..services.protocols import ProtocolDecision
from ..services.response_generator import (
    LLMResponseGenerator, ResponseContext, _build_immediate_question,
    filter_tool_calls, _has_abnormal_vital, _should_inject_watchdog,
)

_ENV_KEYS = (
    "USE_STUB_MEDGEMMA", "USE_REPLAN_PASS", "USE_LOCAL_MEDGEMMA",
    "USE_STUB_RESPONSE_GEN", "MEDGEMMA_TEMPERATURE",
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Save and restore env vars to prevent cross-test pollution."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _profile(user_id: str, conditions: list[str]) -> UserProfile:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return UserProfile(
        user_id=user_id,
        conditions=conditions,
        allergies=[],
        regular_medications=[],
        patterns=[],
        health_summary=None,
        created_at=now,
        updated_at=now,
    )


def test_asthma_cough_question_is_reachable_when_severity_missing():
    """Asthma + cough should ask inhaler question even though cough is not in generic high-priority list."""
    context = ResponseContext(
        extraction=ExtractionResult(
            transcript="I am coughing more today.",
            symptoms=[SymptomEntity(symptom="cough")],
            missing_fields=["severity"],
        ),
        recent_med_logs=[],
        recent_symptom_logs=[],
        user_id="test_user",
        user_profile=_profile("test_user", ["Asthma"]),
    )

    question = _build_immediate_question(context=context, has_scheduled_checkin=False)

    assert question is not None
    assert "rescue inhaler" in question.lower()


def test_duplicate_medication_checkin_detected_by_medication_and_time_window():
    """Repeated medication check-ins for the same med within a short window should be deduped."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    candidate = ScheduledCheckin(
        id="checkin_new",
        user_id="test_user",
        checkin_type=CheckinType.MEDICATION_FOLLOWUP,
        scheduled_for=now + timedelta(hours=2),
        message="How are you feeling after ibuprofen?",
        context={"medication_name": "ibuprofen"},
        created_at=now,
    )
    existing = ScheduledCheckin(
        id="checkin_existing",
        user_id="test_user",
        checkin_type=CheckinType.MEDICATION_FOLLOWUP,
        scheduled_for=now + timedelta(hours=3),
        message="How are you feeling after ibuprofen?",
        context={"medication_name": "Ibuprofen"},
        created_at=now - timedelta(minutes=1),
    )

    assert _is_duplicate_checkin(candidate, [existing]) is True


def test_different_medication_checkins_are_not_deduped():
    """Different medications should remain separate follow-ups."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    candidate = ScheduledCheckin(
        id="checkin_new",
        user_id="test_user",
        checkin_type=CheckinType.MEDICATION_FOLLOWUP,
        scheduled_for=now + timedelta(hours=2),
        message="How are you feeling after ibuprofen?",
        context={"medication_name": "ibuprofen"},
        created_at=now,
    )
    existing = ScheduledCheckin(
        id="checkin_existing",
        user_id="test_user",
        checkin_type=CheckinType.MEDICATION_FOLLOWUP,
        scheduled_for=now + timedelta(hours=2),
        message="How are you feeling after acetaminophen?",
        context={"medication_name": "acetaminophen"},
        created_at=now,
    )

    assert _is_duplicate_checkin(candidate, [existing]) is False


def test_protocol_mode_suppresses_immediate_question_when_checkin_exists():
    """Protocol mode should never emit both an immediate question and a check-in."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    generator = LLMResponseGenerator()

    context = ResponseContext(
        extraction=ExtractionResult(
            transcript="My breathing is worse today.",
            symptoms=[SymptomEntity(symptom="shortness of breath", severity_1_10=8)],
            missing_fields=[],
        ),
        recent_med_logs=[],
        recent_symptom_logs=[],
        user_id="test_user",
        user_profile=_profile("test_user", ["Asthma"]),
    )
    decision = ProtocolDecision(
        protocol_id="asthma_respiratory_protocol",
        schedule_checkin=True,
        checkin_type=CheckinType.SYMPTOM_PROGRESSION,
        checkin_hours=1,
        checkin_message="Checking in soon on your breathing.",
        reason_code="asthma_respiratory_worsening_or_severe",
    )
    question, checkin, _, _ = generator._apply_protocol_mode(
        context=context,
        protocol_decision=decision,
        fallback_question="How bad is it now?",
        fallback_checkin=None,
    )

    assert checkin is not None
    assert question is None


def test_llm_fallback_question_blocked_for_safety_sensitive_long_tail():
    """Unsafe long-tail fallback questions should be blocked when no protocol matched."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    generator = LLMResponseGenerator()

    safety_context = ResponseContext(
        extraction=ExtractionResult(
            transcript="I have chest pain and feel faint.",
            symptoms=[SymptomEntity(symptom="chest pain")],
            missing_fields=[],
        ),
        recent_med_logs=[],
        recent_symptom_logs=[],
        user_id="test_user",
    )
    blocked = generator._sanitize_llm_question(
        "Can you describe the pain and when it started?",
        context=safety_context,
        protocol_matched=False,
    )
    assert blocked is None

    long_tail_context = ResponseContext(
        extraction=ExtractionResult(
            transcript="My skin feels itchy after showering.",
            symptoms=[SymptomEntity(symptom="itchy skin")],
            missing_fields=[],
        ),
        recent_med_logs=[],
        recent_symptom_logs=[],
        user_id="test_user",
    )
    sanitized = generator._sanitize_llm_question(
        "Can you describe when this started? Also, does anything make it better?",
        context=long_tail_context,
        protocol_matched=False,
    )
    assert sanitized == "Can you describe when this started?"


def test_protocol_match_without_followup_does_not_use_llm_fallback():
    """Matched protocol paths stay deterministic even when model offers extra follow-up."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    generator = LLMResponseGenerator()

    context = ResponseContext(
        extraction=ExtractionResult(
            transcript="I had a fever of 100 F",
            symptoms=[SymptomEntity(symptom="fever", severity_1_10=3)],
            missing_fields=[],
        ),
        recent_med_logs=[],
        recent_symptom_logs=[],
        user_id="test_user",
    )
    decision = ProtocolDecision(
        protocol_id="fever_protocol",
        immediate_question=None,
        schedule_checkin=False,
        reason_code="fever_no_followup_needed",
    )
    question, checkin, safety_mode, fallback_used = generator._apply_protocol_mode(
        context=context,
        protocol_decision=decision,
        fallback_question="Can you tell me if you had chills too?",
        fallback_checkin=ScheduledCheckin(
            id="checkin_candidate",
            user_id="test_user",
            checkin_type=CheckinType.SYMPTOM_PROGRESSION,
            scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=2),
            message="Checking in later.",
            context={},
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        ),
    )

    assert question is None
    assert checkin is None
    assert safety_mode == "protocol"
    assert fallback_used is False


# --- Tool-calls parsing guardrail tests ---


def test_tool_calls_parse_empty_array():
    """Empty tool_calls array should pass through as empty."""
    assert filter_tool_calls([]) == []


def test_tool_calls_parse_valid_tool():
    """Known tool name should pass the allowlist filter."""
    assert filter_tool_calls(["run_watchdog_now"]) == ["run_watchdog_now"]


def test_tool_calls_parse_filters_unknown_tools():
    """Unknown or malformed tool names should be stripped by the allowlist."""
    raw = ["run_watchdog_now", "unknown_tool", 42, None, "run_watchdog_now"]
    result = filter_tool_calls(raw)
    assert result == ["run_watchdog_now", "run_watchdog_now"]


def test_tool_calls_parse_string_input_returns_empty():
    """String input (not array) should return empty list, not iterate chars."""
    assert filter_tool_calls("run_watchdog_now") == []


def test_watchdog_force_cooldown_is_shorter_than_normal():
    """Force-triggered watchdog should have a shorter but non-zero cooldown."""
    from ..services.watchdog import WATCHDOG_COOLDOWN_HOURS, WATCHDOG_FORCE_COOLDOWN_HOURS
    assert 0 < WATCHDOG_FORCE_COOLDOWN_HOURS < WATCHDOG_COOLDOWN_HOURS


def test_watchdog_force_run_not_blocked_by_normal_run():
    """Force-triggered run should only check force cooldown, not normal cooldown."""
    from ..services.storage import get_last_watchdog_run, get_last_force_watchdog_run
    from ..services.watchdog import WATCHDOG_COOLDOWN_HOURS, WATCHDOG_FORCE_COOLDOWN_HOURS
    # Verify the two functions exist and are distinct
    assert get_last_watchdog_run is not get_last_force_watchdog_run
    # Verify force cooldown is independent (shorter than normal)
    assert WATCHDOG_FORCE_COOLDOWN_HOURS < WATCHDOG_COOLDOWN_HOURS


def test_safety_sensitive_long_tail_blocks_llm_fallback_checkin():
    """Safety-sensitive non-protocol cases should not schedule LLM fallback check-ins."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    generator = LLMResponseGenerator()

    context = ResponseContext(
        extraction=ExtractionResult(
            transcript="I have chest pain and I feel faint.",
            symptoms=[SymptomEntity(symptom="chest pain")],
            missing_fields=[],
        ),
        recent_med_logs=[],
        recent_symptom_logs=[],
        user_id="test_user",
    )
    decision = ProtocolDecision(
        protocol_id=None,
        immediate_question=None,
        schedule_checkin=False,
        reason_code="no_protocol_match",
    )
    question, checkin, safety_mode, fallback_used = generator._apply_protocol_mode(
        context=context,
        protocol_decision=decision,
        fallback_question="When did the pain start?",
        fallback_checkin=ScheduledCheckin(
            id="checkin_candidate",
            user_id="test_user",
            checkin_type=CheckinType.SYMPTOM_PROGRESSION,
            scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1),
            message="How is the chest pain now?",
            context={},
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        ),
    )

    assert question is None
    assert checkin is None
    assert safety_mode == "protocol"
    assert fallback_used is False


# --- Deterministic watchdog injection tests ---


def test_has_abnormal_vital_glucose_high():
    """Blood glucose > 200 should be flagged as abnormal."""
    vitals = [VitalSignEntry(name="blood sugar", value="230", unit="mg/dL")]
    assert _has_abnormal_vital(vitals) is True


def test_has_abnormal_vital_glucose_low():
    """Blood glucose < 70 should be flagged as abnormal."""
    vitals = [VitalSignEntry(name="blood sugar", value="55", unit="mg/dL")]
    assert _has_abnormal_vital(vitals) is True


def test_has_abnormal_vital_normal():
    """Blood glucose in normal range should not be flagged."""
    vitals = [VitalSignEntry(name="blood sugar", value="120", unit="mg/dL")]
    assert _has_abnormal_vital(vitals) is False


def test_should_inject_watchdog_high_severity():
    """Severity >= 8 with enough history should trigger injection."""
    extraction = ExtractionResult(
        transcript="test",
        symptoms=[SymptomEntity(symptom="cramps", severity_1_10=9)],
    )
    assert _should_inject_watchdog(extraction, num_prior_logs=6) is True


def test_should_inject_watchdog_abnormal_vital_moderate_severity():
    """Abnormal vital + severity >= 5 with enough history should trigger."""
    extraction = ExtractionResult(
        transcript="test",
        symptoms=[SymptomEntity(symptom="dizziness", severity_1_10=6)],
        vital_signs=[VitalSignEntry(name="blood sugar", value="230", unit="mg/dL")],
    )
    assert _should_inject_watchdog(extraction, num_prior_logs=6) is True


def test_should_inject_watchdog_not_enough_history():
    """High severity but insufficient history should NOT trigger."""
    extraction = ExtractionResult(
        transcript="test",
        symptoms=[SymptomEntity(symptom="cramps", severity_1_10=9)],
    )
    assert _should_inject_watchdog(extraction, num_prior_logs=3) is False


def test_should_inject_watchdog_low_severity_normal_vitals():
    """Low severity with normal vitals should NOT trigger."""
    extraction = ExtractionResult(
        transcript="test",
        symptoms=[SymptomEntity(symptom="headache", severity_1_10=3)],
        vital_signs=[VitalSignEntry(name="blood sugar", value="120", unit="mg/dL")],
    )
    assert _should_inject_watchdog(extraction, num_prior_logs=6) is False


def test_should_inject_watchdog_red_flags_no_trigger():
    """Red flags do NOT trigger injection — handled by static_safety path instead."""
    extraction = ExtractionResult(
        transcript="test",
        symptoms=[SymptomEntity(symptom="shaking")],
        red_flags=["passed out"],
    )
    assert _should_inject_watchdog(extraction, num_prior_logs=6) is False


# --- schedule_checkin validation tests ---


def test_schedule_checkin_valid():
    """Valid schedule_checkin tool call should pass filter."""
    assert filter_tool_calls(["schedule_checkin:4:How is the pain?"]) == [
        "schedule_checkin:4:How is the pain?"
    ]


def test_schedule_checkin_hours_out_of_range():
    """Hours outside 1-24 should be rejected."""
    assert filter_tool_calls(["schedule_checkin:0:msg"]) == []
    assert filter_tool_calls(["schedule_checkin:25:msg"]) == []


def test_schedule_checkin_missing_message():
    """Empty message after colon should be rejected."""
    assert filter_tool_calls(["schedule_checkin:4:"]) == []


def test_schedule_checkin_missing_parts():
    """Missing message segment should be rejected."""
    assert filter_tool_calls(["schedule_checkin:4"]) == []


def test_schedule_checkin_non_integer_hours():
    """Non-integer hours should be rejected."""
    assert filter_tool_calls(["schedule_checkin:abc:msg"]) == []


# --- escalate_clinician_alert validation tests ---


def test_escalate_alert_valid():
    """Valid escalation reasons should pass filter."""
    assert filter_tool_calls(["escalate_clinician_alert:worsening_trajectory"]) == [
        "escalate_clinician_alert:worsening_trajectory"
    ]


def test_escalate_alert_all_valid_reasons():
    """All defined escalation reasons should be accepted."""
    from ..services.response_generator import ALLOWED_ESCALATION_REASONS
    for reason in ALLOWED_ESCALATION_REASONS:
        result = filter_tool_calls([f"escalate_clinician_alert:{reason}"])
        assert len(result) == 1, f"Reason '{reason}' should be accepted"


def test_escalate_alert_invalid_reason():
    """Unknown escalation reason should be rejected."""
    assert filter_tool_calls(["escalate_clinician_alert:made_up_reason"]) == []


# --- invoke_protocol validation tests ---


def test_invoke_protocol_valid():
    """Valid protocol ID should pass filter."""
    result = filter_tool_calls(["invoke_protocol:fever_protocol"])
    assert result == ["invoke_protocol:fever_protocol"]


def test_invoke_protocol_invalid():
    """Unknown protocol ID should be rejected."""
    assert filter_tool_calls(["invoke_protocol:fake_protocol"]) == []


# --- Mixed tool_calls tests ---


def test_mixed_tool_calls_filters_correctly():
    """Mix of valid, invalid, and different tool types should filter properly."""
    raw = [
        "run_watchdog_now",
        "schedule_checkin:2:Check pain",
        "escalate_clinician_alert:patient_distress",
        "bad_tool",
        "schedule_checkin:0:invalid_hours",
    ]
    result = filter_tool_calls(raw)
    assert len(result) == 3
    assert "bad_tool" not in result
    assert "schedule_checkin:0:invalid_hours" not in result


# --- Dispatcher unit tests ---


def _make_log_entry(**overrides) -> LogEntry:
    """Build a minimal LogEntry for testing."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    defaults = dict(
        id="log_test123",
        user_id="test_user",
        recorded_at=now,
        transcript="test transcript",
        extracted=ExtractionResult(transcript="test transcript"),
    )
    defaults.update(overrides)
    return LogEntry(**defaults)


def test_dispatch_escalate_sets_clinician_note():
    """escalate_clinician_alert should set contact_clinician_note on the log entry."""
    log_entry = _make_log_entry()
    bg = MagicMock()
    results, pending_checkins = _dispatch_agent_tools(
        ["escalate_clinician_alert:medication_concern"],
        "test_user", log_entry, bg,
    )
    assert "clinician note added" in results["escalate_clinician_alert:medication_concern"]
    assert log_entry.contact_clinician_note is not None
    assert log_entry.contact_clinician_reason == "medication_concern"


def test_dispatch_escalate_skips_if_already_escalated():
    """Should not overwrite existing clinician note from protocol."""
    log_entry = _make_log_entry(
        contact_clinician_note="Existing protocol note",
        contact_clinician_reason="red_flags_detected",
    )
    bg = MagicMock()
    results, pending_checkins = _dispatch_agent_tools(
        ["escalate_clinician_alert:worsening_trajectory"],
        "test_user", log_entry, bg,
    )
    assert "already escalated" in results["escalate_clinician_alert:worsening_trajectory"]
    assert log_entry.contact_clinician_note == "Existing protocol note"


def test_dispatch_watchdog_schedules_background_task():
    """run_watchdog_now should add a background task."""
    log_entry = _make_log_entry()
    bg = MagicMock()
    results, pending_checkins = _dispatch_agent_tools(["run_watchdog_now"], "test_user", log_entry, bg)
    assert "scheduled" in results["run_watchdog_now"]
    bg.add_task.assert_called_once()


def test_dispatch_ignores_invoke_protocol():
    """invoke_protocol is a routing label, not an executable tool."""
    log_entry = _make_log_entry()
    bg = MagicMock()
    results, pending_checkins = _dispatch_agent_tools(
        ["invoke_protocol:fever_protocol"], "test_user", log_entry, bg,
    )
    assert len(results) == 0


def test_dispatch_empty_tool_calls():
    """Empty tool_calls should produce empty results."""
    log_entry = _make_log_entry()
    bg = MagicMock()
    results, pending_checkins = _dispatch_agent_tools([], "test_user", log_entry, bg)
    assert results == {}


# --- Replan deterministic tests ---


def test_replan_deterministic_appends_checkin_note():
    """Deterministic replan should append check-in note to acknowledgment."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    os.environ["USE_REPLAN_PASS"] = "true"
    generator = LLMResponseGenerator()

    agent_response = AgentResponse(
        acknowledgment="Logged your headache.",
        tool_calls=["schedule_checkin:4:How is the pain?"],
    )
    tool_results = {"schedule_checkin:4:How is the pain?": "created checkin_abc123"}

    result = generator._replan_deterministic(agent_response, tool_results)
    assert "check back" in result.acknowledgment.lower()
    assert "4 hours" in result.acknowledgment
    assert result.agent_trace["replan"]["original_acknowledgment"] == "Logged your headache."


def test_replan_deterministic_appends_escalation_note():
    """Deterministic replan should append clinician review note."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    os.environ["USE_REPLAN_PASS"] = "true"
    generator = LLMResponseGenerator()

    agent_response = AgentResponse(
        acknowledgment="I hear you.",
        tool_calls=["escalate_clinician_alert:worsening_trajectory"],
    )
    tool_results = {"escalate_clinician_alert:worsening_trajectory": "clinician note added"}

    result = generator._replan_deterministic(agent_response, tool_results)
    assert "clinician review" in result.acknowledgment.lower()


def test_replan_noop_when_no_tool_results():
    """Replan with empty tool_results should return response unchanged."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    os.environ["USE_REPLAN_PASS"] = "true"
    generator = LLMResponseGenerator()

    agent_response = AgentResponse(
        acknowledgment="Logged your entry.",
        tool_calls=[],
    )
    result = generator._replan_deterministic(agent_response, {})
    assert result.acknowledgment == "Logged your entry."
    assert result.agent_trace["replan"]["kept_original"] is True


def test_replan_preserves_safety_fields():
    """Replan should not modify protocol_id, reason_code, or safety_mode."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    os.environ["USE_REPLAN_PASS"] = "true"
    generator = LLMResponseGenerator()

    agent_response = AgentResponse(
        acknowledgment="Noted the fever.",
        protocol_id="fever_protocol",
        reason_code="fever_high",
        safety_mode="protocol",
        tool_calls=["schedule_checkin:2:Check temp"],
    )
    tool_results = {"schedule_checkin:2:Check temp": "created checkin_xyz"}

    result = generator._replan_deterministic(agent_response, tool_results)
    assert result.protocol_id == "fever_protocol"
    assert result.reason_code == "fever_high"
    assert result.safety_mode == "protocol"


def test_replan_watchdog_only_no_user_visible_note():
    """Watchdog-only tool results should not modify acknowledgment."""
    os.environ["USE_STUB_MEDGEMMA"] = "true"
    os.environ["USE_REPLAN_PASS"] = "true"
    generator = LLMResponseGenerator()

    agent_response = AgentResponse(
        acknowledgment="Logged your entry.",
        tool_calls=["run_watchdog_now"],
    )
    tool_results = {"run_watchdog_now": "scheduled (force=True)"}

    result = generator._replan_deterministic(agent_response, tool_results)
    assert result.acknowledgment == "Logged your entry."
    assert result.agent_trace["replan"]["kept_original"] is True


# --- Factual check replan tests ---


def test_factual_check_replan_rejects_false_checkin_claim():
    """Reject text claiming check-in when none was created."""
    generator = LLMResponseGenerator()
    assert generator._factual_check_replan("I'll check back in 2 hours", {}) is False


def test_factual_check_replan_accepts_checkin_claim_when_created():
    """Accept text claiming check-in when one was created."""
    generator = LLMResponseGenerator()
    results = {"schedule_checkin:4:msg": "created checkin_abc123"}
    assert generator._factual_check_replan("I'll check back in 4 hours", results) is True


def test_factual_check_replan_allows_legitimate_clinician_reference():
    """'Discuss with your clinician' is advice, not an escalation claim."""
    generator = LLMResponseGenerator()
    assert generator._factual_check_replan(
        "This is something your clinician should evaluate.", {}
    ) is True


def test_factual_check_replan_rejects_false_escalation_claim():
    """Reject text claiming escalation when none happened."""
    generator = LLMResponseGenerator()
    assert generator._factual_check_replan(
        "I've flagged this for your clinician review.", {}
    ) is False


# --- HPI demographics post-validation ---


def test_fix_hpi_demographics_corrects_age():
    """_fix_hpi_demographics replaces hallucinated age with profile age."""
    from ..services.medgemma.stub import StubMedGemmaClient
    client = StubMedGemmaClient()
    profile = MagicMock(age=29, gender="female")
    hpi = "A 34-year-old female with a history of menorrhagia presents with worsening pelvic pain."
    fixed = client._fix_hpi_demographics(hpi, profile)
    assert "29-year-old" in fixed
    assert "34-year-old" not in fixed


def test_fix_hpi_demographics_corrects_gender():
    """_fix_hpi_demographics replaces wrong gender."""
    from ..services.medgemma.stub import StubMedGemmaClient
    client = StubMedGemmaClient()
    profile = MagicMock(age=61, gender="male")
    hpi = "A 61-year-old female with COPD presents with cough."
    fixed = client._fix_hpi_demographics(hpi, profile)
    assert "male" in fixed.lower()[:50]


def test_fix_hpi_demographics_no_profile():
    """_fix_hpi_demographics is a no-op without a profile."""
    from ..services.medgemma.stub import StubMedGemmaClient
    client = StubMedGemmaClient()
    hpi = "A 50-year-old male presents."
    assert client._fix_hpi_demographics(hpi, None) == hpi


# --- Speculative conditions filter ---


def test_filter_speculative_conditions():
    """Speculative prefixes are removed from condition suggestions."""
    from ..routes.ingest import _filter_speculative_conditions
    conditions = [
        "Suspected Endometriosis",
        "Menorrhagia (heavy menstrual bleeding)",
        "Possible PCOS",
        "Mild anemia",
        "Rule out lupus",
    ]
    filtered = _filter_speculative_conditions(conditions)
    assert "Menorrhagia (heavy menstrual bleeding)" in filtered
    assert "Mild anemia" in filtered
    assert len(filtered) == 2


# --- Static safety response ---


def test_static_safety_response_already_seeking_care():
    """When transcript mentions going to ER, response should NOT tell them to go to ER."""
    from ..routes.ingest import _static_safety_response
    extraction = MagicMock()
    extraction.transcript = "I can't breathe. Maria is taking me to the ER right now."
    extraction.red_flags = ["difficulty breathing", "can't breathe"]
    profile = MagicMock()
    profile.name = "Frank"
    response = _static_safety_response(extraction, profile)
    assert "Frank" in response
    assert "doing the right thing" in response.lower()
    assert "emergency room" not in response.lower()


def test_static_safety_response_not_seeking_care():
    """When transcript does NOT mention ER, response should use red flag note."""
    from ..routes.ingest import _static_safety_response
    extraction = MagicMock()
    extraction.transcript = "I have terrible chest pain"
    extraction.red_flags = ["chest pain"]
    profile = MagicMock()
    profile.name = "Elena"
    response = _static_safety_response(extraction, profile)
    assert "Elena" in response
    assert "chest pain" in response.lower()


def test_static_safety_response_no_name():
    """Without a patient name, should still produce a valid response."""
    from ..routes.ingest import _static_safety_response
    extraction = MagicMock()
    extraction.transcript = "I had a seizure"
    extraction.red_flags = ["seizure"]
    profile = MagicMock()
    profile.name = None
    response = _static_safety_response(extraction, profile)
    assert response.startswith("I hear you.")
    assert "seizure" in response.lower()


# --- JSON parse robustness ---


def test_parse_json_response_valid_json():
    """Valid JSON parses normally."""
    gen = LLMResponseGenerator.__new__(LLMResponseGenerator)
    result = gen._parse_json_response('{"acknowledgment": "Got it"}')
    assert result["acknowledgment"] == "Got it"


def test_parse_json_response_with_markdown_fence():
    """JSON in markdown code block should parse."""
    gen = LLMResponseGenerator.__new__(LLMResponseGenerator)
    result = gen._parse_json_response('```json\n{"acknowledgment": "Got it"}\n```')
    assert result["acknowledgment"] == "Got it"


def test_parse_json_response_trailing_comma():
    """Trailing comma before } should be repaired."""
    gen = LLMResponseGenerator.__new__(LLMResponseGenerator)
    result = gen._parse_json_response('{"acknowledgment": "Got it",}')
    assert result["acknowledgment"] == "Got it"


def test_parse_json_response_preamble_text():
    """JSON with preamble text should be extracted via boundary detection."""
    gen = LLMResponseGenerator.__new__(LLMResponseGenerator)
    result = gen._parse_json_response('Here is the response:\n{"acknowledgment": "Got it"}')
    assert result["acknowledgment"] == "Got it"


def test_parse_json_response_empty_raises():
    """Empty string should raise ValueError."""
    gen = LLMResponseGenerator.__new__(LLMResponseGenerator)
    with pytest.raises(ValueError, match="Empty"):
        gen._parse_json_response("")


# --- Timeline bullet truncation ---


def test_timeline_bullet_long_transcript_truncated():
    """Transcripts over 200 chars should be truncated at word boundary."""
    from ..services.medgemma.stub import StubMedGemmaClient
    client = StubMedGemmaClient()
    long_text = "word " * 50  # 250 chars
    log = MagicMock()
    log.recorded_at = datetime(2026, 1, 1)
    log.transcript = long_text.strip()
    log.extracted.symptoms = []
    log.extracted.actions_taken = []
    log.extracted.red_flags = []
    bullets = client._build_timeline_bullets([log])
    assert len(bullets) == 1
    # Should end with ...
    assert bullets[0].endswith("...")
    # Should be under 220 chars (200 + date prefix + ...)
    bullet_content = bullets[0].split(": ", 1)[1]
    assert len(bullet_content) < 210


def test_timeline_bullet_short_transcript_not_truncated():
    """Transcripts under 200 chars should not be truncated."""
    from ..services.medgemma.stub import StubMedGemmaClient
    client = StubMedGemmaClient()
    log = MagicMock()
    log.recorded_at = datetime(2026, 1, 1)
    log.transcript = "Short entry about feeling fine today."
    log.extracted.symptoms = []
    log.extracted.actions_taken = []
    log.extracted.red_flags = []
    bullets = client._build_timeline_bullets([log])
    assert "..." not in bullets[0]
    assert "Short entry" in bullets[0]
