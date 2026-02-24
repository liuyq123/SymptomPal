"""Tests for clinician-contact alert note generation."""

from ..models import ExtractionResult, SymptomEntity
from ..services.clinician_alerts import clinician_note_for_log, get_red_flag_note


def test_red_flags_generate_clinician_note():
    extraction = ExtractionResult(
        transcript="I have chest pain and shortness of breath",
        symptoms=[SymptomEntity(symptom="chest pain")],
        red_flags=["chest pain"],
    )
    note, reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id=None,
        reason_code=None,
        image_analysis=None,
    )
    assert note is not None
    assert "chest pain" in note.lower()
    assert "medical evaluation" in note.lower()
    assert reason == "red_flags_detected"


def test_high_fever_reason_code_generates_clinician_note():
    extraction = ExtractionResult(
        transcript="Fever 103 F",
        symptoms=[SymptomEntity(symptom="fever", severity_1_10=8)],
    )
    note, reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id="fever_protocol",
        reason_code="high_fever_indicator",
        image_analysis=None,
    )
    assert note is not None
    assert "clinician" in note.lower()
    assert reason == "high_fever_indicator"


def test_non_escalation_reason_code_generates_no_note():
    extraction = ExtractionResult(
        transcript="Mild headache",
        symptoms=[SymptomEntity(symptom="headache", severity_1_10=3)],
    )
    note, reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id="headache_protocol",
        reason_code="headache_no_followup_needed",
        image_analysis=None,
    )
    assert note is None
    assert reason is None


def test_skin_guardrail_reason_code_generates_clinician_note():
    extraction = ExtractionResult(
        transcript="I noticed a mole",
        symptoms=[SymptomEntity(symptom="skin mole")],
    )
    note, reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id="skin_lesion_escalation",
        reason_code="skin_always_escalate_guardrail",
        image_analysis=None,
    )
    assert note is not None
    assert "skin finding" in note.lower()
    assert reason == "skin_always_escalate_guardrail"


def test_red_flag_passed_out_specific_note():
    """'passed out' red flag should get a targeted note about loss of consciousness."""
    extraction = ExtractionResult(
        transcript="I almost passed out",
        symptoms=[SymptomEntity(symptom="near-syncope")],
        red_flags=["passed out"],
    )
    note, reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id=None,
        reason_code=None,
        image_analysis=None,
    )
    assert note is not None
    assert "consciousness" in note.lower()
    assert reason == "red_flags_detected"


def test_red_flag_unknown_gets_generic_note():
    """Unrecognized red flag should get the generic note."""
    extraction = ExtractionResult(
        transcript="Something unusual happened",
        symptoms=[],
        red_flags=["unknown_flag"],
    )
    note, reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id=None,
        reason_code=None,
        image_analysis=None,
    )
    assert note is not None
    assert "some symptoms may be concerning" in note.lower()
    assert reason == "red_flags_detected"


def test_high_severity_symptom_generates_clinician_note():
    """High severity reason code should generate a clinician note."""
    extraction = ExtractionResult(
        transcript="Dizziness is a 7",
        symptoms=[SymptomEntity(symptom="dizziness", severity_1_10=7)],
    )
    note, reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id="generic_high_severity_escalation",
        reason_code="high_severity_symptom",
        image_analysis=None,
    )
    assert note is not None
    assert "severe" in note.lower()
    assert reason == "high_severity_symptom"


def test_nsaid_ace_inhibitor_generates_clinician_note():
    """NSAID + ACE inhibitor reason code should generate a clinician note."""
    extraction = ExtractionResult(
        transcript="Took ibuprofen for knee pain",
        symptoms=[SymptomEntity(symptom="knee pain")],
    )
    note, reason = clinician_note_for_log(
        extraction=extraction,
        protocol_id="medication_interaction",
        reason_code="nsaid_ace_inhibitor_interaction",
        image_analysis=None,
    )
    assert note is not None
    assert "blood pressure" in note.lower()
    assert reason == "nsaid_ace_inhibitor_interaction"


# --- get_red_flag_note helper ---


def test_get_red_flag_note_known_flag():
    """Known red flag returns targeted note."""
    note = get_red_flag_note(["chest pain"])
    assert "chest pain" in note.lower()
    assert "medical evaluation" in note.lower()


def test_get_red_flag_note_unknown_flag():
    """Unknown red flag returns generic note."""
    note = get_red_flag_note(["mystery_flag"])
    assert "some symptoms may be concerning" in note.lower()


def test_get_red_flag_note_first_known_wins():
    """First matching known flag is used."""
    note = get_red_flag_note(["unknown_one", "passed out", "chest pain"])
    assert "consciousness" in note.lower()
