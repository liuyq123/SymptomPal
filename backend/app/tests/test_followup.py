"""Tests for follow-up question logic."""

from ..models import ExtractionResult, SymptomEntity
from ..services.followup import choose_followup


def test_missing_severity_asks_severity():
    """When severity is missing, should ask about severity."""
    extraction = ExtractionResult(
        transcript="I have a headache",
        symptoms=[SymptomEntity(symptom="headache")],
        missing_fields=["severity", "onset", "duration"],
    )
    result = choose_followup(extraction)
    assert result is not None
    assert "1-10" in result.lower() or "severity" in result.lower()


def test_missing_onset_asks_onset():
    """When only onset is missing (severity present), should ask about onset."""
    extraction = ExtractionResult(
        transcript="I have a headache, severity 7",
        symptoms=[SymptomEntity(symptom="headache", severity_1_10=7)],
        missing_fields=["onset", "duration"],
    )
    result = choose_followup(extraction)
    assert result is not None
    assert "when" in result.lower() or "start" in result.lower()


def test_missing_duration_asks_duration():
    """When only duration is missing, should ask about duration."""
    extraction = ExtractionResult(
        transcript="I have a headache since Tuesday",
        symptoms=[SymptomEntity(symptom="headache", onset_time_text="Tuesday")],
        missing_fields=["duration"],
    )
    result = choose_followup(extraction)
    assert result is not None
    assert "long" in result.lower() or "duration" in result.lower()


def test_no_missing_fields_no_question():
    """When nothing is missing, should return None."""
    extraction = ExtractionResult(
        transcript="Complete info here",
        symptoms=[SymptomEntity(
            symptom="headache",
            severity_1_10=7,
            onset_time_text="Tuesday",
            duration_text="2 hours"
        )],
        missing_fields=[],
    )
    result = choose_followup(extraction)
    assert result is None


def test_priority_severity_over_onset():
    """Severity should be asked before onset."""
    extraction = ExtractionResult(
        transcript="I have a headache",
        symptoms=[SymptomEntity(symptom="headache")],
        missing_fields=["onset", "severity"],  # Both missing
    )
    result = choose_followup(extraction)
    assert result is not None
    # Should ask severity first (priority)
    assert "1-10" in result.lower() or "severe" in result.lower()
