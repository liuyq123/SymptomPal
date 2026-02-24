"""Unit tests for protocolized follow-up decisions."""

from datetime import datetime, timezone

from ..models import (
    ActionEntity, ExtractionResult, SymptomEntity, UserProfile,
    ImageAnalysisResult, SkinLesionDescription, VitalSignEntry,
)
from ..services.protocols import ProtocolContext, get_protocol_registry


def _profile(*conditions: str) -> UserProfile:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return UserProfile(
        user_id="protocol_test_user",
        conditions=list(conditions),
        allergies=[],
        regular_medications=[],
        patterns=[],
        health_summary=None,
        created_at=now,
        updated_at=now,
    )


def test_priority_medication_missing_dose_outranks_symptom_protocols():
    # After protocol reorder, safety-critical protocols (fever, respiratory) take
    # priority over dose-missing questions.  With an Asthma profile + respiratory
    # symptom missing, SpO2 request fires first.  Use a non-respiratory profile
    # with no fever to verify MedicationMissingDosePriority still fires.
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="My knee hurts, I took ibuprofen.",
            symptoms=[SymptomEntity(symptom="knee pain")],
            actions_taken=[ActionEntity(name="ibuprofen", dose_text=None)],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        user_profile=_profile(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "medication_missing_dose_priority"
    assert "dose of ibuprofen" in (decision.immediate_question or "").lower()


def test_fever_protocol_asks_temperature_when_missing():
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I have fever and chills",
            symptoms=[SymptomEntity(symptom="fever")],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "fever_protocol"
    assert decision.immediate_question == "What was your temperature?"


def test_asthma_respiratory_protocol_asks_spo2_when_missing():
    """Without SpO2 data, respiratory protocol should prioritize SpO2 request."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I've been coughing all day",
            symptoms=[SymptomEntity(symptom="cough")],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        user_profile=_profile("Asthma"),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "asthma_respiratory_protocol"
    assert "oxygen" in (decision.immediate_question or "").lower()


def test_asthma_respiratory_protocol_asks_inhaler_when_spo2_present():
    """With SpO2 already provided, respiratory protocol falls through to inhaler question."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I've been coughing all day, spo2 is 95%",
            symptoms=[SymptomEntity(symptom="cough")],
            vital_signs=[VitalSignEntry(name="spo2", value="95", unit="%")],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        user_profile=_profile("Asthma"),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "asthma_respiratory_protocol"
    assert "rescue inhaler" in (decision.immediate_question or "").lower()


def test_copd_patient_hits_respiratory_protocol():
    """COPD patients should get respiratory-specific follow-up (SpO2 request when missing)."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Coughing more today, chest is tight",
            symptoms=[SymptomEntity(symptom="cough")],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        user_profile=_profile("COPD (moderate, GOLD Stage II)"),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "asthma_respiratory_protocol"
    assert "oxygen" in (decision.immediate_question or "").lower()


def test_headache_protocol_asks_severity_when_missing():
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I have a bad headache",
            symptoms=[SymptomEntity(symptom="headache")],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "headache_protocol"
    assert "1-10" in (decision.immediate_question or "")


# --- Skin Lesion Escalation Protocol Tests ---

def _lesion_result(
    desc: str = "Erythematous circular lesion, approx 2cm",
    predicted_condition: str | None = None,
    condition_confidence: float | None = None,
) -> ImageAnalysisResult:
    return ImageAnalysisResult(
        clinical_description=desc,
        confidence=0.8,
        lesion_detected=True,
        skin_lesion=SkinLesionDescription(
            lesion_type="circular lesion",
            color="erythematous",
            size_estimate="approximately 2cm",
            texture="smooth",
            predicted_condition=predicted_condition,
            condition_confidence=condition_confidence,
            confidence_scores={"type": 0.8, "color": 0.7, "size": 0.6, "texture": 0.5},
        ),
    )


def test_skin_lesion_with_fever_triggers_fever_protocol_first():
    """Fever protocol has higher priority than skin lesion protocol."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I have a rash and a fever",
            symptoms=[SymptomEntity(symptom="rash"), SymptomEntity(symptom="fever")],
        ),
        user_id="protocol_test_user",
        image_analysis=_lesion_result(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "fever_protocol"


def test_skin_lesion_with_high_severity_escalates():
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="My skin lesion is really painful",
            symptoms=[SymptomEntity(symptom="skin pain", severity_1_10=8)],
        ),
        user_id="protocol_test_user",
        image_analysis=_lesion_result(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "skin_lesion_escalation"
    assert decision.schedule_checkin is True
    assert decision.reason_code == "skin_lesion_with_compounding_factors"


def test_skin_lesion_with_worsening_escalates():
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="My rash is getting worse and spreading",
            symptoms=[SymptomEntity(symptom="rash")],
        ),
        user_id="protocol_test_user",
        image_analysis=_lesion_result(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "skin_lesion_escalation"
    assert decision.schedule_checkin is True


def test_skin_lesion_without_compounding_factors_no_match():
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I noticed a small mark on my arm",
            symptoms=[SymptomEntity(symptom="skin mark")],
        ),
        user_id="protocol_test_user",
        image_analysis=_lesion_result(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id != "skin_lesion_escalation"


def test_no_image_analysis_skips_skin_protocol():
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I have a bad rash that is getting worse",
            symptoms=[SymptomEntity(symptom="rash", severity_1_10=8)],
        ),
        user_id="protocol_test_user",
        image_analysis=None,
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id != "skin_lesion_escalation"


# --- Melanoma Guardrail Tests ---

def test_melanoma_always_escalates():
    """Melanoma guardrail fires even without compounding factors."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I noticed a mole on my back",
            symptoms=[SymptomEntity(symptom="skin mole")],
        ),
        user_id="protocol_test_user",
        image_analysis=_lesion_result(
            predicted_condition="suspicious mole or possible melanoma",
            condition_confidence=0.65,
        ),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "skin_lesion_escalation"
    assert decision.escalation_flag is True
    assert decision.reason_code == "skin_always_escalate_guardrail"
    assert decision.schedule_checkin is True
    assert decision.checkin_hours == 1


def test_non_melanoma_condition_no_guardrail():
    """Eczema is not in ALWAYS_ESCALATE_CONDITIONS, so without compounding factors, no match."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I have a dry itchy patch on my arm",
            symptoms=[SymptomEntity(symptom="skin patch")],
        ),
        user_id="protocol_test_user",
        image_analysis=_lesion_result(
            predicted_condition="eczema or atopic dermatitis",
            condition_confidence=0.72,
        ),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id != "skin_lesion_escalation"


def test_compounding_factors_still_work_with_condition():
    """Existing compounding-factor path works even when a non-serious condition is predicted."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="My rash is getting worse and spreading",
            symptoms=[SymptomEntity(symptom="rash")],
        ),
        user_id="protocol_test_user",
        image_analysis=_lesion_result(
            predicted_condition="contact dermatitis",
            condition_confidence=0.50,
        ),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "skin_lesion_escalation"
    assert decision.reason_code == "skin_lesion_with_compounding_factors"


def test_protocol_checkin_messages_are_clinician_only():
    forbidden_terms = ("urgent care", "emergency", "call 911")

    cases = [
        ProtocolContext(
            extraction=ExtractionResult(
                transcript="My fever is 103 F and getting worse.",
                symptoms=[SymptomEntity(symptom="fever", severity_1_10=8)],
            ),
            user_id="protocol_test_user",
            symptom_history={},
        ),
        ProtocolContext(
            extraction=ExtractionResult(
                transcript="Breathing has been difficult all day.",
                symptoms=[SymptomEntity(symptom="shortness of breath", severity_1_10=8)],
            ),
            user_id="protocol_test_user",
            user_profile=_profile("Asthma"),
            symptom_history={},
        ),
        ProtocolContext(
            extraction=ExtractionResult(
                transcript="Headache is worse and keeps building.",
                symptoms=[SymptomEntity(symptom="headache", severity_1_10=8)],
            ),
            user_id="protocol_test_user",
            symptom_history={
                "symptom_counts_24h": {"headache": 4},
                "symptom_counts_yesterday": {"headache": 2},
            },
        ),
        ProtocolContext(
            extraction=ExtractionResult(
                transcript="I have a rash and fever at 103 F.",
                symptoms=[SymptomEntity(symptom="rash"), SymptomEntity(symptom="fever", severity_1_10=8)],
            ),
            user_id="protocol_test_user",
            image_analysis=_lesion_result(),
            symptom_history={},
        ),
    ]

    # Include new protocol scenarios
    cases.append(
        ProtocolContext(
            extraction=ExtractionResult(
                transcript="Blood pressure 185 over 125, having a headache",
                symptoms=[SymptomEntity(symptom="headache", severity_1_10=6)],
                vital_signs=[VitalSignEntry(name="blood pressure", value="185/125", unit="mmHg")],
            ),
            user_id="protocol_test_user",
            symptom_history={},
        ),
    )
    cases.append(
        ProtocolContext(
            extraction=ExtractionResult(
                transcript="Stomach pain and blood in stool for 3 days",
                symptoms=[SymptomEntity(symptom="stomach pain", severity_1_10=7)],
            ),
            user_id="protocol_test_user",
            symptom_history={},
        ),
    )

    for context in cases:
        decision = get_protocol_registry().evaluate(context)
        if not decision.schedule_checkin:
            continue
        assert decision.checkin_message is not None
        message = decision.checkin_message.lower()
        for term in forbidden_terms:
            assert term not in message


# --- Hypertension Protocol Tests ---


def test_hypertension_protocol_asks_symptoms_on_elevated_bp():
    """Elevated BP without associated symptoms should ask about them."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Blood pressure 152 over 94 today",
            symptoms=[],
            vital_signs=[VitalSignEntry(name="blood pressure", value="152/94", unit="mmHg")],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "hypertension_protocol"
    assert decision.immediate_question is not None
    assert "headache" in decision.immediate_question.lower() or "vision" in decision.immediate_question.lower()
    assert decision.reason_code == "hypertension_elevated_ask_symptoms"


def test_hypertension_protocol_schedules_checkin_with_symptoms():
    """Elevated BP with associated symptoms should schedule a check-in."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Blood pressure 155 over 98 and I have a headache",
            symptoms=[SymptomEntity(symptom="headache")],
            vital_signs=[VitalSignEntry(name="blood pressure", value="155/98", unit="mmHg")],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "hypertension_protocol"
    assert decision.schedule_checkin is True
    assert decision.reason_code == "hypertension_elevated_with_symptoms"


def test_hypertension_protocol_escalates_crisis_level():
    """BP >= 180/120 should set escalation_flag."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Blood pressure 185 over 122",
            symptoms=[],
            vital_signs=[VitalSignEntry(name="blood pressure", value="185/122", unit="mmHg")],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "hypertension_protocol"
    assert decision.escalation_flag is True


def test_hypertension_protocol_no_match_normal_bp():
    """Normal BP should not trigger hypertension protocol."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Blood pressure 118 over 76 today",
            symptoms=[],
            vital_signs=[VitalSignEntry(name="blood pressure", value="118/76", unit="mmHg")],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id != "hypertension_protocol"


# --- Gastrointestinal Protocol Tests ---


def test_gi_protocol_asks_hydration_on_persistent_symptoms():
    """3+ GI symptom logs in 7 days should trigger hydration question."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Still having diarrhea today",
            symptoms=[SymptomEntity(symptom="diarrhea")],
        ),
        user_id="protocol_test_user",
        symptom_history={
            "symptom_counts_7d": {"diarrhea": 4},
        },
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "gastrointestinal_protocol"
    assert "hydration" in (decision.immediate_question or "").lower() or "fluids" in (decision.immediate_question or "").lower()
    assert decision.reason_code == "gi_persistent_ask_hydration"


def test_gi_protocol_escalates_on_blood_in_stool():
    """Blood in stool is a red flag and should escalate."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="I've had stomach pain and noticed blood in stool",
            symptoms=[SymptomEntity(symptom="stomach pain")],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "gastrointestinal_protocol"
    assert decision.escalation_flag is True
    assert decision.schedule_checkin is True
    assert decision.reason_code == "gi_red_flag_blood"


def test_gi_protocol_fever_routes_to_fever_protocol():
    """GI symptoms + fever should route to FeverProtocol (higher priority)."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Throwing up all day and I have a fever",
            symptoms=[
                SymptomEntity(symptom="vomiting", severity_1_10=7),
                SymptomEntity(symptom="fever"),
            ],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    # Fever protocol has higher priority in the registry
    assert decision.protocol_id == "fever_protocol"


def test_gi_protocol_no_match_mild_single_episode():
    """A single mild GI episode should not trigger the protocol."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Felt a bit nauseous this morning",
            symptoms=[SymptomEntity(symptom="nausea", severity_1_10=3)],
        ),
        user_id="protocol_test_user",
        symptom_history={
            "symptom_counts_7d": {"nausea": 1},
        },
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id != "gastrointestinal_protocol"


# --- Chest tightness respiratory protocol tests ---


def test_chest_tightness_matches_respiratory_protocol():
    """Chest tightness in a COPD patient should match AsthmaRespiratoryProtocol."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="My chest is tight, like there's a band around it",
            symptoms=[SymptomEntity(symptom="chest tightness")],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        user_profile=_profile("COPD (moderate, GOLD Stage II)"),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "asthma_respiratory_protocol"
    assert "oxygen" in (decision.immediate_question or "").lower()


# --- Fever protocol vital signs check ---


def test_fever_protocol_uses_vital_sign_temperature():
    """FeverProtocol should not ask for temp when it's already in vital signs."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Fever. Took my temperature because I was sweating.",
            symptoms=[SymptomEntity(symptom="fever")],
            vital_signs=[VitalSignEntry(name="temperature", value="101.2", unit="F")],
            missing_fields=[],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "fever_protocol"
    # Should NOT ask "What was your temperature?" since it's already provided
    assert decision.immediate_question is None or "temperature" not in decision.immediate_question.lower()


def test_fever_with_copd_schedules_checkin():
    """Moderate fever (101+) in COPD patient should schedule check-in."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Fever. 101.2. Still coughing.",
            symptoms=[SymptomEntity(symptom="fever")],
            vital_signs=[VitalSignEntry(name="temperature", value="101.2", unit="F")],
            missing_fields=[],
        ),
        user_id="protocol_test_user",
        user_profile=_profile("COPD (moderate, GOLD Stage II)"),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "fever_protocol"
    assert decision.schedule_checkin is True
    assert decision.escalation_flag is True
    assert decision.reason_code == "fever_with_respiratory_condition"


# --- Menstrual cycle protocol cooldown ---


def test_cycle_protocol_suppresses_after_recent_fire():
    """Cycle awareness question should be suppressed if already asked recently."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Pelvic pain again",
            symptoms=[SymptomEntity(symptom="pelvic pain", severity_1_10=6)],
        ),
        user_id="protocol_test_user",
        symptom_history={
            "cycle_context": {
                "cycle_day": 24,
                "cycle_phase": "luteal",
                "cycle_number": 1,
                "has_prior_correlation": False,
            },
        },
        recent_protocol_ids=["menstrual_cycle_protocol"],
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "menstrual_cycle_protocol"
    # Question suppressed (no immediate_question)
    assert decision.immediate_question is None
    assert decision.reason_code == "cycle_phase_symptom_awareness_suppressed"


def test_cycle_protocol_fires_on_correlation_despite_cooldown():
    """Cycle correlation detected should ALWAYS fire regardless of cooldown."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Pelvic pain again",
            symptoms=[SymptomEntity(symptom="pelvic pain", severity_1_10=7)],
        ),
        user_id="protocol_test_user",
        symptom_history={
            "cycle_context": {
                "cycle_day": 28,
                "cycle_phase": "luteal",
                "cycle_number": 2,
                "has_prior_correlation": True,
            },
        },
        recent_protocol_ids=["menstrual_cycle_protocol", "menstrual_cycle_protocol"],
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "menstrual_cycle_protocol"
    assert decision.immediate_question is not None
    assert "pattern" in decision.immediate_question.lower()
    assert decision.reason_code == "cycle_correlation_detected"


# --- Medication dose matching ---


def test_medication_dose_skipped_for_profile_medications():
    """Medication with dose known from profile should not trigger dose question."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Took my Spiriva and Advair this morning",
            symptoms=[SymptomEntity(symptom="cough")],
            actions_taken=[
                ActionEntity(name="Spiriva", dose_text=None),
            ],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        user_profile=UserProfile(
            user_id="protocol_test_user",
            conditions=["COPD"],
            allergies=[],
            regular_medications=["Tiotropium (Spiriva) 18mcg inhaler once daily"],
            patterns=[],
            health_summary=None,
            created_at=now,
            updated_at=now,
        ),
        symptom_history={},
        known_medication_doses={"tiotropium": "18mcg", "spiriva": "18mcg"},
    )
    decision = get_protocol_registry().evaluate(context)
    # Should NOT be medication_missing_dose (Spiriva dose is known from profile)
    assert decision.protocol_id != "medication_missing_dose_priority"


def test_medication_dose_skipped_for_finished_medication():
    """Finished medication should not trigger dose question."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Finished the prednisone yesterday. Feeling better.",
            symptoms=[SymptomEntity(symptom="cough")],
            actions_taken=[
                ActionEntity(name="prednisone", dose_text=None),
            ],
            missing_fields=[],
        ),
        user_id="protocol_test_user",
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id != "medication_missing_dose_priority"


# --- Headache beats medication_missing_dose_priority ---


def test_headache_protocol_beats_missing_dose():
    """Headache protocol should fire even when a medication is missing its dose."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Had a bad headache after lunch so I took some ibuprofen.",
            symptoms=[SymptomEntity(symptom="headache")],
            actions_taken=[ActionEntity(name="ibuprofen", dose_text=None)],
            missing_fields=["severity"],
        ),
        user_id="protocol_test_user",
        user_profile=_profile(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "headache_protocol"


# --- High severity general escalation ---


def test_high_severity_escalation_fires_at_7():
    """Severity >= 7 for any symptom should trigger escalation."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Dizziness is really bad, a 7 out of 10",
            symptoms=[SymptomEntity(symptom="dizziness", severity_1_10=7)],
        ),
        user_id="protocol_test_user",
        user_profile=_profile(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "generic_high_severity_escalation"
    assert decision.escalation_flag is True
    assert decision.schedule_checkin is True
    assert decision.reason_code == "high_severity_symptom"


def test_high_severity_escalation_does_not_fire_at_6():
    """Severity 6 should NOT trigger the high severity escalation."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Dizziness is moderate, a 6 out of 10",
            symptoms=[SymptomEntity(symptom="dizziness", severity_1_10=6)],
        ),
        user_id="protocol_test_user",
        user_profile=_profile(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id != "generic_high_severity_escalation"


# --- NSAID + ACE inhibitor interaction ---


def test_nsaid_ace_inhibitor_interaction_fires():
    """Ibuprofen + Lisinopril in profile should trigger medication interaction."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Knee hurts, took ibuprofen 400mg.",
            symptoms=[SymptomEntity(symptom="knee pain")],
            actions_taken=[ActionEntity(name="ibuprofen", dose_text="400mg")],
        ),
        user_id="protocol_test_user",
        user_profile=UserProfile(
            user_id="protocol_test_user",
            conditions=["Hypertension"],
            allergies=[],
            regular_medications=["Lisinopril 10mg daily"],
            patterns=[],
            health_summary=None,
            created_at=now,
            updated_at=now,
        ),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.protocol_id == "medication_interaction"
    assert decision.reason_code == "nsaid_ace_inhibitor_interaction"
    assert "blood pressure" in (decision.immediate_question or "").lower()


def test_nsaid_without_ace_inhibitor_no_interaction():
    """Ibuprofen without ACE inhibitor should not trigger the NSAID+ACE path."""
    context = ProtocolContext(
        extraction=ExtractionResult(
            transcript="Took ibuprofen 400mg for my knee.",
            symptoms=[SymptomEntity(symptom="knee pain")],
            actions_taken=[ActionEntity(name="ibuprofen", dose_text="400mg")],
        ),
        user_id="protocol_test_user",
        user_profile=_profile(),
        symptom_history={},
    )
    decision = get_protocol_registry().evaluate(context)
    assert decision.reason_code != "nsaid_ace_inhibitor_interaction"


# --- Protocol registry priority index ---


def test_protocol_registry_priority_index():
    """get_priority_index returns correct ordering."""
    registry = get_protocol_registry()
    assert registry.get_priority_index("medication_interaction") < registry.get_priority_index("headache_protocol")
    assert registry.get_priority_index("headache_protocol") < registry.get_priority_index("medication_missing_dose_priority")
    assert registry.get_priority_index("generic_high_severity_escalation") < registry.get_priority_index("generic_severity_fallback")
    # Unknown protocol gets lowest priority
    assert registry.get_priority_index("nonexistent_protocol") == len(registry.protocols)
