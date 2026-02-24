"""Deterministic clinician-contact notes for non-diagnostic UX."""

from typing import Optional, Tuple

from ..models import ExtractionResult, ImageAnalysisResult


_REASON_CODE_TO_NOTE = {
    "high_fever_indicator": "Because your temperature appears high or worsening, consider contacting a clinician promptly.",
    "fever_with_respiratory_condition": "Because you have a fever alongside a respiratory condition, consider contacting a clinician promptly.",
    "asthma_respiratory_worsening_or_severe": "If your breathing feels difficult or is worsening, consider contacting a clinician promptly.",
    "asthma_respiratory_sputum_color_change": "Because you're reporting a change in sputum color alongside respiratory symptoms, consider contacting a clinician promptly.",
    "corticosteroid_htn_bp_elevated": "Because corticosteroids can raise blood pressure and yours appears elevated, consider discussing this with your clinician.",
    "corticosteroid_htn_bp_request": "Corticosteroids can affect blood pressure — worth monitoring closely and discussing with your clinician.",
    "compound_escalation_abnormal_vitals": "You're reporting multiple symptoms along with abnormal vital signs. Consider contacting a clinician to discuss.",
    "compound_escalation_medication_nonadherence": "You're reporting multiple symptoms and may have missed medications. Consider contacting a clinician to discuss.",
    "compound_escalation_abnormal_vitals_medication_nonadherence": "You're reporting multiple symptoms, abnormal vital signs, and possible missed medications. Consider contacting a clinician promptly.",
    "headache_severe_worsening_no_relief": "Because this headache appears severe or worsening, consider contacting a clinician promptly.",
    "skin_always_escalate_guardrail": "Consider having a clinician review this skin finding promptly.",
    "skin_lesion_with_fever": "Because this skin finding is paired with fever symptoms, consider contacting a clinician promptly.",
    "skin_lesion_with_compounding_factors": "Consider having a clinician review this skin finding soon.",
    "high_severity_symptom": "Because you're reporting a symptom as quite severe, consider discussing this with your clinician.",
    "nsaid_ace_inhibitor_interaction": "NSAIDs can reduce the effectiveness of blood pressure medications and affect kidney function. Consider discussing this with your clinician.",
}

# Targeted clinician notes for specific red flag types (non-diagnostic).
_RED_FLAG_NOTES = {
    "passed out": "You reported losing consciousness or nearly losing consciousness. Consider seeking prompt medical evaluation.",
    "fainted": "You reported losing consciousness or nearly losing consciousness. Consider seeking prompt medical evaluation.",
    "fainting": "You reported feeling faint or nearly losing consciousness. Consider seeking prompt medical evaluation.",
    "chest pain": "Chest pain can indicate a serious condition. Consider seeking prompt medical evaluation.",
    "seizure": "A seizure warrants prompt medical evaluation.",
    "convulsion": "A seizure warrants prompt medical evaluation.",
    "difficulty breathing": "Significant breathing difficulty warrants prompt medical evaluation.",
    "can't breathe": "Significant breathing difficulty warrants prompt medical evaluation.",
    "severe headache": "A severe or sudden headache warrants prompt medical evaluation.",
    "worst headache": "A severe or sudden headache warrants prompt medical evaluation.",
    "suicidal": "If you're having thoughts of self-harm, please contact a crisis line or emergency services immediately.",
    "want to die": "If you're having thoughts of self-harm, please contact a crisis line or emergency services immediately.",
    "hurt myself": "If you're having thoughts of self-harm, please contact a crisis line or emergency services immediately.",
}
_GENERIC_RED_FLAG_NOTE = "Some symptoms may be concerning. Consider contacting a clinician promptly."


def get_red_flag_note(red_flags: list[str]) -> str:
    """Return a context-appropriate static note for the given red flags."""
    for flag in red_flags:
        if flag in _RED_FLAG_NOTES:
            return _RED_FLAG_NOTES[flag]
    return _GENERIC_RED_FLAG_NOTE


def clinician_note_for_log(
    *,
    extraction: ExtractionResult,
    protocol_id: Optional[str],
    reason_code: Optional[str],
    image_analysis: Optional[ImageAnalysisResult],
) -> Tuple[Optional[str], Optional[str]]:
    """Return a user-facing clinician note and machine reason."""
    if extraction.red_flags:
        # Use a targeted note for specific red flags, fall back to generic
        for flag in extraction.red_flags:
            if flag in _RED_FLAG_NOTES:
                return _RED_FLAG_NOTES[flag], "red_flags_detected"
        return _GENERIC_RED_FLAG_NOTE, "red_flags_detected"

    if reason_code in _REASON_CODE_TO_NOTE:
        return _REASON_CODE_TO_NOTE[reason_code], reason_code

    if protocol_id == "skin_lesion_escalation" and image_analysis and image_analysis.lesion_detected:
        return "Consider having a clinician review this skin finding soon.", "skin_lesion_protocol"

    return None, None
