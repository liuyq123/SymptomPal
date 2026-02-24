"""
Protocolized follow-up rules for hybrid agent behavior.

The registry is evaluated in order (first match wins) to keep behavior deterministic.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models import CheckinType, ExtractionResult, ImageAnalysisResult, UserProfile


ASTHMA_LIKE_TERMS = {
    "asthma",
    "reactive airway",
    "bronchospasm",
    "copd",
    "chronic obstructive",
    "emphysema",
    "chronic bronchitis",
}

RESPIRATORY_TERMS = {
    "cough",
    "coughing",
    "wheeze",
    "wheezing",
    "shortness of breath",
    "breathing difficult",
    "hard to breathe",
    "chest tightness",
    "chest tight",
    "tight chest",
    "difficulty breathing",
    "can't breathe",
    "breathless",
    "dyspnea",
    "winded",
    "out of breath",
    "phlegm",
    "sputum",
}

HEADACHE_TERMS = {
    "headache",
    "migraine",
    "head pain",
}

FEVER_TERMS = {
    "fever",
    "temperature",
    "chills",
}

HIGH_PRIORITY_SEVERITY_TERMS = {
    "pain",
    "headache",
    "migraine",
    "chest pain",
    "abdominal pain",
    "nausea",
    "dizziness",
    "queasy", "nauseous",
    "woozy", "lightheaded",
    "tired", "exhausted", "fatigue", "wiped out", "drained",
    "bloated", "bloating",
    "ache", "cramps",
    "winded", "breathless",
}

HYPERTENSION_SYMPTOM_TERMS = {
    "headache",
    "head pain",
    "vision changes",
    "blurred vision",
    "chest pain",
    "chest tightness",
    "dizziness",
    "lightheaded",
    "nosebleed",
}

GI_PERSISTENT_TERMS = {
    "nausea", "nauseous", "vomiting", "diarrhea", "diarrhoea",
    "stomach pain", "stomach ache", "abdominal pain",
    "upset stomach", "bloating", "cramping",
}

# Defense-in-depth: vital signs/measurements should never trigger severity questions.
NON_SYMPTOM_VITAL_SIGNS = {
    "sugar", "glucose", "blood sugar", "fasting sugar", "blood pressure",
    "bp", "heart rate", "bpm", "pulse", "oxygen", "saturation", "spo2",
    "weight", "temperature",
}

# Defense-in-depth: patterns that indicate a non-medication action.
# Even if extraction mislabels food/lifestyle as an action, the protocol won't fire.
NON_MEDICATION_PATTERNS = {
    "eating", "drinking", "breakfast", "lunch", "dinner", "food", "meal",
    "juice", "water", "tea", "coffee", "milk",
    "skipped", "resting", "sleeping", "walking",
    "consulted", "called doctor", "saw doctor", "dark room",
    "fasting sugar", "fasting",
    "meds", "pills", "medication", "medications", "medicine", "medicines",
    "my pills", "my meds", "my medication",
}

# Supplements and OTC items where exact dosing is clinically low-priority.
LOW_PRIORITY_MEDICATION_PATTERNS = {
    "multivitamin", "multi-vitamin", "vitamin", "vitamin d", "vitamin c",
    "vitamin b", "vitamin b12", "vitamin e", "vitamin a",
    "supplement", "supplements",
    "fish oil", "omega-3", "omega 3", "coq10", "probiotics", "probiotic",
    "melatonin", "zinc", "magnesium",
    "nicotine patch", "nicotine gum", "nicotine lozenge", "nicotine",
    "fiber", "fibre", "metamucil", "psyllium",
    "collagen", "biotin", "turmeric", "glucosamine",
    "eye drops", "eyedrops", "nasal spray", "saline spray",
}

# Corticosteroids that can elevate blood pressure in hypertensive patients.
CORTICOSTEROID_NAMES = {
    "prednisone", "prednisolone", "dexamethasone", "methylprednisolone",
    "hydrocortisone", "cortisone", "betamethasone",
}

# ACE inhibitors — NSAID co-administration increases renal risk and blunts antihypertensive effect.
ACE_INHIBITOR_NAMES = {
    "lisinopril", "enalapril", "ramipril", "benazepril", "captopril",
    "fosinopril", "quinapril", "perindopril", "trandolapril", "moexipril",
}

HYPERTENSION_CONDITION_TERMS = {"hypertension", "high blood pressure", "htn"}

STEROID_HTN_SYMPTOM_TERMS = {
    "headache", "head pain", "facial flushing", "flushing",
    "dizziness", "dizzy", "lightheaded", "wired", "jittery",
    "vision changes", "blurred vision",
}

# Sputum color change indicators (possible infection in COPD/asthma).
SPUTUM_COLOR_CHANGE_TERMS = {
    "green sputum", "yellow sputum", "purulent", "bloody sputum",
    "rusty sputum", "brown sputum", "green mucus", "yellow mucus",
    "green phlegm", "yellow phlegm", "blood in sputum", "coughing blood",
    "hemoptysis",
}
# Color words that indicate infection when found in a phlegm/sputum symptom's character field.
SPUTUM_CONCERNING_COLORS = {"green", "yellow", "brown", "rusty", "bloody", "purulent"}
SPUTUM_SYMPTOM_NAMES = {"phlegm", "sputum", "mucus"}

# Worsening language that distinguishes active exacerbation from chronic baseline.
RESPIRATORY_WORSENING_LANGUAGE = {
    "worse", "worsened", "worsening", "getting bad", "getting worse",
    "more than usual", "increased", "not normal", "unusual",
    "different", "harder to breathe", "struggling", "can't catch",
    "flare", "flaring", "exacerbation", "attack",
    "new symptom", "never had", "first time",
}

# Normalcy language that suggests a symptom is the patient's established baseline.
RESPIRATORY_NORMALCY_LANGUAGE = {
    "usual", "my usual", "normal", "same as always", "back to normal",
    "my usual deal", "morning routine", "always there", "usual morning",
    "nothing weird", "nothing new", "same old", "typical",
}

# Medication non-adherence signals for compound escalation.
MEDICATION_NONADHERENCE_TERMS = {
    "skipped", "missed", "forgot", "didn't take", "ran out",
    "stopped taking", "haven't taken", "out of",
}

# Contextual root-cause questions for recurring symptoms (replaces repeated severity asks).
RECURRING_SYMPTOM_CONTEXT_QUESTIONS = [
    ({"nausea", "nauseous", "queasy"},
     "Are you taking your medications with food, or on an empty stomach?"),
    ({"headache", "head pain"},
     "Is this the same type of headache as before, or does it feel different?"),
    ({"dizziness", "dizzy", "lightheaded", "woozy"},
     "Does the dizziness happen mainly when you stand up or change position?"),
    ({"fatigue", "tired", "exhausted", "wiped out", "drained"},
     "How has your sleep been lately? Are you getting enough rest?"),
    ({"bloating", "bloated"},
     "Have you noticed if certain foods seem to trigger the bloating?"),
    ({"pain", "ache"},
     "Is there a particular activity or time of day when the pain is worst?"),
]


def _get_recurring_context_question(symptom_label: str) -> Optional[str]:
    """Return a contextual root-cause question for a recurring symptom, or None."""
    symptom_lower = symptom_label.lower()
    for terms, question in RECURRING_SYMPTOM_CONTEXT_QUESTIONS:
        if any(term in symptom_lower for term in terms):
            return question
    return None


def _looks_like_medication(action_name: str) -> bool:
    """Return True if the action name plausibly refers to a clinically significant medication."""
    name_lower = action_name.strip().lower()
    if len(name_lower) <= 2:
        return False
    if any(p in name_lower for p in NON_MEDICATION_PATTERNS):
        return False
    if any(p in name_lower for p in LOW_PRIORITY_MEDICATION_PATTERNS):
        return False
    return True


class ProtocolContext(BaseModel):
    """Structured inputs for protocol evaluation."""

    extraction: ExtractionResult
    user_id: str
    user_profile: Optional[UserProfile] = None
    symptom_history: Dict[str, Any] = Field(default_factory=dict)
    image_analysis: Optional[ImageAnalysisResult] = None
    known_medication_doses: Dict[str, str] = Field(default_factory=dict)
    recent_protocol_ids: List[str] = Field(default_factory=list)


class ProtocolDecision(BaseModel):
    """Single decision emitted by a protocol."""

    protocol_id: Optional[str] = None
    immediate_question: Optional[str] = None
    schedule_checkin: bool = False
    checkin_type: Optional[CheckinType] = None
    checkin_hours: Optional[int] = None
    checkin_message: Optional[str] = None
    escalation_flag: bool = False
    reason_code: str = "no_protocol_match"

    def has_followup(self) -> bool:
        return bool(self.immediate_question) or self.schedule_checkin


class FollowupProtocol(ABC):
    """Protocol interface."""

    protocol_id: str

    @abstractmethod
    def matches(self, context: ProtocolContext) -> bool:
        raise NotImplementedError

    @abstractmethod
    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        raise NotImplementedError


def _contains_any(text: str, candidates: set[str]) -> bool:
    text_l = text.lower()
    return any(token in text_l for token in candidates)


def _current_symptom_names(context: ProtocolContext) -> list[str]:
    return [s.symptom.lower() for s in context.extraction.symptoms]


def _is_severity_missing(context: ProtocolContext) -> bool:
    if "severity" in context.extraction.missing_fields:
        return True
    # Also treat as missing if all symptoms have severity_1_10 == None
    if context.extraction.symptoms and all(
        s.severity_1_10 is None for s in context.extraction.symptoms
    ):
        return True
    return False


def _max_severity_for_terms(context: ProtocolContext, terms: set[str]) -> Optional[int]:
    severities: list[int] = []
    for symptom in context.extraction.symptoms:
        symptom_name = symptom.symptom.lower()
        if any(term in symptom_name for term in terms) and symptom.severity_1_10 is not None:
            severities.append(symptom.severity_1_10)
    return max(severities) if severities else None


def _history_count(history: Dict[str, Any], key: str, symptom_terms: set[str]) -> int:
    counter = history.get(key) or {}
    total = 0
    for term in symptom_terms:
        total += int(counter.get(term, 0))
    return total


def _is_worsening(history: Dict[str, Any], symptom_terms: set[str]) -> bool:
    recent = _history_count(history, "symptom_counts_24h", symptom_terms)
    yesterday = _history_count(history, "symptom_counts_yesterday", symptom_terms)
    return recent > yesterday and yesterday > 0


def _has_relief_context(extraction: ExtractionResult) -> bool:
    transcript = extraction.transcript.lower()
    relief_terms = ["better", "improved", "relief", "helped", "resolved", "gone"]

    if any(term in transcript for term in relief_terms):
        return True

    for action in extraction.actions_taken:
        if action.effect_text and any(term in action.effect_text.lower() for term in relief_terms):
            return True
    return False


def _extract_temperature_f(transcript: str) -> Optional[float]:
    """Extract explicit temperature from transcript and convert to Fahrenheit."""
    text = transcript.lower()
    match = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(°?\s*[fc])", text)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).replace(" ", "").replace("°", "")
    if unit == "c":
        return (value * 9.0 / 5.0) + 32.0
    return value


def _has_migraine_history(context: ProtocolContext) -> bool:
    if not context.user_profile:
        return False
    return any("migraine" in condition.lower() for condition in context.user_profile.conditions)


def _has_asthma_like_condition(context: ProtocolContext) -> bool:
    if not context.user_profile:
        return False
    for condition in context.user_profile.conditions:
        lower = condition.lower()
        if any(term in lower for term in ASTHMA_LIKE_TERMS):
            return True
    return False


def _first_symptom_label(context: ProtocolContext) -> str:
    if context.extraction.symptoms:
        return context.extraction.symptoms[0].symptom
    return "symptom"


_MED_FINISHED_MARKERS = {
    "finished", "completed", "done with", "stopped", "last pill",
    "ran out", "ended", "finished yesterday", "last dose",
}
_MED_NOT_STARTED_MARKERS = {
    "haven't started", "bought", "not yet", "going to start",
    "picked up", "haven't opened", "plan to start",
}


def _is_med_inactive_in_transcript(action_name: str, transcript: str) -> bool:
    """Check if the transcript indicates this medication is finished or not yet started."""
    t = transcript.lower()
    name_lower = action_name.strip().lower()
    # Find approximate region around the medication name
    idx = t.find(name_lower)
    if idx < 0:
        # Check the whole transcript for markers
        return (any(marker in t for marker in _MED_FINISHED_MARKERS)
                or any(marker in t for marker in _MED_NOT_STARTED_MARKERS))
    # Check a window around the medication mention
    start = max(0, idx - 60)
    end = min(len(t), idx + len(name_lower) + 60)
    window = t[start:end]
    return (any(marker in window for marker in _MED_FINISHED_MARKERS)
            or any(marker in window for marker in _MED_NOT_STARTED_MARKERS))


def _is_known_dose_fuzzy(action_name: str, known_doses: Dict[str, str], user_profile) -> bool:
    """Check if a medication's dose is already known, using fuzzy matching against profile."""
    name_lower = action_name.strip().lower()
    # Exact match
    if name_lower in known_doses:
        return True
    # Fuzzy match: check if the action name appears in any profile medication string
    if user_profile and user_profile.regular_medications:
        for med_str in user_profile.regular_medications:
            med_lower = med_str.lower()
            if name_lower in med_lower:
                return True  # Known regular medication — don't nag about dose
    return False


class MedicationMissingDosePriority(FollowupProtocol):
    protocol_id = "medication_missing_dose_priority"

    def _needs_dose(self, action, known_doses: Dict[str, str], context: ProtocolContext) -> bool:
        """True if this action is a real medication with an unknown dose."""
        if not action.name or action.dose_text:
            return False
        if not _looks_like_medication(action.name):
            return False
        # Skip if dose already known from profile (including fuzzy matching)
        if _is_known_dose_fuzzy(action.name, known_doses, context.user_profile):
            return False
        # Skip if medication is finished or not yet started
        if _is_med_inactive_in_transcript(action.name, context.extraction.transcript):
            return False
        return True

    def matches(self, context: ProtocolContext) -> bool:
        return any(
            self._needs_dose(action, context.known_medication_doses, context)
            for action in context.extraction.actions_taken
        )

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        action = next(
            (a for a in context.extraction.actions_taken
             if self._needs_dose(a, context.known_medication_doses, context)),
            None,
        )
        if action is None:
            return ProtocolDecision(protocol_id=self.protocol_id, reason_code="medication_missing_dose_not_found")
        return ProtocolDecision(
            protocol_id=self.protocol_id,
            immediate_question=f"What dose of {action.name} did you take?",
            reason_code="medication_missing_dose",
        )


class FeverProtocol(FollowupProtocol):
    protocol_id = "fever_protocol"

    def matches(self, context: ProtocolContext) -> bool:
        if _contains_any(context.extraction.transcript, FEVER_TERMS):
            return True
        return any(_contains_any(symptom, FEVER_TERMS) for symptom in _current_symptom_names(context))

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        explicit_temp_f = _extract_temperature_f(context.extraction.transcript)

        # Also check extracted vital signs for temperature
        if explicit_temp_f is None:
            for vs in context.extraction.vital_signs:
                if vs.name and vs.name.lower() in ("temperature", "temp"):
                    try:
                        explicit_temp_f = float(vs.value)
                    except (ValueError, TypeError):
                        pass

        transcript = context.extraction.transcript.lower()
        high_fever_hint = "high fever" in transcript or "burning up" in transcript

        if explicit_temp_f is None:
            escalation = high_fever_hint
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                immediate_question="What was your temperature?",
                escalation_flag=escalation,
                reason_code="fever_missing_temperature",
            )

        if explicit_temp_f >= 103.0 or high_fever_hint:
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                schedule_checkin=True,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                checkin_hours=1,
                checkin_message="How is your fever now? If it is rising, worsening, or accompanied by concerning symptoms, please contact a clinician promptly.",
                escalation_flag=True,
                reason_code="high_fever_indicator",
            )

        # Moderate fever (101+) in a patient with respiratory conditions warrants check-in
        if explicit_temp_f >= 101.0 and _has_asthma_like_condition(context):
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                schedule_checkin=True,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                checkin_hours=2,
                checkin_message="Checking back on your fever. If it rises or breathing worsens, please contact a clinician promptly.",
                escalation_flag=True,
                reason_code="fever_with_respiratory_condition",
            )

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            reason_code="fever_no_followup_needed",
        )


class AsthmaRespiratoryProtocol(FollowupProtocol):
    protocol_id = "asthma_respiratory_protocol"

    def matches(self, context: ProtocolContext) -> bool:
        if not _has_asthma_like_condition(context):
            return False
        return any(_contains_any(symptom, RESPIRATORY_TERMS) for symptom in _current_symptom_names(context))

    def _has_spo2_in_extraction(self, context: ProtocolContext) -> bool:
        """Check if SpO2 was reported in the current log's vital signs."""
        for vs in context.extraction.vital_signs:
            if vs.name and ("spo2" in vs.name.lower() or "oxygen" in vs.name.lower()):
                return True
        return False

    def _has_sputum_color_change(self, context: ProtocolContext) -> bool:
        """Detect sputum color change in transcript or symptoms."""
        transcript = context.extraction.transcript.lower()
        if any(term in transcript for term in SPUTUM_COLOR_CHANGE_TERMS):
            return True
        for symptom in context.extraction.symptoms:
            name_lower = symptom.symptom.lower()
            if any(term in name_lower for term in SPUTUM_COLOR_CHANGE_TERMS):
                return True
            # Check character field for color words on phlegm/sputum/mucus symptoms
            # Handles cases like symptom="phlegm", character="green" (words not adjacent in transcript)
            if symptom.character:
                char_lower = symptom.character.lower()
                if any(term in char_lower for term in SPUTUM_COLOR_CHANGE_TERMS):
                    return True
                if any(s in name_lower for s in SPUTUM_SYMPTOM_NAMES):
                    if any(color in char_lower for color in SPUTUM_CONCERNING_COLORS):
                        return True
        return False

    def _is_baseline_respiratory(self, context: ProtocolContext) -> bool:
        """Detect if respiratory symptoms appear to be chronic baseline (not worsening).

        Two paths to baseline:
        1. Frequent reports: symptom reported 3+ times in 7 days with no worsening
        2. Normalcy language: patient uses phrases like "usual", "normal", "same old"
           while having a chronic respiratory condition

        Safety guards: worsening language or abnormal SpO2 always bypass baseline.
        """
        transcript = context.extraction.transcript.lower()

        # Safety: any worsening language means NOT baseline
        if any(term in transcript for term in RESPIRATORY_WORSENING_LANGUAGE):
            return False

        # Safety: abnormal SpO2 means NOT baseline
        for vs in context.extraction.vital_signs:
            if vs.name and ("spo2" in vs.name.lower() or "oxygen" in vs.name.lower()):
                try:
                    spo2 = float(str(vs.value).replace("%", ""))
                    if spo2 < 94:
                        return False
                except (ValueError, TypeError):
                    pass

        # Path 1: frequent reports in 7-day window
        history = context.symptom_history
        counts_7d = history.get("symptom_counts_7d") or {}
        resp_count_7d = sum(int(counts_7d.get(term, 0)) for term in RESPIRATORY_TERMS)
        if resp_count_7d >= 3:
            return True

        # Path 2: normalcy language from a patient with chronic respiratory condition
        if any(term in transcript for term in RESPIRATORY_NORMALCY_LANGUAGE):
            return True

        return False

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        symptom_label = _first_symptom_label(context)
        history = context.symptom_history
        worsening = _is_worsening(history, RESPIRATORY_TERMS)
        severity = _max_severity_for_terms(context, RESPIRATORY_TERMS)

        # Priority 0: Sputum color change = possible infection = clinician contact
        if self._has_sputum_color_change(context):
            question = None
            if not self._has_spo2_in_extraction(context):
                question = "Can you check your oxygen level with your pulse oximeter?"
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                immediate_question=question,
                schedule_checkin=True,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                checkin_hours=2,
                checkin_message=(
                    "Checking back on your breathing and sputum changes. "
                    "If you develop fever, increased breathlessness, or feel worse, "
                    "please contact a clinician promptly."
                ),
                escalation_flag=True,
                reason_code="asthma_respiratory_sputum_color_change",
            )

        # Baseline suppression: don't interrogate chronic COPD cough that isn't worsening
        if self._is_baseline_respiratory(context):
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                reason_code="baseline_respiratory_suppressed",
            )

        # Priority 1: Always ask for SpO2 if not provided — critical for COPD/asthma
        if not self._has_spo2_in_extraction(context):
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                immediate_question="Can you check your oxygen level with your pulse oximeter?",
                reason_code="asthma_respiratory_spo2_request",
            )

        if _is_severity_missing(context):
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                immediate_question=f"How bad is the {symptom_label} (1-10)? Have you used your rescue inhaler today?",
                reason_code="asthma_respiratory_missing_severity",
            )

        if (severity is not None and severity >= 7) or worsening:
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                schedule_checkin=True,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                checkin_hours=1,
                checkin_message="Checking back soon on your breathing symptoms. If breathing worsens or feels difficult, please contact a clinician promptly.",
                escalation_flag=True,
                reason_code="asthma_respiratory_worsening_or_severe",
            )

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            reason_code="asthma_respiratory_no_followup_needed",
        )


class HeadacheProtocol(FollowupProtocol):
    protocol_id = "headache_protocol"

    def matches(self, context: ProtocolContext) -> bool:
        return any(_contains_any(symptom, HEADACHE_TERMS) for symptom in _current_symptom_names(context))

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        symptom_label = _first_symptom_label(context)
        history = context.symptom_history
        worsening = _is_worsening(history, HEADACHE_TERMS)
        severity = _max_severity_for_terms(context, HEADACHE_TERMS)
        has_relief = _has_relief_context(context.extraction)

        immediate_question: Optional[str] = None
        reason_code = "headache_no_followup_needed"

        if _is_severity_missing(context):
            immediate_question = f"On a scale of 1-10, how bad is the {symptom_label}?"
            reason_code = "headache_missing_severity"
        elif _has_migraine_history(context):
            immediate_question = f"Does this feel like your usual migraines?"
            reason_code = "headache_migraine_history_check"

        severe = severity is not None and severity >= 7
        if severe and worsening and not has_relief and immediate_question is None:
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                schedule_checkin=True,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                checkin_hours=1,
                checkin_message="I will check back soon on your headache. If symptoms suddenly worsen or feel alarming, please contact a clinician promptly.",
                escalation_flag=True,
                reason_code="headache_severe_worsening_no_relief",
            )

        escalation_flag = severe and worsening and not has_relief
        if escalation_flag and reason_code == "headache_no_followup_needed":
            reason_code = "headache_severe_worsening_no_relief"

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            immediate_question=immediate_question,
            escalation_flag=escalation_flag,
            reason_code=reason_code,
        )


CYCLE_LINKED_SYMPTOMS = {
    "headache", "migraine", "head pain",
    "bloating", "bloated",
    "cramps", "cramping", "abdominal pain",
    "fatigue", "tired", "exhausted",
    "mood changes", "irritability", "anxiety",
    "breast tenderness", "breast pain",
    "back pain", "lower back pain",
    "nausea", "acne", "insomnia",
    "panic attack", "panic",
    "pelvic pain", "painful intercourse", "dyspareunia",
    "heavy bleeding", "painful periods", "dysmenorrhea",
}


class MenstrualCycleProtocol(FollowupProtocol):
    """Cycle-aware follow-up when user has active cycle tracking."""

    protocol_id = "menstrual_cycle_protocol"

    def _get_cycle_context(self, context: ProtocolContext) -> Optional[dict]:
        return context.symptom_history.get("cycle_context")

    def matches(self, context: ProtocolContext) -> bool:
        cycle_ctx = self._get_cycle_context(context)
        if not cycle_ctx:
            return False
        symptom_names = _current_symptom_names(context)
        return any(
            _contains_any(name, CYCLE_LINKED_SYMPTOMS)
            for name in symptom_names
        )

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        cycle_ctx = self._get_cycle_context(context)
        cycle_day = cycle_ctx.get("cycle_day", "?")
        cycle_phase = cycle_ctx.get("cycle_phase", "unknown")
        symptom_label = _first_symptom_label(context)
        has_correlation = cycle_ctx.get("has_prior_correlation", False)

        # Suppress awareness question if cycle protocol already fired recently
        # (prevents asking the same template 4x in 5 days)
        recent_cycle_count = context.recent_protocol_ids.count(self.protocol_id)

        # Fire correlation question, but with cooldown: suppress after 2 fires in 7 days
        if has_correlation:
            recent_correlation_count = context.recent_protocol_ids.count(
                "menstrual_cycle_correlation"
            )
            if recent_correlation_count >= 2:
                return ProtocolDecision(
                    protocol_id=self.protocol_id,
                    reason_code="cycle_correlation_suppressed",
                )
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                immediate_question=(
                    f"You mentioned {symptom_label} — you're on Day {cycle_day} "
                    f"of your cycle ({cycle_phase} phase). "
                    f"I've noticed a pattern of this around these cycle days. "
                    f"Does this feel similar to previous months?"
                ),
                reason_code="cycle_correlation_detected",
            )

        if recent_cycle_count >= 1:
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                reason_code="cycle_phase_symptom_awareness_suppressed",
            )

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            immediate_question=(
                f"You mentioned {symptom_label} — you're on Day {cycle_day} "
                f"of your cycle ({cycle_phase} phase). "
                f"Have you noticed this symptom at a similar point in your cycle before?"
            ),
            reason_code="cycle_phase_symptom_awareness",
        )


GI_SYMPTOM_TERMS = {
    "nausea", "nauseous", "stomach pain", "stomach ache", "abdominal pain",
    "vomiting", "upset stomach", "indigestion", "heartburn",
}

NSAID_NAMES = {
    "ibuprofen", "advil", "motrin", "naproxen", "aleve", "aspirin",
    "diclofenac", "celecoxib", "celebrex", "meloxicam",
}

IRON_NAMES = {"iron", "ferrous", "iron supplement", "iron pill", "iron pills"}


class MedicationInteractionProtocol(FollowupProtocol):
    """Detect dangerous medication combinations before asking for missing doses."""

    protocol_id = "medication_interaction"

    def _get_nsaid_dose_mg(self, context: ProtocolContext) -> Optional[tuple[str, float]]:
        """Return (nsaid_name, dose_mg) if a high-dose NSAID is in actions_taken."""
        for action in context.extraction.actions_taken:
            if not action.name:
                continue
            name_lower = action.name.strip().lower()
            if any(nsaid in name_lower for nsaid in NSAID_NAMES):
                if action.dose_text:
                    dose_match = re.search(r"(\d+)\s*mg", action.dose_text.lower())
                    if dose_match:
                        return name_lower, float(dose_match.group(1))
                # No dose but NSAID present — still check for interaction
                return name_lower, 0.0
        return None

    def _has_iron_or_anemia(self, context: ProtocolContext) -> bool:
        """Check if iron supplement in current actions OR anemia in profile."""
        for action in context.extraction.actions_taken:
            if action.name and any(iron in action.name.lower() for iron in IRON_NAMES):
                return True
        if context.user_profile:
            for cond in context.user_profile.conditions:
                if "anemia" in cond.lower() or "iron deficiency" in cond.lower():
                    return True
        return False

    def _has_gi_symptoms(self, context: ProtocolContext) -> bool:
        """Check for GI symptoms in current extraction."""
        for symptom in context.extraction.symptoms:
            if any(term in symptom.symptom.lower() for term in GI_SYMPTOM_TERMS):
                return True
        return _contains_any(context.extraction.transcript, GI_SYMPTOM_TERMS)

    def _has_corticosteroid(self, context: ProtocolContext) -> Optional[str]:
        """Return corticosteroid name if found in actions_taken or profile meds."""
        for action in context.extraction.actions_taken:
            if not action.name:
                continue
            name_lower = action.name.strip().lower()
            if any(steroid in name_lower for steroid in CORTICOSTEROID_NAMES):
                return name_lower
        # Also check profile medications for active steroid regimen
        if context.user_profile and context.user_profile.regular_medications:
            for med_str in context.user_profile.regular_medications:
                med_lower = med_str.lower()
                if any(steroid in med_lower for steroid in CORTICOSTEROID_NAMES):
                    # Return just the steroid name
                    for steroid in CORTICOSTEROID_NAMES:
                        if steroid in med_lower:
                            return steroid
        return None

    def _has_hypertension_condition(self, context: ProtocolContext) -> bool:
        """Check if user has hypertension in their conditions."""
        if not context.user_profile:
            return False
        for condition in context.user_profile.conditions:
            if any(term in condition.lower() for term in HYPERTENSION_CONDITION_TERMS):
                return True
        return False

    def _has_steroid_htn_symptoms(self, context: ProtocolContext) -> bool:
        """Check for symptoms suggestive of steroid-induced BP elevation."""
        for symptom in context.extraction.symptoms:
            if any(term in symptom.symptom.lower() for term in STEROID_HTN_SYMPTOM_TERMS):
                return True
        return _contains_any(context.extraction.transcript, STEROID_HTN_SYMPTOM_TERMS)

    def _has_bp_in_vitals(self, context: ProtocolContext) -> bool:
        """Check if BP was reported in current vital signs."""
        for vs in context.extraction.vital_signs:
            if vs.name and ("blood pressure" in vs.name.lower() or vs.name.lower() == "bp"):
                return True
        return False

    def _has_nsaid_in_actions(self, context: ProtocolContext) -> Optional[str]:
        """Return NSAID name if found in current actions_taken."""
        for action in context.extraction.actions_taken:
            if not action.name:
                continue
            name_lower = action.name.strip().lower()
            if any(nsaid in name_lower for nsaid in NSAID_NAMES):
                return action.name
        return None

    def _has_ace_inhibitor(self, context: ProtocolContext) -> bool:
        """Check if ACE inhibitor is in current actions or profile medications."""
        for action in context.extraction.actions_taken:
            if action.name and action.name.strip().lower() in ACE_INHIBITOR_NAMES:
                return True
        if context.user_profile and context.user_profile.regular_medications:
            for med_str in context.user_profile.regular_medications:
                med_lower = med_str.lower()
                if any(name in med_lower for name in ACE_INHIBITOR_NAMES):
                    return True
        return False

    def matches(self, context: ProtocolContext) -> bool:
        # Path 1: NSAID + iron/anemia + GI symptoms
        nsaid_info = self._get_nsaid_dose_mg(context)
        if nsaid_info is not None:
            nsaid_name, dose_mg = nsaid_info
            if dose_mg >= 600 and self._has_iron_or_anemia(context) and self._has_gi_symptoms(context):
                return True

        # Path 2: Corticosteroid + hypertension + suggestive symptoms
        steroid_name = self._has_corticosteroid(context)
        if steroid_name and self._has_hypertension_condition(context):
            if self._has_steroid_htn_symptoms(context):
                return True

        # Path 3: NSAID + ACE inhibitor (renal risk, reduced antihypertensive effect)
        nsaid_name = self._has_nsaid_in_actions(context)
        if nsaid_name and self._has_ace_inhibitor(context):
            return True

        return False

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        # Check steroid+HTN path first (proactive BP request)
        steroid_name = self._has_corticosteroid(context)
        if steroid_name and self._has_hypertension_condition(context) and self._has_steroid_htn_symptoms(context):
            if self._has_bp_in_vitals(context):
                return ProtocolDecision(
                    protocol_id=self.protocol_id,
                    schedule_checkin=True,
                    checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                    checkin_hours=4,
                    checkin_message=(
                        f"Checking back on your blood pressure while on {steroid_name}. "
                        "If your headache worsens or you feel dizzy, please contact a clinician promptly."
                    ),
                    escalation_flag=True,
                    reason_code="corticosteroid_htn_bp_elevated",
                )
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                immediate_question=(
                    f"Corticosteroids like {steroid_name} can sometimes affect blood pressure. "
                    "Can you check your BP and let me know the reading?"
                ),
                reason_code="corticosteroid_htn_bp_request",
            )

        # NSAID + ACE inhibitor path
        nsaid_name = self._has_nsaid_in_actions(context)
        if nsaid_name and self._has_ace_inhibitor(context):
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                immediate_question=(
                    f"NSAIDs like {nsaid_name} can reduce the effectiveness of blood pressure "
                    "medications and may affect kidney function. Worth discussing with your "
                    "clinician, especially if you're taking them regularly."
                ),
                reason_code="nsaid_ace_inhibitor_interaction",
            )

        # NSAID + iron path (existing)
        nsaid_info = self._get_nsaid_dose_mg(context)
        nsaid_name = nsaid_info[0] if nsaid_info else "NSAID"
        dose_mg = nsaid_info[1] if nsaid_info else 0

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            immediate_question=(
                f"Taking {int(dose_mg)}mg {nsaid_name} with iron while feeling nauseous "
                f"can increase GI bleeding risk — have you noticed any dark stools or stomach pain? "
                f"Worth discussing this combination with your doctor."
            ),
            escalation_flag=True,
            reason_code="nsaid_iron_gi_bleed_risk",
        )


class GenericHighSeverityEscalation(FollowupProtocol):
    """Escalate when any symptom is reported at severity >= 7, regardless of condition."""

    protocol_id = "generic_high_severity_escalation"

    def matches(self, context: ProtocolContext) -> bool:
        if not context.extraction.symptoms:
            return False
        return any(
            s.severity_1_10 is not None and s.severity_1_10 >= 7
            for s in context.extraction.symptoms
        )

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        severe = [s for s in context.extraction.symptoms
                  if s.severity_1_10 is not None and s.severity_1_10 >= 7]
        symptom_label = severe[0].symptom if severe else "symptom"
        return ProtocolDecision(
            protocol_id=self.protocol_id,
            schedule_checkin=True,
            checkin_type=CheckinType.SYMPTOM_PROGRESSION,
            checkin_hours=4,
            checkin_message=(
                f"Checking back on your {symptom_label}. "
                "If it worsens or you feel unwell, please contact a clinician promptly."
            ),
            escalation_flag=True,
            reason_code="high_severity_symptom",
        )


class GenericSeverityFallback(FollowupProtocol):
    protocol_id = "generic_severity_fallback"

    def matches(self, context: ProtocolContext) -> bool:
        if not _is_severity_missing(context):
            return False
        if not context.extraction.symptoms:
            return False
        symptom_lower = _first_symptom_label(context).lower()
        # Never ask "severity" for vital signs / measurements
        if any(term in symptom_lower for term in NON_SYMPTOM_VITAL_SIGNS):
            return False
        return any(term in symptom_lower for term in HIGH_PRIORITY_SEVERITY_TERMS)

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        symptom_label = _first_symptom_label(context)
        symptom_lower = symptom_label.lower()

        # If this symptom has been reported 3+ times and severity is already known,
        # ask a root-cause question instead of severity again.
        counts_7d = (context.symptom_history or {}).get("symptom_counts_7d", {})
        recent_severities = (context.symptom_history or {}).get("recent_severities", {})

        # Count prior reports for this symptom
        prior_count = int(counts_7d.get(symptom_lower, 0))
        # Also check if a broader term matches (e.g., "queasy" counted under "nausea")
        if prior_count < 3:
            for term in HIGH_PRIORITY_SEVERITY_TERMS:
                if term in symptom_lower:
                    prior_count = max(prior_count, int(counts_7d.get(term, 0)))

        has_prior_severity = symptom_lower in recent_severities and len(recent_severities[symptom_lower]) >= 1

        if prior_count >= 3 and has_prior_severity:
            context_question = _get_recurring_context_question(symptom_label)
            if context_question:
                return ProtocolDecision(
                    protocol_id=self.protocol_id,
                    immediate_question=context_question,
                    reason_code="recurring_symptom_context_question",
                )

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            immediate_question=f"On a scale of 1-10, how bad is the {symptom_label}?",
            reason_code="generic_missing_severity_high_priority",
        )


class CompoundSymptomEscalationProtocol(FollowupProtocol):
    """Escalate when multiple concerning signals coincide (e.g., 3+ symptoms + abnormal vitals)."""

    protocol_id = "compound_symptom_escalation"

    def _has_abnormal_vitals(self, context: ProtocolContext) -> bool:
        for vs in context.extraction.vital_signs:
            name = vs.name.lower() if vs.name else ""
            # Glucose check
            if any(term in name for term in ("sugar", "glucose")):
                try:
                    val = float(vs.value)
                    if val > 200 or val < 70:
                        return True
                except (ValueError, TypeError):
                    pass
            # BP check
            if "blood pressure" in name or name == "bp":
                match = re.match(r"(\d+)\s*/\s*(\d+)", vs.value or "")
                if match:
                    systolic, diastolic = int(match.group(1)), int(match.group(2))
                    if systolic >= 160 or diastolic >= 100:
                        return True
        return False

    def _has_medication_nonadherence(self, context: ProtocolContext) -> bool:
        transcript = context.extraction.transcript.lower()
        return any(term in transcript for term in MEDICATION_NONADHERENCE_TERMS)

    def matches(self, context: ProtocolContext) -> bool:
        if len(context.extraction.symptoms) < 3:
            return False
        return self._has_abnormal_vitals(context) or self._has_medication_nonadherence(context)

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        has_abnormal = self._has_abnormal_vitals(context)
        has_nonadherence = self._has_medication_nonadherence(context)

        parts = []
        if has_abnormal:
            parts.append("abnormal_vitals")
        if has_nonadherence:
            parts.append("medication_nonadherence")

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            schedule_checkin=True,
            checkin_type=CheckinType.SYMPTOM_PROGRESSION,
            checkin_hours=4,
            checkin_message=(
                "You reported several symptoms along with some concerning findings. "
                "How are you feeling now? If symptoms worsen or you feel unwell, "
                "please contact a clinician promptly."
            ),
            escalation_flag=True,
            reason_code=f"compound_escalation_{'_'.join(parts)}",
        )


class HypertensionProtocol(FollowupProtocol):
    """Follow-up when blood pressure reading is elevated."""

    protocol_id = "hypertension_protocol"

    def _get_bp_reading(self, context: ProtocolContext) -> Optional[tuple]:
        """Extract systolic/diastolic from vital signs. Returns (systolic, diastolic) or None."""
        for vs in context.extraction.vital_signs:
            if vs.name and ("blood pressure" in vs.name.lower() or vs.name.lower() == "bp"):
                match = re.match(r"(\d+)\s*/\s*(\d+)", vs.value or "")
                if match:
                    return int(match.group(1)), int(match.group(2))
        return None

    def matches(self, context: ProtocolContext) -> bool:
        bp = self._get_bp_reading(context)
        if bp is None:
            return False
        systolic, diastolic = bp
        return systolic >= 140 or diastolic >= 90

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        bp = self._get_bp_reading(context)
        systolic, diastolic = bp  # type: ignore[misc]

        symptom_names = _current_symptom_names(context)
        has_associated = any(
            _contains_any(name, HYPERTENSION_SYMPTOM_TERMS) for name in symptom_names
        )

        if has_associated:
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                schedule_checkin=True,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                checkin_hours=2,
                checkin_message=(
                    "Checking back on your blood pressure reading. "
                    "If you experience worsening headache, vision changes, or chest pain, "
                    "please contact a clinician promptly."
                ),
                escalation_flag=systolic >= 180 or diastolic >= 120,
                reason_code="hypertension_elevated_with_symptoms",
            )

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            immediate_question=(
                "Your blood pressure reading is elevated. "
                "Are you experiencing any headache, vision changes, dizziness, or chest discomfort?"
            ),
            escalation_flag=systolic >= 180 or diastolic >= 120,
            reason_code="hypertension_elevated_ask_symptoms",
        )


class GastrointestinalProtocol(FollowupProtocol):
    """Follow-up when GI symptoms persist or are severe."""

    protocol_id = "gastrointestinal_protocol"

    _RED_FLAGS = {
        "blood in stool", "bloody stool", "black stool",
        "melena", "hematemesis", "blood in vomit",
    }

    def matches(self, context: ProtocolContext) -> bool:
        symptom_names = _current_symptom_names(context)
        has_gi = any(_contains_any(name, GI_PERSISTENT_TERMS) for name in symptom_names)
        if not has_gi:
            return False

        # Severe current episode
        for s in context.extraction.symptoms:
            if any(term in s.symptom.lower() for term in GI_PERSISTENT_TERMS):
                if s.severity_1_10 is not None and s.severity_1_10 >= 7:
                    return True

        # Persistent: 3+ GI logs in 7 days
        counts_7d = (context.symptom_history or {}).get("symptom_counts_7d", {})
        gi_count = sum(
            counts_7d.get(term, 0) for term in GI_PERSISTENT_TERMS
        )
        if gi_count >= 3:
            return True

        # Red flag keywords in transcript
        transcript = context.extraction.transcript.lower()
        if any(flag in transcript for flag in self._RED_FLAGS):
            return True

        return False

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        transcript = context.extraction.transcript.lower()
        has_red_flag = any(flag in transcript for flag in self._RED_FLAGS)

        if has_red_flag:
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                schedule_checkin=True,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                checkin_hours=1,
                checkin_message=(
                    "Following up on your GI symptoms. Blood in stool or vomit "
                    "can be serious — if symptoms persist or worsen, "
                    "please contact a clinician promptly."
                ),
                escalation_flag=True,
                reason_code="gi_red_flag_blood",
            )

        symptom_names = _current_symptom_names(context)
        has_fever = any(_contains_any(name, FEVER_TERMS) for name in symptom_names)

        if has_fever:
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                immediate_question=(
                    "GI symptoms with fever can sometimes indicate an infection. "
                    "Are you able to keep fluids down? "
                    "Have you noticed any blood in your stool?"
                ),
                escalation_flag=True,
                reason_code="gi_with_fever",
            )

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            immediate_question=(
                "How is your hydration? Are you able to keep fluids down? "
                "Any blood in your stool or severe cramping?"
            ),
            reason_code="gi_persistent_ask_hydration",
        )


SKIN_LESION_COMPOUNDING_SYMPTOMS = {
    "fever", "temperature", "chills",
    "spreading", "growing", "worsening",
    "pain", "swelling", "warmth",
    "red streak", "pus", "drainage",
}

# Hard guardrail: always escalate regardless of MedGemma reasoning
ALWAYS_ESCALATE_CONDITIONS = {"melanoma", "suspicious mole"}


class SkinLesionEscalationProtocol(FollowupProtocol):
    """Escalate when a photo shows a lesion AND compounding clinical factors exist."""

    protocol_id = "skin_lesion_escalation"

    def matches(self, context: ProtocolContext) -> bool:
        if not context.image_analysis or not context.image_analysis.lesion_detected:
            return False

        # Hard guardrail: always escalate potential melanoma
        if (context.image_analysis.skin_lesion
                and context.image_analysis.skin_lesion.predicted_condition):
            cond = context.image_analysis.skin_lesion.predicted_condition.lower()
            if any(term in cond for term in ALWAYS_ESCALATE_CONDITIONS):
                return True

        # Check for compounding factors
        symptom_names = _current_symptom_names(context)
        has_fever = any(_contains_any(s, FEVER_TERMS) for s in symptom_names)
        has_concerning = any(
            _contains_any(s, SKIN_LESION_COMPOUNDING_SYMPTOMS) for s in symptom_names
        )

        # Check transcript for worsening language
        transcript = context.extraction.transcript.lower()
        worsening_terms = {"worse", "worsening", "spreading", "growing", "bigger", "larger"}
        is_worsening_text = any(term in transcript for term in worsening_terms)

        # Check for high severity on any symptom
        max_sev = None
        for s in context.extraction.symptoms:
            if s.severity_1_10 is not None:
                max_sev = max(max_sev or 0, s.severity_1_10)
        has_high_severity = max_sev is not None and max_sev >= 7

        return has_fever or has_high_severity or is_worsening_text or has_concerning

    def decide(self, context: ProtocolContext) -> ProtocolDecision:
        lesion_desc = context.image_analysis.clinical_description if context.image_analysis else "skin lesion"

        # Melanoma guardrail — always escalate with doctor-visit recommendation
        if (context.image_analysis and context.image_analysis.skin_lesion
                and context.image_analysis.skin_lesion.predicted_condition):
            cond = context.image_analysis.skin_lesion.predicted_condition.lower()
            if any(term in cond for term in ALWAYS_ESCALATE_CONDITIONS):
                return ProtocolDecision(
                    protocol_id=self.protocol_id,
                    schedule_checkin=True,
                    checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                    checkin_hours=1,
                    checkin_message=(
                        "The photo analysis detected a skin finding that warrants prompt medical attention. "
                        "Please consider scheduling a dermatology appointment soon."
                    ),
                    escalation_flag=True,
                    reason_code="skin_always_escalate_guardrail",
                )

        symptom_names = _current_symptom_names(context)
        has_fever = any(_contains_any(s, FEVER_TERMS) for s in symptom_names)

        if has_fever:
            return ProtocolDecision(
                protocol_id=self.protocol_id,
                schedule_checkin=True,
                checkin_type=CheckinType.SYMPTOM_PROGRESSION,
                checkin_hours=2,
                checkin_message=(
                    f"Following up on the skin finding ({lesion_desc}) and your fever. "
                    "If the area is spreading or fever worsens, please contact a clinician promptly."
                ),
                escalation_flag=True,
                reason_code="skin_lesion_with_fever",
            )

        return ProtocolDecision(
            protocol_id=self.protocol_id,
            schedule_checkin=True,
            checkin_type=CheckinType.SYMPTOM_PROGRESSION,
            checkin_hours=4,
            checkin_message=(
                f"Checking back on the skin finding ({lesion_desc}). "
                "Has it changed in size, color, or how it feels? Take another photo if you can."
            ),
            reason_code="skin_lesion_with_compounding_factors",
        )


class ProtocolRegistry:
    """Ordered protocol evaluator (first match wins)."""

    def __init__(self, protocols: Optional[List[FollowupProtocol]] = None):
        self.protocols: List[FollowupProtocol] = protocols or [
            MedicationInteractionProtocol(),  # Safety-critical: drug interactions first
            FeverProtocol(),
            AsthmaRespiratoryProtocol(),      # SpO2 priority for COPD/asthma
            CompoundSymptomEscalationProtocol(),  # Multi-signal escalation
            HypertensionProtocol(),           # Elevated BP follow-up
            GastrointestinalProtocol(),       # Persistent/severe GI symptoms
            SkinLesionEscalationProtocol(),
            HeadacheProtocol(),
            MenstrualCycleProtocol(),
            MedicationMissingDosePriority(),   # Dose questions after safety protocols
            GenericHighSeverityEscalation(),   # Escalate severity >= 7 for any symptom
            GenericSeverityFallback(),
        ]

    def get_priority_index(self, protocol_id: str) -> int:
        """Return the priority index of a protocol (lower = higher priority)."""
        for i, p in enumerate(self.protocols):
            if p.protocol_id == protocol_id:
                return i
        return len(self.protocols)

    def evaluate(self, context: ProtocolContext) -> ProtocolDecision:
        for protocol in self.protocols:
            if protocol.matches(context):
                decision = protocol.decide(context)
                if decision.protocol_id is None:
                    decision.protocol_id = protocol.protocol_id
                return decision
        return ProtocolDecision(reason_code="no_protocol_match")


_default_registry: Optional[ProtocolRegistry] = None


def get_protocol_registry() -> ProtocolRegistry:
    """Return singleton protocol registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ProtocolRegistry()
    return _default_registry
