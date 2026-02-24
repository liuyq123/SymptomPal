"""
Proactive Agent Response Generator

Uses MedGemma via HAI-DEF to generate intelligent, context-aware responses that feel like
interacting with a thoughtful assistant rather than filling out a form.

Now includes historical analysis for insight nudges like:
- "You're coughing more than yesterday"
- "This is your third headache this week"
"""

import json
import re
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from collections import Counter
import uuid

from ..models import (
    ExtractionResult, LogEntry, MedicationLogEntry,
    AgentResponse, ScheduledCheckin, CheckinType, UserProfile,
    ImageAnalysisResult, CycleDayTag,
)
from .medgemma_client import get_medgemma_client
from .protocols import ProtocolContext, ProtocolDecision, get_protocol_registry
from .logging import log_debug

VALID_PROTOCOL_IDS = frozenset({
    "medication_interaction", "fever_protocol", "asthma_respiratory_protocol",
    "compound_symptom_escalation", "hypertension_protocol", "gastrointestinal_protocol",
    "skin_lesion_escalation", "headache_protocol", "menstrual_cycle_protocol",
    "medication_missing_dose_priority", "generic_high_severity_escalation",
    "generic_severity_fallback",
})
SAFETY_CRITICAL_PROTOCOLS = frozenset({
    "medication_interaction", "compound_symptom_escalation",
})

logger = logging.getLogger(__name__)

RESPIRATORY_ASTHMA_SYMPTOMS = {
    "cough",
    "coughing",
    "wheeze",
    "wheezing",
    "shortness of breath",
}

HIGH_PRIORITY_SEVERITY_KEYWORDS = [
    "pain",
    "headache",
    "migraine",
    "chest pain",
    "abdominal pain",
    "nausea",
    "dizziness",
]

_USER_QUESTION_SIGNALS = [
    "?",
    "do you think",
    "is that normal",
    "should i",
    "could this be",
    "is it because",
    "is this from",
    "should i be worried",
    "is that why",
    "can you",
    "what do you think",
]


def _user_asked_question(transcript: str) -> bool:
    """Detect if the user asked a direct question in their input."""
    lower = transcript.lower()
    return any(signal in lower for signal in _USER_QUESTION_SIGNALS)


def _extract_user_question(transcript: str) -> str:
    """Extract the question portion from a transcript."""
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    questions = [s.strip() for s in sentences if '?' in s]
    if questions:
        return questions[-1]
    lower = transcript.lower()
    for signal in _USER_QUESTION_SIGNALS[1:]:
        if signal in lower:
            for s in sentences:
                if signal in s.lower():
                    return s.strip()
    return transcript


# LLM-based safety classifier prompt — replaces brittle regex patterns.
# Catches medication advice, diagnoses, and definitive causation that regex would miss.
_SAFETY_CLASSIFIER_PROMPT = """You are a medical compliance reviewer. Does the following health-app response contain ANY of these violations?

1. MEDICATION ADVICE: Recommending to start, stop, change, increase, decrease, or skip any medication or dose.
   NOT a violation: acknowledging medication was taken, noting a medication's known side effects with hedging, or observing what happened when medication was missed.
2. DIAGNOSIS: Telling the patient they have (or probably have) a specific condition, syndrome, disease, or disorder
3. DEFINITIVE CAUSATION: Stating that X is causing/caused Y without hedging language ("can be", "sometimes", "may be associated with")

If the text merely acknowledges what the patient did or asks a question, it is SAFE.

Text to review:
"{text}"

Respond with EXACTLY one word: SAFE or UNSAFE"""

# Strip count-based insight spam from LLM output.
# Matches any clause containing "N times" + a time reference (this week, in the last month, etc.)
# Does NOT match dosing instructions like "take 3 times daily".
_COUNT_SPAM_PATTERN = re.compile(
    r"[^.!—]*\b\d+\s+times?\s+(?:this\s+\w+|in\s+the\s+(?:last|past)\s+\w+|"
    r"(?:—|,)\s*(?:worth|that'?s|which))\b[^.!—]*[.!—]?\s*",
    re.IGNORECASE,
)
_YOUVE_COUNT_PATTERN = re.compile(
    r"[^.!—]*\bYou'?v?e?\s+(?:logged|mentioned|reported|noted|had)\b[^.!—]*"
    r"\b\d+\s+times?\b[^.!—]*[.!—]?\s*",
    re.IGNORECASE,
)


def _strip_count_spam(text: str) -> str:
    """Strip count-based insight spam from LLM output (cosmetic, not safety-critical)."""
    result = _COUNT_SPAM_PATTERN.sub("", text)
    return _YOUVE_COUNT_PATTERN.sub("", result).strip()


_TREND_LABEL_PATTERN = re.compile(r'\s*TREND:\s*.+$', re.MULTILINE)


def _strip_trend_labels(text: str) -> str:
    """Strip echoed TREND/SEVERITY TREND labels from LLM output."""
    return _TREND_LABEL_PATTERN.sub("", text).strip()


# Pattern to find vital sign claims in LLM output
_VITAL_CLAIM_PATTERN = re.compile(
    r'(?:spo2|oxygen\s*(?:saturation|level)?|blood\s*sugar|glucose|blood\s*pressure|bp)'
    r'[^.!?]*?(\d{2,3}(?:\.\d+)?)',
    re.IGNORECASE,
)


def _strip_hallucinated_vitals(text: str, recent_logs: list, current_extraction=None) -> str:
    """Strip sentences that claim vital sign values not supported by actual log data."""
    # Collect all actual vital sign values from recent logs + current extraction
    actual_values: set[str] = set()
    for log in (recent_logs or [])[:30]:
        if hasattr(log, 'extracted') and hasattr(log.extracted, 'vital_signs'):
            for v in log.extracted.vital_signs:
                if v.value:
                    # Extract numeric parts
                    for m in re.finditer(r'\d+(?:\.\d+)?', v.value):
                        actual_values.add(m.group())
    # Include vitals from the current log being processed
    if current_extraction and hasattr(current_extraction, 'vital_signs'):
        for v in current_extraction.vital_signs:
            if v.value:
                for m in re.finditer(r'\d+(?:\.\d+)?', v.value):
                    actual_values.add(m.group())

    if not actual_values:
        # No vitals in logs — any specific vital claim is unsupported
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cleaned = []
        for sentence in sentences:
            claims = _VITAL_CLAIM_PATTERN.findall(sentence)
            if claims:
                logger.warning(f"Stripped hallucinated vital sign claim (no vitals in logs): {sentence[:100]}")
                continue
            cleaned.append(sentence)
        return " ".join(cleaned) if cleaned else text

    # Find claims in the text
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for sentence in sentences:
        claims = _VITAL_CLAIM_PATTERN.findall(sentence)
        if claims:
            # Check if any claimed value is NOT in actual data
            hallucinated = False
            for claimed_val in claims:
                if claimed_val not in actual_values:
                    hallucinated = True
                    break
            if hallucinated:
                logger.warning(f"Stripped hallucinated vital sign claim: {sentence[:100]}")
                continue
        cleaned.append(sentence)

    return " ".join(cleaned) if cleaned else text


_CLINICAL_NOTE_PREFIXES = [
    "patient reports", "patient has", "patient is", "patient describes",
    "consistent cyclical", "symptoms consistently", "cyclical symptoms",
    "this is consistent with", "this pattern is consistent",
]


def _strip_clinical_notes(text: str) -> str:
    """Strip clinical-note-style sentences from patient-facing text."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = [s for s in sentences if not any(
        s.lower().strip().startswith(p) for p in _CLINICAL_NOTE_PREFIXES
    )]
    return ' '.join(filtered).strip() or text


# --- Context tag keyword lists (deterministic transcript classification) ---
_IDENTITY_GRIEF_KEYWORDS = [
    "used to", "can't even", "i was always", "couldn't imagine",
    "not the same", "barely walk", "low bar", "who i was",
]
_BOUNDARY_KEYWORDS = [
    "one thing at a time", "don't want to talk", "not right now",
    "enough about", "i know i know", "let's focus on",
    "not ready", "can we not",
]
_SELF_RESEARCH_KEYWORDS = [
    "looked it up", "read about", "researched", "googled",
    "found online", "coworker suggested", "friend said", "i read that",
]
_BEHAVIOR_CHANGE_KEYWORDS = [
    "cutting down", "quit", "stopped", "started the",
    "trying to quit", "switched to", "signed up", "down to",
    "nicotine patch", "i'm trying",
]

_CHEERLEADER_OPENERS = re.compile(
    r"(?i)^that'?s\s+(?:fantastic|wonderful|amazing|incredible|excellent|great)\b"
)
_CHEERLEADER_CLOSERS = re.compile(
    r"(?i)^keep\s+up\s+the\s+(?:great|good|excellent)\s+work\b"
)
_CHEERLEADER_EMBEDDED = re.compile(
    r"(?i),?\s*which\s+is\s+(?:fantastic|wonderful|amazing|excellent)\s+news[,!.]*\s*"
)


def _strip_cheerleader_phrases(text: str) -> str:
    """Remove cheerleader-tone sentences/phrases from patient-facing text."""
    # First pass: strip embedded mid-sentence cheerleading
    text = _CHEERLEADER_EMBEDDED.sub(" ", text)
    # Second pass: remove entire cheerleader sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = [
        s for s in sentences
        if not _CHEERLEADER_OPENERS.search(s.strip())
        and not _CHEERLEADER_CLOSERS.search(s.strip())
    ]
    return ' '.join(filtered).strip() or text  # never return empty


_SOUNDS_LIKE_RE = re.compile(
    r'(?:^|(?<=\. ))(?:[Ii]t |[Tt]hat )?[Ss]ounds like ')
_THAT_SOUNDS_RE = re.compile(
    r'(?:^|(?<=\. ))[Tt]hat sounds (?!like\b)')


def _diversify_opening(text: str) -> str:
    """Replace sentence-initial 'sounds like' to reduce repetitive openings."""
    # "That sounds [adj]" → "That must be [adj]"
    m = _THAT_SOUNDS_RE.search(text)
    if m:
        return text[:m.start()] + "That must be " + text[m.end():]
    # "It sounds like X" / "That sounds like X" / "Sounds like X" → X (capitalized)
    m = _SOUNDS_LIKE_RE.search(text)
    if m:
        rest = text[m.end():]
        if rest:
            rest = rest[0].upper() + rest[1:]
        return text[:m.start()] + rest
    return text


def clean_patient_text(text: str) -> str:
    """Apply universal deterministic filters to any patient-facing text."""
    text = _strip_count_spam(text)
    text = _strip_trend_labels(text)
    text = _strip_clinical_notes(text)
    text = _strip_cheerleader_phrases(text)
    text = _diversify_opening(text)
    return text


ALLOWED_TOOLS = {"run_watchdog_now"}
PROTOCOL_TOOL_PREFIX = "invoke_protocol:"
SCHEDULE_CHECKIN_PREFIX = "schedule_checkin:"
ESCALATE_ALERT_PREFIX = "escalate_clinician_alert:"

ALLOWED_ESCALATION_REASONS = {
    "worsening_trajectory",
    "medication_concern",
    "vital_sign_concern",
    "patient_distress",
    "multi_system_involvement",
}

ESCALATION_NOTES = {
    "worsening_trajectory": "Your recent symptom pattern shows a worsening trend. Consider discussing this with your healthcare provider.",
    "medication_concern": "There may be a medication-related concern worth discussing with your healthcare provider.",
    "vital_sign_concern": "Some of your vital signs may warrant attention. Consider following up with your healthcare provider.",
    "patient_distress": "If you're feeling overwhelmed by your symptoms, reaching out to your healthcare provider can help.",
    "multi_system_involvement": "You're experiencing symptoms across multiple areas. Consider discussing the overall picture with your healthcare provider.",
}


def filter_tool_calls(raw: object) -> list[str]:
    """Filter tool_calls from LLM output to only allowed tools and protocol invocations."""
    if not isinstance(raw, list):
        return []
    result = []
    for t in raw:
        if not isinstance(t, str):
            continue
        if t in ALLOWED_TOOLS:
            result.append(t)
        elif t.startswith(PROTOCOL_TOOL_PREFIX):
            protocol_id = t[len(PROTOCOL_TOOL_PREFIX):]
            if protocol_id in VALID_PROTOCOL_IDS:
                result.append(t)
        elif t.startswith(SCHEDULE_CHECKIN_PREFIX):
            parts = t.split(":", 2)
            if len(parts) == 3:
                try:
                    hours = int(parts[1])
                    if 1 <= hours <= 24 and parts[2].strip():
                        result.append(t)
                except ValueError:
                    pass
        elif t.startswith(ESCALATE_ALERT_PREFIX):
            reason = t[len(ESCALATE_ALERT_PREFIX):]
            if reason in ALLOWED_ESCALATION_REASONS:
                result.append(t)
    return result


def _has_abnormal_vital(vitals: list) -> bool:
    """Check if any vital sign is outside standard clinical ranges."""
    for v in vitals:
        name = v.name.lower()
        try:
            if "glucose" in name or "sugar" in name or "bg" in name:
                val = float(re.sub(r'[^\d.]', '', v.value))
                if val > 200 or val < 70:
                    return True
            elif "spo2" in name or "oxygen" in name or "o2" in name:
                val = float(re.sub(r'[^\d.]', '', v.value))
                if val < 93:
                    return True
            elif "blood pressure" in name or "bp" in name:
                parts = v.value.split("/")
                if parts:
                    systolic = float(re.sub(r'[^\d.]', '', parts[0]))
                    if systolic > 180 or systolic < 90:
                        return True
        except (ValueError, IndexError):
            continue
    return False


def _should_inject_watchdog(extraction, num_prior_logs: int) -> bool:
    """Deterministic safety net for watchdog injection.

    Uses standard clinical thresholds to catch cases where the LLM
    recognizes escalation but fails to emit tool_calls.
    """
    if num_prior_logs < 5:
        return False

    max_severity = max(
        (s.severity_1_10 for s in extraction.symptoms if s.severity_1_10 is not None),
        default=0,
    )

    # Rule 1: High severity — strong signal of worsening
    if max_severity >= 8:
        return True

    # Rule 2: Abnormal vital + moderate severity
    if max_severity >= 5 and _has_abnormal_vital(extraction.vital_signs):
        return True

    return False


SAFETY_SENSITIVE_TERMS = {
    "chest pain",
    "shortness of breath",
    "can't breathe",
    "cannot breathe",
    "trouble breathing",
    "breathing difficult",
    "faint",
    "passed out",
    "seizure",
    "confusion",
    "weakness",
    "numbness",
    "slurred speech",
    "vision loss",
}


def _is_known_regular_med(med_name: str, profile) -> bool:
    """Check if medication is in the user's regular_medications profile."""
    if not profile or not profile.regular_medications:
        return False
    name_lower = med_name.strip().lower()
    return any(name_lower in med.lower() for med in profile.regular_medications)


def _build_immediate_question(
    context: "ResponseContext",
    has_scheduled_checkin: bool,
) -> Optional[str]:
    """Deterministic follow-up question selection for consistency across paths."""
    if has_scheduled_checkin:
        return None

    symptoms = context.extraction.symptoms
    actions = context.extraction.actions_taken
    profile = context.user_profile

    # Priority 1: Medication without dose (skip if known regular med in profile)
    if actions and not actions[0].dose_text:
        if not _is_known_regular_med(actions[0].name, profile):
            return f"What dose of {actions[0].name} did you take?"

    # Priority 2: Fever without temperature
    if symptoms and any(s.symptom.lower() in ["fever", "temperature"] for s in symptoms):
        fever_symptom = next((s for s in symptoms if s.symptom.lower() in ["fever", "temperature"]), None)
        if fever_symptom and not fever_symptom.severity_1_10:
            return "What was your temperature?"

    # Priority 3: Severity missing for important symptoms
    if "severity" not in context.extraction.missing_fields or not symptoms:
        return None

    primary_symptom = symptoms[0].symptom
    symptom_lower = primary_symptom.lower()
    conditions_lower = [c.lower() for c in profile.conditions] if profile and profile.conditions else []

    # Ensure asthma-specific respiratory questions remain reachable.
    if "asthma" in conditions_lower and any(term in symptom_lower for term in RESPIRATORY_ASTHMA_SYMPTOMS):
        return f"How bad is the {primary_symptom} (1-10)? Have you used your rescue inhaler today?"

    if any(keyword in symptom_lower for keyword in HIGH_PRIORITY_SEVERITY_KEYWORDS):
        if any("migraine" in c for c in conditions_lower) and symptom_lower in ["headache", "head pain"]:
            return f"On a scale of 1-10, how bad is the {primary_symptom}? Does it feel like your usual migraines?"
        return f"On a scale of 1-10, how bad is the {primary_symptom}?"

    return None


_PROFILE_DOSE_PATTERN = re.compile(
    r'^([\w-]+)\s+(\d+\s*(?:mg|ml|mcg|µg|g|iu|units?))',
    re.IGNORECASE,
)
# Also matches "GenericName (BrandName) 18mcg ..."
_PROFILE_DOSE_PATTERN_PAREN = re.compile(
    r'^([\w-]+)\s+\(([\w-]+)\)\s+(\d+\s*(?:mg|ml|mcg|µg|g|iu|units?|/\d+))',
    re.IGNORECASE,
)


def parse_known_medication_doses(
    user_profile: Optional[UserProfile] = None,
    recent_symptom_logs: Optional[List[LogEntry]] = None,
) -> Dict[str, str]:
    """Parse known medication doses from profile and recent log extractions.

    Shared utility used by both ResponseContext and _protocol_only_fallback().
    Returns a dict of med_name_lower -> dose_text.
    """
    known: Dict[str, str] = {}

    # Source 1: UserProfile.regular_medications
    # Handles both "Metformin 500mg" and "Tiotropium (Spiriva) 18mcg"
    if user_profile and user_profile.regular_medications:
        for med_str in user_profile.regular_medications:
            stripped = med_str.strip()
            # Try parenthetical format first: "GenericName (BrandName) dose"
            paren_match = _PROFILE_DOSE_PATTERN_PAREN.match(stripped)
            if paren_match:
                generic = paren_match.group(1).lower()
                brand = paren_match.group(2).lower()
                dose = paren_match.group(3).strip()
                known[generic] = dose
                known[brand] = dose
                continue
            # Standard format: "MedName dose"
            match = _PROFILE_DOSE_PATTERN.match(stripped)
            if match:
                known[match.group(1).lower()] = match.group(2).strip()

    # Source 2: Prior log extractions (most recent wins, don't overwrite profile)
    for log in reversed(recent_symptom_logs or []):
        for action in log.extracted.actions_taken:
            if action.dose_text and action.name:
                name_lower = action.name.strip().lower()
                if name_lower not in known:
                    known[name_lower] = action.dose_text

    return known


class ResponseContext:
    """Context gathered for response generation."""

    def __init__(
        self,
        extraction: ExtractionResult,
        recent_med_logs: List[MedicationLogEntry],
        recent_symptom_logs: List[LogEntry],
        user_id: str,
        user_profile: Optional[UserProfile] = None,
        image_analysis: Optional[ImageAnalysisResult] = None,
        cycle_tag: Optional[CycleDayTag] = None,
        has_cycle_correlation: bool = False,
        ambient_summary: Optional[str] = None,
    ):
        self.extraction = extraction
        self.recent_med_logs = recent_med_logs
        self.recent_symptom_logs = recent_symptom_logs
        self.user_id = user_id
        self.user_profile = user_profile
        self.image_analysis = image_analysis
        self._cycle_tag = cycle_tag
        self._has_cycle_correlation = has_cycle_correlation
        self.ambient_summary = ambient_summary

    def _build_known_medication_doses(self) -> Dict[str, str]:
        """Build med_name_lower -> dose from profile and prior log extractions."""
        return parse_known_medication_doses(self.user_profile, self.recent_symptom_logs)

    def _compute_context_tags(self) -> Dict[str, bool]:
        """Compute boolean context tags from the transcript for prompt steering."""
        text = self.extraction.transcript.lower()
        return {
            "has_identity_grief": any(kw in text for kw in _IDENTITY_GRIEF_KEYWORDS),
            "has_boundary_pushback": any(kw in text for kw in _BOUNDARY_KEYWORDS),
            "has_self_research": any(kw in text for kw in _SELF_RESEARCH_KEYWORDS),
            "has_behavior_change": any(kw in text for kw in _BEHAVIOR_CHANGE_KEYWORDS),
        }

    def _analyze_symptom_history(self) -> Dict:
        """Analyze historical symptom data for insights."""
        now = datetime.now(timezone.utc)
        analysis = {
            "total_logs": len(self.recent_symptom_logs),
            "symptom_counts_7d": Counter(),
            "symptom_counts_24h": Counter(),
            "symptom_counts_yesterday": Counter(),
            "recent_severities": {},
            "trends": [],
        }

        for log in self.recent_symptom_logs:
            # Parse recorded_at timestamp
            try:
                if isinstance(log.recorded_at, str):
                    recorded = datetime.fromisoformat(log.recorded_at.replace('Z', '+00:00'))
                else:
                    recorded = log.recorded_at
                if recorded.tzinfo is None:
                    recorded = recorded.replace(tzinfo=timezone.utc)
                else:
                    recorded = recorded.astimezone(timezone.utc)
            except (ValueError, AttributeError):
                continue

            days_ago = (now - recorded).days
            hours_ago = (now - recorded).total_seconds() / 3600

            for symptom in log.extracted.symptoms:
                symptom_name = symptom.symptom.lower()

                # Count symptoms in different time windows
                if days_ago < 7:
                    analysis["symptom_counts_7d"][symptom_name] += 1

                if hours_ago < 24:
                    analysis["symptom_counts_24h"][symptom_name] += 1

                # Yesterday = 24-48 hours ago
                if 24 <= hours_ago < 48:
                    analysis["symptom_counts_yesterday"][symptom_name] += 1

                # Track severity trends
                if symptom.severity_1_10:
                    if symptom_name not in analysis["recent_severities"]:
                        analysis["recent_severities"][symptom_name] = []
                    analysis["recent_severities"][symptom_name].append({
                        "value": symptom.severity_1_10,
                        "hours_ago": hours_ago
                    })

        # Inject cycle context for protocol evaluation
        if self._cycle_tag:
            analysis["cycle_context"] = {
                "cycle_day": self._cycle_tag.cycle_day,
                "cycle_phase": self._cycle_tag.cycle_phase,
                "cycle_number": self._cycle_tag.cycle_number,
                "has_prior_correlation": self._has_cycle_correlation,
            }

        return analysis

    def to_prompt_context(self) -> str:
        """Convert context to a string for the LLM prompt with historical analysis."""
        parts = []

        # User profile (long-term memory)
        if self.user_profile:
            profile = self.user_profile
            has_profile_data = (
                profile.conditions or profile.allergies or
                profile.regular_medications or
                profile.health_summary or
                getattr(profile, 'name', None)
            )
            if has_profile_data:
                parts.append("USER PROFILE (Long-term health context):")
                if getattr(profile, 'name', None):
                    parts.append(f"  Patient name: {profile.name} (address them by first name)")
                if profile.conditions:
                    parts.append(f"  Known conditions: {', '.join(profile.conditions)}")
                if profile.allergies:
                    parts.append(f"  Allergies: {', '.join(profile.allergies)}")
                if profile.regular_medications:
                    parts.append(f"  Regular medications: {', '.join(profile.regular_medications)}")
                if profile.health_summary:
                    parts.append(f"  Health summary: {profile.health_summary}")
                parts.append("")

        # Current extraction
        symptoms_str = ", ".join([
            f"{s.symptom}"
            + (f" (severity {s.severity_1_10})" if s.severity_1_10 else "")
            + (f" [triggers: {', '.join(s.triggers)}]" if s.triggers else "")
            + (f" [relieved by: {', '.join(s.relievers)}]" if s.relievers else "")
            for s in self.extraction.symptoms
        ]) or "none detected"
        parts.append(f"CURRENT LOG:")
        parts.append(f"  Symptoms: {symptoms_str}")

        actions_str = ", ".join([
            f"{a.name} ({a.dose_text})" if a.dose_text else a.name
            for a in self.extraction.actions_taken
        ]) or "none"
        parts.append(f"  Medications/actions taken: {actions_str}")

        # Vital signs from current log
        if self.extraction.vital_signs:
            vitals_str = ", ".join([
                f"{v.name}: {v.value}{(' ' + v.unit) if v.unit else ''}"
                for v in self.extraction.vital_signs
            ])
            parts.append(f"  Vital signs: {vitals_str}")

        parts.append(f"  Missing information: {', '.join(self.extraction.missing_fields) or 'none'}")

        # Transcript context tags (deterministic classification)
        tags = self._compute_context_tags()
        active_tags = [k for k, v in tags.items() if v]
        if active_tags:
            parts.append(f"\nCONTEXT TAGS (active for this transcript):")
            for tag in active_tags:
                parts.append(f"  - {tag}")

        # Historical analysis
        parts.append(f"\nHISTORICAL CONTEXT (last 30 days):")
        analysis = self._analyze_symptom_history()

        if analysis["total_logs"] > 0:
            parts.append(f"  Total previous logs: {analysis['total_logs']}")

            # Current symptoms' trend analysis (no raw counts — only actionable info)
            current_symptoms = [s.symptom.lower() for s in self.extraction.symptoms]
            for symptom in current_symptoms:
                count_7d = analysis["symptom_counts_7d"].get(symptom, 0)
                count_24h = analysis["symptom_counts_24h"].get(symptom, 0)
                count_yesterday = analysis["symptom_counts_yesterday"].get(symptom, 0)

                if count_7d > 0 or count_24h > 0:
                    parts.append(f"\n  '{symptom}' history:")

                    # Only show ACTIONABLE trend info — NEVER include raw counts
                    if count_7d >= 3 and count_24h > 0 and count_yesterday == 0:
                        parts.append(f"    - NEW PATTERN: {symptom} has become a recurring issue this week")
                    elif count_24h > count_yesterday and count_yesterday > 0:
                        parts.append(f"    - TREND: Increasing (more frequent than yesterday)")
                    elif count_24h < count_yesterday:
                        parts.append(f"    - TREND: Decreasing (less frequent than yesterday)")
                    elif count_7d >= 3:
                        parts.append(f"    - TREND: Stable (recurring pattern, already discussed with patient)")
                    else:
                        parts.append(f"    - TREND: Stable")

                    # Severity trend (always useful — shows change)
                    if symptom in analysis["recent_severities"]:
                        severities = analysis["recent_severities"][symptom]
                        if len(severities) >= 2:
                            recent = [s for s in severities if s["hours_ago"] < 24]
                            older = [s for s in severities if s["hours_ago"] >= 24]
                            if recent and older:
                                avg_recent = sum(s["value"] for s in recent) / len(recent)
                                avg_older = sum(s["value"] for s in older) / len(older)
                                if avg_recent > avg_older + 1:
                                    parts.append(f"    - SEVERITY TREND: Getting worse (avg {avg_recent:.1f} vs {avg_older:.1f})")
                                elif avg_recent < avg_older - 1:
                                    parts.append(f"    - SEVERITY TREND: Improving (avg {avg_recent:.1f} vs {avg_older:.1f})")

            # Top symptoms (names only, no counts — prevents LLM from parroting numbers)
            if analysis["symptom_counts_7d"]:
                top_symptoms = analysis["symptom_counts_7d"].most_common(3)
                parts.append(f"\n  Top symptoms this week: {', '.join([s for s, c in top_symptoms])}")

            # Vital sign trends from historical logs
            vital_sign_history: dict[str, list[str]] = {}
            for log in self.recent_symptom_logs:
                if hasattr(log.extracted, 'vital_signs') and log.extracted.vital_signs:
                    for v in log.extracted.vital_signs:
                        key = v.name.lower()
                        vital_sign_history.setdefault(key, []).append(v.value)
            if vital_sign_history:
                parts.append(f"\n  VITAL SIGN TRENDS:")
                for name, values in vital_sign_history.items():
                    trend_str = " → ".join(values[-5:])  # Last 5 readings
                    parts.append(f"    - {name}: {trend_str}")
        else:
            parts.append("  No previous logs (this is the first entry)")

        # Recent medications
        parts.append(f"\nMEDICATION HISTORY:")
        if self.recent_med_logs:
            for med in self.recent_med_logs[:5]:
                try:
                    taken_at = med.taken_at.replace(tzinfo=None) if med.taken_at.tzinfo else med.taken_at
                    now = datetime.now(timezone.utc).replace(tzinfo=None)
                    hours_ago = (now - taken_at).total_seconds() / 3600
                    parts.append(f"  - {med.medication_name}: {hours_ago:.1f} hours ago")
                except Exception:
                    parts.append(f"  - {med.medication_name}: recently")
        else:
            parts.append("  No recent medication logs")

        # Known medication doses (from profile and history)
        known_doses = self._build_known_medication_doses()
        if known_doses:
            parts.append(f"\nKNOWN MEDICATION DOSES (already established — do NOT ask for these):")
            for med, dose in known_doses.items():
                parts.append(f"  - {med}: {dose}")

        # Image analysis context (for medical urgency assessment)
        if self.image_analysis:
            parts.append(f"\nIMAGE ANALYSIS (current photo):")
            parts.append(f"  Clinical description: {self.image_analysis.clinical_description}")
            parts.append(f"  Lesion detected: {self.image_analysis.lesion_detected}")
            if self.image_analysis.skin_lesion:
                sl = self.image_analysis.skin_lesion
                parts.append(f"  Type: {sl.lesion_type}")
                parts.append(f"  Color: {sl.color}")
                parts.append(f"  Size: {sl.size_estimate}")
                parts.append(f"  Texture: {sl.texture}")
                if sl.predicted_condition:
                    parts.append(f"  Predicted condition: {sl.predicted_condition} (confidence: {sl.condition_confidence:.0%})")

            # Previous image conditions for persistence tracking
            prev_conditions = []
            for log in (self.recent_symptom_logs or []):
                if (log.image_analysis and log.image_analysis.skin_lesion
                        and log.image_analysis.skin_lesion.predicted_condition):
                    prev_conditions.append(log.image_analysis.skin_lesion.predicted_condition)
            if prev_conditions:
                parts.append(f"  Previous image conditions (recent history): {', '.join(prev_conditions[:5])}")
            else:
                parts.append(f"  No previous image analysis on record (first photo)")

        # Cycle context (for hormonal pattern awareness)
        if self._cycle_tag:
            tag = self._cycle_tag
            parts.append(f"\nCYCLE CONTEXT:")
            parts.append(f"  Current cycle day: Day {tag.cycle_day}")
            parts.append(f"  Cycle phase: {tag.cycle_phase}")
            if self._has_cycle_correlation:
                parts.append(f"  NOTE: Historical data shows symptom-cycle correlations around these days.")
            parts.append(f"  Use this to provide cycle-aware context in your response.")

        # Ambient monitoring data (HeAR sessions)
        if self.ambient_summary:
            parts.append(f"\nAMBIENT MONITORING DATA (HeAR):")
            parts.append(f"  {self.ambient_summary}")

        return "\n".join(parts)


class LLMResponseGenerator:
    """Uses MedGemma to generate context-aware, natural responses with insight nudges."""

    SYSTEM_PROMPT = """<role>
You are a warm, perceptive health companion who helps people track their symptoms. You talk to a real person — not a data feed. When someone tells you they're in pain, your first instinct is to acknowledge them as a person, not summarize their data back to them.

ANTI-PATTERN — never open with: "I understand you're experiencing [symptom] ([severity]) and [symptom] today." This is a robotic data readback that ignores the human being. Instead, respond to what matters most in what they said — the story, the frustration, the effort, the fear — then weave in clinical context naturally.

Be concise (1-2 short sentences). PERSONALIZE based on their profile, conditions, and history. You are NOT a passive recorder — you are a thoughtful companion who notices patterns and asks smart questions.

Tone: Talk to the patient as an equal adult. NEVER use cheerleader language.
BAD: "That's fantastic progress!", "Keep up the great work!", "That's wonderful news!", "What a great first step!"
GOOD: "That's a real change from two weeks ago.", "Sounds like the rehab is making a difference.", "Down to two cigarettes — that's not nothing."
Match the emotional register of what they shared — don't be more excited about their health than they are. If they're understated about progress, you should be too.
</role>

<critical_safety_rules>
- NEVER recommend stopping, skipping, or changing a medication or dose
- NEVER attribute causation definitively — use hedging: "can be", "is sometimes associated with", "some people experience"
- ALWAYS include "worth discussing with your doctor" when connecting symptoms to medications or conditions
- NEVER suggest a diagnosis ("you might have X", "this could be Y condition")
- NEVER contradict advice the patient received from their doctor
- NEVER state specific vital sign numbers or trends unless they appear in the patient's actual log data provided above. If you're unsure of exact values, say "your readings have been fluctuating" instead of citing numbers. Do NOT invent or extrapolate vital sign values.
- When referencing medication doses from the patient's profile, use the EXACT dose text (e.g., "Metformin 500mg", never "Metformin 5mg"). Do not abbreviate or round doses.
- When in doubt, say LESS, not more — track the data, don't interpret it
- NEVER fabricate clinical history. Only reference events documented in the patient's actual logs. Do NOT say "history of recurring X" or "multiple episodes of X" unless multiple distinct episodes appear in the log data.
</critical_safety_rules>

<pattern_handling>
Observed patterns in the patient's profile are for YOUR clinical reasoning only.
NEVER quote, paraphrase, or read back pattern text to the patient.
ANTI-PATTERNS — never say any variation of:
- "Respiratory symptoms often occur together"
- "Upper respiratory infection symptoms can precede or coincide with..."
- "Rescue inhaler use increases during exacerbations"
- Any sentence that reads like a textbook description of the patient's condition
Instead, use pattern knowledge to inform your questions and acknowledgments WITHOUT stating the pattern itself.
</pattern_handling>

<emotional_attunement>
When the patient shares something emotionally significant, your FIRST sentence must acknowledge the emotion — before any clinical content or questions.

Recognize these emotional beats:
- IDENTITY GRIEF: Patient references lost abilities or past self [identity-loss statement]. Acknowledge the specific loss — this is grief, not small talk.
- FEAR/VULNERABILITY: "I'm scared", "what if it happens again", being driven to the ER. Meet fear with reassurance, not clinical data.
- FRUSTRATION WITH LIMITATIONS: Patient expresses resignation about their condition [frustration statement]. Validate the frustration without silver-lining it.
- RELATIONSHIP MOMENTS: Family concern (spouse, child, caregiver). Acknowledge the care and support around them.
- MILESTONE MOMENTS: Starting rehab, quitting a habit, showing doctor their data. Acknowledge genuinely without cheerleading.

When CONTEXT TAG `has_identity_grief` is present, your first sentence MUST name what they lost.
BAD: "I'm glad the medication helped. How bad is the fatigue on a 1-10 scale?" (ignores emotion, jumps to data)
GOOD: "Not being able to do what you used to — that's a real loss, and it makes sense to feel frustrated. How are you holding up?"
</emotional_attunement>

<boundary_respect>
When the patient sets a boundary — explicitly or implicitly — RESPECT IT IMMEDIATELY.

Boundary signals: "one thing at a time", "I don't want to talk about that", "let's focus on X", "not right now", "enough about that", "I know, I know", "okay?" at the end of a statement.

When you detect a boundary:
- Do NOT reinforce the topic they're pushing back on (even if medically important)
- Do NOT validate their doctor's advice about the avoided topic
- Acknowledge their boundary with genuine respect: "Fair enough" / "Understood" / "That makes sense"
- Redirect to what they ARE willing to engage with
- You can return to the topic in a FUTURE session if it's medically important — not in this one

When CONTEXT TAG `has_boundary_pushback` is present, respect it immediately.
Example — patient sets a boundary [boundary statement]:
  BAD: [Reinforces the topic they're pushing back on with medical facts or pattern text]
  GOOD: "Fair enough — you've got a lot on your plate. How's [redirect to what they ARE engaging with]?"
</boundary_respect>

<response_logic>
<if_user_asks_question>
If input contains "?", "is that normal", "should I", "do you think", "could this be":
- In "acknowledgment": ADDRESS their question with safety hedging and profile context. You cannot diagnose.
- In "immediate_question": STILL ask for any missing info (dose, severity). Do BOTH — answer AND ask.
Example: User asks about a side effect of their medication →
  acknowledgment: "[Side effect] can be a common side effect of [medication]. Worth mentioning to your doctor if it persists."
  immediate_question: "On a scale of 1-10, how bad is the [symptom]?"
</if_user_asks_question>

<if_recurring_symptoms>
If a symptom appears 3+ times in history, ask a CONTEXTUAL question about WHY — not just "how bad is it".
Examples: "Are you taking your pills on an empty stomach or with food?", "Does the dizziness happen when you stand up quickly?"
</if_recurring_symptoms>

<if_missing_info>
If dose of a NEW medication or severity for pain/nausea/dizziness is missing, ask via immediate_question. Do NOT schedule a check-in instead. Do NOT ask for doses already listed under KNOWN MEDICATION DOSES — acknowledge naturally: "Noted you took your [medication] [dose]."
</if_missing_info>

<if_image_analysis>
If the log includes IMAGE ANALYSIS with a predicted skin condition:
- Assess urgency (melanoma/cellulitis → prompt attention; mild eczema/insect bite → manageable).
- Check PERSISTENCE: same condition in prior images = ongoing = more concerning.
- Consider profile (pre-existing conditions, allergies, medications).
- If doctor visit warranted: set should_schedule_checkin true, use general terms ("the skin finding"), do NOT name the specific condition.
- First-time non-urgent: check-in in 48-72 hours. Persistent/urgent: recommend doctor visit.
</if_image_analysis>

<if_drug_disease_interaction>
When a NEW medication is reported AND the user's profile has conditions/medications that interact:
- Flag the specific interaction risk with hedging language and "worth discussing with your doctor"
- Ask for the relevant vital sign measurement (BP, glucose, etc.)
- If the patient DISMISSES a concerning vital sign, DO NOT accept — acknowledge but note the elevated value and recommend monitoring.
</if_drug_disease_interaction>

<if_new_medication>
When a NEW medication is reported that is NOT in the user's known regular_medications list:
- Cross-reference against the user's ALLERGY list in their profile
- If the medication is SAFE despite an allergy (e.g., azithromycin for a penicillin-allergic patient): proactively confirm safety: "Good — azithromycin is in a different drug class from penicillin, so your allergy isn't a concern here."
- If potentially UNSAFE (same drug class as allergy): flag immediately with escalation
- Always verify dose and frequency for new medications
</if_new_medication>

<if_ambient_data>
If AMBIENT MONITORING DATA shows concerning trends (cough spike, wheezing, irregular rhythm, sleep disruption):
- Reference the pattern objectively ("Your overnight monitoring detected a change"). Do NOT quote raw numbers.
- Use ambient data to inform smarter questions. If data contradicts self-report, gently probe.
</if_ambient_data>

<if_patient_narrative>
When the patient shares a personal story, external suggestion (coworker, family member, article), self-research about a condition, or connects their own dots about their health:
- Center your acknowledgment on THEIR insight and agency — they did the work of noticing, researching, or connecting patterns. Honor that.
- Validate their effort and concern. Reference specific dates or entries from their logs as evidence, without restating stored pattern text or confirming any specific diagnosis.
- Do NOT recite their symptoms or severities back at them — they already know.
- If they mention bringing data to a doctor, affirm this as a strong next step.
- The question should build on their discovery, not retreat to basic clinical data gathering.
When CONTEXT TAG `has_self_research` is present, center your response on their agency.
Example — patient shares self-research about a condition [self-research statement]:
  BAD: [Recites symptom data back at them, ignoring their initiative]
  GOOD: "That took real initiative to look into. Your tracking data supports what you're noticing — [reference specific logged evidence, no pattern parroting]. That's solid information to bring to your doctor."
</if_patient_narrative>

</response_logic>

<insight_rules>
Use the history labels to inform your tone, not your wording:
- "NEW PATTERN" → acknowledge this is new for them. "TREND: Increasing" → express concern about worsening.
- "SEVERITY TREND: Improving" → note improvement warmly. New correlation → mention it naturally.
- "Stable (recurring, already noted)" → set insight_nudge to null.
NEVER echo TREND labels, codes, or system annotations in your response. NEVER say "You've logged X times."
</insight_rules>

<question_priority>
Priority order — ask ONE question, pick highest applicable:
1. User asked a question → address it
2. Respiratory patient + respiratory symptoms, no SpO2 → ask SpO2
3. Drug interaction detected → ask relevant vital
4. BP >= 140/90 → ask about headache/vision/chest pain
5. New medication without dose → ask dose
6. Patient shares self-discovery, external suggestion, or health narrative → ask a question that builds on their insight (NOT a question the tracked data already answers)
7. Fever without temperature → ask temperature
8. High-priority symptom without severity → ask 1-10
9. Recurring symptom (3+) → contextual WHY
10. Behavioral change (smoking, exercise, diet, rehab) → ask about experience/impact
11. Emotional state or life event shared → ask how it's affecting them
12. Patient followed a recommendation → ask about the outcome ("How has that been working?")
CRITICAL: Exactly ONE question. If multiple missing, pick highest priority.
</question_priority>

<tool_calling>
You have two categories of tools you can invoke autonomously:

CLINICAL PROTOCOL TOOLS — invoke when your clinical assessment matches a protocol.
Format: "invoke_protocol:<protocol_id>"
Set reason_code to a short snake_case description of why (e.g., "fever_missing_temperature").

Safety-critical (always invoke if criteria met):
- invoke_protocol:medication_interaction — Drug-disease or drug-drug interaction detected (NSAID+iron+GI, steroid+HTN, anticoagulant bleeding risk)
- invoke_protocol:compound_symptom_escalation — 3+ symptoms with abnormal vitals (glucose >200/<70, BP >=160/100) or medication nonadherence

Clinical protocols:
- invoke_protocol:fever_protocol — Fever, temperature, or chills reported. Ask temperature if missing.
- invoke_protocol:asthma_respiratory_protocol — Respiratory symptoms in patient with asthma/COPD. Ask SpO2 if missing.
- invoke_protocol:hypertension_protocol — Blood pressure >= 140/90. Ask about headache, vision changes, chest discomfort.
- invoke_protocol:gastrointestinal_protocol — Severe or persistent GI symptoms (nausea, vomiting, diarrhea, blood in stool).
- invoke_protocol:skin_lesion_escalation — Skin lesion with concerning features (melanoma, spreading, fever). Always escalate melanoma.
- invoke_protocol:headache_protocol — Headache or migraine. Ask severity if missing.
- invoke_protocol:menstrual_cycle_protocol — Cycle-linked symptom in patient with active cycle tracking.
- invoke_protocol:medication_missing_dose_priority — Medication taken without dose specified. Ask what dose.
- invoke_protocol:generic_severity_fallback — High-priority symptom (pain, nausea, dizziness) without severity. Ask 1-10 scale.

Invoke the HIGHEST priority protocol that applies. Safety-critical over clinical.
Omit protocol tool from tool_calls if the log is routine with no clinical concern.

ANALYSIS TOOLS:
- run_watchdog_now — Deep longitudinal pattern analysis across the patient's full history.

Review the patient's history provided in context. Invoke run_watchdog_now when your clinical judgment indicates the patient is on a WORSENING trajectory that warrants deeper longitudinal review — for example:
- Vital signs deteriorating across recent logs
- Symptom severity escalating over days or weeks
- Medication failing, being overused, or skipped with consequences
- A dangerous pattern repeating (e.g., same symptoms recurring each cycle with increasing impact)
- Multiple concerning signals co-occurring in a single log

The watchdog performs cross-history analysis that you cannot do in a single response. When in doubt about a worsening trend, invoke it — false positives are acceptable, missed deterioration is not.

DO NOT invoke for: routine stable logs, minor single-symptom reports, or first entries with no prior history.

You may invoke BOTH a protocol tool AND run_watchdog_now in the same response.

SCHEDULING TOOLS:
- schedule_checkin:<hours>:<message> — Schedule a proactive check-in with the patient.
  <hours>: integer 1-24. <message>: personalized check-in question.
  Use when: medication was taken and you want to check efficacy, symptom is concerning but not urgent and you want to monitor progression, or patient reported a new pattern worth revisiting.
  Example: "schedule_checkin:4:How is the headache since taking the ibuprofen?"
  Example: "schedule_checkin:6:How is your breathing compared to this morning?"

ESCALATION TOOLS:
- escalate_clinician_alert:<reason> — Flag this log entry for clinician review.
  Allowed reasons: worsening_trajectory, medication_concern, vital_sign_concern, patient_distress, multi_system_involvement
  Use when:
  - Symptoms are worsening across multiple logs despite treatment
  - Vital signs are abnormal or trending in a concerning direction
  - Patient is in distress or struggling to manage
  - Multiple body systems are involved simultaneously
  - Medication may not be working or may be causing problems
  Protocols handle clinical follow-up questions — escalation flags the entry for a human clinician to review. These are complementary, not competing. You should use BOTH when appropriate.
  This does NOT replace emergency advice — always tell patients to call emergency services for acute emergencies.
  Example: "escalate_clinician_alert:worsening_trajectory"

MULTIMODAL SENSING TOOLS (automatic — you do not invoke these, but their results appear in your context):
- analyze_respiratory_audio (HeAR) — When the patient uploads audio, HeAR classifies lung sounds (normal/crackle/wheeze), cough type (dry/wet), severity, and detects acoustic events (cough spikes, wheezing episodes, sleep apnea indicators). Results appear in the AMBIENT MONITORING DATA section. Use these objective measurements to probe discrepancies with self-report and inform clinical follow-up.
- analyze_skin_image (MedSigLIP) — When the patient uploads a photo, MedSigLIP classifies skin findings across five dimensions: lesion type, color, size, texture, and predicted condition via zero-shot medical image-text matching. Results appear in the IMAGE ANALYSIS section. Use these to assess urgency and track progression across visits.

These tools run automatically when media is present. Reference their findings naturally in your response — they provide objective clinical data that complements the patient's subjective report.

tool_calls format: ["invoke_protocol:fever_protocol", "run_watchdog_now", "schedule_checkin:4:How is the pain?", "escalate_clinician_alert:worsening_trajectory"] or []
</tool_calling>

<checkin_rules>
Schedule check-ins when proactive follow-up adds value. You MAY combine a check-in with an immediate question if both are clinically useful.

Medication efficacy follow-ups (use the timing below):
- ibuprofen/advil/motrin: 2h, tylenol/acetaminophen: 1h, naproxen/aleve: 4h
- aspirin: 1h, tramadol: 2h, cyclobenzaprine: 2h, sumatriptan: 1h

Symptom progression checks (4-8 hours):
- Concerning but not urgent symptom worth monitoring
- New pattern the patient should track
- Patient reports a change they're uncertain about

Personalize check-in messages based on the patient's situation and language.
</checkin_rules>

<output_format>
Respond with valid JSON only. IMPORTANT — tool_calls MUST be the first field:
{{
    "tool_calls": [],
    "reason_code": null,
    "acknowledgment": "1-2 sentences. Respond to the PERSON — their story, effort, fear, frustration — not their data points. NEVER open with 'I understand you're experiencing [symptom] ([severity])'. If they shared something personal, acknowledge THAT, not their numbers.",
    "insight_nudge": "Pattern insight from history (null if stable/no pattern). Never raw counts.",
    "should_ask_question": true,
    "immediate_question": "Single follow-up question per priority order above.",
    "question_rationale": "One-sentence clinical reason for asking this question, or null.",
    "should_schedule_checkin": false,
    "checkin_hours": 2,
    "checkin_message": "Personalized check-in message."
}}
tool_calls: REQUIRED. Always include. Use "invoke_protocol:<id>" for clinical protocols, "run_watchdog_now" for longitudinal analysis, "schedule_checkin:<hours>:<message>" for proactive follow-up, and "escalate_clinician_alert:<reason>" to flag for clinician review. Empty [] only for completely routine, uneventful logs.
reason_code: Short snake_case reason for protocol invocation (e.g., "fever_missing_temperature"), or null.
question_rationale: Why the agent chose this specific follow-up question (e.g., "Fever reported without temperature reading — need numeric value for severity assessment"). Null when should_ask_question is false.
should_ask_question = true when: user asked a question, dose missing, severity missing, recurring symptom needs context, patient reports a behavioral change or emotional state worth exploring, patient shares progress worth acknowledging with a follow-up, or patient reports adherence to a recommendation. False ONLY for completely routine, uneventful check-ins with no new information.
</output_format>"""

    def __init__(self):
        self.client = get_medgemma_client()
        # Check mode
        self._use_stub = os.environ.get("USE_STUB_MEDGEMMA", "").lower() == "true"
        self._use_local = os.environ.get("USE_LOCAL_MEDGEMMA", "").lower() == "true"
        self._use_stub_response = os.environ.get("USE_STUB_RESPONSE_GEN", "").lower() == "true"
        self._use_protocol_followup = os.environ.get("USE_PROTOCOL_FOLLOWUP", "").lower() == "true"
        self._protocol_shadow_mode = os.environ.get("PROTOCOL_SHADOW_MODE", "").lower() == "true"
        self._use_replan = os.environ.get("USE_REPLAN_PASS", "").lower() == "true"
        self._protocol_registry = get_protocol_registry()
        # Note: per-request metadata (fallback_reason, mode) is returned from
        # generate() rather than stored on self, to avoid thread-safety issues
        # with the singleton pattern.

    def _build_protocol_context(self, context: ResponseContext) -> ProtocolContext:
        """Build protocol context from response context."""
        # Extract recent protocol IDs from recent logs' followup questions
        # (LogEntry doesn't store protocol_id, so we infer from question patterns)
        recent_protocol_ids: list[str] = []
        # Use the most recent log's recorded_at as reference instead of
        # wall-clock time.  This ensures cooldown works correctly when
        # processing historical / simulated logs whose dates are in the past.
        recent_logs = context.recent_symptom_logs or []
        if recent_logs:
            ref = recent_logs[0].recorded_at
            if isinstance(ref, str):
                ref = datetime.fromisoformat(ref.replace('Z', '+00:00'))
            if ref.tzinfo is None:
                ref = ref.replace(tzinfo=timezone.utc)
            now = ref
        else:
            now = datetime.now(timezone.utc)
        for log in recent_logs[:14]:
            try:
                recorded = log.recorded_at
                if isinstance(recorded, str):
                    recorded = datetime.fromisoformat(recorded.replace('Z', '+00:00'))
                if recorded.tzinfo is None:
                    recorded = recorded.replace(tzinfo=timezone.utc)
                days_ago = (now - recorded).days
                if days_ago > 7:
                    continue
                q = (log.followup_question or "").lower()
                if "cycle" in q and "phase" in q:
                    recent_protocol_ids.append("menstrual_cycle_protocol")
                    # Track correlation-specific fires separately
                    if "pattern" in q:
                        recent_protocol_ids.append("menstrual_cycle_correlation")
            except Exception:
                continue

        return ProtocolContext(
            extraction=context.extraction,
            user_id=context.user_id,
            user_profile=context.user_profile,
            symptom_history=context._analyze_symptom_history(),
            image_analysis=context.image_analysis,
            known_medication_doses=context._build_known_medication_doses(),
            recent_protocol_ids=recent_protocol_ids,
        )

    def _contains_safety_sensitive_symptoms(self, context: ResponseContext) -> bool:
        transcript = context.extraction.transcript.lower()
        if any(term in transcript for term in SAFETY_SENSITIVE_TERMS):
            return True
        for symptom in context.extraction.symptoms:
            name = symptom.symptom.lower()
            if any(term in name for term in SAFETY_SENSITIVE_TERMS):
                return True
        return False

    async def _llm_safety_check(self, text: str) -> bool:
        """Validate generated text doesn't contain medical advice using LLM classifier.

        Returns True if safe, False if unsafe. Fails closed (returns False on error).
        """
        try:
            response = await self.client.generate_agent_response(
                _SAFETY_CLASSIFIER_PROMPT.format(text=text),
                max_tokens=10,
            )
            return "UNSAFE" not in response.upper()
        except Exception as e:
            logger.warning(f"Safety classifier failed, blocking output: {e}")
            return False  # Fail closed

    def _sanitize_llm_question(
        self,
        question: Optional[str],
        context: ResponseContext,
        protocol_matched: bool,
    ) -> Optional[str]:
        """Constrain LLM fallback question to one safe, concise question."""
        if not question:
            return None

        if (not protocol_matched) and self._contains_safety_sensitive_symptoms(context):
            return None

        clean = " ".join(question.strip().split())
        if not clean:
            return None

        # Keep only the first question if model emitted multi-question chains.
        if "?" in clean:
            clean = clean.split("?", 1)[0].strip() + "?"
        elif len(clean) > 220:
            clean = clean[:220].rstrip(". ") + "?"
        elif not clean.endswith("?"):
            clean = clean.rstrip(". ") + "?"

        if len(clean.split()) < 3:
            return None

        return clean

    def _build_llm_candidate_checkin(
        self,
        data: dict,
        context: ResponseContext,
    ) -> Optional[ScheduledCheckin]:
        """Build candidate check-in directly from model output."""
        if not data.get("should_schedule_checkin") or not data.get("checkin_message"):
            return None

        hours = data.get("checkin_hours", 2)
        try:
            hours = max(1, min(24, int(hours)))
        except (TypeError, ValueError):
            hours = 2

        checkin_type = (
            CheckinType.MEDICATION_FOLLOWUP
            if context.extraction.actions_taken
            else CheckinType.SYMPTOM_PROGRESSION
        )
        checkin_context = {
            "hours": hours,
            "transcript": context.extraction.transcript[:200],
        }
        if context.extraction.actions_taken:
            checkin_context["medication_name"] = context.extraction.actions_taken[0].name

        return ScheduledCheckin(
            id=f"checkin_{uuid.uuid4().hex[:12]}",
            user_id=context.user_id,
            checkin_type=checkin_type,
            scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=hours),
            message=data["checkin_message"],
            context=checkin_context,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    def _build_protocol_checkin(
        self,
        decision: ProtocolDecision,
        context: ResponseContext,
    ) -> Optional[ScheduledCheckin]:
        """Create check-in object from protocol decision."""
        if not decision.schedule_checkin:
            return None

        hours = decision.checkin_hours if decision.checkin_hours is not None else 2
        hours = max(1, min(24, hours))
        checkin_type = decision.checkin_type or (
            CheckinType.MEDICATION_FOLLOWUP
            if context.extraction.actions_taken
            else CheckinType.SYMPTOM_PROGRESSION
        )

        message = decision.checkin_message
        if not message:
            if context.extraction.actions_taken:
                med_name = context.extraction.actions_taken[0].name
                message = f"Checking back after {med_name}. How are you feeling now?"
            else:
                symptom = context.extraction.symptoms[0].symptom if context.extraction.symptoms else "your symptoms"
                message = f"I'll check back in a bit. How is {symptom} now?"

        checkin_context = {
            "hours": hours,
            "transcript": context.extraction.transcript[:200],
            "protocol_id": decision.protocol_id,
            "reason_code": decision.reason_code,
        }
        if context.extraction.actions_taken:
            checkin_context["medication_name"] = context.extraction.actions_taken[0].name

        return ScheduledCheckin(
            id=f"checkin_{uuid.uuid4().hex[:12]}",
            user_id=context.user_id,
            checkin_type=checkin_type,
            scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=hours),
            message=message,
            context=checkin_context,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    def _apply_protocol_mode(
        self,
        context: ResponseContext,
        protocol_decision: ProtocolDecision,
        fallback_question: Optional[str],
        fallback_checkin: Optional[ScheduledCheckin],
    ) -> Tuple[Optional[str], Optional[ScheduledCheckin], str, bool]:
        """Apply protocol output with constrained LLM fallback for long-tail cases."""
        question = protocol_decision.immediate_question
        checkin = self._build_protocol_checkin(protocol_decision, context)
        safety_mode = "protocol"
        llm_fallback_used = False
        protocol_matched = bool(protocol_decision.protocol_id)

        # Detect if user asked a direct question — they deserve a response
        user_asked = _user_asked_question(context.extraction.transcript)

        if not question and not checkin:
            # Fallback is only for true long-tail (no protocol matched).
            if not protocol_matched:
                if self._contains_safety_sensitive_symptoms(context):
                    fallback_checkin = None
                    sanitized_question = None
                else:
                    sanitized_question = self._sanitize_llm_question(
                        fallback_question,
                        context=context,
                        protocol_matched=False,
                    )
                if sanitized_question or fallback_checkin:
                    question = sanitized_question
                    checkin = fallback_checkin
                    safety_mode = "llm_fallback"
                    llm_fallback_used = True

        # If user asked a direct question and LLM generated a response for it,
        # always prefer the LLM's answer over protocol questions.
        # The LLM's system prompt already instructs it to ask about missing
        # critical info (doses, severity), so safety is maintained via the prompt.
        if user_asked and fallback_question:
            sanitized = self._sanitize_llm_question(
                fallback_question,
                context=context,
                protocol_matched=protocol_matched,
            )
            if sanitized:
                question = sanitized
                safety_mode = "llm_user_question_priority"
                llm_fallback_used = True

        # Protocol questions take absolute priority over check-ins.
        # A protocol that asks a question needs the answer NOW (e.g., medication dose).
        # EXCEPTION: when a protocol explicitly sets BOTH question AND check-in
        # (e.g., sputum color change needs SpO2 answer NOW + scheduled follow-up).
        if checkin is not None and question is not None:
            if protocol_matched and protocol_decision.immediate_question and protocol_decision.schedule_checkin:
                # Protocol explicitly designed both — keep BOTH
                pass
            elif protocol_matched and protocol_decision.immediate_question and question == protocol_decision.immediate_question:
                # Protocol asked this question but didn't want a check-in — drop check-in
                checkin = None
            else:
                # LLM/user-question — keep it, drop the check-in
                checkin = None

        return question, checkin, safety_mode, llm_fallback_used

    def _apply_followup_policy(
        self,
        context: ResponseContext,
        candidate_question: Optional[str],
        candidate_checkin: Optional[ScheduledCheckin],
        llm_protocol_id: Optional[str] = None,
        llm_reason_code: Optional[str] = None,
        llm_question_rationale: Optional[str] = None,
    ) -> Dict[str, object]:
        """Compute final follow-up decision using LLM-first protocol selection.

        Three paths:
        1. Safety override — hardcoded safety-critical protocol forces over LLM choice
        2. LLM protocol — LLM selected a valid protocol, trust its selection + question
        3. Fallback — no LLM protocol, use existing hardcoded ProtocolRegistry path
        """
        protocol_context = self._build_protocol_context(context)
        protocol_decision = self._protocol_registry.evaluate(protocol_context)

        # Validate LLM protocol_id against known catalog
        if llm_protocol_id and llm_protocol_id not in VALID_PROTOCOL_IDS:
            logger.warning("LLM returned invalid protocol_id: %s", llm_protocol_id)
            llm_protocol_id = None

        # Safety-critical override: hardcoded protocols take precedence
        use_safety_override = (
            protocol_decision.protocol_id in SAFETY_CRITICAL_PROTOCOLS
            and llm_protocol_id != protocol_decision.protocol_id
        )

        if use_safety_override:
            # PATH 1: Safety override — force hardcoded protocol
            logger.info(
                "Safety override: LLM chose %s, forcing %s",
                llm_protocol_id, protocol_decision.protocol_id,
            )
            question, checkin, _, _ = self._apply_protocol_mode(
                context=context,
                protocol_decision=protocol_decision,
                fallback_question=candidate_question,
                fallback_checkin=candidate_checkin,
            )
            safety_mode = "safety_override"
            llm_fallback_used = False
            final_protocol_id = protocol_decision.protocol_id
            final_reason_code = protocol_decision.reason_code
            question_rationale = f"Safety override: {final_protocol_id} — {final_reason_code}"
        elif llm_protocol_id:
            # PATH 2: LLM protocol — but check if the hardcoded registry matched
            # a higher-priority protocol. The registry ordering is a guaranteed
            # floor: the LLM cannot downgrade to a lower-priority protocol.
            registry_id = protocol_decision.protocol_id
            use_registry_override = False
            if registry_id and registry_id != llm_protocol_id:
                reg_idx = self._protocol_registry.get_priority_index(registry_id)
                llm_idx = self._protocol_registry.get_priority_index(llm_protocol_id)
                if reg_idx < llm_idx:
                    use_registry_override = True

            if use_registry_override:
                # Registry matched a higher-priority protocol — override LLM
                logger.info(
                    "Registry priority override: LLM chose %s (idx %d), forcing %s (idx %d)",
                    llm_protocol_id, llm_idx, registry_id, reg_idx,
                )
                question, checkin, _, _ = self._apply_protocol_mode(
                    context=context,
                    protocol_decision=protocol_decision,
                    fallback_question=candidate_question,
                    fallback_checkin=candidate_checkin,
                )
                safety_mode = "registry_priority_override"
                llm_fallback_used = False
                final_protocol_id = protocol_decision.protocol_id
                final_reason_code = protocol_decision.reason_code
                question_rationale = f"Registry priority override: {final_protocol_id} — {final_reason_code}"
            else:
                # LLM agrees with registry (or no registry match).
                # Prefer the protocol's deterministic question when available —
                # it's clinically authored vs. LLM-generated.
                if protocol_decision.immediate_question is not None:
                    question = protocol_decision.immediate_question
                else:
                    question = candidate_question
                checkin = candidate_checkin
                safety_mode = "llm_protocol"
                llm_fallback_used = False
                final_protocol_id = llm_protocol_id
                # When the hardcoded protocol wants to escalate, use its precise
                # reason_code so it matches the deterministic clinician alert mappings.
                # The hardcoded protocol is the safety authority regardless of which
                # protocol the LLM chose.
                if protocol_decision.escalation_flag and protocol_decision.reason_code:
                    final_reason_code = protocol_decision.reason_code
                else:
                    final_reason_code = llm_reason_code or "llm_selected"
                question_rationale = llm_question_rationale
                # Question takes priority over checkin
                if checkin and question:
                    checkin = None
        else:
            # PATH 3: No LLM protocol — existing hardcoded path (unchanged)
            legacy_question = _build_immediate_question(
                context=context,
                has_scheduled_checkin=candidate_checkin is not None,
            )
            legacy_checkin = candidate_checkin

            if self._use_protocol_followup:
                question, checkin, safety_mode, llm_fallback_used = self._apply_protocol_mode(
                    context=context,
                    protocol_decision=protocol_decision,
                    fallback_question=candidate_question,
                    fallback_checkin=candidate_checkin,
                )
            else:
                question, checkin = legacy_question, legacy_checkin
                safety_mode = "legacy"
                llm_fallback_used = False

            final_protocol_id = protocol_decision.protocol_id
            final_reason_code = protocol_decision.reason_code
            # Use protocol rationale if protocol provided question, else LLM's rationale
            if final_protocol_id and safety_mode == "protocol":
                question_rationale = f"Protocol: {final_protocol_id} — {final_reason_code}"
            else:
                question_rationale = llm_question_rationale

        # Shadow mode logging
        if self._protocol_shadow_mode:
            shadow_question, shadow_checkin, shadow_mode, shadow_fallback = self._apply_protocol_mode(
                context=context,
                protocol_decision=protocol_decision,
                fallback_question=candidate_question,
                fallback_checkin=candidate_checkin,
            )
            log_debug(
                "protocol_shadow_decision",
                protocol_id=protocol_decision.protocol_id,
                llm_protocol_id=llm_protocol_id,
                reason_code=protocol_decision.reason_code,
                protocol_question=shadow_question,
                protocol_has_checkin=bool(shadow_checkin),
                shadow_safety_mode=shadow_mode,
                shadow_llm_fallback_used=shadow_fallback,
                active_safety_mode=safety_mode,
                active_question=question,
                active_has_checkin=bool(checkin),
            )

        # Post-processing guards: suppress redundant LLM-generated questions
        # that duplicate information already present in the extraction.
        # Safety override uses protocol-crafted questions — skip suppression.
        if question and safety_mode != "safety_override":
            question = self._suppress_redundant_question(question, context)

        # Clear rationale when question was suppressed
        if question is None:
            question_rationale = None

        # Defensive: ensure rationale is populated when protocol provided a question
        if question is not None and question_rationale is None and final_protocol_id:
            question_rationale = f"Protocol: {final_protocol_id} — {final_reason_code}"

        return {
            "immediate_question": question,
            "scheduled_checkin": checkin,
            "protocol_id": final_protocol_id,
            "reason_code": final_reason_code,
            "safety_mode": safety_mode,
            "llm_fallback_used": llm_fallback_used,
            "question_rationale": question_rationale,
        }

    # ------------------------------------------------------------------
    # Post-processing guards
    # ------------------------------------------------------------------

    _SEVERITY_ASK_RE = re.compile(
        r"scale of 1|rate your|how severe|severity|1.to.10|1.10",
        re.IGNORECASE,
    )
    _DOSE_ASK_RE = re.compile(
        r"what dose|what dosage|how (?:much|many)|what strength|mg|milligram",
        re.IGNORECASE,
    )
    _MED_NOT_STARTED_MARKERS = [
        "haven't started", "havent started", "not yet", "going to start",
        "plan to start", "bought but", "picked up but", "not started",
    ]

    def _suppress_redundant_question(
        self,
        question: str,
        context: ResponseContext,
    ) -> Optional[str]:
        """Suppress LLM-generated questions that duplicate extracted data."""
        # Guard 1: Don't re-ask severity when already extracted
        if self._SEVERITY_ASK_RE.search(question):
            has_severity = any(
                s.severity_1_10 is not None
                for s in (context.extraction.symptoms or [])
            )
            if has_severity:
                return None

        # Guard 2: Don't ask dose for medications the patient hasn't started
        if self._DOSE_ASK_RE.search(question):
            transcript_lower = context.extraction.transcript.lower()
            if any(m in transcript_lower for m in self._MED_NOT_STARTED_MARKERS):
                return None

        return question

    async def generate(self, context: ResponseContext) -> tuple[AgentResponse, dict]:
        """Generate a context-aware agent response using MedGemma.

        Returns:
            Tuple of (AgentResponse, metadata dict with 'mode' and 'fallback_reason' keys).
        """
        # If using stub mode for responses, skip LLM call and use smart stub logic
        if self._use_stub_response or (self._use_stub and not self._use_local):
            return self._generate_stub_response(context), {
                "mode": "stub",
                "fallback_reason": "response_generator_stub_mode",
            }

        response = await self._generate_llm_response(context)
        if response.agent_trace.get("safety_blocked"):
            return response, {
                "mode": "llm_safety_fallback",
                "fallback_reason": "safety_classifier_blocked",
            }
        return response, {"mode": "llm", "fallback_reason": None}

    async def replan(
        self,
        agent_response: AgentResponse,
        tool_results: dict[str, str],
        context: "ResponseContext",
    ) -> AgentResponse:
        """Second pass: refine acknowledgment based on tool execution results."""
        if not self._use_replan or not tool_results:
            return agent_response
        # Stub/test mode: deterministic template-based refinement
        if self._use_stub_response or (self._use_stub and not self._use_local):
            return self._replan_deterministic(agent_response, tool_results)
        try:
            return await self._replan_llm(agent_response, tool_results, context)
        except Exception as e:
            logger.warning("Replan LLM call failed, keeping original: %s", e)
            agent_response.agent_trace["replan"] = {"error": str(e)}
            return agent_response

    async def _replan_llm(
        self,
        agent_response: AgentResponse,
        tool_results: dict[str, str],
        context: "ResponseContext",
    ) -> AgentResponse:
        """LLM-based replan: refine acknowledgment with tool execution context."""
        # Build concise tool results summary
        results_lines = []
        for tool, result in tool_results.items():
            results_lines.append(f"- {tool} → {result}")
        results_text = "\n".join(results_lines)

        prompt = (
            "You are refining a health companion response after tool execution.\n\n"
            f'Patient said: "{context.extraction.transcript}"\n'
            f'Your initial response: "{agent_response.acknowledgment}"\n'
        )
        if agent_response.immediate_question:
            prompt += f'Your follow-up question: "{agent_response.immediate_question}"\n'
        prompt += (
            f"\nTools executed and their results:\n{results_text}\n\n"
            "Revise ONLY the acknowledgment to naturally reference what actions were taken "
            '(e.g. "I\'ll check back in 4 hours..." or "I\'ve flagged this for your clinician..."). '
            "Do NOT change medical content, add diagnoses, or remove safety language. "
            "Keep a matter-of-fact tone. No cheerleader phrases ('fantastic', 'wonderful', 'keep up the great work').\n\n"
            'Respond with JSON only:\n{"revised_acknowledgment": "...", "revision_reason": "..."}'
        )

        response_text = await self.client.generate_agent_response(prompt, max_tokens=150)
        data = self._parse_json_response(response_text)
        revised = data.get("revised_acknowledgment", "").strip()

        if not revised:
            agent_response.agent_trace["replan"] = {"kept_original": True, "reason": "empty_revision"}
            return agent_response

        # Post-process same as first pass
        revised = clean_patient_text(revised)
        revised = _strip_hallucinated_vitals(revised, context.recent_symptom_logs, context.extraction)

        # Safety check revised text
        if not await self._llm_safety_check(revised):
            logger.warning("Replan safety check failed, keeping original")
            agent_response.agent_trace["replan"] = {"kept_original": True, "reason": "safety_blocked"}
            return agent_response

        # Factual consistency: reject if claims don't match tool outcomes
        if not self._factual_check_replan(revised, tool_results):
            logger.warning("Replan factual check failed, keeping original")
            agent_response.agent_trace["replan"] = {"kept_original": True, "reason": "factual_mismatch"}
            return agent_response

        original_ack = agent_response.acknowledgment
        agent_response.acknowledgment = revised
        agent_response.agent_trace["replan"] = {
            "original_acknowledgment": original_ack,
            "revised_acknowledgment": revised,
            "revision_reason": data.get("revision_reason", ""),
            "tool_results_seen": tool_results,
        }
        return agent_response

    def _factual_check_replan(self, revised: str, tool_results: dict[str, str]) -> bool:
        """Reject revised text that claims tool actions which didn't actually succeed."""
        revised_lower = revised.lower()

        # Check-in claim without a successful creation
        checkin_claimed = any(
            kw in revised_lower
            for kw in ("check back", "check in", "check-in", "follow up in")
        )
        checkin_created = any(
            k.startswith("schedule_checkin:") and "created" in v
            for k, v in tool_results.items()
        )
        if checkin_claimed and not checkin_created:
            return False

        # Clinician escalation claim without a successful escalation
        # Use action-claiming phrases (not bare "clinician" which appears in legitimate advice)
        escalation_claimed = any(
            kw in revised_lower
            for kw in (
                "i've flagged this",
                "i've noted this for",
                "flagged for your clinician",
                "notified your clinician",
                "alerted your clinician",
                "escalated to",
            )
        )
        escalation_created = any(
            k.startswith("escalate_clinician_alert:") and "added" in v
            for k, v in tool_results.items()
        )
        if escalation_claimed and not escalation_created:
            return False

        return True

    def _replan_deterministic(
        self,
        agent_response: AgentResponse,
        tool_results: dict[str, str],
    ) -> AgentResponse:
        """Template-based replan for stub/test mode."""
        original_ack = agent_response.acknowledgment
        additions: list[str] = []
        for tool_key, result in tool_results.items():
            if tool_key.startswith("schedule_checkin:") and "created" in result:
                parts = tool_key.split(":", 2)
                hours = parts[1] if len(parts) > 1 else "a few"
                additions.append(f"I'll check back with you in {hours} hours.")
            elif tool_key.startswith("escalate_clinician_alert:") and "added" in result:
                additions.append("I've noted this for clinician review.")
        if additions:
            agent_response.acknowledgment = f"{original_ack} {' '.join(additions)}"
            agent_response.agent_trace["replan"] = {
                "original_acknowledgment": original_ack,
                "revised_acknowledgment": agent_response.acknowledgment,
                "revision_reason": "deterministic_template",
                "tool_results_seen": tool_results,
            }
        else:
            agent_response.agent_trace["replan"] = {"kept_original": True, "reason": "no_user_visible_tools"}
        return agent_response

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from model response, with repair for common LLM errors."""
        text = response_text.strip()
        if not text:
            raise ValueError("Empty LLM response")
        # Strip markdown fences
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        # Find JSON object boundaries
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        # Try parsing as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Repair: trailing commas before } or ]
        repaired = re.sub(r',\s*([}\]])', r'\1', text)
        # Repair: missing commas between "value"\n"key" pairs
        repaired = re.sub(r'"\s*\n\s*"', '",\n"', repaired)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        # Last resort: truncate at last complete key-value pair and close
        last_complete = repaired.rfind('",')
        if last_complete > 0:
            truncated = repaired[:last_complete + 1] + "}"
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON from LLM response (len={len(response_text)})")

    async def _generate_llm_response(self, context: ResponseContext) -> AgentResponse:
        """Generate response using MedGemma via Vertex AI."""
        user_message = f"""The user just logged their symptoms. Here's the full context:

{context.to_prompt_context()}

Original transcript: "{context.extraction.transcript}"

Generate your response as JSON. Include an insight_nudge if the historical data shows a notable pattern."""

        # Inject explicit question signal so the LLM cannot miss user questions
        if _user_asked_question(context.extraction.transcript):
            user_question = _extract_user_question(context.extraction.transcript)
            user_message += (
                f'\n\nIMPORTANT - THE PATIENT ASKED A DIRECT QUESTION: "{user_question}"\n'
                f"Your acknowledgment MUST start by addressing this question. "
                f"Share relevant medical context from their profile and medication history "
                f"(you cannot diagnose, but you can note known side effects, medication interactions, etc.). "
                f"ALWAYS use hedging language ('can be', 'sometimes', 'worth discussing with your doctor') — "
                f"NEVER state causation definitively ('is from', 'is causing'). "
                f"Then your immediate_question should ask for any missing data (dose, severity)."
            )

        prompt = f"{self.SYSTEM_PROMPT}\n\n{user_message}"
        response_text = await self.client.generate_agent_response(prompt, max_tokens=320)
        data = self._parse_json_response(response_text)

        # Build acknowledgment with optional insight
        acknowledgment = data.get("acknowledgment", "Got it, logged your entry.")
        acknowledgment = clean_patient_text(acknowledgment)
        acknowledgment = _strip_hallucinated_vitals(acknowledgment, context.recent_symptom_logs, context.extraction)
        insight = data.get("insight_nudge")
        if insight:
            insight = clean_patient_text(insight)
            insight = _strip_hallucinated_vitals(insight, context.recent_symptom_logs, context.extraction)

        candidate_question = data.get("immediate_question") if data.get("should_ask_question") else None

        # Parse tool_calls — MedGemma can request downstream tools and invoke protocols
        raw_tools = data.get("tool_calls", [])
        tool_calls = filter_tool_calls(raw_tools)
        trace: dict = {"llm_proposed_tools": raw_tools if isinstance(raw_tools, list) else []}
        deterministic_actions: list[str] = []

        # Extract LLM protocol selection from tool_calls (invoke_protocol:<id>)
        llm_protocol_id = None
        for tc in tool_calls:
            if tc.startswith(PROTOCOL_TOOL_PREFIX):
                llm_protocol_id = tc[len(PROTOCOL_TOOL_PREFIX):]
                break
        # Backward compat: fall back to explicit protocol_id field if present
        if not llm_protocol_id:
            llm_protocol_id = data.get("protocol_id")
        llm_reason_code = data.get("reason_code")
        llm_question_rationale = data.get("question_rationale")

        # Deterministic gate: only honor run_watchdog_now when enough history exists
        # for longitudinal analysis (mirrors WATCHDOG_MIN_LOGS). Prevents over-triggering
        # on early logs where the model hallucinates a pattern from insufficient data.
        if "run_watchdog_now" in tool_calls and len(context.recent_symptom_logs) < 5:
            logger.debug("Suppressed run_watchdog_now — only %d prior logs", len(context.recent_symptom_logs))
            deterministic_actions.append(f"suppressed run_watchdog_now (only {len(context.recent_symptom_logs)} prior logs)")
            tool_calls = [t for t in tool_calls if t != "run_watchdog_now"]

        # Deterministic safety net: inject watchdog for clinically significant entries
        # when the LLM fails to emit the tool call despite recognizing escalation.
        if "run_watchdog_now" not in tool_calls and _should_inject_watchdog(
            context.extraction, len(context.recent_symptom_logs)
        ):
            tool_calls.append("run_watchdog_now")
            deterministic_actions.append("injected run_watchdog_now — clinical threshold met")
            logger.info("Injected run_watchdog_now — deterministic clinical threshold met")

        # Safety classifier — blocks response delivery if unsafe content detected.
        # Skip for benign entries (no symptoms, no red flags) where false-positive
        # risk outweighs the near-zero chance of harmful content.
        # Also skip when a clinical protocol already fired — the protocol system
        # handles safety (clinician alert + check-in), and the acknowledgment text
        # for escalated scenarios is prone to false positives from the classifier.
        protocol_already_escalated = llm_protocol_id in (
            "compound_symptom_escalation", "medication_interaction",
            "fever_protocol", "red_flag_static",
        )
        # Also skip when extraction shows medication non-adherence + abnormal vitals.
        # The deterministic protocol engine will fire compound_symptom_escalation for
        # these cases even if the LLM didn't propose it. The classifier's false-positive
        # rate is highest here because acknowledging missed meds + elevated readings
        # looks like "medication advice" to the classifier despite being observation.
        if not protocol_already_escalated:
            has_missed_med = any(
                a.effect_text and "skip" in a.effect_text.lower()
                for a in (context.extraction.actions_taken or [])
            )
            has_abnormal_vitals = bool(context.extraction.vital_signs)
            if has_missed_med and has_abnormal_vitals:
                protocol_already_escalated = True
        has_clinical_content = (
            context.extraction.symptoms
            or context.extraction.red_flags
            or self._contains_safety_sensitive_symptoms(context)
        )
        if has_clinical_content and not protocol_already_escalated:
            parts_to_check = [acknowledgment]
            if insight:
                parts_to_check.append(insight)
            if candidate_question:
                parts_to_check.append(candidate_question)
            text_to_check = " ".join(parts_to_check)

            if not await self._llm_safety_check(text_to_check):
                logger.warning("Safety classifier blocked response: %s", text_to_check[:200])
                safe_response = self._generate_stub_response(context)
                safe_response.acknowledgment += (
                    " If your symptoms feel severe or concerning, please contact a clinician."
                )
                safe_response.agent_trace = {"safety_blocked": True}
                return safe_response

        if insight and insight not in acknowledgment:
            acknowledgment = f"{acknowledgment} {insight}"
        candidate_checkin = self._build_llm_candidate_checkin(data, context)

        followup_policy = self._apply_followup_policy(
            context=context,
            candidate_question=candidate_question,
            candidate_checkin=candidate_checkin,
            llm_protocol_id=llm_protocol_id,
            llm_reason_code=llm_reason_code,
            llm_question_rationale=llm_question_rationale,
        )

        # Reconcile tool_calls with final protocol decision (safety override
        # may have forced a different protocol than the LLM chose).
        final_protocol_id = followup_policy["protocol_id"]
        if final_protocol_id:
            protocol_tool = f"{PROTOCOL_TOOL_PREFIX}{final_protocol_id}"
            tool_calls = [t for t in tool_calls if not t.startswith(PROTOCOL_TOOL_PREFIX)]
            tool_calls.insert(0, protocol_tool)
            if final_protocol_id != llm_protocol_id:
                deterministic_actions.append(
                    f"safety override: protocol {llm_protocol_id} → {final_protocol_id}"
                )
        else:
            # No protocol applies — remove any LLM-emitted protocol tool call
            tool_calls = [t for t in tool_calls if not t.startswith(PROTOCOL_TOOL_PREFIX)]

        # Prefer tool-based scheduling over legacy checkin fields to avoid duplicates
        has_tool_checkin = any(t.startswith(SCHEDULE_CHECKIN_PREFIX) for t in tool_calls)
        if has_tool_checkin and followup_policy["scheduled_checkin"]:
            deterministic_actions.append("legacy checkin suppressed — tool-based checkin takes priority")

        trace["deterministic_actions"] = deterministic_actions

        return AgentResponse(
            acknowledgment=acknowledgment,
            immediate_question=followup_policy["immediate_question"],
            scheduled_checkin=None if has_tool_checkin else followup_policy["scheduled_checkin"],
            protocol_id=followup_policy["protocol_id"],
            reason_code=followup_policy["reason_code"],
            safety_mode=followup_policy["safety_mode"],
            tool_calls=tool_calls,
            question_rationale=followup_policy["question_rationale"],
            agent_trace=trace,
        )

    def _generate_stub_response(self, context: ResponseContext) -> AgentResponse:
        """Fallback stub response when LLM is unavailable."""
        symptoms = context.extraction.symptoms
        actions = context.extraction.actions_taken
        profile = context.user_profile

        # Analyze history for stub insights
        analysis = context._analyze_symptom_history()
        insight = None

        if symptoms:
            symptom_name = symptoms[0].symptom.lower()
            count_24h = analysis["symptom_counts_24h"].get(symptom_name, 0)
            count_yesterday = analysis["symptom_counts_yesterday"].get(symptom_name, 0)
            count_7d = analysis["symptom_counts_7d"].get(symptom_name, 0)

            # Generate insight based on trend changes only (no raw counts)
            if count_24h > count_yesterday and count_yesterday > 0:
                insight = f"You're reporting {symptom_name} more frequently than yesterday."
            elif count_7d >= 3 and count_yesterday == 0:
                insight = f"{symptom_name.capitalize()} has become a recurring pattern this week — worth noting for your doctor."

        # Generate acknowledgment
        transcript_lower = context.extraction.transcript.lower()

        # Detect emotional content FIRST — applies whether or not symptoms are present
        emotional_keywords = ["frustrated", "scared", "anxious", "overwhelmed", "struggling",
                              "terrible", "dread", "not normal", "nobody", "ruining"]
        has_emotion = any(w in transcript_lower for w in emotional_keywords)

        if not symptoms:
            if has_emotion:
                acknowledgment = "I hear you. It sounds like you're going through a lot right now."
            elif any(phrase in transcript_lower for phrase in ["put together", "summary", "doctor appointment", "help me put"]):
                acknowledgment = "Of course — I'll help organize everything for your appointment."
            else:
                acknowledgment = "Got it, logged your entry."
        else:
            primary_symptom = symptoms[0].symptom
            if has_emotion:
                acknowledgment = f"I hear you — that sounds really tough. Logged your {primary_symptom}."
            elif actions:
                action = actions[0]
                verb = "skipped" if action.effect_text and "skip" in action.effect_text.lower() else "took"
                acknowledgment = f"Logged your {primary_symptom}. I see you {verb} {action.name}."
            elif symptoms[0].severity_1_10 and symptoms[0].severity_1_10 >= 7:
                acknowledgment = f"That's significant — logged your {primary_symptom} at severity {symptoms[0].severity_1_10}."
            elif symptoms[0].severity_1_10:
                acknowledgment = f"Logged your {primary_symptom} at severity {symptoms[0].severity_1_10}."
            else:
                acknowledgment = f"Got it, logged your {primary_symptom}."

        # Append insight if found
        if insight:
            acknowledgment = f"{acknowledgment} {insight}"

        # Check for medication -> schedule check-in
        scheduled_checkin = None
        if actions:
            action = actions[0]
            med_hours = {"ibuprofen": 2, "advil": 2, "tylenol": 1, "acetaminophen": 1, "naproxen": 4, "aleve": 4}
            hours = med_hours.get(action.name.lower(), 2)

            # Personalize check-in message based on profile
            checkin_message = f"Hey, it's been {hours} hours since you took {action.name}. How are you feeling? Did it help?"
            if profile and profile.conditions:
                # Add profile-aware follow-up
                conditions_lower = [c.lower() for c in profile.conditions]
                if "asthma" in conditions_lower and symptoms:
                    symptom_lower = symptoms[0].symptom.lower()
                    if symptom_lower in ["cough", "coughing", "wheeze", "wheezing", "shortness of breath"]:
                        checkin_message += " Have you needed your rescue inhaler?"

            scheduled_checkin = ScheduledCheckin(
                id=f"checkin_{uuid.uuid4().hex[:12]}",
                user_id=context.user_id,
                checkin_type=CheckinType.MEDICATION_FOLLOWUP,
                scheduled_for=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=hours),
                message=checkin_message,
                context={"medication_name": action.name},
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )

        candidate_question = _build_immediate_question(
            context=context,
            has_scheduled_checkin=scheduled_checkin is not None,
        )
        followup_policy = self._apply_followup_policy(
            context=context,
            candidate_question=candidate_question,
            candidate_checkin=scheduled_checkin,
        )

        return AgentResponse(
            acknowledgment=acknowledgment,
            immediate_question=followup_policy["immediate_question"],
            scheduled_checkin=followup_policy["scheduled_checkin"],
            protocol_id=followup_policy["protocol_id"],
            reason_code=followup_policy["reason_code"],
            safety_mode=followup_policy["safety_mode"],
            tool_calls=[],
            question_rationale=followup_policy["question_rationale"],
        )


# Keep backward compatibility alias
ResponseGenerator = LLMResponseGenerator


_response_generator_instance: Optional[LLMResponseGenerator] = None
_response_generator_lock = threading.Lock()


def get_response_generator() -> LLMResponseGenerator:
    """Factory function for response generator (cached singleton)."""
    global _response_generator_instance
    with _response_generator_lock:
        if _response_generator_instance is None:
            _response_generator_instance = LLMResponseGenerator()
        return _response_generator_instance


async def llm_safety_check(text: str) -> bool:
    """Standalone safety check for use outside the response generator (e.g. watchdog).

    Returns True if the text is safe, False if it contains medical advice/diagnoses.
    Fails closed (returns False on error).
    """
    gen = get_response_generator()
    return await gen._llm_safety_check(text)
