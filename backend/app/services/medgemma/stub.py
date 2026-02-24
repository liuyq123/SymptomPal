"""Stub implementation of MedGemmaClient for MVP/testing."""

import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional

from .base import MedGemmaClient, logger

from ...models import (
    ExtractionResult,
    MenstrualStatus,
    SymptomEntity,
    ActionEntity,
    VitalSignEntry,
    DoctorPacket,
    TimelineReveal,
    TimelinePoint,
    LogEntry,
    WatchdogResult,
)


def _to_utc_naive(value: datetime) -> datetime:
    """Normalize datetimes for stable comparisons/serialization."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


class StubMedGemmaClient(MedGemmaClient):
    """Stub implementation for MVP/testing that returns deterministic results."""

    async def extract(self, transcript: str) -> ExtractionResult:
        """Smarter stub extraction for demo purposes."""
        transcript_lower = transcript.lower()

        symptoms = []
        missing_fields = []

        # Symptom patterns: (keywords to match, symptom name to use)
        symptom_patterns = [
            # Head
            (["headache", "head hurts", "head ache", "migraine"], "headache"),
            # Stomach/digestive
            (["stomach hurts", "stomach ache", "stomachache", "stomach pain", "belly", "abdominal"], "stomach pain"),
            (["nausea", "nauseous", "queasy", "sick to my stomach", "feel nauseous"], "nausea"),
            (["vomit", "throwing up", "can't keep food down", "threw up"], "vomiting"),
            (["bloating", "bloated", "gassy"], "bloating"),
            (["diarrhea", "loose stool"], "diarrhea"),
            (["constipat"], "constipation"),
            # Respiratory
            (["cough", "coughing"], "cough"),
            (["wheez", "wheezy"], "wheezing"),
            (["shortness of breath", "hard to breathe", "can't breathe", "breathing difficult"], "shortness of breath"),
            (["congestion", "congested", "stuffy nose", "runny nose"], "congestion"),
            (["sore throat", "throat hurts"], "sore throat"),
            # General
            (["fever", "temperature", "chills"], "fever"),
            (["fatigue", "tired", "exhausted", "no energy", "low energy"], "fatigue"),
            (["dizzy", "dizziness", "lightheaded"], "dizziness"),
            (["anxious", "anxiety", "worried", "nervous", "panic"], "anxiety"),
            (["insomnia", "can't sleep", "trouble sleeping", "couldn't sleep"], "insomnia"),
            # Pain
            (["back pain", "back hurts", "lower back", "upper back"], "back pain"),
            (["chest pain", "chest hurts", "chest tight"], "chest pain"),
            (["joint pain", "joints hurt", "arthritis"], "joint pain"),
            (["muscle pain", "sore muscles", "muscle ache"], "muscle pain"),
            # Sensitivities (migraine-related)
            (["light hurts", "photophobia", "sensitive to light", "bright light", "lights bother"], "light sensitivity"),
            (["sound makes it worse", "phonophobia", "sensitive to sound", "noise bothers", "loud noise"], "sound sensitivity"),
        ]

        # Extract severity (X out of 10, X/10, or colloquial "about a 4")
        severity = None
        severity_match = re.search(r'(\d+)\s*(?:out of|/)\s*10', transcript_lower)
        if severity_match:
            severity = int(severity_match.group(1))
        else:
            severity_match = re.search(
                r'(?:about|around|maybe|like|roughly|say)\s+(?:a\s+)?(\d)\b',
                transcript_lower,
            )
            if severity_match:
                severity = int(severity_match.group(1))

        for keywords, symptom_name in symptom_patterns:
            for keyword in keywords:
                if keyword in transcript_lower:
                    # Avoid duplicates
                    if not any(s.symptom == symptom_name for s in symptoms):
                        symptoms.append(SymptomEntity(
                            symptom=symptom_name,
                            severity_1_10=severity,  # Include extracted severity
                            onset_time_text=None,  # Don't hallucinate onset
                        ))
                    break

        # Fallback: try to extract something meaningful if nothing matched
        if not symptoms:
            # Check if this is describing an action (doctor visit, etc.) rather than a symptom
            is_action_only = any(phrase in transcript_lower for phrase in [
                "called doctor", "called my doctor", "saw doctor", "saw my doctor",
                "visited doctor", "went to doctor", "appointment with",
                "prescribed", "prescription", "doctor said", "doctor told"
            ])

            if is_action_only:
                # This is primarily about an action, not a symptom
                # Create a neutral symptom entry
                symptoms.append(SymptomEntity(symptom="check-up"))
            # Look for generic pain indicators
            elif "hurt" in transcript_lower or "pain" in transcript_lower or "ache" in transcript_lower:
                # Try to find what hurts
                words = transcript_lower.split()
                for i, word in enumerate(words):
                    if word in ["hurts", "hurt", "aching", "aches", "painful"]:
                        # Look at previous word for body part
                        if i > 0:
                            body_part = words[i-1]
                            if body_part not in ["it", "really", "my", "the", "a", "so"]:
                                symptoms.append(SymptomEntity(symptom=f"{body_part} pain"))
                                break
                if not symptoms:
                    symptoms.append(SymptomEntity(symptom="pain"))
            else:
                # Last resort: use first few words as symptom description
                clean_words = [w for w in transcript_lower.split()[:5]
                              if w not in ["i", "my", "have", "a", "the", "am", "feel", "feeling", "been", "having"]]
                if clean_words:
                    symptoms.append(SymptomEntity(symptom=" ".join(clean_words[:3])))
                else:
                    symptoms.append(SymptomEntity(symptom="general discomfort"))

        # Detect actions taken (medications only — not food, drinks, or lifestyle)
        actions = []

        # Note: Doctor visits are NOT medications, so we don't add them to actions_taken.
        # The transcript still captures this context for the LLM response generator.

        # Detect prescribed medications
        prescription_keywords = [
            "prednisone", "antibiotics", "amoxicillin", "azithromycin", "steroid",
            "prescription", "inhaler", "albuterol",
        ]
        for med in prescription_keywords:
            if med in transcript_lower:
                if "prescribed" in transcript_lower or "prescription" in transcript_lower:
                    actions.append(ActionEntity(name=f"prescribed {med}"))
                else:
                    actions.append(ActionEntity(name=med))
                break  # Only capture first medication mentioned

        # Detect medications by name (OTC and prescription)
        if not actions:
            medication_keywords = [
                # Pain/fever
                "ibuprofen", "tylenol", "aspirin", "advil", "acetaminophen", "aleve", "naproxen",
                # GI
                "pepto", "tums", "antacid", "omeprazole", "pantoprazole",
                # Sleep/anxiety
                "melatonin", "benadryl", "lorazepam", "xanax", "ativan", "zoloft", "sertraline",
                # Migraine/headache
                "sumatriptan", "rizatriptan", "eletriptan", "topiramate", "propranolol", "amitriptyline",
                # Other common
                "ondansetron", "metoclopramide",
            ]
            for med in medication_keywords:
                if med in transcript_lower:
                    # Try to extract dose if mentioned
                    words = transcript_lower.split()
                    med_index = None
                    for i, word in enumerate(words):
                        if med in word:
                            med_index = i
                            break

                    dose = None
                    if med_index is not None and med_index + 1 < len(words):
                        next_word = words[med_index + 1]
                        if "mg" in next_word or "mcg" in next_word:
                            dose = next_word

                    actions.append(ActionEntity(name=med, dose_text=dose))
                    break  # Only capture first medication mentioned

        # Generic "took something" detection — only for medication-like words
        # Skip verbs that indicate non-medication actions (ate, drank, skipped)
        _non_med_verbs = {"ate", "eating", "drank", "drinking", "skipped", "skipping"}
        if not actions and ("took" in transcript_lower or "take" in transcript_lower or "used" in transcript_lower):
            words = transcript_lower.split()
            skip_words = {
                "a", "an", "my", "the", "it", "some", "regular", "usual", "daily",
                "morning", "evening", "preventive", "rescue",
                # Non-medications that commonly follow "took/ate/drank"
                "breakfast", "lunch", "dinner", "food", "meal", "snack",
                "juice", "water", "tea", "coffee", "milk", "soda",
                "nap", "rest", "walk", "shower", "bath",
                "bigger", "smaller", "another", "more", "less",
                # Context words that aren't medication names
                "everything", "pills", "meds", "medications", "medicine",
                # Timing/preposition words that follow "took"
                "time", "on", "before", "after", "today", "yesterday",
            }

            for i, word in enumerate(words):
                # Only match actual "took/take" verbs, not skipped/ate/drank
                if word in ["took", "take", "taken", "used", "using"] and word not in _non_med_verbs:
                    # Look ahead for the medication name, skipping modifier words
                    for j in range(i + 1, min(i + 5, len(words))):
                        candidate = words[j].rstrip(".,;:!?")
                        if candidate not in skip_words and len(candidate) > 2:
                            if any(c.isalpha() for c in candidate):
                                dose = None
                                if j + 1 < len(words):
                                    next_word = words[j + 1]
                                    if "mg" in next_word or "mcg" in next_word or next_word.replace(".", "").isdigit():
                                        dose = next_word

                                actions.append(ActionEntity(
                                    name=candidate,
                                    dose_text=dose
                                ))
                                break
                    if actions:
                        break

        # Determine missing fields
        if not any(s.severity_1_10 for s in symptoms):
            missing_fields.append("severity")
        if not any(s.onset_time_text for s in symptoms):
            missing_fields.append("onset")
        if not any(s.duration_text for s in symptoms):
            missing_fields.append("duration")

        # Extract vital signs via regex patterns
        vital_signs: list[VitalSignEntry] = []
        # Blood sugar / glucose
        for m in re.finditer(r'(?:blood\s*sugar|glucose|sugar)\s*(?:was|is|of|at|:)?\s*(\d+)', transcript_lower):
            vital_signs.append(VitalSignEntry(name="blood sugar", value=m.group(1), unit="mg/dL"))
        # SpO2
        for m in re.finditer(r'(?:spo2|sp02|oxygen|o2\s*sat)\s*(?:was|is|of|at|:)?\s*(\d+)', transcript_lower):
            vital_signs.append(VitalSignEntry(name="spo2", value=m.group(1), unit="%"))
        # Blood pressure
        for m in re.finditer(r'(?:blood\s*pressure|bp)\s*(?:was|is|of|at|:)?\s*(\d+)\s*/\s*(\d+)', transcript_lower):
            vital_signs.append(VitalSignEntry(name="blood pressure", value=f"{m.group(1)}/{m.group(2)}", unit="mmHg"))
        # Temperature
        for m in re.finditer(r'(?:temp|temperature)\s*(?:was|is|of|at|:)?\s*(\d+\.?\d*)', transcript_lower):
            vital_signs.append(VitalSignEntry(name="temperature", value=m.group(1), unit="F"))

        # Detect menstrual status from keywords
        menstrual_status = None
        _period_skip = re.search(
            r'\b(last\s+period|previous\s+period|period\s+was\b|period\s+is\s+late|missed\s+my\s+period|no\s+period|period\s+of\s+time)\b',
            transcript_lower,
        )
        if not _period_skip:
            _period_match = re.search(
                r'\b(period\s+started|started\s+my\s+period|got\s+my\s+period|on\s+my\s+period|day\s+\d+\s+of\s+my\s+period|started\s+menstruating|am\s+menstruating)\b',
                transcript_lower,
            )
            if _period_match:
                flow = "medium"
                if re.search(r'\b(heavy\s+flow|super\s+heavy|really\s+heavy|soaking)\b', transcript_lower):
                    flow = "heavy"
                elif re.search(r'\b(light\s+flow|spotting|just\s+spotting)\b', transcript_lower):
                    flow = "spotting" if "spotting" in transcript_lower else "light"
                menstrual_status = MenstrualStatus(is_period_day=True, flow_level=flow)

        return ExtractionResult(
            transcript=transcript,
            symptoms=symptoms,
            actions_taken=actions,
            vital_signs=vital_signs,
            missing_fields=missing_fields,
            red_flags=[],
            menstrual_status=menstrual_status,
        )

    async def doctor_packet(self, logs: List[LogEntry], days: int, user_id: str | None = None, user_profile=None) -> DoctorPacket:
        """Generate a deterministic Doctor Packet."""
        # Fetch watchdog observations for system flags
        flags: list[str] = []
        if user_id:
            from ...services.storage import get_watchdog_observations
            flags = get_watchdog_observations(user_id)

        if not logs:
            return DoctorPacket(
                hpi="No symptom logs recorded for this period.",
                pertinent_positives=[],
                pertinent_negatives=[],
                timeline_bullets=[],
                questions_for_clinician=[],
                system_longitudinal_flags=flags,
            )

        all_symptoms = []
        all_actions = []
        for log in logs:
            all_symptoms.extend([s.symptom for s in log.extracted.symptoms])
            all_actions.extend([a.name for a in log.extracted.actions_taken])

        unique_symptoms = list(set(all_symptoms))
        unique_actions = list(set(all_actions))

        if not unique_symptoms:
            return DoctorPacket(
                hpi="Patient logged entries but no specific symptoms were extracted.",
                pertinent_positives=[],
                pertinent_negatives=[],
                timeline_bullets=[f"Log recorded on {log.recorded_at.strftime('%Y-%m-%d')}" for log in logs[:5]],
                questions_for_clinician=[],
                system_longitudinal_flags=flags,
            )

        # Build HPI from actual data only
        hpi = f"Patient reports {', '.join(unique_symptoms)}."
        if unique_actions:
            hpi += f" Treatment included {', '.join(unique_actions)}."

        return DoctorPacket(
            hpi=hpi,
            pertinent_positives=unique_symptoms[:5],
            pertinent_negatives=[],
            timeline_bullets=self._build_timeline_bullets(logs),
            questions_for_clinician=["What could be causing these symptoms?"] if unique_symptoms else [],
            system_longitudinal_flags=flags,
        )

    async def timeline(self, logs: List[LogEntry], days: int) -> TimelineReveal:
        """Generate timeline story points from logs."""
        story_points = []

        if logs:
            sorted_logs = sorted(logs, key=lambda x: _to_utc_naive(x.recorded_at))

            for i, log in enumerate(sorted_logs):
                symptoms = ", ".join([s.symptom for s in log.extracted.symptoms]) or "symptoms"
                label = "Onset" if i == 0 else f"Update {i}"
                story_points.append(TimelinePoint(
                    timestamp=_to_utc_naive(log.recorded_at),
                    label=label,
                    details=f"Reported: {symptoms}",
                ))

        return TimelineReveal(story_points=story_points)

    async def generate_agent_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Stubbed agent response generation - returns empty to trigger fallback to smart stub logic."""
        return ""

    async def generate_profile_update(self, logs: List[LogEntry], current_profile: dict) -> dict:
        """
        Stub profile update generation.

        Uses simple heuristics to detect recurring patterns:
        - If same symptom appears 3+ times, suggest adding as condition
        - If symptoms cluster around certain triggers, suggest pattern
        """
        if not logs:
            return {"add_conditions": [], "add_patterns": [], "health_summary": None}

        # Count symptoms
        symptom_counter = Counter()
        for log in logs:
            for symptom in log.extracted.symptoms:
                symptom_counter[symptom.symptom.lower()] += 1

        # Find recurring symptoms (3+ occurrences)
        add_conditions = []
        current_conditions = [c.lower() for c in current_profile.get("conditions", [])]

        for symptom, count in symptom_counter.most_common(5):
            if count >= 3 and symptom not in current_conditions:
                # Map common symptoms to condition names
                condition_map = {
                    "headache": "Recurrent Headaches",
                    "migraine": "Recurrent Migraines",
                    "cough": "Chronic Cough",
                    "fatigue": "Chronic Fatigue",
                    "back pain": "Chronic Back Pain",
                    "nausea": "Recurrent Nausea",
                }
                condition = condition_map.get(symptom, f"Recurrent {symptom.title()}")
                if condition.lower() not in current_conditions:
                    add_conditions.append(condition)

        # Generate health summary if we have enough data
        health_summary = None
        if len(logs) >= 5 and symptom_counter:
            top_symptom, top_count = symptom_counter.most_common(1)[0]
            health_summary = f"Primary concern: {top_symptom} ({top_count} occurrences in recent logs)."

        return {
            "add_conditions": add_conditions[:2],  # Limit to 2 new conditions
            "add_patterns": [],  # Stub doesn't detect patterns
            "health_summary": health_summary,
        }

    async def respond_to_followup(
        self,
        original_transcript: str,
        followup_question: str,
        followup_answer: str,
        patient_name: Optional[str] = None,
        user_profile=None,
    ) -> str:
        """Stub: return a generic warm acknowledgment."""
        return "Thanks for sharing that — it's all logged and building your pattern picture."

    async def watchdog_analysis(self, history_context: str) -> WatchdogResult:
        """Keyword-based watchdog stub for testing."""
        text = history_context.lower()
        concerning_keywords = [
            "recurring", "worsening", "increasing", "every month",
            "every cycle", "progressive", "refractory",
        ]
        detected = any(kw in text for kw in concerning_keywords)
        if detected:
            return WatchdogResult(
                concerning_pattern_detected=True,
                internal_clinical_rationale="Stub: recurring pattern keywords detected in history.",
                safe_patient_nudge=(
                    "SymptomPal noticed a recurring pattern in your symptoms. "
                    "Consider generating a Doctor Packet to share with your physician."
                ),
                clinician_facing_observation="Stub: recurring symptom pattern detected in longitudinal data.",
            )
        return WatchdogResult(
            concerning_pattern_detected=False,
            internal_clinical_rationale="Stub: no concerning patterns detected.",
            safe_patient_nudge=None,
            clinician_facing_observation=None,
        )
