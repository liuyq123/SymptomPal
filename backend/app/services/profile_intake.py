"""Profile intake orchestration with LLM-powered answer parsing."""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..models import CheckinType, ScheduledCheckin, UserProfile

logger = logging.getLogger(__name__)

_NONE_TOKENS = {
    "none",
    "no",
    "nope",
    "n/a",
    "na",
    "nothing",
    "no known",
    "dont know",
    "don't know",
    "unknown",
    "skip",
    "prefer not to say",
}

_DEMOGRAPHIC_QUESTION_IDS = {"name", "age_sex"}

_QUESTION_FLOW: List[Tuple[str, str]] = [
    ("name", "Hi! I'm your health tracking assistant. What's your name?"),
    ("age_sex", "Nice to meet you! To help personalize your health tracking, how old are you and what's your biological sex?"),
    ("conditions", "To personalize tracking, do you have any chronic conditions?"),
    ("allergies", "Any medication or food allergies I should record?"),
    ("regular_medications", "What regular medications or supplements do you take?"),
    ("surgeries", "Any major surgeries or procedures in the past?"),
    ("family_history", "Any important family health history to note?"),
    ("social_history", "Any smoking, alcohol, or lifestyle factors to track?"),
    ("patterns", "Any symptom triggers or patterns you've already noticed?"),
    ("health_summary", "Anything else about your baseline health I should keep in mind?"),
]

_TOPIC_METADATA = {
    "name": ("name", "the patient's name"),
    "age_sex": ("age and biological sex", "the patient's age and biological sex for clinical context"),
    "conditions": ("chronic conditions", "chronic or recurring health conditions"),
    "allergies": ("allergies", "medication or food allergies"),
    "regular_medications": ("regular medications", "medications or supplements taken regularly"),
    "medication_doses": ("medication doses", "doses and frequencies for each medication"),
    "surgeries": ("surgical history", "major surgeries or procedures"),
    "family_history": ("family health history", "important health conditions in their family"),
    "social_history": ("lifestyle factors", "smoking, alcohol, or lifestyle factors"),
    "patterns": ("symptom patterns", "triggers or patterns in symptoms"),
    "health_summary": ("baseline health", "anything else about overall health"),
}

_INTAKE_PARSE_AND_NEXT_PROMPT = """You are a health assistant conducting a patient intake interview. You just asked about their {topic_label}.

<question_asked>
{question_text}
</question_asked>

<patient_response>
{response_text}
</patient_response>

<profile_so_far>
{profile_context}
</profile_so_far>

<next_topic>
Topic: {next_topic_label}
Description: {next_topic_description}
</next_topic>

Do TWO things:

1. PARSE the patient's response into a structured list for the "{question_id}" field.
   - ONLY include information the patient explicitly stated in their response above. Do NOT infer, extrapolate, or add details from the profile context — the profile is provided for generating the next question only, not for enriching parsed items.
   - Normalize medical terminology (e.g., "high blood pressure" → "Hypertension", "sugar disease" → "Type 2 Diabetes")
   - For medications, include name + dose + frequency ONLY if explicitly stated by the patient. Do NOT infer doses from drug names. If dose was not mentioned, omit it. (e.g., patient says "metformin 500mg twice a day" → "Metformin 500mg twice daily"; patient says "my blood pressure pill" → "Antihypertensive medication (dose not specified)")
   - Remove filler words and speech artifacts ("um", "like", "you know", "basically")
   - If the patient clearly declined or said nothing relevant (e.g., "none", "no", "skip"), return an empty list
   - If the patient starts with a denial but then shares information (e.g., "Nothing diagnosed officially, but I have mild anemia"), extract the information — do NOT return an empty list
   - Return a JSON array of clean, concise strings

2. GENERATE a natural follow-up question for the next topic: "{next_question_id}".
   - Reference what the patient has already shared to make it conversational
   - Keep it to 1-2 sentences max
   - Ask about the next topic, just personalized based on context
   - If no profile context yet, ask a warm generic question

Respond with valid JSON only:
{{
    "parsed_items": ["item1", "item2"],
    "next_question": "Your personalized question text here"
}}"""


def intake_enabled() -> bool:
    return os.environ.get("ENABLE_PROFILE_INTAKE", "true").lower() == "true"


def intake_max_questions() -> int:
    try:
        value = int(os.environ.get("PROFILE_INTAKE_MAX_QUESTIONS", "10"))
        return max(1, min(20, value))
    except Exception:
        return 10


def should_start_intake(profile: UserProfile, total_recent_logs: int) -> bool:
    if not intake_enabled():
        return False
    if profile.intake_completed:
        return False
    if profile.intake_questions_asked > 0:
        return False
    return total_recent_logs <= 1


def _medications_needing_doses(profile: UserProfile) -> List[str]:
    """Return medication names from profile that lack dose information."""
    if not profile.regular_medications:
        return []
    return [
        med.split(" (")[0].strip()
        for med in profile.regular_medications
        if "(dose not specified)" in med.lower()
    ]


def get_next_intake_question(profile: UserProfile) -> Optional[Tuple[str, str]]:
    if not intake_enabled() or profile.intake_completed:
        return None

    max_questions = intake_max_questions()
    if profile.intake_questions_asked >= max_questions:
        return None

    answered = set(profile.intake_answered_question_ids or [])

    # After medications are answered, check for missing doses before moving on
    if "regular_medications" in answered and "medication_doses" not in answered:
        meds = _medications_needing_doses(profile)
        if meds:
            names = ", ".join(meds)
            return "medication_doses", f"You mentioned {names} — do you know the doses for each?"

    for question_id, message in _QUESTION_FLOW:
        if question_id not in answered:
            # Personalize age_sex question with name if available
            if question_id == "age_sex" and profile.name:
                message = f"Nice to meet you, {profile.name}! To help personalize your health tracking, how old are you and what's your biological sex?"
            return question_id, message
    return None


def create_intake_checkin(user_id: str, question_id: str, message: str) -> ScheduledCheckin:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return ScheduledCheckin(
        id=f"checkin_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        checkin_type=CheckinType.PROFILE_INTAKE,
        scheduled_for=now,
        message=message,
        context={"question_id": question_id, "kind": "profile_intake"},
        created_at=now,
    )


def _parse_name(text: str) -> Optional[str]:
    """Extract name from a conversational response."""
    cleaned = text.strip().rstrip(".")
    for prefix in ["my name is", "i'm", "im", "it's", "its", "call me", "i am", "hey i'm", "hi i'm"]:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip().lstrip(",").strip()
            break
    parts = cleaned.split()[:3]
    return " ".join(w.capitalize() for w in parts) if parts else None


def _parse_age_sex(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract age and biological sex from a conversational response."""
    text_lower = text.lower()
    age_match = re.search(r'\b(\d{1,3})\b', text)
    age = int(age_match.group(1)) if age_match and 0 < int(age_match.group(1)) < 130 else None
    gender = None
    if any(w in text_lower for w in ["female", "woman", "girl"]):
        gender = "female"
    elif any(w in text_lower for w in ["male", "man", "boy", "guy", "dude"]):
        gender = "male"
    elif "non-binary" in text_lower or "nonbinary" in text_lower or "non binary" in text_lower:
        gender = "non-binary"
    return age, gender


def _is_none_answer(value: str) -> bool:
    return value.lower().strip() in _NONE_TOKENS


def _split_medication_doses(text: str) -> List[str]:
    """Basic sentence splitting for medication dose answers when LLM parsing fails."""
    parts = re.split(r'(?<=[.!])\s+', text.strip())
    expanded: List[str] = []
    for p in parts:
        expanded.extend(re.split(r'\b(?:[Aa]nd|[Oo]h and)\b', p))
    return [item.strip().rstrip('.') for item in expanded if len(item.strip()) > 5]


async def _parse_medication_doses_via_extract(
    medgemma_client,
    response_text: str,
    profile: UserProfile,
    next_question_id: Optional[str],
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Parse medication doses using the extraction pipeline (same as daily logs).

    The extraction prompt has focused, example-rich dose instructions that
    small quantized models handle well — unlike the generic intake prompt.
    """
    try:
        extraction = await medgemma_client.extract(response_text)
        if not extraction.actions_taken:
            # Extraction found no medications — fall back to sentence splitting
            items = _split_medication_doses(response_text)
            return (items if items else None), _get_hardcoded_question(next_question_id)

        # Build dose-enriched strings. For each extracted medication, find its
        # sentence in the original text to preserve frequency/timing info.
        items = []
        text_lower = response_text.lower()

        for action in extraction.actions_taken:
            name_lower = action.name.lower()
            idx = text_lower.find(name_lower)
            if idx >= 0:
                # Pull the full sentence containing this medication
                dot_before = response_text.rfind('.', 0, idx)
                start = dot_before + 1 if dot_before >= 0 else 0
                dot_after = response_text.find('.', idx)
                end = dot_after if dot_after >= 0 else len(response_text)
                sentence = response_text[start:end].strip()
                # Clean leading filler ("The ", "And ")
                sentence = re.sub(r'^(?:the|and|oh and)\s+', '', sentence, flags=re.IGNORECASE)
                if sentence and len(sentence) > 3:
                    items.append(sentence)
                    continue
            # Fallback: construct from extracted fields
            if action.dose_text:
                items.append(f"{action.name} {action.dose_text}")
            else:
                items.append(action.name)

        return items, _get_hardcoded_question(next_question_id)

    except Exception as e:
        logger.warning("Medication dose extraction failed, falling back: %s", e)
        return None, _get_hardcoded_question(next_question_id)


def _apply_parsed_items_to_patch(patch: Dict[str, object], question_id: str, items: List[str]) -> None:
    """Apply parsed items to a profile patch for the given question type."""
    if question_id == "conditions":
        patch.setdefault("add_conditions", []).extend(items)
    elif question_id == "allergies":
        patch.setdefault("add_allergies", []).extend(items)
    elif question_id == "regular_medications":
        patch.setdefault("add_regular_medications", []).extend(items)
    elif question_id == "medication_doses":
        # Dose-enriched items replace old entries in build_intake_profile_patch
        patch.setdefault("add_regular_medications", []).extend(items)
    elif question_id == "surgeries":
        patch.setdefault("add_surgeries", []).extend(items)
    elif question_id == "family_history":
        patch.setdefault("add_family_history", []).extend(items)
    elif question_id == "social_history":
        patch.setdefault("add_social_history", []).extend(items)
    elif question_id == "patterns":
        patch.setdefault("add_patterns", []).extend(items)
    elif question_id == "health_summary":
        patch["health_summary"] = " ".join(items) if items else None


def build_intake_profile_patch(
    profile: UserProfile,
    question_id: str,
    response_text: str,
    parsed_items: Optional[List[str]] = None,
) -> Dict[str, object]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    answered = set(profile.intake_answered_question_ids or [])
    answered.add(question_id)

    max_questions = intake_max_questions()
    asked_count = min(max_questions, max(profile.intake_questions_asked + 1, len(answered)))

    # medication_doses is a follow-up, not a core question — don't count toward asked limit
    is_followup_question = question_id not in {qid for qid, _ in _QUESTION_FLOW}

    patch: Dict[str, object] = {
        "intake_questions_asked": asked_count if not is_followup_question else profile.intake_questions_asked,
        "intake_answered_question_ids": sorted(answered),
        "intake_last_question_id": question_id,
        "intake_started_at": profile.intake_started_at or now,
    }

    # For medication_doses, remove ALL existing medications.
    # The dose follow-up covers every listed medication, so the new
    # dose-enriched items are a complete replacement.
    if question_id == "medication_doses":
        existing_meds = list(profile.regular_medications or [])
        if existing_meds:
            patch["remove_regular_medications"] = existing_meds

    # Demographics are scalar fields — parse directly, no LLM needed
    if question_id == "name":
        name = _parse_name(response_text)
        if name:
            patch["name"] = name
    elif question_id == "age_sex":
        age, gender = _parse_age_sex(response_text)
        if age:
            patch["age"] = age
        if gender:
            patch["gender"] = gender
    elif parsed_items is not None:
        _apply_parsed_items_to_patch(patch, question_id, parsed_items)
    else:
        # Model unavailable — for medication_doses, do basic splitting
        # to avoid losing dose information entirely
        if question_id == "medication_doses":
            items = _split_medication_doses(response_text)
            _apply_parsed_items_to_patch(patch, question_id, items)
        else:
            # Save raw text for deferred parsing, don't regex-parse
            patch["intake_pending_raw_add"] = {question_id: response_text.strip()}
            _apply_parsed_items_to_patch(patch, question_id, [])
        # health_summary fallback: store raw text directly instead of empty
        if question_id == "health_summary":
            patch["health_summary"] = None if _is_none_answer(response_text) else response_text.strip()

    core_answered = {qid for qid, _ in _QUESTION_FLOW} & answered
    completed = len(core_answered) >= len(_QUESTION_FLOW) or asked_count >= max_questions
    if completed:
        patch["intake_completed"] = True
        patch["intake_completed_at"] = now

    return patch


async def drain_pending_intake_raw(
    medgemma_client,
    profile: UserProfile,
) -> Dict[str, object]:
    """Reparse queued raw intake answers now that MedGemma is available.

    Returns a profile patch with parsed items and cleared pending_raw.
    Returns empty dict if nothing to drain or all reparses fail.
    """
    pending = profile.intake_pending_raw
    if not pending:
        return {}

    patch: Dict[str, object] = {}
    remaining = dict(pending)

    for question_id, raw_text in list(pending.items()):
        question_text = _get_hardcoded_question(question_id) or question_id
        try:
            parsed_items, _ = await parse_answer_and_generate_next_question(
                medgemma_client, profile, question_id, question_text,
                raw_text, None,
            )
            if parsed_items is not None:
                _apply_parsed_items_to_patch(patch, question_id, parsed_items)
                del remaining[question_id]
        except Exception:
            pass  # Keep in pending for next attempt

    if remaining == pending:
        return {}  # Nothing was drained

    patch["intake_pending_raw"] = remaining  # {} clears the field in update_user_profile
    return patch


# ---------------------------------------------------------------------------
# LLM-powered intake: answer parsing + contextual question generation
# ---------------------------------------------------------------------------

def _build_profile_context(profile: UserProfile) -> str:
    """Format populated profile fields as text for the LLM prompt."""
    parts = []
    if profile.conditions:
        parts.append(f"Conditions: {', '.join(profile.conditions)}")
    if profile.allergies:
        parts.append(f"Allergies: {', '.join(profile.allergies)}")
    if profile.regular_medications:
        parts.append(f"Medications: {', '.join(profile.regular_medications)}")
    if profile.surgeries:
        parts.append(f"Surgeries: {', '.join(profile.surgeries)}")
    if profile.family_history:
        parts.append(f"Family history: {', '.join(profile.family_history)}")
    if profile.social_history:
        parts.append(f"Social history: {', '.join(profile.social_history)}")
    if profile.patterns:
        parts.append(f"Patterns: {', '.join(profile.patterns)}")
    return "\n".join(parts) if parts else "No information collected yet."


def _get_hardcoded_question(question_id: Optional[str]) -> Optional[str]:
    """Look up the hardcoded question text for a given question_id."""
    if question_id is None:
        return None
    for qid, msg in _QUESTION_FLOW:
        if qid == question_id:
            return msg
    return None


def _extract_json(raw: str) -> str:
    """Extract JSON object from LLM response, stripping markdown fences."""
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


async def parse_answer_and_generate_next_question(
    medgemma_client,
    profile: UserProfile,
    question_id: str,
    question_text: str,
    response_text: str,
    next_question_id: Optional[str],
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Use LLM to parse the user's answer AND generate the next contextual question.

    Returns (parsed_items, next_question_text).
    Falls back to (regex result, hardcoded question) on any LLM failure.
    """
    # Demographics are parsed with simple regex — skip LLM entirely
    if question_id in _DEMOGRAPHIC_QUESTION_IDS:
        return None, _get_hardcoded_question(next_question_id)

    # For medication_doses, use the extraction pipeline (same as daily logs)
    # instead of the generic intake prompt. The focused extraction prompt with
    # dose examples works much better on small quantized models.
    if question_id == "medication_doses":
        return await _parse_medication_doses_via_extract(
            medgemma_client, response_text, profile, next_question_id
        )

    profile_context = _build_profile_context(profile)
    topic_label, _ = _TOPIC_METADATA.get(question_id, (question_id, ""))

    if next_question_id is not None:
        next_topic_label, next_topic_desc = _TOPIC_METADATA.get(
            next_question_id, (next_question_id, "")
        )
    else:
        next_topic_label = "none"
        next_topic_desc = "This is the last question — no follow-up needed."

    prompt = _INTAKE_PARSE_AND_NEXT_PROMPT.format(
        topic_label=topic_label,
        question_text=question_text,
        response_text=response_text,
        profile_context=profile_context,
        next_topic_label=next_topic_label,
        next_topic_description=next_topic_desc,
        question_id=question_id,
        next_question_id=next_question_id or "none",
    )

    try:
        raw = await medgemma_client.generate_agent_response(prompt, max_tokens=300)
        data = json.loads(_extract_json(raw))

        parsed = data.get("parsed_items", [])
        if not isinstance(parsed, list):
            parsed = [str(parsed)]
        parsed = [str(item).strip() for item in parsed if str(item).strip()]

        next_q = data.get("next_question")
        if next_question_id is None:
            next_q = None
        elif not next_q or not isinstance(next_q, str) or len(next_q.strip()) < 10:
            next_q = _get_hardcoded_question(next_question_id)

        return parsed, next_q

    except Exception as e:
        logger.warning("LLM intake parse unavailable, deferring raw response for question=%s: %s", question_id, e)
        return None, _get_hardcoded_question(next_question_id)

