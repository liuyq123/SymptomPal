"""
Safety Observer - Non-Diagnostic Red Flag Detection

This module implements a synonym-aware keyword + co-occurrence detector for potential
red flags. It does NOT diagnose - it only flags patterns that may warrant
prompt clinical attention.

Important: All outputs should be accompanied by the standard disclaimer:
"If symptoms feel severe, rapidly worsening, or concerning, consider
contacting a clinician promptly."
"""

# Synonym groups: canonical name → list of phrases that should match.
# Each phrase is checked via substring match on the lowercased transcript.
_SYNONYM_GROUPS: dict[str, list[str]] = {
    "chest pain": [
        "chest pain", "chest hurts", "chest is hurting", "chest hurting",
        "pain in my chest", "pain in chest", "pressure in my chest",
        "chest pressure", "chest tightness", "tight chest",
        "tightness in my chest", "squeezing in my chest", "chest discomfort",
    ],
    "shortness of breath": [
        "shortness of breath", "short of breath", "difficulty breathing",
        "trouble breathing", "can't breathe", "cannot breathe", "can not breathe",
        "hard to breathe", "gasping", "gasping for air", "can't get air",
        "struggling to breathe", "labored breathing", "laboured breathing",
        "breathless", "out of breath", "sob",
    ],
    "arm pain": [
        "arm pain", "arm hurts", "pain in my arm", "pain in arm",
        "arm ache", "arm aching", "left arm pain", "left arm hurts",
        "pain radiating to arm", "tingling in arm",
    ],
    "jaw pain": [
        "jaw pain", "jaw hurts", "pain in my jaw", "jaw ache", "jaw aching",
    ],
    "fainted": [
        "fainted", "faint", "feeling faint",
    ],
    "fainting": [
        "fainting", "about to faint", "almost fainted", "nearly fainted",
    ],
    "passed out": [
        "passed out", "lost consciousness", "blacked out",
    ],
    "face drooping": [
        "face drooping", "face is drooping", "facial drooping",
        "one side of my face", "face feels numb", "face droop",
    ],
    "facial droop": [
        "facial droop", "face drooped", "droopy face",
    ],
    "slurred speech": [
        "slurred speech", "slurring words", "slurring my words",
        "can't speak clearly", "speech is slurred", "talking funny",
        "words coming out wrong",
    ],
    "sudden numbness": [
        "sudden numbness", "suddenly numb", "numbness came on suddenly",
        "went numb", "sudden tingling",
    ],
    "sudden weakness": [
        "sudden weakness", "suddenly weak", "weakness came on suddenly",
        "can't move my arm", "can't move my leg", "leg gave out",
    ],
    "severe headache": [
        "severe headache", "worst headache", "terrible headache",
        "excruciating headache", "blinding headache", "thunderclap headache",
        "splitting headache",
    ],
    "sudden": [
        "sudden", "suddenly", "came on fast", "out of nowhere", "all of a sudden",
    ],
    "worst headache": [
        "worst headache", "worst headache of my life", "never had a headache this bad",
    ],
    "difficulty breathing": [
        "difficulty breathing", "hard to breathe", "struggling to breathe",
        "trouble breathing", "can't breathe", "cannot breathe",
        "labored breathing", "laboured breathing", "respiratory distress",
        "can't catch my breath", "can't get enough air",
    ],
    "can't breathe": [
        "can't breathe", "cannot breathe", "unable to breathe",
        "i can't breathe", "struggling to breathe", "can't catch my breath",
    ],
    "coughing blood": [
        "coughing blood", "coughing up blood", "blood when i cough",
        "hemoptysis", "blood in sputum", "bloody sputum",
    ],
    "vomiting blood": [
        "vomiting blood", "throwing up blood", "blood in vomit",
        "hematemesis", "bloody vomit",
    ],
    "blood in stool": [
        "blood in stool", "bloody stool", "blood in my stool",
        "rectal bleeding", "blood when i go to the bathroom",
        "black stool", "tarry stool", "melena",
    ],
    "seizure": [
        "seizure", "seizures", "having a seizure", "had a seizure",
    ],
    "convulsion": [
        "convulsion", "convulsions", "convulsing",
    ],
    "high fever": [
        "high fever", "very high fever", "fever over 103", "fever over 104",
        "fever of 103", "fever of 104", "fever of 105", "burning up",
    ],
    "stiff neck": [
        "stiff neck", "neck stiffness", "neck is stiff", "can't move my neck",
    ],
    "suicidal": [
        "suicidal", "suicidal thoughts", "thinking about suicide",
        "want to kill myself", "end my life", "don't want to live",
    ],
    "want to die": [
        "want to die", "wanna die", "wish i was dead", "wish i were dead",
        "better off dead", "rather be dead",
    ],
    "hurt myself": [
        "hurt myself", "harm myself", "self harm", "self-harm",
        "cutting myself", "hurting myself",
    ],
}

# Red flag rules: (primary_group_key, [required_secondary_group_keys])
# A flag is raised if primary matches AND all secondary groups match.
RED_FLAG_RULES: list[tuple[str, list[str]]] = [
    ("chest pain", []),  # Standalone — chest pain alone is an emergency indicator
    ("chest pain", ["shortness of breath"]),
    ("chest pain", ["difficulty breathing"]),
    ("chest pain", ["arm pain"]),
    ("chest pain", ["jaw pain"]),
    ("fainted", []),
    ("fainting", []),
    ("passed out", []),
    ("face drooping", []),
    ("facial droop", []),
    ("slurred speech", []),
    ("sudden numbness", []),
    ("sudden weakness", []),
    ("severe headache", []),  # Standalone — severe headache alone warrants escalation
    ("severe headache", ["sudden"]),
    ("worst headache", []),
    ("difficulty breathing", []),
    ("can't breathe", []),
    ("coughing blood", []),
    ("vomiting blood", []),
    ("blood in stool", []),
    ("seizure", []),
    ("convulsion", []),
    ("high fever", []),  # Standalone — high fever alone warrants monitoring
    ("high fever", ["stiff neck"]),
    ("suicidal", []),
    ("want to die", []),
    ("hurt myself", []),
]


def _matches_any(transcript_lower: str, group_key: str) -> bool:
    """Check if any synonym in the group matches the transcript."""
    synonyms = _SYNONYM_GROUPS.get(group_key)
    if synonyms is None:
        # Fallback: treat group_key itself as a literal substring
        return group_key in transcript_lower
    return any(synonym in transcript_lower for synonym in synonyms)


def detect_red_flags(transcript: str) -> list[str]:
    """
    Detect potential red flags in a transcript.

    This is a non-diagnostic keyword detector. It does NOT provide medical
    advice or diagnoses. It simply identifies patterns that may warrant
    prompt clinical attention.

    Args:
        transcript: The text to analyze

    Returns:
        A list of detected red flag keywords (empty if none found)
    """
    t = transcript.lower()
    flags: list[str] = []

    for primary, secondary in RED_FLAG_RULES:
        if _matches_any(t, primary) and all(_matches_any(t, s) for s in secondary):
            if primary not in flags:
                flags.append(primary)

    return flags


# Standard safety disclaimer - should be displayed whenever red flags are shown
SAFETY_DISCLAIMER = (
    "If symptoms feel severe, rapidly worsening, or concerning, "
    "consider contacting a clinician promptly."
)

# Educational disclaimer - should always be visible in the app
EDUCATIONAL_DISCLAIMER = (
    "This app is for educational purposes only and does not provide "
    "medical advice, diagnosis, or treatment recommendations. "
    "Always consult a qualified healthcare provider."
)
