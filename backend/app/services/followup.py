from typing import Optional
from ..models import ExtractionResult


def choose_followup(extracted: ExtractionResult) -> Optional[str]:
    """
    Choose at most 1 follow-up question based on missing fields.

    Priority order: severity → onset → duration

    Rules:
    - Ask at most 1 question per log
    - Do not claim medication onset times (avoid pharmacokinetics assertions)
    """
    if "severity" in extracted.missing_fields:
        return "Quick check: how severe is it (1-10)?"

    if "onset" in extracted.missing_fields:
        return "About when did this start?"

    if "duration" in extracted.missing_fields:
        return "Roughly how long did it last?"

    return None
