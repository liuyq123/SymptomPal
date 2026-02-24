"""Tests for safety red flag detection."""

from ..services.safety import detect_red_flags, SAFETY_DISCLAIMER


def test_chest_pain_with_shortness_of_breath():
    """Chest pain + shortness of breath should trigger red flag."""
    transcript = "I have chest pain and shortness of breath"
    flags = detect_red_flags(transcript)
    assert "chest pain" in flags


def test_chest_pain_with_trouble_breathing():
    """Chest pain + trouble breathing should trigger red flag."""
    transcript = "Having chest pain and trouble breathing"
    flags = detect_red_flags(transcript)
    assert "chest pain" in flags


def test_chest_pain_alone_flags():
    """Chest pain alone should trigger red flag (standalone emergency indicator)."""
    transcript = "I have some chest pain"
    flags = detect_red_flags(transcript)
    assert "chest pain" in flags


def test_fainted_standalone():
    """Fainting alone should trigger red flag."""
    transcript = "I fainted yesterday"
    flags = detect_red_flags(transcript)
    assert "fainted" in flags


def test_face_drooping():
    """Face drooping should trigger red flag (stroke symptom)."""
    transcript = "My face drooping on one side"
    flags = detect_red_flags(transcript)
    assert "face drooping" in flags


def test_no_red_flags():
    """Regular symptoms should not trigger red flags."""
    transcript = "I have a mild headache and some nausea"
    flags = detect_red_flags(transcript)
    assert len(flags) == 0


def test_case_insensitive():
    """Detection should be case insensitive."""
    transcript = "I FAINTED and had CHEST PAIN with SHORTNESS OF BREATH"
    flags = detect_red_flags(transcript)
    assert "fainted" in flags
    assert "chest pain" in flags


def test_seizure():
    """Seizure should trigger red flag."""
    transcript = "I had a seizure this morning"
    flags = detect_red_flags(transcript)
    assert "seizure" in flags


def test_head_is_pounding_not_red_flag():
    """'Head is pounding a little' should NOT trigger severe headache red flag."""
    transcript = "My head is pounding a little but I can breathe better"
    flags = detect_red_flags(transcript)
    assert "severe headache" not in flags


def test_severe_headache_still_flags():
    """Genuine severe headache terms should still trigger red flag."""
    transcript = "I have the worst headache of my life"
    flags = detect_red_flags(transcript)
    assert "severe headache" in flags


def test_safety_disclaimer_exists():
    """Safety disclaimer should be defined."""
    assert SAFETY_DISCLAIMER is not None
    lower = SAFETY_DISCLAIMER.lower()
    assert "clinician" in lower
    assert "urgent care" not in lower
    assert "emergency" not in lower
