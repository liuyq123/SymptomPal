"""Tests for LLM-based safety classifier (replaces brittle regex patterns)."""

import pytest
from unittest.mock import AsyncMock, patch

from ..services.response_generator import (
    LLMResponseGenerator,
    _strip_count_spam,
    _SAFETY_CLASSIFIER_PROMPT,
)


@pytest.mark.asyncio
async def test_safety_check_passes_safe_text():
    """LLM classifier returns True for safe text."""
    gen = LLMResponseGenerator()
    with patch.object(gen.client, "generate_agent_response", new_callable=AsyncMock, return_value="SAFE"):
        assert await gen._llm_safety_check("Got it, logged your headache.") is True


@pytest.mark.asyncio
async def test_safety_check_blocks_unsafe_text():
    """LLM classifier returns False for text containing medical advice."""
    gen = LLMResponseGenerator()
    with patch.object(gen.client, "generate_agent_response", new_callable=AsyncMock, return_value="UNSAFE"):
        assert await gen._llm_safety_check("You should stop taking metformin immediately.") is False


@pytest.mark.asyncio
async def test_safety_check_fails_closed_on_error():
    """If the classifier call fails, assume unsafe (fail closed)."""
    gen = LLMResponseGenerator()
    with patch.object(gen.client, "generate_agent_response", new_callable=AsyncMock, side_effect=RuntimeError("model down")):
        assert await gen._llm_safety_check("Some text") is False


@pytest.mark.asyncio
async def test_safety_check_handles_mixed_case_response():
    """Classifier should handle 'Unsafe' or 'unsafe' responses."""
    gen = LLMResponseGenerator()
    with patch.object(gen.client, "generate_agent_response", new_callable=AsyncMock, return_value="Unsafe - contains medication advice"):
        assert await gen._llm_safety_check("You must start taking aspirin daily.") is False


def test_strip_count_spam_removes_logged_times():
    """Count spam patterns are still stripped (cosmetic, not safety-critical)."""
    text = "Got it. You've logged headaches 5 times this week. Let me know how you feel."
    result = _strip_count_spam(text)
    assert "5 times" not in result
    assert "Let me know" in result


def test_strip_count_spam_preserves_clean_text():
    """Clean text passes through unchanged."""
    text = "Noted your headache. Worth mentioning to your doctor."
    assert _strip_count_spam(text) == text


def test_safety_classifier_prompt_exists():
    """The LLM safety classifier prompt is defined."""
    assert "MEDICATION ADVICE" in _SAFETY_CLASSIFIER_PROMPT
    assert "DIAGNOSIS" in _SAFETY_CLASSIFIER_PROMPT
    assert "CAUSATION" in _SAFETY_CLASSIFIER_PROMPT


def test_regex_patterns_removed():
    """Verify _UNSAFE_ADVICE_PATTERNS no longer exists — regex is not a safety net."""
    from ..services import response_generator
    assert not hasattr(response_generator, "_UNSAFE_ADVICE_PATTERNS")
    assert not hasattr(response_generator, "_sanitize_acknowledgment")
