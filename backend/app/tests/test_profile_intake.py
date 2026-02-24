"""Unit tests for profile intake helpers."""

import pytest
from datetime import datetime, timezone

from ..models import UserProfile
from unittest.mock import AsyncMock
import json

from ..services.profile_intake import (
    build_intake_profile_patch,
    drain_pending_intake_raw,
    get_next_intake_question,
    should_start_intake,
    _build_profile_context,
    _extract_json,
    _get_hardcoded_question,
    parse_answer_and_generate_next_question,
)


def _empty_profile() -> UserProfile:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return UserProfile(
        user_id="test_user",
        conditions=[],
        allergies=[],
        regular_medications=[],
        surgeries=[],
        family_history=[],
        social_history=[],
        patterns=[],
        health_summary=None,
        created_at=now,
        updated_at=now,
    )


def test_should_start_intake_for_first_log():
    profile = _empty_profile()
    assert should_start_intake(profile, total_recent_logs=1) is True


def test_get_next_question_skips_answered():
    profile = _empty_profile()
    profile.intake_answered_question_ids = ["name", "age_sex", "conditions", "allergies"]
    next_question = get_next_intake_question(profile)
    assert next_question is not None
    question_id, _ = next_question
    assert question_id == "regular_medications"


def test_build_patch_for_none_answer_marks_progress_without_values():
    profile = _empty_profile()
    patch = build_intake_profile_patch(profile, "allergies", "none")
    assert patch["intake_questions_asked"] == 1
    assert patch["intake_answered_question_ids"] == ["allergies"]
    assert patch.get("add_allergies") == []


# --- LLM-powered intake helpers ---


def test_build_patch_with_pre_parsed_items():
    """LLM-parsed items should be used instead of regex parsing."""
    profile = _empty_profile()
    patch = build_intake_profile_patch(
        profile, "conditions", "I have high blood pressure and sugar",
        parsed_items=["Hypertension", "Type 2 Diabetes"],
    )
    assert patch["add_conditions"] == ["Hypertension", "Type 2 Diabetes"]


def test_build_patch_without_parsed_items_defers_raw():
    """When parsed_items is None, raw text is saved for deferred parsing."""
    profile = _empty_profile()
    patch = build_intake_profile_patch(profile, "conditions", "asthma and diabetes")
    assert patch["add_conditions"] == []  # No regex parsing attempted
    assert patch["intake_pending_raw_add"] == {"conditions": "asthma and diabetes"}


def test_build_patch_health_summary_with_parsed_items():
    """Health summary joins parsed items into a single string."""
    profile = _empty_profile()
    patch = build_intake_profile_patch(
        profile, "health_summary", "generally healthy, exercise daily",
        parsed_items=["Generally healthy", "Exercises daily"],
    )
    assert patch["health_summary"] == "Generally healthy Exercises daily"


def test_build_patch_health_summary_empty_parsed_items():
    """Empty parsed items for health_summary sets None."""
    profile = _empty_profile()
    patch = build_intake_profile_patch(
        profile, "health_summary", "nothing really",
        parsed_items=[],
    )
    assert patch["health_summary"] is None


def test_build_profile_context_populated():
    profile = _empty_profile()
    profile.conditions = ["Asthma", "Diabetes"]
    profile.allergies = ["Penicillin"]
    context = _build_profile_context(profile)
    assert "Asthma" in context
    assert "Penicillin" in context


def test_build_profile_context_empty():
    profile = _empty_profile()
    context = _build_profile_context(profile)
    assert "No information" in context


def test_get_hardcoded_question_found():
    q = _get_hardcoded_question("allergies")
    assert q is not None
    assert "allerg" in q.lower()


def test_get_hardcoded_question_none():
    assert _get_hardcoded_question(None) is None
    assert _get_hardcoded_question("nonexistent") is None


def test_extract_json_plain():
    raw = '{"parsed_items": ["Asthma"], "next_question": "Any allergies?"}'
    result = _extract_json(raw)
    assert '"parsed_items"' in result


def test_extract_json_with_markdown_fences():
    raw = '```json\n{"parsed_items": ["Asthma"]}\n```'
    result = _extract_json(raw)
    assert '"parsed_items"' in result


def test_extract_json_with_prefix_text():
    raw = 'Here is my response:\n{"parsed_items": ["COPD"]}'
    result = _extract_json(raw)
    assert '"parsed_items"' in result


@pytest.mark.asyncio
async def test_parse_none_answer_returns_empty():
    """'None' answers go to LLM and return empty parsed_items."""
    mock_client = AsyncMock()
    mock_client.generate_agent_response.return_value = json.dumps({
        "parsed_items": [],
        "next_question": "Any medication or food allergies?",
    })
    parsed, next_q = await parse_answer_and_generate_next_question(
        mock_client,
        _empty_profile(),
        "conditions",
        "Any chronic conditions?",
        "none",
        "allergies",
    )
    mock_client.generate_agent_response.assert_called_once()
    assert parsed == []
    assert next_q is not None


@pytest.mark.asyncio
async def test_parse_none_answer_last_question():
    """'None' answer on last question returns None for next question."""
    mock_client = AsyncMock()
    mock_client.generate_agent_response.return_value = json.dumps({
        "parsed_items": [],
        "next_question": None,
    })
    parsed, next_q = await parse_answer_and_generate_next_question(
        mock_client, _empty_profile(), "health_summary",
        "Anything else?", "skip", None,
    )
    assert parsed == []
    assert next_q is None


# --- Deferred reparse (drain) tests ---


@pytest.mark.asyncio
async def test_drain_pending_raw_reparses_items():
    """drain_pending_intake_raw reparses stored raw and returns patch."""
    profile = _empty_profile()
    profile.intake_pending_raw = {"conditions": "high blood pressure and sugar"}
    mock_client = AsyncMock()
    mock_client.generate_agent_response.return_value = json.dumps({
        "parsed_items": ["Hypertension", "Type 2 Diabetes"],
        "next_question": None,
    })
    patch = await drain_pending_intake_raw(mock_client, profile)
    assert patch["add_conditions"] == ["Hypertension", "Type 2 Diabetes"]
    assert patch["intake_pending_raw"] == {}  # cleared — empty dict passes storage's is-not-None guard


@pytest.mark.asyncio
async def test_drain_empty_pending_returns_empty():
    """No pending raw returns empty dict."""
    profile = _empty_profile()
    patch = await drain_pending_intake_raw(None, profile)
    assert patch == {}


@pytest.mark.asyncio
async def test_drain_keeps_failed_entries():
    """Entries that fail to reparse remain in pending_raw."""
    profile = _empty_profile()
    profile.intake_pending_raw = {"conditions": "stuff", "allergies": "penicillin"}
    mock_client = AsyncMock()
    mock_client.generate_agent_response.side_effect = [
        json.dumps({"parsed_items": ["Asthma"], "next_question": None}),
        Exception("model unavailable"),
    ]
    patch = await drain_pending_intake_raw(mock_client, profile)
    assert "add_conditions" in patch
    assert patch["intake_pending_raw"] == {"allergies": "penicillin"}


# --- Qualified denial LLM parsing ---


@pytest.mark.asyncio
async def test_parse_qualified_denial_extracts_items():
    """Qualified denial like 'Nothing but X' should extract X via LLM."""
    mock_client = AsyncMock()
    mock_client.generate_agent_response.return_value = json.dumps({
        "parsed_items": ["Menstrual irregularity", "Mild anemia"],
        "next_question": "Any medication allergies?",
    })
    parsed, next_q = await parse_answer_and_generate_next_question(
        mock_client, _empty_profile(), "conditions",
        "Any chronic conditions?",
        "Nothing diagnosed officially, but my periods have gotten really bad since I stopped birth control six months ago. And mild anemia.",
        "allergies",
    )
    # Verify LLM was called (not short-circuited by _is_none_answer)
    mock_client.generate_agent_response.assert_called_once()
    assert parsed == ["Menstrual irregularity", "Mild anemia"]
