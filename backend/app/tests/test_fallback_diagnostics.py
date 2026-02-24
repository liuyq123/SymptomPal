"""Tests for structured fallback diagnostics metadata."""

from typing import List

from ..models import (
    DoctorPacket,
    ExtractionResult,
    LogEntry,
    TimelineReveal,
    WatchdogResult,
)
from ..services.medgemma.base import MedGemmaClient


class _DummyClient(MedGemmaClient):
    MODEL_ID = "unsloth/medgemma-27b-text-it-unsloth-bnb-4bit"
    _endpoint_id = "endpoint-123"
    _region = "us-central1"

    async def extract(self, transcript: str) -> ExtractionResult:  # pragma: no cover - not used
        raise NotImplementedError

    async def doctor_packet(self, logs: List[LogEntry], days: int) -> DoctorPacket:  # pragma: no cover - not used
        raise NotImplementedError

    async def timeline(self, logs: List[LogEntry], days: int) -> TimelineReveal:  # pragma: no cover - not used
        raise NotImplementedError

    async def generate_agent_response(self, prompt: str, max_tokens: int = 512) -> str:  # pragma: no cover - not used
        raise NotImplementedError

    async def generate_profile_update(self, logs: List[LogEntry], current_profile: dict) -> dict:  # pragma: no cover - not used
        raise NotImplementedError

    async def watchdog_analysis(self, history_context: str) -> WatchdogResult:  # pragma: no cover - not used
        raise NotImplementedError


def test_format_fallback_reason_contains_runtime_metadata():
    client = _DummyClient()
    formatted = client.format_fallback_reason("timeline_stub_fallback:vertex_not_implemented")

    assert "stage=timeline" in formatted
    assert "fallback=timeline_stub_fallback" in formatted
    assert "reason=vertex_not_implemented" in formatted
    assert "client=_DummyClient" in formatted
    assert "model=unsloth/medgemma-27b-text-it-unsloth-bnb-4bit" in formatted
    assert "endpoint=endpoint-123" in formatted
    assert "region=us-central1" in formatted


def test_set_last_fallback_auto_formats_reason():
    client = _DummyClient()
    client._set_last_fallback("extract_stub_fallback:RuntimeError")
    reason = client.consume_last_fallback_reason()

    assert reason is not None
    assert "stage=extract" in reason
    assert "fallback=extract_stub_fallback" in reason
    assert "reason=RuntimeError" in reason
