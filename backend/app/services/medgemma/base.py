"""Base class and shared utilities for MedGemma clients."""

import os
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

from ...models import (
    ExtractionResult,
    SymptomEntity,
    ActionEntity,
    DoctorPacket,
    TimelineReveal,
    TimelinePoint,
    LogEntry,
    WatchdogResult,
)

logger = logging.getLogger(__name__)

# Lazy load Vertex AI dependencies
_aiplatform = None
_Endpoint = None


def _load_vertex_ai():
    """Lazy load Vertex AI SDK."""
    global _aiplatform, _Endpoint
    if _aiplatform is None:
        try:
            from google.cloud import aiplatform
            from google.cloud.aiplatform import Endpoint
            _aiplatform = aiplatform
            _Endpoint = Endpoint
            logger.info("Vertex AI SDK loaded successfully")
            return True
        except ImportError as e:
            logger.warning(f"Vertex AI SDK not available: {e}")
            return False
    return True


class MedGemmaClient(ABC):
    """Abstract base class for MedGemma extraction and summarization."""

    def __init__(self):
        self.last_fallback_reason: Optional[str] = None
        # Cache for build_full_history_context — avoids redundant map-reduce
        # when multiple endpoints (timeline, doctor packet) request the same logs.
        # Dict keyed on frozenset(log_ids) so ordering doesn't matter.
        self._history_cache: Dict[frozenset, tuple] = {}  # {frozenset: (result_str, timestamp)}

    @staticmethod
    def _sanitize_reason_token(value: object, *, max_len: int = 120) -> str:
        """Make fallback metadata safe for headers/log tokens."""
        token = str(value).strip()
        token = token.replace("\n", " ").replace("\r", " ")
        token = token.replace("|", "/").replace(";", "/")
        if len(token) > max_len:
            token = token[:max_len]
        return token

    @staticmethod
    def _infer_stage_from_fallback(fallback: str) -> str:
        if fallback.endswith("_stub_fallback"):
            return fallback[: -len("_stub_fallback")]
        if fallback.startswith("agent_response"):
            return "agent_response"
        if fallback.startswith("response_generator"):
            return "response_generator"
        return fallback

    def describe_runtime(self) -> Dict[str, str]:
        """Describe the active client/runtime in a compact, non-sensitive form."""
        provider_map = {
            "VertexAIMedGemmaClient": "vertex",
            "LocalMedGemmaClient": "local",
            "StubMedGemmaClient": "stub",
        }
        runtime: Dict[str, str] = {
            "client": self.__class__.__name__,
            "provider": provider_map.get(self.__class__.__name__, "unknown"),
        }

        attr_candidates = {
            "model": ("MODEL_ID", "model"),
            "endpoint": ("_endpoint_id", "endpoint_id"),
            "dedicated_dns": ("_dedicated_endpoint_dns", "dedicated_endpoint_dns"),
            "region": ("_region", "region"),
            "project": ("_project_id", "project_id"),
        }

        for output_key, attrs in attr_candidates.items():
            for attr in attrs:
                value = getattr(self, attr, None)
                if value:
                    runtime[output_key] = self._sanitize_reason_token(value)
                    break

        return runtime

    def format_fallback_reason(
        self,
        reason: str,
        *,
        stage: Optional[str] = None,
        error: Optional[Exception] = None,
    ) -> str:
        """Build structured fallback metadata for diagnostics and UI surfacing."""
        raw_reason = self._sanitize_reason_token(reason or "unknown_fallback")
        fallback = raw_reason
        detail = type(error).__name__ if error else "unknown"
        error_detail = None
        if error is not None:
            msg = str(error).strip()
            if msg:
                error_detail = msg

        if ":" in raw_reason:
            fallback, parsed_detail = raw_reason.split(":", 1)
            if parsed_detail:
                detail = parsed_detail

        stage_name = stage or self._infer_stage_from_fallback(fallback)
        runtime = self.describe_runtime()
        ordered_fields = {
            "stage": stage_name,
            "fallback": fallback,
            "reason": detail,
            "detail": error_detail,
            "client": runtime.get("client"),
            "provider": runtime.get("provider"),
            "model": runtime.get("model"),
            "endpoint": runtime.get("endpoint"),
            "region": runtime.get("region"),
            "project": runtime.get("project"),
        }

        tokens = []
        for key, value in ordered_fields.items():
            if value:
                tokens.append(f"{key}={self._sanitize_reason_token(value)}")
        return "|".join(tokens)

    def _clear_last_fallback(self) -> None:
        self.last_fallback_reason = None

    def _set_last_fallback(self, reason: str) -> None:
        if "fallback=" in reason and "client=" in reason:
            formatted_reason = reason
        else:
            formatted_reason = self.format_fallback_reason(reason)
        self.last_fallback_reason = formatted_reason
        logger.warning("MedGemma fallback activated: %s", formatted_reason)

    def get_last_fallback_reason(self) -> Optional[str]:
        return self.last_fallback_reason

    def consume_last_fallback_reason(self) -> Optional[str]:
        reason = self.last_fallback_reason
        self.last_fallback_reason = None
        return reason

    @abstractmethod
    async def extract(self, transcript: str) -> ExtractionResult:
        """Extract structured symptom information from transcript."""
        raise NotImplementedError

    @abstractmethod
    async def doctor_packet(self, logs: List[LogEntry], days: int, user_id: str | None = None, user_profile=None) -> DoctorPacket:
        """Generate a clinician-facing packet."""
        raise NotImplementedError

    @abstractmethod
    async def timeline(self, logs: List[LogEntry], days: int) -> TimelineReveal:
        """Generate timeline story points."""
        raise NotImplementedError

    @abstractmethod
    async def generate_agent_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text for agent response prompts."""
        raise NotImplementedError

    @abstractmethod
    async def generate_profile_update(self, logs: List[LogEntry], current_profile: dict) -> dict:
        """
        Analyze recent logs and generate profile updates.

        Returns a dict with optional keys:
        - add_conditions: list of conditions to add
        - add_allergies: list of allergies to add
        - add_patterns: list of patterns to add
        - health_summary: updated summary text
        """
        raise NotImplementedError

    @abstractmethod
    async def watchdog_analysis(self, history_context: str) -> "WatchdogResult":
        """Run background diagnostic analysis on longitudinal history.

        Accepts the pre-built history context string (from build_full_history_context),
        not raw logs. Returns a WatchdogResult with internal reasoning (logged only)
        and a safe patient-facing nudge.
        """
        raise NotImplementedError

    async def respond_to_followup(
        self,
        original_transcript: str,
        followup_question: str,
        followup_answer: str,
        patient_name: Optional[str] = None,
        user_profile=None,
    ) -> str:
        """Generate a brief acknowledgment of the patient's followup answer.

        Default implementation returns a generic response. Override in subclasses
        that have LLM access for context-aware responses.
        """
        return "Thanks for sharing that — it's noted."

    def _format_patient_profile(self, user_profile, include_patterns: bool = True) -> str:
        """Format user profile for inclusion in prompts."""
        if not user_profile:
            return "No patient profile available."
        parts = []
        if getattr(user_profile, 'name', None):
            parts.append(f"Name: {user_profile.name}")
        if getattr(user_profile, 'age', None) is not None:
            parts.append(f"Age: {user_profile.age}")
        if getattr(user_profile, 'gender', None):
            parts.append(f"Gender: {user_profile.gender}")
        if getattr(user_profile, 'health_summary', None):
            parts.append(f"Summary: {user_profile.health_summary}")
        if getattr(user_profile, 'conditions', None):
            parts.append(f"Conditions: {', '.join(user_profile.conditions)}")
        if getattr(user_profile, 'allergies', None):
            parts.append(f"Allergies: {', '.join(user_profile.allergies)}")
        if getattr(user_profile, 'regular_medications', None):
            parts.append("Regular medications:\n" + "\n".join(f"  - {m}" for m in user_profile.regular_medications))
        if getattr(user_profile, 'surgeries', None):
            parts.append(f"Surgeries/procedures: {', '.join(user_profile.surgeries)}")
        if getattr(user_profile, 'family_history', None):
            parts.append("Family history:\n" + "\n".join(f"  - {h}" for h in user_profile.family_history))
        if getattr(user_profile, 'social_history', None):
            parts.append("Social history:\n" + "\n".join(f"  - {s}" for s in user_profile.social_history))
        if include_patterns and getattr(user_profile, 'patterns', None):
            parts.append("Known patterns:\n" + "\n".join(f"  - {p}" for p in user_profile.patterns))
        return "\n".join(parts) if parts else "No patient profile available."

    def _fix_hpi_demographics(self, hpi: str, user_profile) -> str:
        """Post-process HPI to correct any hallucinated age or gender."""
        if not user_profile or not hpi:
            return hpi
        age = getattr(user_profile, 'age', None)
        if age is not None:
            hpi = re.sub(r'\b\d{1,3}-year-old\b', f'{age}-year-old', hpi, count=1)
        gender = getattr(user_profile, 'gender', None)
        if gender:
            wrong_map = {"female": "male", "male": "female"}
            wrong_gender = wrong_map.get(gender)
            if wrong_gender and re.search(rf'\b{wrong_gender}\b', hpi[:120], re.IGNORECASE):
                hpi = re.sub(rf'\b{wrong_gender}\b', gender, hpi, count=1, flags=re.IGNORECASE)
        return hpi

    def _fix_pertinent_positives_demographics(self, positives: list, user_profile) -> list:
        """Correct hallucinated age in pertinent_positives entries."""
        if not user_profile or not positives:
            return positives
        age = getattr(user_profile, 'age', None)
        if age is None:
            return positives
        return [
            re.sub(r'\b\d{1,3}-year-old\b', f'{age}-year-old', entry, count=1)
            for entry in positives
        ]

    _DATE_RE = re.compile(r'\b(\d{4})-(\d{2}-\d{2})\b')

    def _fix_hpi_dates(self, hpi: str, logs: List[LogEntry]) -> str:
        """Fix year-off-by-one hallucinations in HPI dates.

        Collects actual log dates, then for any YYYY-MM-DD in the HPI that
        doesn't match a log date, tries year ±1.  Only replaces when the
        corrected date provably exists in the logs.
        """
        if not logs or not hpi:
            return hpi
        log_dates: set[str] = set()
        for log in logs:
            dt = getattr(log, 'recorded_at', None)
            if dt:
                log_dates.add(dt.strftime('%Y-%m-%d'))
        if not log_dates:
            return hpi

        def _fix(m: re.Match) -> str:
            full = m.group(0)
            if full in log_dates:
                return full
            year = int(m.group(1))
            md = m.group(2)
            for delta in (-1, 1):
                candidate = f"{year + delta}-{md}"
                if candidate in log_dates:
                    return candidate
            return full

        return self._DATE_RE.sub(_fix, hpi)

    def _format_logs_for_prompt(self, logs: List[LogEntry], max_logs: int = 20) -> str:
        """Format logs for inclusion in prompts with full clinical detail."""
        formatted = []
        for log in logs[:max_logs]:  # Configurable limit (default 20 for responses, 50 for doctor packets)
            # Format symptoms with all available details
            symptom_details = []
            for s in log.extracted.symptoms:
                detail = s.symptom
                if hasattr(s, 'location') and s.location:
                    detail += f" ({s.location})"
                if hasattr(s, 'character') and s.character:
                    detail += f", {s.character}"
                if s.severity_1_10:
                    detail += f", severity {s.severity_1_10}/10"
                if s.onset_time_text:
                    detail += f", onset: {s.onset_time_text}"
                if s.duration_text:
                    detail += f", duration: {s.duration_text}"
                if hasattr(s, 'triggers') and s.triggers:
                    detail += f", triggers: {', '.join(s.triggers)}"
                if hasattr(s, 'relievers') and s.relievers:
                    detail += f", relieved by: {', '.join(s.relievers)}"
                symptom_details.append(detail)

            # Format actions with doses
            action_details = []
            for a in log.extracted.actions_taken:
                detail = a.name
                if a.dose_text:
                    detail += f" {a.dose_text}"
                if a.effect_text:
                    detail += f" ({a.effect_text})"
                action_details.append(detail)

            # Include raw transcript for context
            transcript_preview = log.extracted.transcript[:200] if log.extracted.transcript else ""

            # Include image analysis if available
            image_line = ""
            if log.image_analysis:
                ia = log.image_analysis
                image_line = f"\n  Image findings: {ia.clinical_description}"
                if ia.lesion_detected and ia.skin_lesion:
                    sl = ia.skin_lesion
                    image_line += f" (type: {sl.lesion_type}, color: {sl.color}, size: {sl.size_estimate})"

            formatted.append(
                f"[{log.recorded_at.strftime('%Y-%m-%d %H:%M')}]\n"
                f"  Raw: \"{transcript_preview}\"\n"
                f"  Symptoms: {', '.join(symptom_details) or 'none extracted'}\n"
                f"  Actions: {', '.join(action_details) or 'none'}"
                f"{image_line}"
            )
        return "\n\n".join(formatted)

    def _build_timeline_bullets(self, logs: List[LogEntry]) -> List[str]:
        """Build timeline bullets deterministically from log data.

        Guarantees every log gets a bullet with the correct date — no LLM
        truncation or date hallucination possible.
        """
        bullets = []
        for log in logs:
            date_str = log.recorded_at.strftime("%Y-%m-%d")
            parts = []
            for s in log.extracted.symptoms:
                detail = s.symptom
                if s.severity_1_10:
                    detail += f" ({s.severity_1_10}/10)"
                parts.append(detail)
            for a in log.extracted.actions_taken:
                detail = a.name
                if a.dose_text:
                    detail += f" {a.dose_text}"
                parts.append(detail)
            if log.extracted.red_flags:
                parts.append(f"RED FLAG: {', '.join(log.extracted.red_flags)}")
            if not parts:
                # Use original transcript, not extracted (which may include followup text)
                transcript = log.transcript or ""
                # Strip any followup contamination that may have leaked
                if ". When asked " in transcript:
                    transcript = transcript.split(". When asked ")[0]
                if transcript:
                    if len(transcript) > 200:
                        truncated = transcript[:200].rsplit(" ", 1)[0]
                        parts.append(truncated + "...")
                    else:
                        parts.append(transcript)
                else:
                    parts.append("entry recorded")
            bullets.append(f"{date_str}: {'; '.join(parts)}")
        return bullets

    # -- Chunked map-reduce for full-history doctor packets --

    CHUNK_SUMMARY_PROMPT = """You are a medical assistant condensing patient symptom logs into a clinical summary.

<logs>
{logs_text}
</logs>

Write ONE concise clinical paragraph (3-5 sentences). Include:
- All unique symptoms with severity ranges (e.g., "nausea rated 3-5/10")
- All medications with doses
- Key patterns (timing, triggers, what helped)
- Any red flags or concerning events
- Date range covered

Plain text paragraph only — no JSON, no bullets."""

    async def _summarize_chunk(self, logs: List[LogEntry]) -> str:
        """Summarize a chunk of logs into a clinical paragraph."""
        logs_text = self._format_logs_for_prompt(logs, max_logs=30)
        prompt = self.CHUNK_SUMMARY_PROMPT.format(logs_text=logs_text)
        return await self.generate_agent_response(prompt, max_tokens=512)

    async def build_full_history_context(self, all_logs: List[LogEntry], user_id: Optional[str] = None) -> str:
        """Build full-history context using baseline compression or chunked summarization.

        If a rolling baseline exists for the user (from compress_history_if_needed),
        uses hybrid context: pre-compressed baseline + recent high-resolution logs.
        This requires ZERO LLM calls for older history.

        Falls back to chunked map-reduce when no baseline is available.

        Results are cached for 5 minutes keyed on frozenset(log_ids).
        """
        if not all_logs:
            return "No patient logs available."

        # Sort oldest-first — callers may pass DESC or ASC
        all_logs = sorted(all_logs, key=lambda l: l.recorded_at)

        # Order-independent cache key
        cache_key = frozenset(log.id for log in all_logs)
        now = time.monotonic()
        cached = self._history_cache.get(cache_key)
        if cached and (now - cached[1]) < 300:
            return cached[0]

        # --- Hybrid path: use rolling baseline if available ---
        if user_id:
            from ...services.storage import get_baseline_info
            baseline = get_baseline_info(user_id)
            if baseline["text"]:
                cursor_date = datetime.fromisoformat(baseline["last_compressed_at"])
                if cursor_date.tzinfo is None:
                    cursor_date = cursor_date.replace(tzinfo=timezone.utc)
                # Only include logs newer than the compression cursor
                uncompressed = [
                    l for l in all_logs
                    if (l.recorded_at if l.recorded_at.tzinfo else l.recorded_at.replace(tzinfo=timezone.utc)) > cursor_date
                ]
                parts = [
                    "=== HISTORICAL BASELINE (> 30 Days Old) ===",
                    baseline["text"],
                    "\n=== RECENT DAILY LOGS (High Resolution) ===",
                    self._format_logs_for_prompt(uncompressed, max_logs=50) if uncompressed else "(no recent logs)",
                ]
                result = "\n".join(parts)
                self._history_cache[cache_key] = (result, now)
                self._evict_stale_cache(now)
                return result

        # --- Fallback: chunked map-reduce (no baseline yet) ---
        latest_time = all_logs[-1].recorded_at
        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=timezone.utc)
        seven_days_ago = latest_time - timedelta(days=7)

        recent_logs = []
        older_logs = []
        for log in all_logs:
            log_time = log.recorded_at
            if log_time.tzinfo is None:
                log_time = log_time.replace(tzinfo=timezone.utc)
            if log_time >= seven_days_ago:
                recent_logs.append(log)
            else:
                older_logs.append(log)

        parts = []

        # Summarize older logs in chunks
        if older_logs:
            chunk_size = 30
            chunks = [older_logs[i:i + chunk_size] for i in range(0, len(older_logs), chunk_size)]
            parts.append("=== PRIOR HISTORY (condensed) ===")
            for chunk in chunks:
                start = chunk[0].recorded_at.strftime('%Y-%m-%d')
                end = chunk[-1].recorded_at.strftime('%Y-%m-%d')
                try:
                    summary = await self._summarize_chunk(chunk)
                    parts.append(f"\n[{start} to {end}]:\n{summary}")
                except Exception as e:
                    logger.warning(f"Chunk summary failed ({start}-{end}): {e}")
                    parts.append(f"\n[{start} to {end}]: (summary unavailable)")

        # Recent logs as raw detail
        if recent_logs:
            parts.append("\n=== RECENT DETAIL (last 7 days) ===")
            parts.append(self._format_logs_for_prompt(recent_logs, max_logs=50))

        result = "\n".join(parts)
        self._history_cache[cache_key] = (result, now)
        self._evict_stale_cache(now)

        return result

    def _evict_stale_cache(self, now: float) -> None:
        """Remove cache entries older than 5 minutes."""
        stale = [k for k, (_, t) in self._history_cache.items() if now - t >= 300]
        for k in stale:
            del self._history_cache[k]

    async def compress_history_if_needed(self, user_id: str) -> None:
        """Compress logs older than 30 days into the permanent rolling baseline.

        Uses a time cursor to track the last compressed position. Only logs
        between the cursor and 30-days-ago are compressed in each run,
        creating a sliding window that enables multi-year patient tracking
        without unbounded token cost.
        """
        from ...services.storage import get_baseline_info, update_baseline, list_logs_in_range

        baseline = get_baseline_info(user_id)
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        logs_to_compress = list_logs_in_range(
            user_id, after=baseline["last_compressed_at"], before=thirty_days_ago
        )
        if not logs_to_compress:
            return

        from .vertex import VertexAIMedGemmaClient
        logs_text = self._format_logs_for_prompt(logs_to_compress, max_logs=50)
        current = baseline["text"] or "No prior baseline."
        prompt = VertexAIMedGemmaClient.BASELINE_COMPRESSION_PROMPT.format(
            current_baseline=current, logs_text=logs_text
        )
        new_baseline = await self.generate_agent_response(prompt, max_tokens=1024)

        latest = max(logs_to_compress, key=lambda l: l.recorded_at)
        cursor_ts = latest.recorded_at
        if cursor_ts.tzinfo is None:
            cursor_ts = cursor_ts.replace(tzinfo=timezone.utc)
        update_baseline(user_id, new_baseline.strip(), cursor_ts.isoformat())
        logger.info("[BASELINE] Compressed %d logs for user=%s", len(logs_to_compress), user_id)
