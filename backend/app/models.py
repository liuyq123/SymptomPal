from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
import re
from typing import Any, Optional, List, Dict
from datetime import datetime


# === Size Limits ===
MAX_AUDIO_B64_SIZE = 10 * 1024 * 1024  # 10MB base64 (~7.5MB raw audio)
MAX_PHOTO_B64_SIZE = 5 * 1024 * 1024   # 5MB base64 (~3.75MB raw image)
MAX_TEXT_SIZE = 10 * 1024              # 10KB text
MAX_USER_ID_SIZE = 256                  # 256 chars for user ID
TIME_HHMM_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

def _strip_data_url(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if "," in value:
        return value.split(",", 1)[1]
    return value

def _validate_image_data_url(value: str) -> None:
    if value.startswith("data:") and not value.startswith("data:image/"):
        raise ValueError("image_b64 must be a data URL with image/* MIME type")


# === Ingest + Log Storage ===

class VoiceIngestRequest(BaseModel):
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    audio_b64: Optional[str] = None  # Voice recording (optional now)
    recorded_at: datetime
    description_text: Optional[str] = Field(default=None, max_length=MAX_TEXT_SIZE)
    description_audio_b64: Optional[str] = None  # Voice description (will be transcribed)
    photo_b64: Optional[str] = None  # Photo attachment

    @field_validator('audio_b64')
    @classmethod
    def validate_audio_size(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > MAX_AUDIO_B64_SIZE:
            raise ValueError(f'audio_b64 exceeds maximum size of {MAX_AUDIO_B64_SIZE // (1024*1024)}MB')
        return v

    @field_validator('description_audio_b64')
    @classmethod
    def validate_description_audio_size(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > MAX_AUDIO_B64_SIZE:
            raise ValueError(f'description_audio_b64 exceeds maximum size of {MAX_AUDIO_B64_SIZE // (1024*1024)}MB')
        return v

    @field_validator('photo_b64')
    @classmethod
    def validate_photo_size(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            _validate_image_data_url(v)
            payload = _strip_data_url(v)
            if payload is not None and len(payload) > MAX_PHOTO_B64_SIZE:
                raise ValueError(f'photo_b64 exceeds maximum size of {MAX_PHOTO_B64_SIZE // (1024*1024)}MB')
        return v


class SymptomEntity(BaseModel):
    symptom: str
    location: Optional[str] = None  # e.g., "left side of head"
    character: Optional[str] = None  # e.g., "throbbing", "dull", "sharp"
    severity_1_10: Optional[int] = None
    onset_time_text: Optional[str] = None
    duration_text: Optional[str] = None
    triggers: List[str] = []
    relievers: List[str] = []
    associated_symptoms: List[str] = []

    @field_validator("severity_1_10", mode="before")
    @classmethod
    def clamp_severity(cls, v):
        """MedGemma sometimes confuses blood sugar values (e.g., 148) with severity.
        Discard values outside 1-10 range rather than crashing."""
        if v is None:
            return None
        try:
            val = int(v)
        except (TypeError, ValueError):
            return None
        if val < 1 or val > 10:
            return None
        return val

    @field_validator("triggers", "relievers", "associated_symptoms", mode="before")
    @classmethod
    def coerce_none_to_list(cls, v):
        """MedGemma sometimes returns null instead of []. Coerce to empty list."""
        if v is None:
            return []
        return v


class ActionEntity(BaseModel):
    name: str
    dose_text: Optional[str] = None
    effect_text: Optional[str] = None


class VitalSignEntry(BaseModel):
    name: str               # "blood sugar", "spo2", "blood pressure"
    value: str              # "143", "88%", "158/95"
    unit: Optional[str] = None  # "mg/dL", "%", "mmHg"


class MenstrualStatus(BaseModel):
    """Auto-detected menstrual status from transcript."""
    is_period_day: bool = False
    flow_level: Optional[str] = None  # "spotting", "light", "medium", "heavy"


class ExtractionResult(BaseModel):
    transcript: str
    symptoms: List[SymptomEntity] = []
    actions_taken: List[ActionEntity] = []
    vital_signs: List[VitalSignEntry] = []
    missing_fields: List[str] = []
    red_flags: List[str] = []
    menstrual_status: Optional[MenstrualStatus] = None


class FollowupExchange(BaseModel):
    question: str
    answer: Optional[str] = None
    agent_response: Optional[str] = None
    answered_at: Optional[datetime] = None


class LogEntry(BaseModel):
    id: str
    user_id: str
    recorded_at: datetime

    transcript: str
    description: Optional[str] = None  # User's text/transcribed description
    photo_b64: Optional[str] = None  # Photo attachment (base64)
    extracted: ExtractionResult
    image_analysis: Optional["ImageAnalysisResult"] = None  # MedSigLIP analysis of photo
    contact_clinician_note: Optional[str] = None
    contact_clinician_reason: Optional[str] = None

    followup_exchanges: List[FollowupExchange] = []

    @computed_field
    @property
    def followup_question(self) -> Optional[str]:
        return self.followup_exchanges[0].question if self.followup_exchanges else None

    @computed_field
    @property
    def followup_answer(self) -> Optional[str]:
        return self.followup_exchanges[0].answer if self.followup_exchanges else None

    @computed_field
    @property
    def followup_answered_at(self) -> Optional[datetime]:
        if self.followup_exchanges and self.followup_exchanges[0].answer:
            return self.followup_exchanges[0].answered_at
        return None

    @computed_field
    @property
    def followup_response(self) -> Optional[str]:
        return self.followup_exchanges[0].agent_response if self.followup_exchanges else None


# === Artifacts (Timeline, Doctor Packet) ===

class DoctorPacket(BaseModel):
    hpi: str  # <= 3 sentences
    pertinent_positives: List[str]
    pertinent_negatives: List[str]
    timeline_bullets: List[str]
    questions_for_clinician: List[str]
    system_longitudinal_flags: List[str] = Field(default_factory=list)


class TimelinePoint(BaseModel):
    timestamp: datetime
    label: str
    details: str


class TimelineReveal(BaseModel):
    story_points: List[TimelinePoint]


# === API Response Models ===

class IngestResponse(BaseModel):
    log: LogEntry
    followup_question: Optional[str] = None


class FollowupRequest(BaseModel):
    answer: str = Field(..., max_length=MAX_TEXT_SIZE)


class SummarizeRequest(BaseModel):
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    days: int = Field(default=7, ge=0, le=365)  # 0 = all history


# === Medication Logging ===

class MedicationEntry(BaseModel):
    id: str
    user_id: str
    name: str
    dose: Optional[str] = None  # e.g., "500mg", "2 tablets"
    frequency: Optional[str] = None  # e.g., "twice daily", "as needed"
    reason: Optional[str] = None  # Why taking it
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None  # None = ongoing
    notes: Optional[str] = None
    is_active: bool = True
    reminder_enabled: bool = False  # Whether reminders are enabled
    reminder_times: List[str] = []  # Times to remind (e.g., ["08:00", "20:00"])
    created_at: datetime
    updated_at: datetime


class MedicationLogEntry(BaseModel):
    """A single instance of taking a medication"""
    id: str
    user_id: str
    medication_id: Optional[str] = None  # Link to MedicationEntry, or None for ad-hoc
    medication_name: str  # Denormalized for convenience
    dose_taken: Optional[str] = None
    taken_at: datetime
    notes: Optional[str] = None
    symptom_log_id: Optional[str] = None  # Optional link to related symptom log


class MedicationCreateRequest(BaseModel):
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    name: str = Field(..., max_length=256)
    dose: Optional[str] = Field(default=None, max_length=256)
    frequency: Optional[str] = Field(default=None, max_length=256)
    reason: Optional[str] = Field(default=None, max_length=1024)
    start_date: Optional[datetime] = None
    notes: Optional[str] = Field(default=None, max_length=MAX_TEXT_SIZE)
    reminder_enabled: bool = False
    reminder_times: List[str] = Field(default_factory=list)

    @field_validator("reminder_times")
    @classmethod
    def validate_reminder_times(cls, values: List[str]) -> List[str]:
        for time_str in values:
            if not TIME_HHMM_PATTERN.match(time_str):
                raise ValueError("reminder_times must use HH:MM 24-hour format")
        return values


class MedicationLogRequest(BaseModel):
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    medication_id: Optional[str] = Field(default=None, max_length=MAX_USER_ID_SIZE)
    medication_name: Optional[str] = Field(default=None, max_length=256)
    dose_taken: Optional[str] = Field(default=None, max_length=256)
    taken_at: Optional[datetime] = None  # Defaults to now
    notes: Optional[str] = Field(default=None, max_length=MAX_TEXT_SIZE)
    symptom_log_id: Optional[str] = Field(default=None, max_length=MAX_USER_ID_SIZE)


class MedicationUpdateRequest(BaseModel):
    """Allowed fields for medication updates."""
    name: Optional[str] = Field(default=None, max_length=256)
    dose: Optional[str] = Field(default=None, max_length=256)
    frequency: Optional[str] = Field(default=None, max_length=256)
    reason: Optional[str] = Field(default=None, max_length=1024)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    notes: Optional[str] = Field(default=None, max_length=MAX_TEXT_SIZE)
    is_active: Optional[bool] = None
    reminder_enabled: Optional[bool] = None
    reminder_times: Optional[List[str]] = None

    @field_validator("reminder_times")
    @classmethod
    def validate_reminder_times(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return values
        for time_str in values:
            if not TIME_HHMM_PATTERN.match(time_str):
                raise ValueError("reminder_times must use HH:MM 24-hour format")
        return values

    model_config = ConfigDict(extra="forbid")


class MedicationVoiceRequest(BaseModel):
    """Request to transcribe + extract medication info from voice."""
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    audio_b64: str

    @field_validator('audio_b64')
    @classmethod
    def validate_audio_size(cls, v: str) -> str:
        if len(v) > MAX_AUDIO_B64_SIZE:
            raise ValueError(f'audio_b64 exceeds maximum size of {MAX_AUDIO_B64_SIZE // (1024*1024)}MB')
        return v


class MedicationVoiceResponse(BaseModel):
    """Extracted medication fields from voice input."""
    transcript: str
    medications: List[ActionEntity] = []
    frequency: Optional[str] = None
    reason: Optional[str] = None
    extraction_failed: bool = False


# === Medication Reminders ===

class PendingMedicationReminder(BaseModel):
    """A medication reminder that is currently due or upcoming"""
    medication_id: str
    medication_name: str
    dose: Optional[str] = None
    scheduled_time: str  # Time string like "08:00"
    due_at: datetime  # The specific datetime this reminder is for
    is_overdue: bool  # True if past due_at


class ReminderActionRequest(BaseModel):
    """Request to take action on a medication reminder"""
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    medication_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    due_at: datetime  # Which specific reminder instance
    snooze_minutes: Optional[int] = Field(default=None, ge=5, le=240)  # For snooze action


# === Ambient Monitoring (HeAR-based) ===

class SessionType(str, Enum):
    SLEEP = "sleep"
    COUGH_MONITOR = "cough_monitor"
    VOICE_BIOMARKER = "voice_biomarker"
    GENERAL = "general"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AmbientEventType(str, Enum):
    COUGH = "cough"
    BREATHING_NORMAL = "breathing_normal"
    BREATHING_IRREGULAR = "breathing_irregular"
    BREATHING_APNEA = "breathing_apnea"
    SNORING = "snoring"
    VOICE_STRESS = "voice_stress"
    VOICE_FATIGUE = "voice_fatigue"
    VOICE_CONGESTION = "voice_congestion"
    SILENCE = "silence"
    NOISE = "noise"


class AmbientSession(BaseModel):
    """An ambient audio monitoring session."""
    id: str
    user_id: str
    session_type: SessionType
    status: SessionStatus
    label: Optional[str] = None
    started_at: datetime
    ended_at: Optional[datetime] = None
    chunk_count: int = 0
    total_duration_seconds: float = 0.0
    created_at: datetime
    updated_at: datetime


class AmbientEvent(BaseModel):
    """A detected event within an ambient session."""
    id: str
    session_id: str
    user_id: str
    event_type: AmbientEventType
    timestamp: datetime
    duration_seconds: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Optional[dict] = None
    chunk_index: int


class AmbientChunk(BaseModel):
    """Metadata for an uploaded audio chunk."""
    id: str
    session_id: str
    user_id: str
    chunk_index: int
    duration_seconds: float
    uploaded_at: datetime
    processed: bool = False
    events_detected: int = 0


# === Ambient Metrics ===

class CoughMetrics(BaseModel):
    total_coughs: int
    coughs_per_hour: float
    peak_cough_period: Optional[str] = None
    cough_intensity_avg: Optional[float] = None
    # Classification results (aggregated from detected coughs)
    dominant_condition: Optional[str] = None  # healthy, abnormal, symptomatic
    condition_confidence: Optional[float] = None
    condition_probabilities: Optional[dict[str, float]] = None
    dominant_cough_type: Optional[str] = None  # dry, wet
    cough_type_confidence: Optional[float] = None
    dominant_severity: Optional[str] = None  # mild, moderate, severe
    severity_confidence: Optional[float] = None
    # Lung sound type classification (normal, wheeze, crackle, rhonchi, stridor, both)
    dominant_lung_sound: Optional[str] = None
    lung_sound_confidence: Optional[float] = None


class SleepQualityMetrics(BaseModel):
    total_sleep_duration_minutes: float
    breathing_regularity_score: float = Field(ge=0.0, le=100.0)
    apnea_events: int = 0
    snoring_minutes: float = 0.0
    restlessness_score: float = Field(ge=0.0, le=100.0)
    quality_rating: str  # "poor", "fair", "good", "excellent"


class VoiceBiomarkers(BaseModel):
    stress_level: float = Field(ge=0.0, le=100.0)
    fatigue_level: float = Field(ge=0.0, le=100.0)
    congestion_detected: bool = False
    voice_clarity_score: float = Field(ge=0.0, le=100.0)


class AmbientSessionResult(BaseModel):
    """Aggregated results from an ambient session."""
    session_id: str
    session_type: SessionType
    duration_minutes: float
    cough_metrics: Optional[CoughMetrics] = None
    sleep_quality: Optional[SleepQualityMetrics] = None
    voice_biomarkers: Optional[VoiceBiomarkers] = None
    events_timeline: List[AmbientEvent]
    summary: str


# === Ambient API Request/Response Models ===

class StartSessionRequest(BaseModel):
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    session_type: SessionType
    label: Optional[str] = Field(default=None, max_length=256)


class StartSessionResponse(BaseModel):
    session: AmbientSession
    upload_interval_seconds: int = 30


class UploadChunkRequest(BaseModel):
    session_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    chunk_index: int = Field(..., ge=0, le=10000)  # Max 10000 chunks per session
    audio_b64: str
    duration_seconds: float = Field(..., ge=0, le=300)  # Max 5 minutes per chunk

    @field_validator('audio_b64')
    @classmethod
    def validate_chunk_audio_size(cls, v: str) -> str:
        # Chunk audio limit: 2MB (30 seconds of audio)
        max_chunk_size = 2 * 1024 * 1024
        if len(v) > max_chunk_size:
            raise ValueError(f'audio_b64 chunk exceeds maximum size of {max_chunk_size // (1024*1024)}MB')
        return v


class UploadChunkResponse(BaseModel):
    chunk_id: str
    events_detected: List[AmbientEvent]


class EndSessionRequest(BaseModel):
    session_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)


class EndSessionResponse(BaseModel):
    session: AmbientSession
    result: AmbientSessionResult


# === Medical Image Analysis (MedSigLIP) ===

class SkinLesionDescription(BaseModel):
    """Detailed description of a detected skin lesion."""
    lesion_type: str  # e.g., "circular lesion", "raised nodule"
    color: str  # e.g., "erythematous", "hyperpigmented"
    size_estimate: str  # e.g., "approximately 2-3cm"
    texture: str  # e.g., "scaly", "smooth"
    predicted_condition: Optional[str] = None  # e.g., "eczema or atopic dermatitis"
    condition_confidence: Optional[float] = None
    confidence_scores: Dict[str, float] = Field(default_factory=dict)


class ImageAnalysisResult(BaseModel):
    """Result from medical image analysis."""
    clinical_description: str  # e.g., "Erythematous circular lesion, approx 3cm"
    confidence: float = Field(ge=0.0, le=1.0)
    lesion_detected: bool = False
    skin_lesion: Optional[SkinLesionDescription] = None
    raw_classifications: Optional[Dict] = None  # Debug: top predictions


class ImageIngestRequest(BaseModel):
    """Request to analyze a medical image."""
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    image_b64: str  # Base64-encoded image (PNG/JPEG)
    context: Optional[str] = Field(default=None, max_length=MAX_TEXT_SIZE)  # e.g., "rash on arm"
    recorded_at: Optional[datetime] = None

    @field_validator('image_b64')
    @classmethod
    def validate_image_size(cls, v: str) -> str:
        _validate_image_data_url(v)
        payload = _strip_data_url(v) or ""
        if len(payload) > MAX_PHOTO_B64_SIZE:
            raise ValueError(f'image_b64 exceeds maximum size of {MAX_PHOTO_B64_SIZE // (1024*1024)}MB')
        return v


class ImageIngestResponse(BaseModel):
    """Response from image analysis."""
    analysis: ImageAnalysisResult
    log_id: Optional[str] = None  # If attached to a symptom log
    transcript_addition: str  # Text to add to log: "Image shows: ..."


# === Watchdog Diagnostic Analysis ===

class WatchdogResult(BaseModel):
    """Internal result from the Clinical Watchdog Agent.

    internal_clinical_rationale is logged to terminal only — never stored in user-facing DB.
    safe_patient_nudge is the only field surfaced to the patient.
    clinician_facing_observation is injected into the Doctor Packet — objective, non-diagnostic.
    """
    concerning_pattern_detected: bool
    internal_clinical_rationale: str
    safe_patient_nudge: Optional[str] = None
    clinician_facing_observation: Optional[str] = None


# === Proactive Agent Response Models ===

class CheckinType(str, Enum):
    """Types of scheduled check-ins the agent can create."""
    MEDICATION_FOLLOWUP = "medication_followup"  # "Did the ibuprofen help?"
    SYMPTOM_PROGRESSION = "symptom_progression"  # "How's the headache now?"
    SCHEDULED_REMINDER = "scheduled_reminder"    # General check-in
    PROFILE_INTAKE = "profile_intake"            # New-user profile-building intake
    HEALTH_INSIGHT = "health_insight"            # Watchdog diagnostic nudge


class ScheduledCheckin(BaseModel):
    """A check-in the agent schedules for later."""
    id: str
    user_id: str
    checkin_type: CheckinType
    scheduled_for: datetime
    message: str  # What the agent will say
    context: Dict = Field(default_factory=dict)  # Related log_id, medication_name, etc.
    created_at: datetime
    triggered: bool = False  # Has been shown to user
    dismissed: bool = False  # User dismissed without responding
    response: Optional[str] = None
    responded_at: Optional[datetime] = None


class AgentResponse(BaseModel):
    """The proactive agent's response to a new symptom log."""
    acknowledgment: str  # Always present: "Got it, logged your headache."
    immediate_question: Optional[str] = None  # Only for critical missing info
    scheduled_checkin: Optional[ScheduledCheckin] = None  # Future check-in
    protocol_id: Optional[str] = None  # Which protocol produced the follow-up decision
    reason_code: Optional[str] = None  # Deterministic reason for auditing
    safety_mode: Optional[str] = None  # protocol | llm_fallback | legacy
    tool_calls: List[str] = []  # Tools MedGemma chose to invoke (e.g. "run_watchdog_now")
    question_rationale: Optional[str] = None  # Why the agent chose this follow-up question
    agent_trace: Dict[str, Any] = Field(default_factory=dict)  # Decision chain for auditability


class EnhancedIngestResponse(BaseModel):
    """Enhanced response that includes agent behavior."""
    log: LogEntry
    agent_response: AgentResponse
    # Backward compatibility - maps to immediate_question
    followup_question: Optional[str] = None
    image_analysis: Optional[ImageAnalysisResult] = None
    degraded_mode: bool = False
    warnings: List[str] = Field(default_factory=list)


class CheckinRespondRequest(BaseModel):
    """Request to respond to a scheduled check-in."""
    response: Optional[str] = Field(default=None, max_length=MAX_TEXT_SIZE)
    response_audio_b64: Optional[str] = None

    @field_validator("response_audio_b64")
    @classmethod
    def validate_response_audio_size(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and len(value) > MAX_AUDIO_B64_SIZE:
            raise ValueError(
                f"response_audio_b64 exceeds maximum size of {MAX_AUDIO_B64_SIZE // (1024 * 1024)}MB"
            )
        return value

    @model_validator(mode="after")
    def validate_has_text_or_audio(self):
        has_text = bool((self.response or "").strip())
        has_audio = bool(self.response_audio_b64)
        if not has_text and not has_audio:
            raise ValueError("Either response text or response_audio_b64 must be provided")
        return self


# === User Profile (Long-Term Memory) ===

class UserProfile(BaseModel):
    """
    Long-term health context that evolves over time.

    Used by the proactive agent to personalize interactions:
    - "How's your asthma? Have you used your rescue inhaler today?"
    - "I noticed you've had migraines 3 times this week..."
    """
    user_id: str

    # Demographics
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None

    # Chronic conditions (e.g., "Asthma", "Type 2 Diabetes", "Recurrent Migraines")
    conditions: List[str] = Field(default_factory=list)

    # Known allergies (e.g., "Penicillin", "Peanuts")
    allergies: List[str] = Field(default_factory=list)

    # Regular medications not captured in medication_log
    # (e.g., "Rescue inhaler", "Daily multivitamin")
    regular_medications: List[str] = Field(default_factory=list)

    # Past surgeries/procedures
    surgeries: List[str] = Field(default_factory=list)

    # Family history risk factors
    family_history: List[str] = Field(default_factory=list)

    # Social history context (e.g., smoking/alcohol/exercise)
    social_history: List[str] = Field(default_factory=list)

    # Observed patterns (e.g., "Migraines often triggered by stress")
    patterns: List[str] = Field(default_factory=list)

    # Key health context (free text, updated by MedGemma)
    health_summary: Optional[str] = None

    # Metadata
    created_at: datetime
    updated_at: datetime
    intake_completed: bool = False
    intake_questions_asked: int = 0
    intake_answered_question_ids: List[str] = Field(default_factory=list)
    intake_last_question_id: Optional[str] = None
    intake_started_at: Optional[datetime] = None
    intake_completed_at: Optional[datetime] = None
    intake_pending_raw: Dict[str, str] = Field(default_factory=dict)


class ProfileUpdateRequest(BaseModel):
    """Request to update a user's profile."""
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    add_conditions: List[str] = Field(default_factory=list)
    remove_conditions: List[str] = Field(default_factory=list)
    add_allergies: List[str] = Field(default_factory=list)
    remove_allergies: List[str] = Field(default_factory=list)
    add_regular_medications: List[str] = Field(default_factory=list)
    remove_regular_medications: List[str] = Field(default_factory=list)
    add_surgeries: List[str] = Field(default_factory=list)
    remove_surgeries: List[str] = Field(default_factory=list)
    add_family_history: List[str] = Field(default_factory=list)
    remove_family_history: List[str] = Field(default_factory=list)
    add_social_history: List[str] = Field(default_factory=list)
    remove_social_history: List[str] = Field(default_factory=list)
    add_patterns: List[str] = Field(default_factory=list)
    remove_patterns: List[str] = Field(default_factory=list)
    health_summary: Optional[str] = None


# === Menstrual Cycle Tracking ===

class FlowLevel(str, Enum):
    NONE = "none"
    SPOTTING = "spotting"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"


class CycleDayLog(BaseModel):
    """A single day's menstrual cycle data."""
    id: str
    user_id: str
    date: str  # ISO date string YYYY-MM-DD
    flow_level: FlowLevel
    is_period_day: bool = True
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CycleInfo(BaseModel):
    """A computed cycle (period start to next period start). Not stored — derived."""
    cycle_number: int
    start_date: str  # YYYY-MM-DD
    end_date: Optional[str] = None  # None if current/open cycle
    length_days: Optional[int] = None  # None if current/open cycle
    period_length_days: int


class CycleDayTag(BaseModel):
    """Cycle-day annotation for a symptom log. Computed, not stored."""
    cycle_day: int  # 1-indexed
    cycle_phase: str  # menstrual, follicular, ovulatory, luteal
    cycle_number: int
    cycle_start_date: str  # YYYY-MM-DD


class CycleSymptomCorrelation(BaseModel):
    """A detected pattern between symptoms and cycle phases/days."""
    symptom: str
    cycle_days: List[int]
    cycle_phase: str
    occurrences: int
    total_cycles: int
    confidence: str  # strong, moderate, weak
    description: str


class CyclePatternReport(BaseModel):
    """Full cycle-symptom correlation report."""
    user_id: str
    analysis_window_cycles: int
    average_cycle_length: Optional[float] = None
    average_period_length: Optional[float] = None
    correlations: List[CycleSymptomCorrelation]
    generated_at: datetime


class CycleDayLogRequest(BaseModel):
    """Request to log a cycle day."""
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    date: str  # YYYY-MM-DD
    flow_level: FlowLevel
    notes: Optional[str] = Field(default=None, max_length=MAX_TEXT_SIZE)


class PeriodStartRequest(BaseModel):
    """Simplified request: mark period start."""
    user_id: str = Field(..., max_length=MAX_USER_ID_SIZE)
    date: str  # YYYY-MM-DD
