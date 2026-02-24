// API Types matching backend Pydantic models

export interface SymptomEntity {
  symptom: string
  location?: string | null
  character?: string | null
  severity_1_10?: number | null
  onset_time_text?: string | null
  duration_text?: string | null
  triggers: string[]
  relievers: string[]
  associated_symptoms: string[]
}

export interface ActionEntity {
  name: string
  dose_text?: string | null
  effect_text?: string | null
}

export interface ExtractionResult {
  transcript: string
  symptoms: SymptomEntity[]
  actions_taken: ActionEntity[]
  missing_fields: string[]
  red_flags: string[]
}

export interface FollowupExchange {
  question: string
  answer?: string | null
  agent_response?: string | null
  answered_at?: string | null
}

export interface LogEntry {
  id: string
  user_id: string
  recorded_at: string
  transcript: string
  description?: string | null
  photo_b64?: string | null
  extracted: ExtractionResult
  image_analysis?: ImageAnalysisResult | null
  contact_clinician_note?: string | null
  contact_clinician_reason?: string | null
  followup_exchanges: FollowupExchange[]
  // Computed backward-compat fields (first exchange)
  followup_question?: string | null
  followup_answer?: string | null
  followup_answered_at?: string | null
  followup_response?: string | null
}

export interface IngestResponse {
  log: LogEntry
  followup_question?: string | null
}

export interface DoctorPacket {
  hpi: string
  pertinent_positives: string[]
  pertinent_negatives: string[]
  timeline_bullets: string[]
  questions_for_clinician: string[]
  system_longitudinal_flags?: string[]
}

export interface TimelinePoint {
  timestamp: string
  label: string
  details: string
}

export interface TimelineReveal {
  story_points: TimelinePoint[]
}

export interface VoiceIngestRequest {
  user_id: string
  audio_b64?: string | null
  recorded_at: string
  description_text?: string | null
  description_audio_b64?: string | null
  photo_b64?: string | null
}

export interface SummarizeRequest {
  user_id: string
  days: number
}

// Medication types
export interface MedicationEntry {
  id: string
  user_id: string
  name: string
  dose?: string | null
  frequency?: string | null
  reason?: string | null
  start_date?: string | null
  end_date?: string | null
  notes?: string | null
  is_active: boolean
  reminder_enabled?: boolean
  reminder_times?: string[]
  created_at: string
  updated_at: string
}

export interface MedicationLogEntry {
  id: string
  user_id: string
  medication_id?: string | null
  medication_name: string
  dose_taken?: string | null
  taken_at: string
  notes?: string | null
  symptom_log_id?: string | null
}

export interface MedicationCreateRequest {
  user_id: string
  name: string
  dose?: string | null
  frequency?: string | null
  reason?: string | null
  start_date?: string | null
  notes?: string | null
  reminder_enabled?: boolean
  reminder_times?: string[]
}

export interface MedicationLogRequest {
  user_id: string
  medication_id?: string | null
  medication_name?: string | null
  dose_taken?: string | null
  taken_at?: string | null
  notes?: string | null
  symptom_log_id?: string | null
}

export interface MedicationVoiceResponse {
  transcript: string
  medications: ActionEntity[]
  frequency?: string | null
  reason?: string | null
}

export interface PendingMedicationReminder {
  medication_id: string
  medication_name: string
  dose?: string | null
  scheduled_time: string
  due_at: string
  is_overdue: boolean
}

export interface ReminderActionRequest {
  user_id: string
  medication_id: string
  due_at: string
  snooze_minutes?: number | null
}

export interface ReminderActionResponse {
  status: string
  message?: string
  medication_log_id?: string
  snoozed_until?: string
}

// === Ambient Monitoring Types ===

export type SessionType = 'sleep' | 'cough_monitor'
export type SessionStatus = 'active' | 'processing' | 'completed' | 'cancelled'
export type AmbientEventType =
  | 'cough'
  | 'breathing_normal'
  | 'breathing_irregular'
  | 'breathing_apnea'
  | 'snoring'
  | 'voice_stress'
  | 'voice_fatigue'
  | 'voice_congestion'
  | 'silence'
  | 'noise'

export interface AmbientSession {
  id: string
  user_id: string
  session_type: SessionType
  status: SessionStatus
  label?: string | null
  started_at: string
  ended_at?: string | null
  chunk_count: number
  total_duration_seconds: number
  created_at: string
  updated_at: string
}

export interface AmbientEvent {
  id: string
  session_id: string
  user_id: string
  event_type: AmbientEventType
  timestamp: string
  duration_seconds?: number | null
  confidence: number
  metadata?: Record<string, unknown> | null
  chunk_index: number
}

export interface AmbientChunk {
  id: string
  session_id: string
  user_id: string
  chunk_index: number
  duration_seconds: number
  uploaded_at: string
  processed: boolean
  events_detected: number
}

// Ambient Metrics
export interface CoughMetrics {
  total_coughs: number
  coughs_per_hour: number
  peak_cough_period?: string | null
  cough_intensity_avg?: number | null
  // Classification results
  dominant_condition?: string | null  // COVID-19, healthy, infection
  condition_confidence?: number | null
  condition_probabilities?: Record<string, number> | null
  dominant_cough_type?: string | null  // dry, wet
  cough_type_confidence?: number | null
  dominant_severity?: string | null  // mild, moderate, severe
  severity_confidence?: number | null
}

export interface SleepQualityMetrics {
  total_sleep_duration_minutes: number
  breathing_regularity_score: number
  apnea_events: number
  snoring_minutes: number
  restlessness_score: number
  quality_rating: 'poor' | 'fair' | 'good' | 'excellent'
}

export interface VoiceBiomarkers {
  stress_level: number
  fatigue_level: number
  congestion_detected: boolean
  voice_clarity_score: number
}

export interface AmbientSessionResult {
  session_id: string
  session_type: SessionType
  duration_minutes: number
  cough_metrics?: CoughMetrics | null
  sleep_quality?: SleepQualityMetrics | null
  voice_biomarkers?: VoiceBiomarkers | null
  events_timeline: AmbientEvent[]
  summary: string
}

// Ambient API Request/Response
export interface StartSessionRequest {
  user_id: string
  session_type: SessionType
  label?: string | null
}

export interface StartSessionResponse {
  session: AmbientSession
  upload_interval_seconds: number
}

export interface UploadChunkRequest {
  session_id: string
  user_id: string
  chunk_index: number
  audio_b64: string
  duration_seconds: number
}

export interface UploadChunkResponse {
  chunk_id: string
  events_detected: AmbientEvent[]
}

export interface EndSessionRequest {
  session_id: string
  user_id: string
}

export interface EndSessionResponse {
  session: AmbientSession
  result: AmbientSessionResult
}

// === Proactive Agent Types ===

export type CheckinType =
  | 'medication_followup'
  | 'symptom_progression'
  | 'scheduled_reminder'
  | 'profile_intake'

export interface ScheduledCheckin {
  id: string
  user_id: string
  checkin_type: CheckinType
  scheduled_for: string
  message: string
  context: Record<string, unknown>
  created_at: string
  triggered: boolean
  dismissed: boolean
  response?: string | null
  responded_at?: string | null
}

export interface AgentResponse {
  acknowledgment: string
  immediate_question?: string | null
  scheduled_checkin?: ScheduledCheckin | null
  protocol_id?: string | null
  reason_code?: string | null
  safety_mode?: string | null
}

export interface EnhancedIngestResponse {
  log: LogEntry
  agent_response: AgentResponse
  followup_question?: string | null  // Backward compatibility
  image_analysis?: ImageAnalysisResult | null
  degraded_mode?: boolean
  warnings?: string[]
}

// === Medical Image Analysis Types ===

export interface SkinLesionDescription {
  lesion_type: string
  color: string
  size_estimate: string
  texture: string
  predicted_condition?: string | null
  condition_confidence?: number | null
  confidence_scores: Record<string, number>
}

export interface ImageAnalysisResult {
  clinical_description: string
  confidence: number
  lesion_detected: boolean
  skin_lesion?: SkinLesionDescription | null
  raw_classifications?: Record<string, unknown> | null
}

export interface ImageIngestRequest {
  user_id: string
  image_b64: string
  context?: string | null
  recorded_at?: string | null
}

export interface ImageIngestResponse {
  analysis: ImageAnalysisResult
  log_id?: string | null
  transcript_addition: string
}

// === User Profile (Long-Term Memory) ===

export interface UserProfile {
  user_id: string
  conditions: string[]      // e.g., ["Asthma", "Recurrent Migraines"]
  allergies: string[]       // e.g., ["Penicillin"]
  regular_medications: string[]  // e.g., ["Rescue inhaler"]
  surgeries: string[]
  family_history: string[]
  social_history: string[]
  patterns: string[]        // e.g., ["Migraines triggered by stress"]
  health_summary?: string | null
  created_at: string
  updated_at: string
  intake_completed: boolean
  intake_questions_asked: number
  intake_answered_question_ids: string[]
  intake_last_question_id?: string | null
  intake_started_at?: string | null
  intake_completed_at?: string | null
}

export interface ProfileUpdateRequest {
  user_id: string
  add_conditions?: string[]
  remove_conditions?: string[]
  add_allergies?: string[]
  remove_allergies?: string[]
  add_regular_medications?: string[]
  remove_regular_medications?: string[]
  add_surgeries?: string[]
  remove_surgeries?: string[]
  add_family_history?: string[]
  remove_family_history?: string[]
  add_social_history?: string[]
  remove_social_history?: string[]
  add_patterns?: string[]
  remove_patterns?: string[]
  health_summary?: string | null
}

// === Menstrual Cycle Tracking ===

export type FlowLevel = 'none' | 'spotting' | 'light' | 'medium' | 'heavy'

export interface CycleDayLog {
  id: string
  user_id: string
  date: string  // YYYY-MM-DD
  flow_level: FlowLevel
  is_period_day: boolean
  notes?: string | null
  created_at: string
  updated_at: string
}

export interface CycleInfo {
  cycle_number: number
  start_date: string
  end_date?: string | null
  length_days?: number | null
  period_length_days: number
}

export interface CycleSymptomCorrelation {
  symptom: string
  cycle_days: number[]
  cycle_phase: string
  occurrences: number
  total_cycles: number
  confidence: 'strong' | 'moderate' | 'weak'
  description: string
}

export interface CyclePatternReport {
  user_id: string
  analysis_window_cycles: number
  average_cycle_length?: number | null
  average_period_length?: number | null
  correlations: CycleSymptomCorrelation[]
  generated_at: string
}

export interface CycleDayLogRequest {
  user_id: string
  date: string
  flow_level: FlowLevel
  notes?: string | null
}

export interface PeriodStartRequest {
  user_id: string
  date: string
}
