export interface DemoPatient {
  name: string
  age: number
  gender: string
  conditions: string[]
  summary: string
}

export interface DemoMessage {
  speaker: 'patient' | 'agent'
  text: string
  audio: string | null
  question?: string
  isFollowup?: boolean
}

export interface DemoWatchdog {
  internal_clinical_rationale: string
  safe_patient_nudge: string
  clinician_facing_observation?: string
}

export interface DemoLogMetadata {
  symptoms: string[]
  actions_taken: string[]
  red_flags: string[]
  protocol: string | null
  clinician_note: string | null
  watchdog?: DemoWatchdog
  // Agent trace fields
  safety_mode?: string | null
  reason_code?: string | null
  tool_calls?: string[]
  question_rationale?: string | null
  vital_signs?: Record<string, number | string>
}

export interface DemoIntakeMetadata {
  question_id: string
  raw_answer: string
  parsed_items: string[]
  profile_field: string
}

export interface DemoLogEntry {
  day: number
  time: string
  phase: string
  messages: DemoMessage[]
  metadata: DemoLogMetadata
  intake?: DemoIntakeMetadata
}

export interface DemoDoctorPacket {
  hpi: string
  pertinent_positives: string[]
  pertinent_negatives: string[]
  timeline_bullets: string[]
  questions_for_clinician: string[]
  system_longitudinal_flags?: string[]
}

export interface DemoWatchdogCheckin {
  id: string
  checkin_type: string
  scheduled_for: string
  message: string
  context: Record<string, unknown>
}

export interface DemoWatchdogResults {
  pending_checkins: DemoWatchdogCheckin[]
  clinician_observations: string[]
}

export interface DemoData {
  patient: DemoPatient
  base_date?: string
  logs: DemoLogEntry[]
  doctor_packet: DemoDoctorPacket | null
  watchdog_results?: DemoWatchdogResults | null
}
