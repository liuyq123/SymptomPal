/**
 * Transforms demo JSON (DemoData) into the API response shapes
 * expected by normal app components.
 */

import type { DemoData, DemoLogEntry, DemoMessage } from '../types/demoPlayer'
import type {
  LogEntry,
  DoctorPacket,
  TimelineReveal,
  TimelinePoint,
  UserProfile,
  SymptomEntity,
  ActionEntity,
  CycleDayLog,
  CycleInfo,
  CyclePatternReport,
} from './types'
import type { DemoAppStore } from './demoInterceptor'
import {
  PATIENT_MEDICATIONS,
  PATIENT_ALLERGIES,
  PATIENT_SOCIAL_HISTORY,
  PATIENT_PATTERNS,
  buildMedicationHistory,
  buildSarahCycleData,
} from './demoFixtures'

// ---------------------------------------------------------------------------
// Date anchoring: day numbers → real ISO timestamps
// ---------------------------------------------------------------------------

function computeBaseDate(logs: DemoLogEntry[], storedBaseDate?: string): Date {
  if (storedBaseDate) {
    const d = new Date(storedBaseDate + 'T00:00:00')
    if (!isNaN(d.getTime())) return d
  }
  const maxDay = Math.max(...logs.map((l) => l.day))
  const base = new Date()
  base.setDate(base.getDate() - maxDay)
  base.setHours(0, 0, 0, 0)
  return base
}

function dayTimeToISO(baseDate: Date, day: number, time: string): string {
  const d = new Date(baseDate)
  d.setDate(baseDate.getDate() + day - 1)
  const [h, m] = time.split(':').map(Number)
  d.setHours(h || 9, m || 0, 0, 0)
  return d.toISOString()
}

// ---------------------------------------------------------------------------
// Log transformation
// ---------------------------------------------------------------------------

function extractFromMessages(messages: DemoMessage[]): {
  transcript: string
  followupQuestion: string | null
  followupAnswer: string | null
} {
  const patientInitial = messages.find((m) => m.speaker === 'patient' && !m.isFollowup)
  const agentWithQuestion = messages.find((m) => m.speaker === 'agent' && m.question)
  const patientFollowup = messages.find((m) => m.speaker === 'patient' && m.isFollowup)

  return {
    transcript: patientInitial?.text || messages.find((m) => m.speaker === 'patient')?.text || '',
    followupQuestion: agentWithQuestion?.question ?? null,
    followupAnswer: patientFollowup?.text ?? null,
  }
}

function transformLog(
  entry: DemoLogEntry,
  baseDate: Date,
  userId: string,
  index: number,
): LogEntry {
  const { transcript, followupQuestion, followupAnswer } = extractFromMessages(entry.messages)
  const meta = entry.metadata

  const symptoms: SymptomEntity[] = meta.symptoms.map((s) => ({
    symptom: s,
    location: null,
    character: null,
    severity_1_10: null,
    onset_time_text: null,
    duration_text: null,
    triggers: [],
    relievers: [],
    associated_symptoms: [],
  }))

  const actions: ActionEntity[] = meta.actions_taken.map((a) => ({
    name: a,
    dose_text: null,
    effect_text: null,
  }))

  const recordedAt = dayTimeToISO(baseDate, entry.day, entry.time)

  return {
    id: `demo_log_${entry.day}_${index}`,
    user_id: userId,
    recorded_at: recordedAt,
    transcript,
    description: null,
    photo_b64: null,
    extracted: {
      transcript,
      symptoms,
      actions_taken: actions,
      missing_fields: [],
      red_flags: meta.red_flags || [],
    },
    image_analysis: null,
    contact_clinician_note: meta.clinician_note ?? null,
    contact_clinician_reason: null,
    followup_exchanges: followupQuestion
      ? [{ question: followupQuestion, answer: followupAnswer ?? undefined, agent_response: null, answered_at: followupAnswer ? recordedAt : null }]
      : [],
    followup_question: followupQuestion,
    followup_answer: followupAnswer,
    followup_answered_at: followupAnswer ? recordedAt : null,
    followup_response: null,
  }
}

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------

function transformProfile(demoData: DemoData, patientId: string, userId: string): UserProfile {
  const p = demoData.patient
  const ts = new Date().toISOString()
  const meds = PATIENT_MEDICATIONS[patientId] || []

  return {
    user_id: userId,
    conditions: p.conditions,
    allergies: PATIENT_ALLERGIES[patientId] || [],
    regular_medications: meds.map((m) => `${m.name}${m.dose ? ' ' + m.dose : ''}${m.frequency ? ' ' + m.frequency : ''}`),
    surgeries: [],
    family_history: [],
    social_history: PATIENT_SOCIAL_HISTORY[patientId] || [],
    patterns: PATIENT_PATTERNS[patientId] || [],
    health_summary: p.summary || null,
    created_at: ts,
    updated_at: ts,
    intake_completed: true,
    intake_questions_asked: 8,
    intake_answered_question_ids: [],
    intake_last_question_id: null,
    intake_started_at: ts,
    intake_completed_at: ts,
  }
}

// ---------------------------------------------------------------------------
// Doctor Packet (direct copy — structures match)
// ---------------------------------------------------------------------------

function transformDoctorPacket(demoData: DemoData): DoctorPacket {
  const dp = demoData.doctor_packet
  if (!dp) {
    return {
      hpi: 'No data available yet.',
      pertinent_positives: [],
      pertinent_negatives: [],
      timeline_bullets: [],
      questions_for_clinician: [],
    }
  }
  return {
    hpi: dp.hpi,
    pertinent_positives: dp.pertinent_positives,
    pertinent_negatives: dp.pertinent_negatives,
    timeline_bullets: dp.timeline_bullets,
    questions_for_clinician: dp.questions_for_clinician,
    system_longitudinal_flags: dp.system_longitudinal_flags,
  }
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

function transformTimeline(logs: LogEntry[]): TimelineReveal {
  // Build one bullet per log (mirrors backend _build_timeline_bullets)
  const points: TimelinePoint[] = logs.map((log) => {
    const parts: string[] = []
    for (const s of log.extracted.symptoms) {
      parts.push(s.symptom)
    }
    for (const a of log.extracted.actions_taken) {
      let detail = a.name
      if (a.dose_text) detail += ` ${a.dose_text}`
      parts.push(detail)
    }
    if (log.extracted.red_flags.length > 0) {
      parts.push(`RED FLAG: ${log.extracted.red_flags.join(', ')}`)
    }
    if (parts.length === 0) {
      parts.push(log.transcript.slice(0, 100) || 'entry recorded')
    }

    const date = new Date(log.recorded_at)
    const label = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })

    return {
      timestamp: log.recorded_at,
      label,
      details: parts.join('; '),
    }
  })

  // Reverse chronological (most recent first)
  points.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  return { story_points: points }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export function transformDemoData(demoData: DemoData, patientId: string): DemoAppStore {
  const userId = `demo_${patientId}`
  const baseDate = computeBaseDate(demoData.logs, demoData.base_date)

  // Transform logs
  const logs = demoData.logs.map((entry, i) => transformLog(entry, baseDate, userId, i))

  // Medications
  const medications = PATIENT_MEDICATIONS[patientId] || []
  const medicationHistory = buildMedicationHistory(
    patientId,
    medications,
    logs.map((l) => ({
      recorded_at: l.recorded_at,
      actions: l.extracted.actions_taken.map((a) => a.name),
    })),
  )

  // Profile
  const profile = transformProfile(demoData, patientId, userId)

  // Summaries
  const doctorPacket = transformDoctorPacket(demoData)
  const timeline = transformTimeline(logs)

  // Cycle data (Sarah only)
  let cycleDays: CycleDayLog[] = []
  let cycles: CycleInfo[] = []
  let cycleCorrelations: CyclePatternReport = {
    user_id: userId,
    analysis_window_cycles: 0,
    average_cycle_length: null,
    average_period_length: null,
    correlations: [],
    generated_at: new Date().toISOString(),
  }

  if (patientId === 'sarah_chen') {
    const cycleData = buildSarahCycleData(baseDate)
    cycleDays = cycleData.cycleDays
    cycles = cycleData.cycles
    cycleCorrelations = cycleData.correlations
  }

  return {
    userId,
    logs,
    medications,
    medicationHistory,
    profile,
    doctorPacket,
    timeline,
    cycleDays,
    cycles,
    cycleCorrelations,
  }
}
