/**
 * Per-patient static data for Demo App Mode.
 *
 * Hardcoded because the demo JSON doesn't carry structured medication/profile
 * data in the shapes the normal app expects. Source of truth: patient JSON
 * files in backend/demo_data/patients/.
 */

import type {
  MedicationEntry,
  MedicationLogEntry,
  CycleDayLog,
  CycleInfo,
  CyclePatternReport,
} from './types'

// ---------------------------------------------------------------------------
// Helper: generate stable IDs and timestamps
// ---------------------------------------------------------------------------

function medId(patient: string, idx: number): string {
  return `demo_med_${patient}_${idx}`
}

function now(): string {
  return new Date().toISOString()
}

// ---------------------------------------------------------------------------
// Medications
// ---------------------------------------------------------------------------

function buildMeds(patient: string, meds: Partial<MedicationEntry>[]): MedicationEntry[] {
  return meds.map((m, i) => ({
    id: medId(patient, i),
    user_id: `demo_${patient}`,
    name: m.name!,
    dose: m.dose ?? null,
    frequency: m.frequency ?? null,
    reason: m.reason ?? null,
    start_date: null,
    end_date: null,
    notes: null,
    is_active: true,
    reminder_enabled: false,
    reminder_times: [],
    created_at: now(),
    updated_at: now(),
    ...m,
  }))
}

export const PATIENT_MEDICATIONS: Record<string, MedicationEntry[]> = {
  frank_russo: buildMeds('frank_russo', [
    { name: 'Tiotropium (Spiriva)', dose: '18mcg', frequency: 'Once daily', reason: 'COPD maintenance' },
    { name: 'Fluticasone-Salmeterol (Advair)', dose: '250/50', frequency: 'Twice daily (morning & evening)', reason: 'COPD maintenance' },
    { name: 'Albuterol (ProAir)', dose: '90mcg/puff', frequency: 'As needed (rescue)', reason: 'COPD rescue inhaler' },
    { name: 'Lisinopril', dose: '20mg', frequency: 'Once daily', reason: 'Hypertension' },
  ]),
  elena_martinez: buildMeds('elena_martinez', [
    { name: 'Metformin', dose: '500mg', frequency: 'Twice daily (with meals)', reason: 'Type 2 Diabetes' },
    { name: 'Glipizide', dose: '5mg', frequency: 'Once daily', reason: 'Type 2 Diabetes (recently added)' },
    { name: 'Lisinopril', dose: '10mg', frequency: 'Once daily', reason: 'Hypertension' },
  ]),
  sarah_chen: buildMeds('sarah_chen', [
    { name: 'Multivitamin', frequency: 'Once daily' },
    { name: 'Iron Supplement', dose: '65mg', frequency: 'Once daily', reason: 'Iron deficiency anemia (heavy periods)' },
    { name: 'Ibuprofen', dose: '400-600mg', frequency: 'As needed', reason: 'Menstrual pain' },
  ]),
}

// ---------------------------------------------------------------------------
// Medication history (sample dose logs)
// ---------------------------------------------------------------------------

export function buildMedicationHistory(
  patientId: string,
  medications: MedicationEntry[],
  logs: { recorded_at: string; actions: string[] }[],
): MedicationLogEntry[] {
  const history: MedicationLogEntry[] = []
  let idx = 0

  for (const log of logs) {
    for (const action of log.actions) {
      const actionLower = action.toLowerCase()
      const matchedMed = medications.find((m) => actionLower.includes(m.name.toLowerCase().split(' ')[0].toLowerCase()))
      history.push({
        id: `demo_medlog_${patientId}_${idx++}`,
        user_id: `demo_${patientId}`,
        medication_id: matchedMed?.id ?? null,
        medication_name: action,
        dose_taken: matchedMed?.dose ?? null,
        taken_at: log.recorded_at,
        notes: null,
        symptom_log_id: null,
      })
    }
  }

  return history
}

// ---------------------------------------------------------------------------
// Patient profile extras
// ---------------------------------------------------------------------------

export const PATIENT_ALLERGIES: Record<string, string[]> = {
  frank_russo: ['Penicillin'],
  elena_martinez: ['Sulfa drugs'],
  sarah_chen: [],
}

export const PATIENT_SOCIAL_HISTORY: Record<string, string[]> = {
  frank_russo: ['Retired construction foreman', 'Current smoker (reducing with patches)', 'Lives with wife Linda', 'Daughter Maria nearby'],
  elena_martinez: ['Retired teacher', 'Lives alone', 'Tracks blood sugar most mornings'],
  sarah_chen: ['Software engineer', 'Married', 'Active runner', 'Stopped combined OCP 6 months ago due to mood side effects'],
}

export const PATIENT_PATTERNS: Record<string, string[]> = {
  frank_russo: [
    'Current smoker, down to ~half pack/day from 2 packs/day historically',
    'Morning productive cough is chronic baseline',
    'Uses rescue inhaler 2-3 times per week at baseline',
    'Avoids stairs when possible',
  ],
  elena_martinez: [
    'Takes medications with breakfast and dinner',
    'Tracks fasting blood sugar most mornings',
    'Recently started Glipizide',
  ],
  sarah_chen: [
    'Stopped combined oral contraceptives 6 months ago due to mood-related side effects',
    'Takes ibuprofen frequently for menstrual pain',
    'Periods have become increasingly painful since stopping BC',
  ],
}

// ---------------------------------------------------------------------------
// Sarah's cycle data (from sarah_chen.json cycle_day_logs)
// ---------------------------------------------------------------------------

export function buildSarahCycleData(baseDate: Date): {
  cycleDays: CycleDayLog[]
  cycles: CycleInfo[]
  correlations: CyclePatternReport
} {
  const userId = 'demo_sarah_chen'

  // Raw cycle day logs from patient JSON
  const rawDays: { day: number; flow_level: 'heavy' | 'medium' | 'light' | 'spotting'; notes?: string }[] = [
    { day: 1, flow_level: 'heavy', notes: 'Period started' },
    { day: 2, flow_level: 'heavy' },
    { day: 3, flow_level: 'medium' },
    { day: 4, flow_level: 'light' },
    { day: 5, flow_level: 'spotting' },
    { day: 30, flow_level: 'heavy', notes: 'Period started — heavier than last month' },
    { day: 31, flow_level: 'heavy' },
    { day: 32, flow_level: 'heavy' },
    { day: 33, flow_level: 'medium' },
    { day: 34, flow_level: 'medium' },
    { day: 35, flow_level: 'light' },
    { day: 36, flow_level: 'spotting' },
    { day: 63, flow_level: 'heavy', notes: 'Period started — 33-day cycle this time' },
    { day: 64, flow_level: 'heavy' },
    { day: 65, flow_level: 'medium' },
    { day: 66, flow_level: 'light' },
    { day: 67, flow_level: 'spotting' },
  ]

  function dayToDate(day: number): string {
    const d = new Date(baseDate)
    d.setDate(baseDate.getDate() + day - 1)
    return d.toISOString().split('T')[0]
  }

  const cycleDays: CycleDayLog[] = rawDays.map((rd, idx) => ({
    id: `demo_cycle_day_${idx}`,
    user_id: userId,
    date: dayToDate(rd.day),
    flow_level: rd.flow_level,
    is_period_day: rd.flow_level !== 'spotting',
    notes: rd.notes ?? null,
    created_at: now(),
    updated_at: now(),
  }))

  const cycles: CycleInfo[] = [
    {
      cycle_number: 1,
      start_date: dayToDate(1),
      end_date: dayToDate(29),
      length_days: 29,
      period_length_days: 5,
    },
    {
      cycle_number: 2,
      start_date: dayToDate(30),
      end_date: dayToDate(62),
      length_days: 33,
      period_length_days: 7,
    },
  ]

  const correlations: CyclePatternReport = {
    user_id: userId,
    analysis_window_cycles: 2,
    average_cycle_length: 31,
    average_period_length: 6,
    correlations: [
      {
        symptom: 'pelvic pain',
        cycle_days: [24, 26, 27, 28, 29, 30, 32],
        cycle_phase: 'luteal/menstrual',
        occurrences: 7,
        total_cycles: 2,
        confidence: 'strong',
        description: 'Deep pelvic pain onset at RCD −5 (5 days before menstruation) in both cycles, persisting through menstruation. Consistent timing despite 29-day vs 33-day cycle lengths.',
      },
      {
        symptom: 'back pain',
        cycle_days: [2, 3, 26, 30],
        cycle_phase: 'menstrual/luteal',
        occurrences: 4,
        total_cycles: 2,
        confidence: 'strong',
        description: 'Back pain in menstrual phase (CD2-3) and late luteal phase (RCD −3) — both patterns consistent across 2 completed cycles.',
      },
      {
        symptom: 'bloating',
        cycle_days: [22, 24, 26, 29],
        cycle_phase: 'luteal',
        occurrences: 4,
        total_cycles: 2,
        confidence: 'moderate',
        description: 'Bloating onset at RCD −7 (7 days before menstruation) in both cycles. Consistent timing despite different cycle lengths.',
      },
      {
        symptom: 'fatigue',
        cycle_days: [22, 26, 27, 30],
        cycle_phase: 'luteal',
        occurrences: 4,
        total_cycles: 2,
        confidence: 'moderate',
        description: 'Fatigue onset at RCD −7 (7 days before menstruation) in both cycles.',
      },
      {
        symptom: 'nausea',
        cycle_days: [2, 3, 27, 32],
        cycle_phase: 'menstrual/luteal',
        occurrences: 4,
        total_cycles: 2,
        confidence: 'moderate',
        description: 'Nausea in menstrual phase (CD2-3) with emerging late luteal pattern.',
      },
    ],
    generated_at: now(),
  }

  return { cycleDays, cycles, correlations }
}
