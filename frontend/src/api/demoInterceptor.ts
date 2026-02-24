/**
 * Demo App Mode API Interceptor
 *
 * When activated, all fetchApi() calls return pre-transformed static data
 * instead of hitting the network. Existing components work unchanged.
 */

import type {
  LogEntry,
  MedicationEntry,
  MedicationLogEntry,
  DoctorPacket,
  TimelineReveal,
  UserProfile,
  CycleDayLog,
  CycleInfo,
  CyclePatternReport,
} from './types'

export interface DemoAppStore {
  userId: string
  logs: LogEntry[]
  medications: MedicationEntry[]
  medicationHistory: MedicationLogEntry[]
  profile: UserProfile
  doctorPacket: DoctorPacket
  timeline: TimelineReveal
  cycleDays: CycleDayLog[]
  cycles: CycleInfo[]
  cycleCorrelations: CyclePatternReport
}

let _active = false
let _store: DemoAppStore | null = null

export function isDemoAppMode(): boolean {
  return _active
}

export function getDemoStore(): DemoAppStore | null {
  return _store
}

export function setDemoAppMode(store: DemoAppStore): void {
  _active = true
  _store = store
}

export function clearDemoAppMode(): void {
  _active = false
  _store = null
}

/**
 * Route a fetchApi endpoint to the appropriate demo data.
 * Called from fetchApi() when demo app mode is active.
 */
export function resolveDemoEndpoint<T>(endpoint: string, options?: RequestInit): T {
  const store = _store!
  const method = options?.method?.toUpperCase() || 'GET'

  // Mutations → silent no-op
  if (['POST', 'PATCH', 'PUT', 'DELETE'].includes(method)) {
    // Summarize endpoints return pre-computed data even for POST
    if (endpoint.includes('/summarize/doctor-packet')) return store.doctorPacket as T
    if (endpoint.includes('/summarize/timeline')) return store.timeline as T
    // Auth session
    if (endpoint.includes('/auth/session')) return {} as T
    // Everything else → no-op success
    return { status: 'ok' } as T
  }

  // --- GET routes ---

  // Logs with date range filtering (CalendarView)
  if (endpoint.includes('/logs')) {
    const url = new URL(endpoint, 'http://localhost')
    const startDate = url.searchParams.get('start_date')
    const endDate = url.searchParams.get('end_date')

    if (startDate && endDate) {
      const start = new Date(startDate).getTime()
      const end = new Date(endDate).getTime()
      const filtered = store.logs.filter((log) => {
        const ts = new Date(log.recorded_at).getTime()
        return ts >= start && ts < end
      })
      return filtered as T
    }

    // Single log by ID: /logs/demo_log_5
    const idMatch = endpoint.match(/\/logs\/([^?/]+)/)
    if (idMatch) {
      const id = idMatch[1]
      return (store.logs.find((l) => l.id === id) || null) as T
    }

    // All logs
    return store.logs as T
  }

  // Profile
  if (endpoint.includes('/profile')) return store.profile as T

  // Medications
  if (endpoint.includes('/medications/log/history')) return store.medicationHistory as T
  if (endpoint.includes('/medications/reminders/pending')) return [] as T
  if (endpoint.includes('/medications')) return store.medications as T

  // Cycle
  if (endpoint.includes('/cycle/correlations')) return store.cycleCorrelations as T
  if (endpoint.includes('/cycle/cycles')) return store.cycles as T
  if (endpoint.includes('/cycle/days')) return store.cycleDays as T

  // Checkins
  if (endpoint.includes('/checkins/pending')) return [] as T

  // Ambient sessions
  if (endpoint.includes('/ambient/sessions')) return [] as T

  // Default
  return [] as T
}
