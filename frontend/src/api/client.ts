import type {
  EnhancedIngestResponse,
  LogEntry,
  DoctorPacket,
  TimelineReveal,
  VoiceIngestRequest,
  SummarizeRequest,
  MedicationEntry,
  MedicationLogEntry,
  MedicationCreateRequest,
  MedicationLogRequest,
  MedicationVoiceResponse,
  PendingMedicationReminder,
  ReminderActionRequest,
  ReminderActionResponse,
  AmbientSession,
  AmbientSessionResult,
  StartSessionRequest,
  StartSessionResponse,
  UploadChunkRequest,
  UploadChunkResponse,
  EndSessionRequest,
  EndSessionResponse,
  ScheduledCheckin,
  ImageIngestRequest,
  ImageIngestResponse,
  UserProfile,
  ProfileUpdateRequest,
  CycleDayLog,
  CycleDayLogRequest,
  PeriodStartRequest,
  CycleInfo,
  CyclePatternReport,
} from './types'
import { isDemoAppMode, resolveDemoEndpoint } from './demoInterceptor'

const API_BASE = '/api'

let _sessionEstablished = false

export type DegradedReason = {
  raw: string
  stage?: string
  fallback?: string
  reason?: string
  detail?: string
  client?: string
  provider?: string
  model?: string
  endpoint?: string
  region?: string
  project?: string
}

export type DegradedEventDetail = {
  endpoint: string
  reasons: string
  parsedReasons: DegradedReason[]
}

function parseOneDegradedReason(rawReason: string): DegradedReason {
  const raw = rawReason.trim()
  if (!raw) return { raw: '' }

  const structured: Record<string, string> = {}
  for (const token of raw.split('|')) {
    const idx = token.indexOf('=')
    if (idx <= 0) continue
    const key = token.slice(0, idx).trim()
    const value = token.slice(idx + 1).trim()
    if (key && value) structured[key] = value
  }

  if (Object.keys(structured).length > 0) {
    return {
      raw,
      stage: structured.stage,
      fallback: structured.fallback,
      reason: structured.reason,
      detail: structured.detail,
      client: structured.client,
      provider: structured.provider,
      model: structured.model,
      endpoint: structured.endpoint,
      region: structured.region,
      project: structured.project,
    }
  }

  if (raw.includes(':')) {
    const [fallback, reason] = raw.split(':', 2)
    return { raw, fallback, reason }
  }

  return { raw, reason: raw }
}

export function parseDegradedReasons(reasonsHeader: string): DegradedReason[] {
  return reasonsHeader
    .split(';')
    .map((entry) => parseOneDegradedReason(entry))
    .filter((entry) => entry.raw.length > 0)
}

/**
 * Establish a session cookie. The backend creates a session and sets an
 * HttpOnly cookie. All subsequent requests authenticate via that cookie.
 */
let _sessionError: string | null = null

export function getSessionError(): string | null {
  return _sessionError
}

export async function initSession(userId?: string): Promise<void> {
  if (isDemoAppMode()) return
  if (_sessionEstablished) return
  _sessionError = null
  try {
    const headers: Record<string, string> = {}
    if (userId) {
      headers['X-User-Id'] = userId
    }
    const res = await fetch(`${API_BASE}/auth/session`, {
      method: 'POST',
      credentials: 'include',
      headers,
    })
    if (res.ok) {
      _sessionEstablished = true
    } else {
      _sessionError = `Session creation failed (${res.status})`
    }
  } catch (err) {
    _sessionError = `Cannot connect to backend: ${err instanceof Error ? err.message : 'unknown error'}`
  }
}

function buildHeaders(userId?: string, extra?: HeadersInit): Headers {
  const headers = new Headers()
  headers.set('Content-Type', 'application/json')
  if (userId) {
    headers.set('X-User-Id', userId)
  }
  if (extra) {
    const extraHeaders = new Headers(extra)
    extraHeaders.forEach((value, key) => headers.set(key, value))
  }
  return headers
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit,
  userId?: string,
  retry401 = true,
): Promise<T> {
  // Demo app mode: return pre-loaded data instead of hitting the network
  if (isDemoAppMode()) return resolveDemoEndpoint<T>(endpoint, options)

  // Create an AbortController for timeout
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 60000) // 60 second timeout

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      headers: buildHeaders(userId, options?.headers),
      ...options,
      credentials: 'include',
      signal: controller.signal,
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      // If the backend restarts, cookie sessions become invalid (in-memory store).
      // Re-establish the session and retry once.
      if (response.status === 401 && retry401) {
        _sessionEstablished = false
        await initSession(userId).catch(() => {})
        return await fetchApi<T>(endpoint, options, userId, false)
      }
      let detail = `HTTP ${response.status}`
      const contentType = response.headers.get('content-type') || ''
      if (contentType.includes('application/json')) {
        const error = await response.json().catch(() => ({ detail: detail }))
        detail = error.detail || detail
      } else {
        const text = await response.text().catch(() => '')
        if (text && text.trim()) {
          detail = text.trim()
        }
      }
      throw new Error(detail)
    }

    const degraded = response.headers.get('X-Degraded-Mode') === 'true'
    const degradedReasons = response.headers.get('X-Degraded-Reasons')
    if (degraded && degradedReasons) {
      const detail: DegradedEventDetail = {
        endpoint,
        reasons: degradedReasons,
        parsedReasons: parseDegradedReasons(degradedReasons),
      }
      console.warn('API degraded mode', detail)
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('mysymptom:degraded', { detail }))
      }
    }

    return response.json()
  } catch (error) {
    clearTimeout(timeoutId)
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error('Request timed out - please try again')
      }
      throw error
    }
    throw new Error('An unexpected error occurred')
  }
}

// Ingest endpoints
export async function ingestVoice(request: VoiceIngestRequest): Promise<EnhancedIngestResponse> {
  return fetchApi<EnhancedIngestResponse>('/ingest/voice', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

// Log endpoints
export async function getLogs(userId: string, limit = 50): Promise<LogEntry[]> {
  return fetchApi<LogEntry[]>(`/logs?user_id=${encodeURIComponent(userId)}&limit=${limit}`, undefined, userId)
}

export async function getLogsByDateRange(
  userId: string,
  startDate: string,
  endDate: string,
  limit = 200
): Promise<LogEntry[]> {
  return fetchApi<LogEntry[]>(
    `/logs?user_id=${encodeURIComponent(userId)}&start_date=${encodeURIComponent(startDate)}&end_date=${encodeURIComponent(endDate)}&limit=${limit}`,
    undefined,
    userId
  )
}

export async function getLog(logId: string, userId: string): Promise<LogEntry> {
  return fetchApi<LogEntry>(`/logs/${logId}`, undefined, userId)
}

export async function deleteLog(
  logId: string,
  userId: string,
  permanent = false
): Promise<{ status: string; log_id: string }> {
  return fetchApi<{ status: string; log_id: string }>(`/logs/${logId}?permanent=${permanent}`, {
    method: 'DELETE',
  }, userId)
}

export async function submitFollowup(logId: string, answer: string, userId: string): Promise<LogEntry> {
  return fetchApi<LogEntry>(`/logs/${logId}/followup`, {
    method: 'POST',
    body: JSON.stringify({ answer }),
  }, userId)
}

// Summarize endpoints
export async function generateDoctorPacket(request: SummarizeRequest): Promise<DoctorPacket> {
  return fetchApi<DoctorPacket>('/summarize/doctor-packet', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function generateTimeline(request: SummarizeRequest): Promise<TimelineReveal> {
  return fetchApi<TimelineReveal>('/summarize/timeline', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

// Utility: Convert blob to base64
export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      const result = reader.result as string
      if (!result || !result.includes(',')) {
        reject(new Error('Invalid file format - could not encode as base64'))
        return
      }
      const base64 = result.split(',')[1]
      if (!base64) {
        reject(new Error('Failed to extract base64 data from file'))
        return
      }
      resolve(base64)
    }
    reader.onerror = (error) => {
      console.error('FileReader error:', error)
      reject(new Error('Failed to read file'))
    }
    reader.readAsDataURL(blob)
  })
}

// Alias for backward compatibility
export const audioToBase64 = blobToBase64

// Utility: Convert File to base64
export function fileToBase64(file: File): Promise<string> {
  return blobToBase64(file)
}

// Utility: Convert File to data URL (preserves MIME)
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      resolve(reader.result as string)
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

// Medication endpoints
export async function getMedications(userId: string, activeOnly = true): Promise<MedicationEntry[]> {
  return fetchApi<MedicationEntry[]>(
    `/medications?user_id=${encodeURIComponent(userId)}&active_only=${activeOnly}`,
    undefined,
    userId
  )
}

export async function extractMedicationFromVoice(
  userId: string,
  audio_b64: string
): Promise<MedicationVoiceResponse> {
  return fetchApi<MedicationVoiceResponse>('/medications/voice-extract', {
    method: 'POST',
    body: JSON.stringify({ user_id: userId, audio_b64 }),
  }, userId)
}

export async function createMedication(request: MedicationCreateRequest): Promise<MedicationEntry> {
  return fetchApi<MedicationEntry>('/medications', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function updateMedication(
  medId: string,
  updates: Partial<MedicationEntry>,
  userId: string
): Promise<MedicationEntry> {
  return fetchApi<MedicationEntry>(`/medications/${medId}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  }, userId)
}

export async function deactivateMedication(medId: string, userId: string): Promise<void> {
  return fetchApi(`/medications/${medId}`, {
    method: 'DELETE',
  }, userId)
}

export async function logMedication(request: MedicationLogRequest): Promise<MedicationLogEntry> {
  return fetchApi<MedicationLogEntry>('/medications/log', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function getMedicationHistory(
  userId: string,
  limit = 50,
  days?: number
): Promise<MedicationLogEntry[]> {
  let url = `/medications/log/history?user_id=${encodeURIComponent(userId)}&limit=${limit}`
  if (days !== undefined) {
    url += `&days=${days}`
  }
  return fetchApi<MedicationLogEntry[]>(url, undefined, userId)
}

export async function getPendingMedicationReminders(
  userId: string
): Promise<PendingMedicationReminder[]> {
  return fetchApi<PendingMedicationReminder[]>(
    `/medications/reminders/pending?user_id=${encodeURIComponent(userId)}`,
    undefined,
    userId
  )
}

export async function takeMedicationReminder(
  request: ReminderActionRequest
): Promise<ReminderActionResponse> {
  return fetchApi<ReminderActionResponse>('/medications/reminders/take', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function dismissMedicationReminder(
  request: ReminderActionRequest
): Promise<ReminderActionResponse> {
  return fetchApi<ReminderActionResponse>('/medications/reminders/dismiss', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function snoozeMedicationReminder(
  request: ReminderActionRequest
): Promise<ReminderActionResponse> {
  return fetchApi<ReminderActionResponse>('/medications/reminders/snooze', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

// Ambient monitoring endpoints
export async function startAmbientSession(request: StartSessionRequest): Promise<StartSessionResponse> {
  return fetchApi<StartSessionResponse>('/ambient/sessions/start', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function uploadAmbientChunk(request: UploadChunkRequest): Promise<UploadChunkResponse> {
  return fetchApi<UploadChunkResponse>('/ambient/sessions/upload', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function endAmbientSession(request: EndSessionRequest): Promise<EndSessionResponse> {
  return fetchApi<EndSessionResponse>('/ambient/sessions/end', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function getAmbientSessions(userId: string, limit = 20): Promise<AmbientSession[]> {
  return fetchApi<AmbientSession[]>(
    `/ambient/sessions?user_id=${encodeURIComponent(userId)}&limit=${limit}`,
    undefined,
    userId
  )
}

export async function getActiveAmbientSession(userId: string): Promise<AmbientSession | null> {
  return fetchApi<AmbientSession | null>(
    `/ambient/sessions/active?user_id=${encodeURIComponent(userId)}`,
    undefined,
    userId
  )
}

export async function cancelAmbientSession(sessionId: string, userId: string): Promise<void> {
  return fetchApi(`/ambient/sessions/${sessionId}/cancel?user_id=${encodeURIComponent(userId)}`, {
    method: 'POST',
  }, userId)
}

export async function getAmbientSessionResult(
  sessionId: string,
  userId: string
): Promise<AmbientSessionResult | null> {
  return fetchApi<AmbientSessionResult | null>(
    `/ambient/sessions/${sessionId}/result?user_id=${encodeURIComponent(userId)}`,
    undefined,
    userId
  )
}

// Check-in endpoints
export async function getPendingCheckins(userId: string): Promise<ScheduledCheckin[]> {
  return fetchApi<ScheduledCheckin[]>(
    `/checkins/pending?user_id=${encodeURIComponent(userId)}`,
    undefined,
    userId
  )
}

export async function respondToCheckin(
  checkinId: string,
  response: string,
  userId: string,
  responseAudioB64?: string | null,
): Promise<ScheduledCheckin> {
  const payload: Record<string, string> = {}
  const trimmed = response.trim()
  if (trimmed) {
    payload.response = trimmed
  }
  if (responseAudioB64) {
    payload.response_audio_b64 = responseAudioB64
  }
  return fetchApi<ScheduledCheckin>(`/checkins/${checkinId}/respond`, {
    method: 'POST',
    body: JSON.stringify(payload),
  }, userId)
}

export async function dismissCheckin(checkinId: string, userId: string): Promise<void> {
  return fetchApi(`/checkins/${checkinId}/dismiss`, {
    method: 'POST',
  }, userId)
}

export async function triggerCheckin(checkinId: string, userId: string): Promise<void> {
  return fetchApi(`/checkins/${checkinId}/trigger`, {
    method: 'POST',
  }, userId)
}

// Image analysis endpoints
export async function analyzeImage(request: ImageIngestRequest): Promise<ImageIngestResponse> {
  return fetchApi<ImageIngestResponse>('/ingest/image', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

// User profile endpoints (long-term memory)
export async function getUserProfile(userId: string): Promise<UserProfile | null> {
  return fetchApi<UserProfile | null>(
    `/profile?user_id=${encodeURIComponent(userId)}`,
    undefined,
    userId
  )
}

export async function updateUserProfile(request: ProfileUpdateRequest): Promise<UserProfile> {
  return fetchApi<UserProfile>('/profile', {
    method: 'PATCH',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function startOnboarding(userId: string): Promise<{ status: string; checkin_id?: string }> {
  return fetchApi<{ status: string; checkin_id?: string }>(
    `/profile/onboarding/start?user_id=${encodeURIComponent(userId)}`,
    { method: 'POST' },
    userId
  )
}

// === Cycle Tracking ===

export async function logCycleDay(request: CycleDayLogRequest): Promise<CycleDayLog> {
  return fetchApi<CycleDayLog>('/cycle/day', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function markPeriodStart(request: PeriodStartRequest): Promise<{ status: string; date: string }> {
  return fetchApi<{ status: string; date: string }>('/cycle/period-start', {
    method: 'POST',
    body: JSON.stringify(request),
  }, request.user_id)
}

export async function getCycleDays(userId: string, limit = 365): Promise<CycleDayLog[]> {
  return fetchApi<CycleDayLog[]>(
    `/cycle/days?user_id=${encodeURIComponent(userId)}&limit=${limit}`,
    undefined, userId
  )
}

export async function getCycles(userId: string): Promise<CycleInfo[]> {
  return fetchApi<CycleInfo[]>(
    `/cycle/cycles?user_id=${encodeURIComponent(userId)}`,
    undefined, userId
  )
}

export async function getCycleCorrelations(userId: string): Promise<CyclePatternReport> {
  return fetchApi<CyclePatternReport>(
    `/cycle/correlations?user_id=${encodeURIComponent(userId)}`,
    undefined, userId
  )
}
