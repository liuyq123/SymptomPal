import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import ErrorBoundary from './components/ErrorBoundary'
import SafetyBanner from './components/SafetyBanner'
import SymptomInput from './components/SymptomInput'
import LogPreview from './components/LogPreview'
import FollowUpChat from './components/FollowUpChat'
import TimelineReveal from './components/TimelineReveal'
import DoctorPacket from './components/DoctorPacket'
import MedicationInput from './components/MedicationInput'
import MedicationList from './components/MedicationList'
import MedicationReminders from './components/MedicationReminders'
import AmbientMonitor from './components/AmbientMonitor'
import CheckinNotification from './components/CheckinNotification'
import CalendarView from './components/CalendarView'
import ProfilePanel from './components/ProfilePanel'
import OnboardingScreen from './components/OnboardingScreen'
import CycleTracker from './components/CycleTracker'
import DemoWelcome from './components/demo/DemoWelcome'
import ScenarioSelector from './components/demo/ScenarioSelector'
import FeatureTour from './components/demo/FeatureTour'
import DemoTimeline from './components/demo/DemoTimeline'
import DemoPlayer from './components/demo/DemoPlayer'
import DemoApp from './components/demo/DemoApp'
import {
  getLogs,
  generateDoctorPacket,
  generateTimeline,
  getMedications,
  getMedicationHistory,
  deleteLog,
  initSession,
  getSessionError,
  getUserProfile,
  parseDegradedReasons,
} from './api/client'
import type { DegradedEventDetail, DegradedReason } from './api/client'
import type {
  LogEntry,
  EnhancedIngestResponse,
  AgentResponse,
  DoctorPacket as DoctorPacketType,
  TimelineReveal as TimelineRevealType,
} from './api/types'
import type { Scenario, DemoStage } from './types/demo'

const configuredUserId = import.meta.env.VITE_USER_ID || 'demo_user'
const allowUserOverride = import.meta.env.VITE_ALLOW_USER_OVERRIDE === 'true'
const queryUserId = new URLSearchParams(window.location.search).get('userId')
const USER_ID = allowUserOverride && queryUserId ? queryUserId : configuredUserId

type Tab = 'record' | 'monitor' | 'meds' | 'cycle' | 'profile' | 'timeline' | 'doctor'

function formatDegradedReason(reason: DegradedReason): string {
  const fields: string[] = []
  if (reason.stage) fields.push(`stage=${reason.stage}`)
  if (reason.client) fields.push(`client=${reason.client}`)
  if (reason.model) fields.push(`model=${reason.model}`)
  if (reason.endpoint) fields.push(`endpoint=${reason.endpoint}`)
  if (reason.region) fields.push(`region=${reason.region}`)
  if (reason.reason) fields.push(`reason=${reason.reason}`)
  if (reason.detail) fields.push(`detail=${reason.detail}`)
  if (fields.length === 0) return reason.raw || 'unknown reason'
  return fields.join(', ')
}

export default function App() {
  const queryClient = useQueryClient()

  // Demo mode state
  const [demoMode, setDemoMode] = useState<boolean>(
    new URLSearchParams(window.location.search).get('demo') !== 'false'
  )
  const [demoStage, setDemoStage] = useState<DemoStage>('welcome')
  const [selectedScenario, setSelectedScenario] = useState<Scenario | null>(null)

  // Session state
  const [sessionError, setSessionError] = useState<string | null>(null)

  // Establish session cookie on mount, bound to the configured user ID.
  // Skip in demo mode — the replay flow is fully static and doesn't need the backend.
  useEffect(() => {
    if (demoMode) return
    initSession(USER_ID).then(() => {
      const err = getSessionError()
      if (err) setSessionError(err)
    })
  }, [demoMode])

  // Profile query — used to gate onboarding before main app
  const profileQuery = useQuery({
    queryKey: ['profile', USER_ID],
    queryFn: () => getUserProfile(USER_ID),
    enabled: !demoMode,
  })

  // Regular app state
  const [activeTab, setActiveTab] = useState<Tab>('record')
  const [latestLog, setLatestLog] = useState<LogEntry | null>(null)
  const [agentResponse, setAgentResponse] = useState<AgentResponse | null>(null)
  const [doctorPacket, setDoctorPacket] = useState<DoctorPacketType | null>(null)
  const [timeline, setTimeline] = useState<TimelineRevealType | null>(null)
  const [ingestWarnings, setIngestWarnings] = useState<string[]>([])
  const [degradedNotice, setDegradedNotice] = useState<string | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<{ logId: string; isPermanent: boolean } | null>(null)

  useEffect(() => {
    const onDegraded = (event: Event) => {
      const custom = event as CustomEvent<DegradedEventDetail>
      const endpoint = custom.detail?.endpoint || 'unknown endpoint'
      const reasons = custom.detail?.reasons || 'unknown reason'
      const parsed = custom.detail?.parsedReasons?.length
        ? custom.detail.parsedReasons
        : parseDegradedReasons(reasons)
      const prettyReasons = parsed.length > 0
        ? parsed.map((reason) => formatDegradedReason(reason)).join(' ; ')
        : reasons
      setDegradedNotice(`Fallback mode active for ${endpoint}: ${prettyReasons}`)
    }
    window.addEventListener('mysymptom:degraded', onDegraded)
    return () => window.removeEventListener('mysymptom:degraded', onDegraded)
  }, [])

  // Fetch logs
  const logsQuery = useQuery({
    queryKey: ['logs', USER_ID],
    queryFn: () => getLogs(USER_ID),
  })

  // Fetch medications
  const medicationsQuery = useQuery({
    queryKey: ['medications', USER_ID],
    queryFn: () => getMedications(USER_ID),
  })

  // Fetch medication history
  const medHistoryQuery = useQuery({
    queryKey: ['medication-history', USER_ID],
    queryFn: () => getMedicationHistory(USER_ID, 50, 7),
  })

  // Shared mutation error state
  const [mutationError, setMutationError] = useState<string | null>(null)
  const handleMutationError = (err: unknown) => {
    const msg = err instanceof Error ? err.message : 'An unexpected error occurred'
    setMutationError(msg)
    setTimeout(() => setMutationError(null), 8000) // Auto-dismiss after 8s
  }

  // Generate summaries
  const doctorPacketMutation = useMutation({
    mutationFn: () => generateDoctorPacket({ user_id: USER_ID, days: 7 }),
    onSuccess: setDoctorPacket,
    onError: handleMutationError,
  })

  const timelineMutation = useMutation({
    mutationFn: () => generateTimeline({ user_id: USER_ID, days: 7 }),
    onSuccess: setTimeline,
    onError: handleMutationError,
  })

  const deleteMutation = useMutation({
    mutationFn: ({ logId, permanent }: { logId: string; permanent: boolean }) =>
      deleteLog(logId, USER_ID, permanent),
    onSuccess: () => {
      // Reset generated artifacts so UI reflects latest dataset after deletion.
      setDoctorPacket(null)
      setTimeline(null)
      setIngestWarnings([])
      setAgentResponse(null)
      // Clear latest log if it was deleted
      if (latestLog && latestLog.id === deleteConfirm?.logId) {
        setLatestLog(null)
      }
      setDeleteConfirm(null)
      // Invalidate and refetch logs list
      queryClient.invalidateQueries({ queryKey: ['logs', USER_ID] })
    },
    onError: handleMutationError,
  })

  // Set latest log when logs are fetched
  useEffect(() => {
    if (logsQuery.data && logsQuery.data.length > 0 && !latestLog) {
      setLatestLog(logsQuery.data[0])
    }
    if (logsQuery.data && logsQuery.data.length === 0) {
      setLatestLog(null)
    }
  }, [logsQuery.data, latestLog])

  const handleRecordingComplete = (response: EnhancedIngestResponse) => {
    setLatestLog(response.log)
    setAgentResponse(response.agent_response)
    setIngestWarnings(response.warnings || [])
    // Reset summaries when new log is added
    setDoctorPacket(null)
    setTimeline(null)
  }

  const handleFollowupAnswered = (updatedLog: LogEntry) => {
    setLatestLog(updatedLog)
    setAgentResponse(null) // Clear agent response after answering
  }

  const handleDeleteLog = (logId: string, isPermanent = false) => {
    setDeleteConfirm({ logId, isPermanent })
  }

  const confirmDelete = () => {
    if (deleteConfirm) {
      deleteMutation.mutate({ logId: deleteConfirm.logId, permanent: deleteConfirm.isPermanent })
    }
  }

  const handleTabChange = (tab: Tab) => {
    setActiveTab(tab)
    // Generate summaries on demand
    if (tab === 'doctor' && !doctorPacket && !doctorPacketMutation.isPending) {
      doctorPacketMutation.mutate()
    }
    if (tab === 'timeline' && !timeline && !timelineMutation.isPending) {
      timelineMutation.mutate()
    }
  }

  const hasRedFlags = latestLog?.extracted.red_flags && latestLog.extracted.red_flags.length > 0
  const totalLogs = logsQuery.data?.length ?? 0
  const activeMeds = (medicationsQuery.data || []).filter((med) => med.is_active).length
  const hasWarnings = ingestWarnings.length > 0 || Boolean(degradedNotice)
  const formattedIngestWarnings = ingestWarnings
    .flatMap((warning) => parseDegradedReasons(warning))
    .map((reason) => formatDegradedReason(reason))
  const displayIngestWarnings = formattedIngestWarnings.length > 0 ? formattedIngestWarnings : ingestWarnings

  return (
    <ErrorBoundary>
    <div className="app-shell min-h-screen">
      {sessionError && (
        <div role="alert" aria-live="assertive" className="mx-auto mt-3 w-[min(980px,92vw)] rounded-xl border border-red-300 bg-red-50/90 px-4 py-3 text-sm text-red-700 shadow-sm">
          {sessionError} - API calls may fail. Check that the backend is running.
        </div>
      )}
      {mutationError && (
        <div role="alert" aria-live="polite" className="mx-auto mt-3 w-[min(980px,92vw)] rounded-xl border border-yellow-300 bg-yellow-50/95 px-4 py-3 text-sm text-yellow-800 shadow-sm">
          Error: {mutationError}
        </div>
      )}
      {demoMode ? (
        // Demo flow
        <>
          {demoStage === 'welcome' && (
            <DemoWelcome onNext={() => setDemoStage('scenario')} />
          )}
          {demoStage === 'scenario' && (
            <ScenarioSelector
              onSelect={(scenario) => {
                setSelectedScenario(scenario)
                // If scenario has pre-recorded demo data, go straight to replay
                if (scenario.demoDataFile) {
                  setDemoStage('replay')
                } else {
                  setDemoStage('tour')
                }
              }}
              onExplore={(scenario) => {
                setSelectedScenario(scenario)
                setDemoStage('explore')
              }}
              onBack={() => setDemoStage('welcome')}
            />
          )}
          {demoStage === 'replay' && selectedScenario && (
            <DemoPlayer
              scenario={selectedScenario}
              onBack={() => {
                setDemoStage('scenario')
                setSelectedScenario(null)
              }}
            />
          )}
          {demoStage === 'explore' && selectedScenario && (
            <DemoApp
              scenario={selectedScenario}
              onBack={() => {
                setDemoStage('scenario')
                setSelectedScenario(null)
              }}
            />
          )}
          {demoStage === 'tour' && selectedScenario && (
            <FeatureTour
              scenario={selectedScenario}
              onStart={() => setDemoStage('live')}
              onBack={() => setDemoStage('scenario')}
            />
          )}
          {demoStage === 'live' && selectedScenario && (
            <DemoTimeline
              scenario={selectedScenario}
              onBack={() => setDemoStage('tour')}
              onExit={() => {
                setDemoMode(false)
                setDemoStage('welcome')
                setSelectedScenario(null)
              }}
            />
          )}
        </>
      ) : profileQuery.data && !profileQuery.data.intake_completed ? (
        // Onboarding flow — profile intake not yet completed
        <OnboardingScreen
          userId={USER_ID}
          onComplete={() => profileQuery.refetch()}
        />
      ) : (
        // Normal app flow
        <>
          {/* Header */}
          <header className="sticky top-0 z-40 border-b border-slate-200/60 bg-white/72 backdrop-blur-xl">
            <div className="mx-auto flex w-[min(1100px,92vw)] items-center justify-between px-1 py-4">
              <div className="flex items-center gap-3">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-cyan-600 to-indigo-700 text-sm font-semibold text-white shadow-lg">
                  SP
                </div>
                <div>
                  <h1 className="text-xl font-bold text-slate-900">SymptomPal</h1>
                  <p className="text-sm text-slate-600">Voice-first health journaling for faster clinical handoffs</p>
                </div>
              </div>
              <button
                onClick={() => { window.location.href = window.location.pathname + '?demo=true' }}
                className="rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-700 px-4 py-2 text-sm font-semibold text-white shadow-lg transition-transform hover:scale-[1.02]"
              >
                Replay Demo
              </button>
            </div>
          </header>

      {/* Main content */}
      <main className="mx-auto w-[min(1100px,92vw)] space-y-4 py-6">
        <section className="hero-strip float-in rounded-2xl px-5 py-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-cyan-700">Hackathon Build</p>
              <h2 className="mt-1 text-2xl font-bold text-slate-900">Capture symptoms, surface trends, prep the doctor packet</h2>
              <p className="mt-2 max-w-2xl text-sm text-slate-700">
                Keep the story tight for judges: one voice log, one safety-aware follow-up, one clinician-ready summary.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="status-chip rounded-full px-3 py-1 text-xs font-semibold">{totalLogs} total logs</span>
              <span className="status-chip rounded-full px-3 py-1 text-xs font-semibold">{activeMeds} active meds</span>
              <span className="status-chip rounded-full px-3 py-1 text-xs font-semibold">{hasWarnings ? 'fallback active' : 'full pipeline'}</span>
            </div>
          </div>
        </section>

        {/* Safety banner - always visible */}
        <SafetyBanner hasRedFlags={hasRedFlags} />
        {degradedNotice && (
          <div className="glass-panel rounded-xl border-amber-300 bg-amber-50/85 px-4 py-3 text-sm text-amber-900">
            <div className="flex items-start justify-between gap-3">
              <span>{degradedNotice}</span>
              <button
                onClick={() => setDegradedNotice(null)}
                className="text-amber-700 hover:text-amber-900"
                aria-label="Dismiss fallback notice"
              >
                x
              </button>
            </div>
          </div>
        )}

        {/* Tab navigation */}
        <nav role="tablist" aria-label="App sections" className="tab-rail flex gap-1 overflow-x-auto rounded-2xl p-1.5">
          {[
            { id: 'record', label: 'Record' },
            { id: 'monitor', label: 'Monitor' },
            { id: 'meds', label: 'Meds' },
            { id: 'cycle', label: 'Cycle' },
            { id: 'profile', label: 'Profile' },
            { id: 'timeline', label: 'Timeline' },
            { id: 'doctor', label: 'Doctor' },
          ].map(({ id, label }) => (
            <button
              key={id}
              role="tab"
              aria-selected={activeTab === id}
              onClick={() => handleTabChange(id as Tab)}
              className={`tab-pill flex-1 whitespace-nowrap rounded-xl px-4 py-2.5 text-sm font-semibold ${
                activeTab === id
                  ? 'tab-pill-active'
                  : ''
              }`}
            >
              {label}
            </button>
          ))}
        </nav>

        {/* Tab content */}
        {activeTab === 'record' && (
          <div className="glass-panel-strong space-y-4 rounded-2xl p-4 sm:p-5">
            {ingestWarnings.length > 0 && (
              <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                Limited model mode used for this log: {displayIngestWarnings.join(' ; ')}
              </div>
            )}
            <SymptomInput userId={USER_ID} onSubmitComplete={handleRecordingComplete} />

            {latestLog && (
              <>
                <LogPreview log={latestLog} onDelete={handleDeleteLog} />
                <FollowUpChat
                  log={latestLog}
                  agentResponse={agentResponse}
                  onAnswerSubmitted={handleFollowupAnswered}
                />
              </>
            )}

            {!latestLog && logsQuery.isLoading && (
              <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">Loading...</div>
            )}

            {!latestLog && !logsQuery.isLoading && (
              <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">
                Record your first symptom log to get started.
              </div>
            )}
          </div>
        )}

        {activeTab === 'monitor' && (
          <div className="glass-panel-strong rounded-2xl p-4 sm:p-5">
            <AmbientMonitor userId={USER_ID} />
          </div>
        )}

        {activeTab === 'meds' && (
          <div className="glass-panel-strong space-y-4 rounded-2xl p-4 sm:p-5">
            <MedicationReminders userId={USER_ID} />
            <MedicationInput
              userId={USER_ID}
              medications={medicationsQuery.data || []}
            />
            <MedicationList
              userId={USER_ID}
              medications={medicationsQuery.data || []}
              history={medHistoryQuery.data || []}
              isLoading={medicationsQuery.isLoading || medHistoryQuery.isLoading}
            />
          </div>
        )}

        {activeTab === 'cycle' && (
          <div className="glass-panel-strong rounded-2xl p-4 sm:p-5">
            <CycleTracker userId={USER_ID} />
          </div>
        )}

        {activeTab === 'profile' && (
          <div className="glass-panel-strong rounded-2xl p-4 sm:p-5">
            <ProfilePanel userId={USER_ID} />
          </div>
        )}

        {activeTab === 'timeline' && (
          <div className="glass-panel-strong space-y-4 rounded-2xl p-4 sm:p-5">
            {timelineMutation.isPending && (
              <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">Generating timeline...</div>
            )}
            {timelineMutation.isError && (
              <div className="rounded-xl border border-red-200 bg-red-50 py-8 text-center text-red-500">
                Error: {(timelineMutation.error as Error).message}
              </div>
            )}
            {timeline && <TimelineReveal timeline={timeline} />}
            {!timeline && !timelineMutation.isPending && !timelineMutation.isError && (
              <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">
                Record some symptoms to see your timeline.
              </div>
            )}
          </div>
        )}

        {activeTab === 'doctor' && (
          <div className="glass-panel-strong space-y-4 rounded-2xl p-4 sm:p-5">
            {doctorPacketMutation.isPending && (
              <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">Generating doctor packet...</div>
            )}
            {doctorPacketMutation.isError && (
              <div className="rounded-xl border border-red-200 bg-red-50 py-8 text-center text-red-500">
                Error: {(doctorPacketMutation.error as Error).message}
              </div>
            )}
            {doctorPacket && <DoctorPacket packet={doctorPacket} />}
            {!doctorPacket && !doctorPacketMutation.isPending && !doctorPacketMutation.isError && (
              <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">
                Record some symptoms to generate your pre-visit packet.
              </div>
            )}
          </div>
        )}

        {/* Calendar log history (shown on Record tab) */}
        {activeTab === 'record' && (
          <div className="glass-panel rounded-2xl p-4 sm:p-5">
            <CalendarView
              userId={USER_ID}
              onDeleteLog={handleDeleteLog}
            />
          </div>
        )}
      </main>

          {/* Footer */}
          <footer className="mx-auto w-[min(1100px,92vw)] px-4 py-8 text-center text-xs text-slate-500">
            SymptomPal — For demonstration purposes only. Not medical advice.
          </footer>

          {/* Check-in notification (floats bottom-right) */}
          <CheckinNotification userId={USER_ID} />

          {/* Delete confirmation dialog */}
          {deleteConfirm && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-labelledby="delete-dialog-title">
              <div className="glass-panel-strong w-full max-w-sm rounded-2xl p-6">
                <h3 id="delete-dialog-title" className="mb-2 text-lg font-semibold text-slate-800">Delete Log Entry?</h3>
                <p className="mb-4 text-slate-600">
                  {deleteConfirm.isPermanent
                    ? 'This will permanently delete the log entry. This action cannot be undone.'
                    : 'This will move the log entry to trash. You can recover it later if needed.'}
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => setDeleteConfirm(null)}
                    disabled={deleteMutation.isPending}
                    className="flex-1 rounded-lg bg-slate-200 px-4 py-2 text-slate-800 transition-colors hover:bg-slate-300 disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={confirmDelete}
                    disabled={deleteMutation.isPending}
                    className="flex-1 rounded-lg bg-red-500 px-4 py-2 text-white transition-colors hover:bg-red-600 disabled:opacity-50"
                  >
                    {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
                {deleteMutation.isError && (
                  <p className="mt-2 text-sm text-red-500">
                    Error: {(deleteMutation.error as Error).message}
                  </p>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
    </ErrorBoundary>
  )
}
