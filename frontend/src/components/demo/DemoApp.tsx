/**
 * DemoApp — Preloaded app mode.
 *
 * Shows the normal app tabs (Record, Monitor, Meds, Cycle, Profile, Timeline,
 * Doctor) pre-filled with demo patient data.  No backend needed —
 * the fetchApi interceptor returns pre-transformed static data.
 *
 * Layout matches the real app as closely as possible: same header, hero strip,
 * all 8 tabs, and placeholder versions of components that require a backend.
 */

import { useState, useEffect, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'

import type { Scenario } from '../../types/demo'
import type { DemoData } from '../../types/demoPlayer'
import type { DoctorPacket as DoctorPacketType, TimelineReveal as TimelineRevealType } from '../../api/types'
import { setDemoAppMode, clearDemoAppMode } from '../../api/demoInterceptor'
import { transformDemoData } from '../../api/demoTransformers'
import { getLogs, getMedications, getMedicationHistory } from '../../api/client'

// Reuse existing components — they work unchanged via the API interceptor
import LogPreview from '../LogPreview'
import CalendarView from '../CalendarView'
import MedicationList from '../MedicationList'
import ProfilePanel from '../ProfilePanel'
import CycleTracker from '../CycleTracker'
import TimelineReveal from '../TimelineReveal'
import DoctorPacket from '../DoctorPacket'
import SafetyBanner from '../SafetyBanner'

type DemoTab = 'record' | 'monitor' | 'meds' | 'cycle' | 'profile' | 'timeline' | 'doctor'

interface DemoAppProps {
  scenario: Scenario
  onBack: () => void
}

export default function DemoApp({ scenario, onBack }: DemoAppProps) {
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<DemoTab>('record')

  // Pre-computed summaries (passed as props, not fetched)
  const [doctorPacket, setDoctorPacket] = useState<DoctorPacketType | null>(null)
  const [timeline, setTimeline] = useState<TimelineRevealType | null>(null)

  const demoUserId = `demo_${scenario.id}`

  // Load demo JSON → transform → activate interceptor
  useEffect(() => {
    if (!scenario.demoDataFile) {
      setError('No demo data file configured for this scenario.')
      return
    }

    let cancelled = false

    async function load() {
      try {
        const resp = await fetch(scenario.demoDataFile!)
        if (!resp.ok) throw new Error(`Failed to load demo data: ${resp.status}`)
        const demoData: DemoData = await resp.json()

        if (cancelled) return

        const store = transformDemoData(demoData, scenario.id)
        setDemoAppMode(store)

        // Set pre-computed summaries
        setDoctorPacket(store.doctorPacket)
        setTimeline(store.timeline)
        setIsReady(true)
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load demo data')
        }
      }
    }

    load()

    return () => {
      cancelled = true
      clearDemoAppMode()
    }
  }, [scenario])

  // Once interceptor is active, React Query hooks in child components
  // will call fetchApi → interceptor returns pre-loaded data.
  // We also fetch at this level for components that receive data as props.
  const logsQuery = useQuery({
    queryKey: ['logs', demoUserId],
    queryFn: () => getLogs(demoUserId),
    enabled: isReady,
    staleTime: Infinity,
  })

  const medicationsQuery = useQuery({
    queryKey: ['medications', demoUserId],
    queryFn: () => getMedications(demoUserId),
    enabled: isReady,
    staleTime: Infinity,
  })

  const medHistoryQuery = useQuery({
    queryKey: ['medication-history', demoUserId],
    queryFn: () => getMedicationHistory(demoUserId, 50, 7),
    enabled: isReady,
    staleTime: Infinity,
  })

  const latestLog = useMemo(() => {
    if (!logsQuery.data?.length) return null
    return logsQuery.data[logsQuery.data.length - 1]
  }, [logsQuery.data])

  const hasRedFlags = latestLog?.extracted.red_flags && latestLog.extracted.red_flags.length > 0

  const totalLogs = logsQuery.data?.length || 0
  const activeMeds = (medicationsQuery.data || []).filter((med) => med.is_active).length

  // Tab definitions — all 8, matching real app
  const tabs: { id: DemoTab; label: string }[] = [
    { id: 'record', label: 'Record' },
    { id: 'monitor', label: 'Monitor' },
    { id: 'meds', label: 'Meds' },
    { id: 'cycle', label: 'Cycle' },
    { id: 'profile', label: 'Profile' },
    { id: 'timeline', label: 'Timeline' },
    { id: 'doctor', label: 'Doctor' },
  ]

  // --- Render ---

  if (error) {
    return (
      <div className="mx-auto max-w-2xl py-20 text-center">
        <p className="text-red-600">{error}</p>
        <button onClick={onBack} className="mt-4 text-cyan-600 hover:underline">Back to scenarios</button>
      </div>
    )
  }

  if (!isReady) {
    return (
      <div className="mx-auto max-w-2xl py-20 text-center">
        <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-cyan-600 border-t-transparent" />
        <p className="mt-4 text-slate-600">Loading {scenario.title} demo data...</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Header — matches real app */}
      <header className="sticky top-0 z-40 border-b border-slate-200/60 bg-white/72 backdrop-blur-xl">
        <div className="mx-auto flex w-[min(1100px,92vw)] items-center justify-between px-1 py-4">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-gradient-to-br from-cyan-600 to-indigo-700 text-sm font-semibold text-white shadow-lg">
              SP
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold text-slate-900">SymptomPal</h1>
                <span className="rounded-full bg-cyan-100 px-2 py-0.5 text-xs font-semibold text-cyan-700">
                  Demo
                </span>
              </div>
              <p className="text-sm text-slate-600">Voice-first health journaling for faster clinical handoffs</p>
            </div>
          </div>
          <button
            onClick={onBack}
            className="rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-700 px-4 py-2 text-sm font-semibold text-white shadow-lg transition-transform hover:scale-[1.02]"
          >
            Back
          </button>
        </div>
      </header>

      {/* Main content */}
      <main className="mx-auto w-[min(1100px,92vw)] space-y-4 py-6">
        {/* Hero strip — matches real app */}
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
              <span className="status-chip rounded-full px-3 py-1 text-xs font-semibold">full pipeline</span>
            </div>
          </div>
        </section>

        {/* Safety banner */}
        <SafetyBanner hasRedFlags={!!hasRedFlags} />

        {/* Tab navigation */}
        <nav role="tablist" aria-label="App sections" className="tab-rail flex gap-1 overflow-x-auto rounded-2xl p-1.5">
          {tabs.map(({ id, label }) => (
            <button
              key={id}
              role="tab"
              aria-selected={activeTab === id}
              onClick={() => setActiveTab(id)}
              className={`tab-pill flex-1 whitespace-nowrap rounded-xl px-4 py-2.5 text-sm font-semibold ${
                activeTab === id ? 'tab-pill-active' : ''
              }`}
            >
              {label}
            </button>
          ))}
        </nav>

        {/* Tab content */}
        {activeTab === 'record' && (
          <>
            <div className="glass-panel-strong space-y-4 rounded-2xl p-4 sm:p-5">
              {/* SymptomInput placeholder (disabled) */}
              <div className="pointer-events-none opacity-60">
                <h3 className="mb-3 text-lg font-semibold text-slate-800">Log Symptoms</h3>
                <div className="mb-4 flex gap-2">
                  <button className="rounded-lg border-2 border-blue-500 bg-blue-100 px-4 py-1.5 text-sm font-medium text-blue-700">
                    Voice
                  </button>
                  <button className="rounded-lg border border-slate-200 px-4 py-1.5 text-sm font-medium text-slate-600">
                    Text
                  </button>
                </div>
                <div className="flex flex-col items-center gap-3 py-4">
                  <button className="flex h-16 w-16 items-center justify-center rounded-full bg-blue-500 text-white shadow-lg">
                    <svg className="h-8 w-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </button>
                  <span className="text-xs text-slate-500">Tap to record</span>
                </div>
                <div className="mb-3 rounded-lg border-2 border-dashed border-gray-300 p-4 text-center text-sm text-gray-400">
                  Attach a photo (optional)
                </div>
                <button className="w-full rounded-xl bg-gray-300 px-4 py-2.5 text-sm font-semibold text-white" disabled>
                  Submit
                </button>
              </div>

              {/* Actual log preview */}
              {latestLog && (
                <LogPreview log={latestLog} onDelete={() => {}} />
              )}
              {!latestLog && (
                <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">
                  No logs available.
                </div>
              )}

              {/* FollowUpChat placeholder (disabled) */}
              {latestLog && latestLog.followup_question && (
                <div className="pointer-events-none opacity-60">
                  <h3 className="mb-3 text-base font-semibold text-slate-800">Quick Follow-up</h3>
                  <div className="space-y-3">
                    <div className="flex items-start gap-2">
                      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-green-100">
                        <svg className="h-4 w-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                      <div className="rounded-lg bg-green-50 p-3 text-sm text-slate-700" style={{ maxWidth: '80%' }}>
                        Got it, thanks for checking in.
                      </div>
                    </div>
                    <div className="flex items-start gap-2">
                      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-blue-100">
                        <svg className="h-4 w-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <div className="rounded-lg bg-blue-50 p-3 text-sm text-slate-700" style={{ maxWidth: '80%' }}>
                        {latestLog.followup_question}
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 flex gap-2">
                    <input
                      type="text"
                      placeholder="Type your answer..."
                      className="flex-1 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm"
                      disabled
                    />
                    <button className="rounded-full bg-blue-500 px-4 py-2 text-sm font-semibold text-white opacity-50" disabled>
                      Send
                    </button>
                  </div>
                </div>
              )}
            </div>
            <div className="glass-panel rounded-2xl p-4 sm:p-5">
              <CalendarView userId={demoUserId} onDeleteLog={() => {}} />
            </div>
          </>
        )}

        {activeTab === 'monitor' && (
          <div className="glass-panel-strong rounded-2xl p-4 sm:p-5">
            <div className="pointer-events-none opacity-60">
              <div className="bg-white rounded-xl shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">
                  Ambient Audio Monitor
                </h3>
                <p className="text-gray-600 mb-4">
                  Select a monitoring type and start a session to track health sounds.
                </p>

                {/* Input mode toggle */}
                <div className="flex rounded-lg bg-gray-100 p-1 mb-6">
                  <button className="flex-1 py-2 px-4 rounded-md text-sm font-medium bg-white text-indigo-600 shadow">
                    Live Microphone
                  </button>
                  <button className="flex-1 py-2 px-4 rounded-md text-sm font-medium text-gray-600">
                    Upload Audio File
                  </button>
                </div>

                {/* Session type selector */}
                <div className="grid grid-cols-2 gap-3 mb-6">
                  {[
                    { emoji: '🫁', label: 'Cough Tracker', desc: 'Count and analyze coughing episodes' },
                    { emoji: '🌬️', label: 'Breath Monitor', desc: 'Track breathing patterns' },
                  ].map((session, i) => (
                    <div
                      key={session.label}
                      className={`p-4 rounded-lg border-2 text-left ${
                        i === 0 ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200'
                      }`}
                    >
                      <div className="text-2xl mb-1">{session.emoji}</div>
                      <div className="font-medium text-gray-800">{session.label}</div>
                      <div className="text-xs text-gray-500">{session.desc}</div>
                    </div>
                  ))}
                </div>

                <button className="w-full py-3 bg-indigo-600 text-white rounded-lg font-medium" disabled>
                  Start Monitoring
                </button>
                <p className="mt-3 text-xs text-gray-500 text-center">
                  Note: Live microphone requires HTTPS or localhost access
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'meds' && (
          <div className="glass-panel-strong space-y-4 rounded-2xl p-4 sm:p-5">
            {/* MedicationReminders placeholder */}
            <div className="pointer-events-none opacity-60">
              <h3 className="mb-2 text-lg font-semibold text-slate-800">Medication Reminders</h3>
              <div className="rounded-xl border border-slate-200 bg-white/70 py-6 text-center text-sm text-slate-500">
                No reminders due right now.
              </div>
            </div>

            {/* MedicationInput placeholder */}
            <div className="pointer-events-none opacity-60">
              <h3 className="mb-3 text-lg font-semibold text-slate-800">Log Medication</h3>
              <div className="mb-3 flex gap-2">
                <button className="rounded-lg border-2 border-green-500 bg-green-100 px-4 py-1.5 text-sm font-medium text-green-700">
                  Quick Log
                </button>
                <button className="rounded-lg border border-slate-200 px-4 py-1.5 text-sm font-medium text-slate-600">
                  Add Medication
                </button>
              </div>
              <div className="space-y-3">
                <select className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-400" disabled>
                  <option>Select medication...</option>
                </select>
                <input
                  type="text"
                  placeholder="Dose taken"
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                  disabled
                />
                <input
                  type="text"
                  placeholder="Notes (optional)"
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm"
                  disabled
                />
                <button className="w-full rounded-xl bg-gray-300 px-4 py-2.5 text-sm font-semibold text-white" disabled>
                  Log Dose
                </button>
              </div>
            </div>

            {/* Actual MedicationList */}
            <MedicationList
              userId={demoUserId}
              medications={medicationsQuery.data || []}
              history={medHistoryQuery.data || []}
              isLoading={medicationsQuery.isLoading || medHistoryQuery.isLoading}
            />
          </div>
        )}

        {activeTab === 'cycle' && (
          <div className="glass-panel-strong rounded-2xl p-4 sm:p-5">
            <CycleTracker userId={demoUserId} />
          </div>
        )}

        {activeTab === 'profile' && (
          <div className="glass-panel-strong rounded-2xl p-4 sm:p-5">
            <ProfilePanel userId={demoUserId} />
          </div>
        )}

        {activeTab === 'timeline' && (
          <div className="glass-panel-strong space-y-4 rounded-2xl p-4 sm:p-5">
            {timeline ? (
              <TimelineReveal timeline={timeline} />
            ) : (
              <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">
                No timeline data available.
              </div>
            )}
          </div>
        )}

        {activeTab === 'doctor' && (
          <div className="glass-panel-strong space-y-4 rounded-2xl p-4 sm:p-5">
            {doctorPacket ? (
              <DoctorPacket packet={doctorPacket} />
            ) : (
              <div className="rounded-xl border border-slate-200 bg-white/70 py-8 text-center text-slate-500">
                No doctor packet available.
              </div>
            )}
          </div>
        )}
      </main>

      {/* Hardcoded scheduled check-in notification for demo (Sarah only) */}
      {scenario.id === 'sarah_chen' && <div className="fixed bottom-4 right-4 max-w-sm bg-white rounded-lg shadow-lg border border-blue-200 p-4 z-50">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
            <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-900 mb-2">
              Earlier you mentioned nausea after taking 800mg ibuprofen with your iron supplement. How are you feeling now — any stomach pain or dark stools?
            </p>
            <div className="flex items-center gap-2 mb-2">
              <input
                type="text"
                placeholder="How are you feeling?"
                className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled
              />
            </div>
            <div className="flex gap-2">
              <button className="flex-1 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md opacity-50 cursor-not-allowed" disabled>
                Reply
              </button>
              <button className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md opacity-50 cursor-not-allowed" disabled>
                Dismiss
              </button>
            </div>
          </div>
        </div>
      </div>}

      {/* Footer */}
      <footer className="mx-auto w-[min(1100px,92vw)] px-4 py-8 text-center text-xs text-slate-500">
        SymptomPal — For demonstration purposes only. Not medical advice.
      </footer>
    </div>
  )
}
