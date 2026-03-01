import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import type { DemoData, DemoLogEntry, DemoMessage, DemoLogMetadata, DemoDoctorPacket } from '../../types/demoPlayer'
import type { Scenario } from '../../types/demo'
import InlineTraceCard from './InlineTraceCard'
import IntakeTraceCard from './IntakeTraceCard'
import PromptInspector from './PromptInspector'

function formatDayDate(baseDate: Date | null, day: number): string {
  if (!baseDate) return `Day ${day}`
  const d = new Date(baseDate)
  d.setDate(baseDate.getDate() + day - 1)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

interface DemoPlayerProps {
  scenario: Scenario
  onBack: () => void
}

/** A flattened message for display, including day/phase context */
interface DisplayMessage {
  type: 'phase' | 'dayHeader' | 'message' | 'traceCard' | 'intakeTraceCard'
  phase?: string
  day?: number
  time?: string
  message?: DemoMessage
  metadata?: DemoLogMetadata
  logEntry?: DemoLogEntry
  logIndex?: number
  hasFollowup?: boolean
}

const LOADING_PHASES = [
  'Waking Demo Sandbox...',
  'Loading Pre-computed MedGemma 27B Trace...',
  'Ready for Replay',
]

export default function DemoPlayer({ scenario, onBack }: DemoPlayerProps) {
  const [demoData, setDemoData] = useState<DemoData | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [loadingPhase, setLoadingPhase] = useState(0)
  const [displayedMessages, setDisplayedMessages] = useState<DisplayMessage[]>([])
  const [isComplete, setIsComplete] = useState(false)
  const [isAudioEnabled, setIsAudioEnabled] = useState(true)
  const [waitTime, setWaitTime] = useState(3000)
  const [showDoctorPacket, setShowDoctorPacket] = useState(false)
  const [accumulatedSymptoms, setAccumulatedSymptoms] = useState<string[]>([])
  const [accumulatedRedFlags, setAccumulatedRedFlags] = useState<string[]>([])
  const [accumulatedProtocols, setAccumulatedProtocols] = useState<string[]>([])
  const [accumulatedClinicianNotes, setAccumulatedClinicianNotes] = useState<string[]>([])
  const [watchdogRevealed, setWatchdogRevealed] = useState(false)

  // Pause/step state
  const [isPaused, setIsPaused] = useState(false)
  const playbackModeRef = useRef<'playing' | 'paused' | 'stepping'>('playing')

  // Inline trace card expand/collapse state
  const [expandedTraceCards, setExpandedTraceCards] = useState<Set<number>>(new Set())

  // Pipeline trace state
  const [activeLogIndex, setActiveLogIndex] = useState(-1)
  const [activePipelineStep, setActivePipelineStep] = useState(0)

  // Calendar date anchoring
  const baseDate = useMemo(() => {
    if (!demoData) return null
    if (demoData.base_date) {
      const d = new Date(demoData.base_date + 'T00:00:00')
      if (!isNaN(d.getTime())) return d
    }
    const maxDay = Math.max(...demoData.logs.map(l => l.day))
    const d = new Date()
    d.setDate(d.getDate() - maxDay)
    d.setHours(0, 0, 0, 0)
    return d
  }, [demoData])

  // Prompt inspector
  const [showPromptInspector, setShowPromptInspector] = useState(false)

  const chatContainerRef = useRef<HTMLDivElement>(null)
  const lastMessageRef = useRef<HTMLDivElement>(null)
  const messageQueue = useRef<DisplayMessage[]>([])
  const totalMessages = useRef(0)
  const currentAudio = useRef<HTMLAudioElement | null>(null)
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isAudioEnabledRef = useRef(isAudioEnabled)
  const waitTimeRef = useRef(waitTime)
  const currentLogMetadataRef = useRef<DemoLogMetadata | null>(null)
  const playbackStarted = useRef(false)
  const currentLogHasFollowupRef = useRef(false)

  useEffect(() => { isAudioEnabledRef.current = isAudioEnabled }, [isAudioEnabled])
  useEffect(() => { waitTimeRef.current = waitTime }, [waitTime])

  const toggleTraceCard = useCallback((logIndex: number) => {
    setExpandedTraceCards(prev => {
      const next = new Set(prev)
      if (next.has(logIndex)) {
        next.delete(logIndex)
      } else {
        next.add(logIndex)
      }
      return next
    })
  }, [])

  // Loading sequence animation
  useEffect(() => {
    if (!demoData && !loadError) {
      const t1 = setTimeout(() => setLoadingPhase(1), 800)
      const t2 = setTimeout(() => setLoadingPhase(2), 2000)
      return () => { clearTimeout(t1); clearTimeout(t2) }
    }
  }, [demoData, loadError])

  // Load demo data
  useEffect(() => {
    if (!scenario.demoDataFile) {
      setLoadError('No demo data file configured for this scenario')
      return
    }
    let cancelled = false
    fetch(scenario.demoDataFile)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load demo data: ${res.status}`)
        return res.json()
      })
      .then((data: DemoData) => {
        if (cancelled) return
        setDemoData(data)
        // Flatten logs into a message queue
        const queue: DisplayMessage[] = []
        let lastPhase = ''
        let logIdx = 0
        for (const log of data.logs) {
          if (log.phase && log.phase !== lastPhase) {
            queue.push({ type: 'phase', phase: log.phase })
            lastPhase = log.phase
          }
          const hasFollowup = log.messages.some(m => m.isFollowup)
          queue.push({ type: 'dayHeader', day: log.day, time: log.time, metadata: log.metadata, hasFollowup })
          for (const msg of log.messages) {
            queue.push({ type: 'message', message: msg, metadata: log.metadata, day: log.day })
          }
          queue.push({
            type: log.intake ? 'intakeTraceCard' : 'traceCard',
            logEntry: log, logIndex: logIdx, metadata: log.metadata, day: log.day,
          })
          logIdx++
        }
        messageQueue.current = queue
        totalMessages.current = queue.length
      })
      .catch(err => { if (!cancelled) setLoadError(err.message) })
    return () => { cancelled = true }
  }, [scenario.demoDataFile])

  // Helper: schedule next message or respect pause/step
  const scheduleNext = useCallback((delay: number, processQueueFn: () => void) => {
    if (playbackModeRef.current === 'paused') {
      return // paused — wait for user to resume or step
    }
    if (playbackModeRef.current === 'stepping') {
      playbackModeRef.current = 'paused'
      setIsPaused(true)
      return // stepped one item, now pause
    }
    // 'playing' — schedule normally
    timeoutRef.current = setTimeout(processQueueFn, delay)
  }, [])

  // Process queue — advance one message at a time
  const processQueue = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }

    if (messageQueue.current.length === 0) {
      setIsComplete(true)
      return
    }

    const next = messageQueue.current.shift()!
    setDisplayedMessages(prev => [...prev, next])

    // Update accumulated metadata for the right panel
    if (next.type === 'dayHeader' && next.metadata) {
      const m = next.metadata
      currentLogMetadataRef.current = m
      currentLogHasFollowupRef.current = next.hasFollowup ?? false
      // Advance pipeline: new log entry
      setActiveLogIndex(prev => prev + 1)
      setActivePipelineStep(0)

      setAccumulatedSymptoms(prev => {
        const newSymptoms = m.symptoms.filter(s => !prev.includes(s))
        return newSymptoms.length > 0 ? [...prev, ...newSymptoms] : prev
      })
      if (m.red_flags.length > 0) {
        setAccumulatedRedFlags(prev => {
          const newFlags = m.red_flags.filter(f => !prev.includes(f))
          return newFlags.length > 0 ? [...prev, ...newFlags] : prev
        })
      }
      if (m.protocol) {
        setAccumulatedProtocols(prev =>
          prev.includes(m.protocol!) ? prev : [...prev, m.protocol!]
        )
      }
      if (m.clinician_note) {
        setAccumulatedClinicianNotes(prev =>
          prev.includes(m.clinician_note!) ? prev : [...prev, m.clinician_note!]
        )
      }
      if (m.tool_calls?.includes('run_watchdog_now')) {
        setWatchdogRevealed(true)
      }
    }

    // Advance pipeline steps based on message type
    if (next.type === 'message' && next.message) {
      if (next.message.speaker === 'patient') {
        if (next.message.isFollowup) {
          // Followup → advance to followup re-extraction step (6)
          setActivePipelineStep(6)
        } else {
          setActivePipelineStep(1)
          // Auto-advance through extraction → safety → protocol
          setTimeout(() => setActivePipelineStep(2), 400)
          setTimeout(() => setActivePipelineStep(3), 2200)
          setTimeout(() => setActivePipelineStep(4), 2600)
        }
      } else if (next.message.speaker === 'agent') {
        setActivePipelineStep(5)
        // Auto-advance to Watchdog — step 7 if followup present, else 6
        if (currentLogMetadataRef.current?.tool_calls?.includes('run_watchdog_now')) {
          const watchdogStep = currentLogHasFollowupRef.current ? 7 : 6
          setTimeout(() => setActivePipelineStep(watchdogStep), 3000)
        }
      }
    }

    // For traceCard/intakeTraceCard, auto-expand it and collapse previous
    if ((next.type === 'traceCard' || next.type === 'intakeTraceCard') && next.logIndex !== undefined) {
      setExpandedTraceCards(new Set([next.logIndex]))
      scheduleNext(800, processQueue)
      return
    }

    // For phase/dayHeader, advance quickly
    if (next.type === 'phase' || next.type === 'dayHeader') {
      scheduleNext(800, processQueue)
      return
    }

    // For messages with audio
    if (next.message?.audio && isAudioEnabledRef.current) {
      if (currentAudio.current) {
        currentAudio.current.pause()
        currentAudio.current.src = ''
      }
      const audio = new Audio(next.message.audio)
      currentAudio.current = audio
      audio.onended = () => {
        currentAudio.current = null
        scheduleNext(0, processQueue)
      }
      audio.onerror = () => {
        currentAudio.current = null
        scheduleNext(0, processQueue)
      }
      audio.play().catch(() => {
        currentAudio.current = null
        scheduleNext(0, processQueue)
      })
    } else {
      // No audio — use timer
      scheduleNext(waitTimeRef.current, processQueue)
    }
  }, [scheduleNext])

  // Start processing when data loads
  useEffect(() => {
    if (demoData && messageQueue.current.length > 0 && !playbackStarted.current) {
      playbackStarted.current = true
      processQueue()
    }
  }, [demoData, processQueue])

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (lastMessageRef.current) {
      lastMessageRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [displayedMessages])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (currentAudio.current) {
        currentAudio.current.pause()
        currentAudio.current = null
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  const handleToggleAudio = () => {
    setIsAudioEnabled(prev => {
      const isNowEnabled = !prev
      if (!isNowEnabled && currentAudio.current) {
        currentAudio.current.pause()
        currentAudio.current.src = ''
        currentAudio.current = null
        processQueue()
      }
      if (!isNowEnabled) setWaitTime(1500)
      return isNowEnabled
    })
  }

  const handleSpeedToggle = () => {
    setWaitTime(prev => prev === 3000 ? 1500 : 3000)
  }

  const handlePauseToggle = () => {
    if (isPaused) {
      // Resume
      playbackModeRef.current = 'playing'
      setIsPaused(false)
      processQueue()
    } else {
      // Pause
      playbackModeRef.current = 'paused'
      setIsPaused(true)
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
        timeoutRef.current = null
      }
      if (currentAudio.current) {
        currentAudio.current.pause()
        currentAudio.current = null
      }
    }
  }

  const handleStep = () => {
    playbackModeRef.current = 'stepping'
    setIsPaused(true)
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
    processQueue()
  }

  const handleSkipToEnd = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
    if (currentAudio.current) {
      currentAudio.current.pause()
      currentAudio.current = null
    }

    const remaining = messageQueue.current.splice(0)
    if (remaining.length === 0) return

    const newSymptoms = new Set(accumulatedSymptoms)
    const newRedFlags = new Set(accumulatedRedFlags)
    const newProtocols = new Set(accumulatedProtocols)
    const newClinicianNotes = new Set(accumulatedClinicianNotes)
    let didWatchdog = watchdogRevealed
    let lastLogIdx = activeLogIndex

    for (const item of remaining) {
      if (item.type === 'dayHeader' && item.metadata) {
        lastLogIdx++
        const m = item.metadata
        m.symptoms.forEach(s => newSymptoms.add(s))
        m.red_flags.forEach(f => newRedFlags.add(f))
        if (m.protocol) newProtocols.add(m.protocol)
        if (m.clinician_note) newClinicianNotes.add(m.clinician_note)
        if (m.tool_calls?.includes('run_watchdog_now')) didWatchdog = true
      }
    }

    setDisplayedMessages(prev => [...prev, ...remaining])
    setAccumulatedSymptoms([...newSymptoms])
    setAccumulatedRedFlags([...newRedFlags])
    setAccumulatedProtocols([...newProtocols])
    setAccumulatedClinicianNotes([...newClinicianNotes])
    setWatchdogRevealed(didWatchdog)
    setActiveLogIndex(lastLogIdx)
    setActivePipelineStep(7)
    setExpandedTraceCards(new Set())
    setIsPaused(false)
    playbackModeRef.current = 'playing'
    setIsComplete(true)
  }, [accumulatedSymptoms, accumulatedRedFlags, accumulatedProtocols, accumulatedClinicianNotes, watchdogRevealed, activeLogIndex])

  if (loadError) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg p-8 max-w-md text-center">
          <div className="text-4xl mb-4">&#x26A0;&#xFE0F;</div>
          <h3 className="text-xl font-semibold text-gray-800 mb-2">Trace Data Not Found</h3>
          <p className="text-gray-600 mb-2">{loadError}</p>
          <p className="text-gray-500 text-sm mb-6">
            Generate trace data first by running:<br />
            <code className="bg-gray-100 px-2 py-1 rounded text-xs">
              python simulate_patient.py {scenario.id} --save-responses
            </code><br />
            then:<br />
            <code className="bg-gray-100 px-2 py-1 rounded text-xs">
              python build_demo_json.py
            </code>
          </p>
          <button
            onClick={onBack}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Back to Trace Selection
          </button>
        </div>
      </div>
    )
  }

  // Staged loading sequence
  if (!demoData) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <div className="text-gray-600 text-lg font-mono">{LOADING_PHASES[loadingPhase]}</div>
          <div className="flex gap-1.5 justify-center mt-4">
            {LOADING_PHASES.map((_, i) => (
              <div
                key={i}
                className={`w-2 h-2 rounded-full transition-colors duration-500 ${
                  i <= loadingPhase ? 'bg-blue-500' : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm px-4 py-3 flex items-center justify-between flex-shrink-0">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-800 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>
        <div className="text-center">
          <h1 className="text-lg font-bold text-gray-800">Execution Sandbox</h1>
          <p className="text-xs text-gray-500">
            {demoData.patient.name} &middot; {demoData.patient.age}{demoData.patient.gender?.toLowerCase() === 'male' ? 'M' : 'F'}
            &middot; {demoData.patient.conditions.join(', ')}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleToggleAudio}
            className={`p-2 rounded-lg transition-colors ${isAudioEnabled ? 'text-blue-500 bg-blue-50' : 'text-gray-400 bg-gray-100'}`}
            title={isAudioEnabled ? 'Disable audio' : 'Enable audio'}
          >
            {isAudioEnabled ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072M12 6l-4 4H4v4h4l4 4V6z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707A1 1 0 0112 5v14a1 1 0 01-1.707.707L5.586 15z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
              </svg>
            )}
          </button>
          {!isAudioEnabled && (
            <button
              onClick={handleSpeedToggle}
              className={`px-2 py-1 text-xs rounded-lg transition-colors ${waitTime === 1500 ? 'text-blue-500 bg-blue-50' : 'text-gray-400 bg-gray-100'}`}
              title={waitTime === 1500 ? 'Normal speed' : 'Fast speed'}
            >
              {waitTime === 1500 ? '2x' : '1x'}
            </button>
          )}
        </div>
      </header>

      {/* Main content — single column */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 flex flex-col min-w-0 mx-auto w-full max-w-3xl">
          {/* Playback controls */}
          <div className="px-4 py-2 bg-gray-50 border-b border-gray-200 flex-shrink-0">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-gray-700">Conversation Trace</h2>
                <p className="text-xs text-gray-500">
                  {demoData.logs.length} logs over {demoData.logs[demoData.logs.length - 1]?.day || 0} days
                </p>
              </div>
              <div className="flex items-center gap-2">
                {!isComplete && (
                  <>
                    <button
                      onClick={handlePauseToggle}
                      className="p-1.5 rounded-lg bg-blue-50 text-blue-600 hover:bg-blue-100 transition-colors"
                      title={isPaused ? 'Resume' : 'Pause'}
                    >
                      {isPaused ? (
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M8 5v14l11-7z" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                        </svg>
                      )}
                    </button>
                    <button
                      onClick={handleStep}
                      className="p-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
                      title="Step forward"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                      </svg>
                    </button>
                    <button
                      onClick={handleSkipToEnd}
                      className="p-1.5 rounded-lg bg-gray-100 text-gray-600 hover:bg-gray-200 transition-colors"
                      title="Skip to end"
                    >
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M5.5 5v14l8-7-8-7zm9 0v14h3V5h-3z" />
                      </svg>
                    </button>
                  </>
                )}
                <span className="text-[10px] text-gray-400 font-mono">
                  {displayedMessages.length}/{totalMessages.current}
                </span>
                {isPaused && !isComplete && (
                  <span className="text-[10px] bg-amber-100 text-amber-600 px-1.5 py-0.5 rounded font-semibold">
                    PAUSED
                  </span>
                )}
              </div>
            </div>
          </div>

          <div ref={chatContainerRef} className="flex-1 overflow-y-auto px-4 py-4 pb-6 space-y-3">
            {displayedMessages.length === 0 && (
              <div className="text-center py-12 text-gray-400">
                Initializing trace replay...
              </div>
            )}
            {displayedMessages.map((item, idx) => (
              <div
                key={idx}
                ref={idx === displayedMessages.length - 1 ? lastMessageRef : null}
                className={idx === displayedMessages.length - 1 ? 'animate-fade-in' : ''}
              >
                {item.type === 'phase' && (
                  <div className="flex items-center gap-3 py-4">
                    <div className="flex-1 border-t border-blue-200" />
                    <span className="text-xs font-semibold text-blue-600 uppercase tracking-wide">
                      {item.phase}
                    </span>
                    <div className="flex-1 border-t border-blue-200" />
                  </div>
                )}
                {item.type === 'dayHeader' && (
                  <div className="flex items-center gap-2 pt-3 pb-1">
                    <span className="text-xs font-medium text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                      {formatDayDate(baseDate, item.day!)}, {item.time}
                    </span>
                    {item.metadata?.red_flags && item.metadata.red_flags.length > 0 && (
                      <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded font-medium">
                        RED FLAG
                      </span>
                    )}
                    {item.metadata?.protocol && (
                      <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded">
                        {item.metadata.protocol}
                      </span>
                    )}
                  </div>
                )}
                {item.type === 'message' && item.message && (
                  <MessageBubble
                    message={item.message}
                    patientName={demoData.patient.name}
                    onInspect={item.message.speaker === 'agent' ? () => setShowPromptInspector(true) : undefined}
                  />
                )}
                {item.type === 'traceCard' && item.logEntry && item.logIndex !== undefined && (
                  <InlineTraceCard
                    log={item.logEntry}
                    logIndex={item.logIndex}
                    isExpanded={expandedTraceCards.has(item.logIndex)}
                    isCurrentLog={item.logIndex === activeLogIndex}
                    activeStep={item.logIndex === activeLogIndex ? activePipelineStep : 7}
                    patientName={demoData.patient.name}
                    onToggle={toggleTraceCard}
                  />
                )}
                {item.type === 'intakeTraceCard' && item.logEntry && item.logIndex !== undefined && (
                  <IntakeTraceCard
                    log={item.logEntry}
                    logIndex={item.logIndex}
                    isExpanded={expandedTraceCards.has(item.logIndex)}
                    onToggle={toggleTraceCard}
                  />
                )}
              </div>
            ))}

            {/* End-of-replay Clinical Intelligence summary */}
            {isComplete && (
              <div className="mt-6 space-y-6 border-t-2 border-green-200 pt-6">
                <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wide">
                  Clinical Intelligence Summary
                </h3>

                {/* Patient info */}
                <div>
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Patient</h4>
                  <p className="text-sm text-gray-800 font-medium">{demoData.patient.name}</p>
                  <p className="text-xs text-gray-500">
                    {demoData.patient.age}{demoData.patient.gender?.toLowerCase() === 'male' ? 'M' : 'F'},
                    {' '}{demoData.patient.conditions.join(', ')}
                  </p>
                  {demoData.patient.summary && (
                    <p className="text-xs text-gray-600 mt-1">{demoData.patient.summary}</p>
                  )}
                </div>

                {/* Extracted symptoms */}
                {accumulatedSymptoms.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                      Extracted Symptoms ({accumulatedSymptoms.length})
                    </h4>
                    <div className="flex flex-wrap gap-1">
                      {accumulatedSymptoms.map((symptom, i) => (
                        <span key={i} className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded">
                          {symptom}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Red flags */}
                {accumulatedRedFlags.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-red-600 uppercase tracking-wide mb-2">
                      Red Flags
                    </h4>
                    <ul className="space-y-1">
                      {accumulatedRedFlags.map((flag, i) => (
                        <li key={i} className="text-xs text-red-700 bg-red-50 px-2 py-1 rounded">
                          {flag}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Protocol matches */}
                {accumulatedProtocols.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                      Protocol Matches
                    </h4>
                    <div className="flex flex-wrap gap-1">
                      {accumulatedProtocols.map((protocol, i) => (
                        <span key={i} className="text-xs bg-amber-50 text-amber-700 px-2 py-1 rounded">
                          {protocol}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Clinician notes */}
                {accumulatedClinicianNotes.length > 0 && (
                  <div>
                    <h4 className="text-xs font-semibold text-orange-600 uppercase tracking-wide mb-2">
                      Clinician Alerts
                    </h4>
                    <ul className="space-y-1">
                      {accumulatedClinicianNotes.map((note, i) => (
                        <li key={i} className="text-xs text-orange-800 bg-orange-50 px-2 py-1 rounded">
                          {note}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Watchdog Longitudinal Analysis */}
                {watchdogRevealed && demoData.watchdog_results && (
                  <div>
                    <h4 className="text-xs font-semibold text-amber-600 uppercase tracking-wide mb-1">
                      Watchdog Longitudinal Analysis
                    </h4>
                    <p className="text-[10px] text-slate-500 mb-2">
                      Run-level findings from end-of-simulation longitudinal review.
                    </p>

                    {demoData.watchdog_results.clinician_observations.map((obs, i) => (
                      <div key={i} className="bg-[#0d1117] rounded-lg border border-slate-700 p-3 mb-2">
                        <div className="flex items-center gap-1.5 mb-1.5">
                          <span className="w-2 h-2 bg-amber-400 rounded-full animate-pulse" />
                          <span className="text-[10px] text-amber-400 font-mono font-semibold uppercase">
                            Longitudinal Flag {i + 1}
                          </span>
                        </div>
                        <p className="text-amber-300 text-xs font-mono leading-relaxed whitespace-pre-wrap">{obs}</p>
                      </div>
                    ))}

                    {demoData.watchdog_results.pending_checkins
                      .filter(c => c.checkin_type === 'health_insight')
                      .map((c, i) => (
                        <div key={i} className="bg-teal-50 border border-teal-200 rounded-lg p-3 mt-2">
                          <div className="flex items-center gap-1.5 mb-1.5">
                            <span className="w-2 h-2 bg-teal-500 rounded-full" />
                            <span className="text-[10px] text-teal-600 font-semibold uppercase">
                              Patient-Facing Health Insight
                            </span>
                          </div>
                          <p className="text-teal-800 text-xs leading-relaxed">{c.message}</p>
                        </div>
                      ))}
                  </div>
                )}

                {/* Doctor packet */}
                {demoData.doctor_packet && (
                  <div>
                    <button
                      onClick={() => setShowDoctorPacket(!showDoctorPacket)}
                      className="w-full text-left flex items-center justify-between bg-green-50 text-green-800 px-3 py-2 rounded-lg hover:bg-green-100 transition-colors"
                    >
                      <span className="text-xs font-semibold uppercase tracking-wide">
                        Doctor Packet
                      </span>
                      <svg
                        className={`w-4 h-4 transition-transform ${showDoctorPacket ? 'rotate-180' : ''}`}
                        fill="none" stroke="currentColor" viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                    {showDoctorPacket && (
                      <DoctorPacketPanel packet={demoData.doctor_packet} />
                    )}
                  </div>
                )}

                <div className="text-center py-4 text-green-600 text-sm font-medium">
                  Trace replay complete
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer disclaimer */}
      <footer className="bg-white border-t border-gray-200 px-4 py-2 text-center text-xs text-gray-400 flex-shrink-0">
        MedGemma 27B Execution Sandbox — pre-computed traces from real backend architecture. Not medical advice.
      </footer>

      {/* Prompt Inspector modal */}
      <PromptInspector
        isOpen={showPromptInspector}
        onClose={() => setShowPromptInspector(false)}
      />

      {/* CSS for fade-in animation */}
      <style>{`
        .animate-fade-in {
          animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}

// Module-level ref so only one message replays audio at a time
let activeReplayAudio: { stop: () => void } | null = null

function MessageBubble({ message, patientName, onInspect }: { message: DemoMessage; patientName: string; onInspect?: () => void }) {
  const isPatient = message.speaker === 'patient'
  const firstName = patientName.split(' ')[0]
  const [isPlaying, setIsPlaying] = useState(false)
  const replayAudioRef = useRef<HTMLAudioElement | null>(null)

  const handleReplayAudio = useCallback(() => {
    if (!message.audio) return
    // Stop any other message's replay first
    activeReplayAudio?.stop()
    const audio = new Audio(message.audio)
    replayAudioRef.current = audio
    setIsPlaying(true)
    const cleanup = () => {
      setIsPlaying(false)
      replayAudioRef.current = null
      if (activeReplayAudio?.stop === stop) activeReplayAudio = null
    }
    const stop = () => { audio.pause(); cleanup() }
    activeReplayAudio = { stop }
    audio.onended = cleanup
    audio.onerror = cleanup
    audio.play().catch(cleanup)
  }, [message.audio])

  return (
    <div className={`flex gap-3 ${isPatient ? '' : 'flex-row-reverse'}`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold ${
        isPatient ? 'bg-blue-100 text-blue-700' : 'bg-green-100 text-green-700'
      }`}>
        {isPatient ? firstName[0] : 'AI'}
      </div>
      {/* Bubble */}
      <div className={`max-w-[75%] ${isPatient ? '' : 'text-right'}`}>
        <div className={`flex items-center gap-2 mb-0.5 ${isPatient ? '' : 'justify-end'}`}>
          <span className="text-xs text-gray-500">
            {isPatient
              ? (message.isFollowup ? `${firstName} (reply)` : firstName)
              : 'SymptomPal'
            }
          </span>
          {message.audio && (
            <button
              onClick={handleReplayAudio}
              className={`transition-colors ${isPlaying ? 'text-blue-500 animate-pulse' : 'text-gray-400 hover:text-blue-500'}`}
              title="Replay audio"
            >
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z" />
              </svg>
            </button>
          )}
          {!isPatient && onInspect && (
            <button
              onClick={onInspect}
              className="text-xs text-gray-400 hover:text-blue-500 font-mono transition-colors"
              title="Inspect prompt template"
            >
              &lt;/&gt;
            </button>
          )}
        </div>
        <div className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isPatient
            ? 'bg-blue-500 text-white rounded-bl-md'
            : 'bg-gray-100 text-gray-800 rounded-br-md'
        }`}>
          {message.text}
        </div>
        {message.question && !message.text.includes(message.question) && (
          <div className="mt-1 rounded-2xl px-4 py-2 text-sm bg-gray-50 text-gray-700 border border-gray-200 rounded-br-md">
            {message.question}
          </div>
        )}
      </div>
    </div>
  )
}

function DoctorPacketPanel({ packet }: { packet: DemoDoctorPacket }) {
  return (
    <div className="mt-2 space-y-3 text-xs">
      <div>
        <h4 className="font-semibold text-gray-700 mb-1">HPI</h4>
        <p className="text-gray-600 leading-relaxed">{packet.hpi}</p>
      </div>
      {packet.system_longitudinal_flags && packet.system_longitudinal_flags.length > 0 && (
        <div>
          <h4 className="font-semibold text-amber-700 mb-1 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 bg-amber-500 rounded-full" />
            Longitudinal System Flags
          </h4>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-2">
            <ul className="space-y-0.5">
              {packet.system_longitudinal_flags.map((flag, i) => (
                <li key={i} className="text-amber-900">&bull; {flag}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
      {packet.pertinent_positives.length > 0 && (
        <div>
          <h4 className="font-semibold text-gray-700 mb-1">Pertinent Positives</h4>
          <ul className="space-y-0.5">
            {packet.pertinent_positives.map((pp, i) => (
              <li key={i} className="text-green-700">&bull; {pp}</li>
            ))}
          </ul>
        </div>
      )}
      {packet.pertinent_negatives.length > 0 && (
        <div>
          <h4 className="font-semibold text-gray-700 mb-1">Pertinent Negatives</h4>
          <ul className="space-y-0.5">
            {packet.pertinent_negatives.map((pn, i) => (
              <li key={i} className="text-red-700">&bull; {pn}</li>
            ))}
          </ul>
        </div>
      )}
      {packet.timeline_bullets.length > 0 && (
        <div>
          <h4 className="font-semibold text-gray-700 mb-1">Timeline</h4>
          <ul className="space-y-0.5">
            {packet.timeline_bullets.map((tb, i) => (
              <li key={i} className="text-gray-600">&bull; {tb}</li>
            ))}
          </ul>
        </div>
      )}
      {packet.questions_for_clinician.length > 0 && (
        <div>
          <h4 className="font-semibold text-gray-700 mb-1">Patient Concerns</h4>
          <ul className="space-y-0.5">
            {packet.questions_for_clinician.map((q, i) => (
              <li key={i} className="text-gray-600">&bull; {q}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
