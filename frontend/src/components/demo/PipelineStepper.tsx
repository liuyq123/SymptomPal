import { Fragment } from 'react'
import type { DemoLogEntry } from '../../types/demoPlayer'
import { useTypewriter } from '../../hooks/useTypewriter'

interface PipelineStepperProps {
  log: DemoLogEntry | null
  activeStep: number
  patientName: string
}

const STEPS = [
  { id: 'input',    label: 'Patient Input',       icon: 'MedASR',  terminalCmd: 'medASR.transcribe(audio_input)...' },
  { id: 'extract',  label: 'MedGemma Extraction', icon: 'Tool #1', terminalCmd: 'medgemma.extract(transcript, profile)...' },
  { id: 'redflag',  label: 'Red Flag Gate',       icon: 'Safety',  terminalCmd: 'safety.detect_red_flags(extraction)...' },
  { id: 'protocol', label: 'Protocol Engine',     icon: 'Rules',   terminalCmd: 'protocol_registry.evaluate(extraction, history)...' },
  { id: 'response', label: 'MedGemma Response',   icon: 'Tool #2', terminalCmd: 'medgemma.generate_response(extraction, context)...' },
  { id: 'watchdog', label: 'Watchdog Analysis',   icon: 'Tool #4', terminalCmd: 'watchdog.analyze_longitudinal(history)...' },
]

const FOLLOWUP_STEP = {
  id: 'followup',
  label: 'Followup Re-extraction',
  icon: 'Tool #1',
  terminalCmd: 'medgemma.extract(followup_answer, merge=true)...',
}

function highlightKeywords(text: string, keywords: string[]): (string | JSX.Element)[] {
  if (keywords.length === 0) return [text]
  const pattern = new RegExp(`(${keywords.map(k => k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  const parts = text.split(pattern)
  return parts.map((part, i) =>
    keywords.some(k => k.toLowerCase() === part.toLowerCase())
      ? <mark key={i} className="bg-yellow-200 px-0.5 rounded">{part}</mark>
      : part
  )
}

/** Decision trace steps for the agent's protocol selection path */
function decisionTraceSteps(mode: string | null | undefined, protocol: string | null | undefined) {
  switch (mode) {
    case 'llm_protocol':
      return [
        { label: 'MedGemma Analysis', color: 'text-blue-500' },
        { label: `Selected: ${protocol}`, color: 'text-blue-600' },
        { label: 'Safety \u2713', color: 'text-green-600' },
      ]
    case 'safety_override':
      return [
        { label: 'Risk Detected', color: 'text-red-500' },
        { label: `Override: ${protocol}`, color: 'text-red-600' },
        { label: 'LLM overridden', color: 'text-red-500' },
      ]
    case 'llm_fallback':
      return [
        { label: 'No Match', color: 'text-gray-500' },
        { label: 'MedGemma Fallback', color: 'text-blue-500' },
      ]
    case 'llm_user_question_priority':
      return [
        { label: 'Patient Question', color: 'text-blue-500' },
        { label: 'User Q Priority', color: 'text-blue-600' },
      ]
    case 'protocol':
      return protocol
        ? [
            { label: `Match: ${protocol}`, color: 'text-amber-600' },
            { label: 'Deterministic', color: 'text-amber-500' },
          ]
        : [
            { label: 'No Match', color: 'text-gray-500' },
            { label: 'No Override', color: 'text-gray-400' },
          ]
    default:
      return [{ label: mode || 'Unknown', color: 'text-gray-500' }]
  }
}

/** Human-readable label for the safety_mode value */
function safetyModeLabel(mode: string | null | undefined, protocol: string | null | undefined): string {
  switch (mode) {
    case 'protocol':
      return protocol ? 'Protocol Override' : 'Protocol Engine (No Override)'
    case 'llm_fallback': return 'MedGemma Response'
    case 'llm_user_question_priority': return 'MedGemma (User Q Priority)'
    case 'static_safety': return 'Static Safety Response'
    case 'protocol_fallback': return 'Protocol Fallback'
    default: return mode || 'Unknown'
  }
}

export default function PipelineStepper({ log, activeStep, patientName }: PipelineStepperProps) {
  // Hooks must be called unconditionally (Rules of Hooks)
  const meta = log?.metadata ?? { symptoms: [], actions_taken: [], red_flags: [], protocol: null, clinician_note: null }
  const patientMessage = log?.messages.find(m => m.speaker === 'patient') ?? null
  const followupMessage = log?.messages.find(m => m.isFollowup) ?? null
  const agentMessage = log?.messages.find(m => m.speaker === 'agent') ?? null

  // Reconstruct extraction JSON from parsed metadata
  const extractionJson = log ? JSON.stringify({
    symptoms: meta.symptoms,
    actions_taken: meta.actions_taken,
    red_flags: meta.red_flags,
    ...(meta.vital_signs && Object.keys(meta.vital_signs).length > 0 ? { vital_signs: meta.vital_signs } : {}),
    protocol_match: meta.protocol || null
  }, null, 2) : ''

  // Typewriter for Step 2: MedGemma extraction JSON
  const { displayedText: jsonStream, isTyping: isJsonTyping } = useTypewriter(
    extractionJson, 5, activeStep >= 2
  )

  // Typewriter for Step 5: Agent response text
  const agentText = agentMessage?.text || ''
  const { displayedText: agentStream, isTyping: isAgentTyping } = useTypewriter(
    agentText, 12, activeStep >= 5
  )

  if (!log) {
    return (
      <div className="text-center py-8 text-gray-400 text-sm">
        Waiting for first log entry...
      </div>
    )
  }

  const hasRedFlags = meta.red_flags.length > 0
  const hasProtocol = !!meta.protocol
  const isStaticSafety = meta.safety_mode === 'static_safety'
  const protocolToolCall = meta.tool_calls?.find(t => t.startsWith('invoke_protocol:'))
  const protocolName = protocolToolCall?.split(':')[1] ?? null
  const hasWatchdogCall = meta.tool_calls?.includes('run_watchdog_now')
  const hasToolCalls = meta.tool_calls && meta.tool_calls.length > 0

  // Build dynamic steps: base 5 + optional followup + optional watchdog
  const stepsToShow = [
    ...STEPS.slice(0, 5),
    ...(followupMessage ? [FOLLOWUP_STEP] : []),
    ...(meta.tool_calls?.includes('run_watchdog_now') ? [STEPS[5]] : []),
  ]

  return (
    <div className="space-y-1">
      {/* Log header with cached trace badge */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-gray-500 font-medium">
          Day {log.day}, {log.time}
        </span>
        <span className="text-[9px] text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded font-mono">
          Cached Trace
        </span>
      </div>

      {stepsToShow.map((step, idx) => {
        const stepNum = idx + 1
        const isCompleted = stepNum < activeStep
        const isActive = stepNum === activeStep
        const isPending = stepNum > activeStep

        return (
          <div key={idx} className="flex gap-3">
            {/* Vertical line + circle */}
            <div className="flex flex-col items-center">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 ${
                isCompleted
                  ? 'bg-green-500 text-white'
                  : isActive
                    ? 'bg-blue-500 text-white animate-pulse'
                    : 'bg-gray-200 text-gray-400'
              }`}>
                {isCompleted ? (
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                ) : stepNum}
              </div>
              {idx < stepsToShow.length - 1 && (
                <div className={`w-0.5 flex-1 min-h-[8px] ${
                  isCompleted ? 'bg-green-300' : 'bg-gray-200'
                }`} />
              )}
            </div>

            {/* Content */}
            <div className={`flex-1 pb-3 ${isPending ? 'opacity-40' : ''}`}>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-semibold text-gray-700">{step.label}</span>
                <span className="text-[10px] text-gray-400 font-mono">{step.icon}</span>
                {isActive && (
                  <span className="text-[10px] bg-blue-100 text-blue-600 px-1.5 py-0.5 rounded font-medium">
                    ACTIVE
                  </span>
                )}
              </div>

              {/* Step-specific content — only show when completed or active */}
              {(isCompleted || isActive) && (
                <div className="text-xs leading-relaxed">
                  {/* Terminal command prefix */}
                  <div className="text-[10px] text-gray-400 font-mono mb-1">
                    $&gt; {step.terminalCmd}
                  </div>

                  {/* Patient Input */}
                  {step.id === 'input' && patientMessage && (
                    <div className="bg-blue-50 rounded-lg p-2 text-gray-700">
                      <span className="text-[10px] text-gray-400 block mb-1">{patientName}:</span>
                      {highlightKeywords(patientMessage.text, meta.symptoms)}
                    </div>
                  )}

                  {/* MedGemma Extraction — streaming JSON */}
                  {step.id === 'extract' && (
                    <div className="space-y-1.5">
                      {/* Terminal-style streaming JSON */}
                      <pre className="p-2 bg-[#0d1117] rounded border border-slate-700 text-emerald-400 font-mono text-[10px] overflow-x-auto whitespace-pre-wrap">
                        <code>{isActive ? jsonStream : extractionJson}</code>
                        {isActive && isJsonTyping && (
                          <span className="inline-block w-1.5 h-3 bg-emerald-400 ml-0.5 align-middle animate-pulse" />
                        )}
                      </pre>
                      {/* Parsed summary chips (show when completed) */}
                      {isCompleted && (
                        <>
                          {meta.symptoms.length > 0 && (
                            <div>
                              <span className="text-[10px] text-gray-500">Symptoms:</span>
                              <div className="flex flex-wrap gap-1 mt-0.5">
                                {meta.symptoms.map((s, i) => (
                                  <span key={i} className="bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded text-[10px]">
                                    {s}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          {meta.actions_taken.length > 0 && (
                            <div>
                              <span className="text-[10px] text-gray-500">Actions:</span>
                              <div className="flex flex-wrap gap-1 mt-0.5">
                                {meta.actions_taken.map((a, i) => (
                                  <span key={i} className="bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded text-[10px]">
                                    {a}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </>
                      )}
                      {meta.symptoms.length === 0 && meta.actions_taken.length === 0 && isCompleted && (
                        <span className="text-gray-400 text-[10px]">No structured entities extracted</span>
                      )}
                    </div>
                  )}

                  {/* Red Flag Gate */}
                  {step.id === 'redflag' && (
                    <div>
                      {hasRedFlags ? (
                        <div className="bg-red-50 border border-red-200 rounded-lg p-2">
                          <div className="flex items-center gap-1.5 mb-1">
                            <span className="w-2 h-2 bg-red-500 rounded-full" />
                            <span className="text-red-700 font-semibold text-[10px] uppercase">Red Flags Detected</span>
                          </div>
                          <div className="text-red-600 text-[10px]">
                            {isStaticSafety
                              ? 'LLM generation halted. Static emergency response activated.'
                              : 'Execution halted. Triggering static emergency protocol.'}
                          </div>
                          {isStaticSafety && (
                            <div className="mt-1">
                              <span className="text-[10px] bg-red-200 text-red-800 px-1.5 py-0.5 rounded font-mono">
                                safety_mode: static_safety
                              </span>
                            </div>
                          )}
                          <div className="flex flex-wrap gap-1 mt-1">
                            {meta.red_flags.map((f, i) => (
                              <span key={i} className="bg-red-100 text-red-700 px-1.5 py-0.5 rounded text-[10px]">
                                {f}
                              </span>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="bg-green-50 border border-green-200 rounded-lg p-2">
                          <div className="flex items-center gap-1.5">
                            <span className="w-2 h-2 bg-green-500 rounded-full" />
                            <span className="text-green-700 font-semibold text-[10px] uppercase">Clear</span>
                          </div>
                          <div className="text-green-600 text-[10px]">
                            No safety flags. LLM generation permitted.
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Protocol Engine */}
                  {step.id === 'protocol' && (
                    <div>
                      {hasProtocol ? (
                        <div className="bg-amber-50 border border-amber-200 rounded-lg p-2">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <span className="bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded text-[10px] font-mono">
                              {meta.protocol}
                            </span>
                            {meta.safety_mode === 'llm_protocol' && (
                              <span className="text-[10px] bg-blue-100 text-blue-600 px-1.5 py-0.5 rounded font-semibold">
                                LLM-SELECTED
                              </span>
                            )}
                            {meta.safety_mode === 'safety_override' && (
                              <span className="text-[10px] bg-red-100 text-red-600 px-1.5 py-0.5 rounded font-semibold">
                                SAFETY OVERRIDE
                              </span>
                            )}
                          </div>
                          {meta.reason_code && (
                            <div className="text-amber-500 text-[10px] font-mono mt-1">
                              reason: {meta.reason_code}
                            </div>
                          )}
                          <div className="text-amber-600 text-[10px] mt-1">
                            {meta.safety_mode === 'llm_protocol'
                              ? 'MedGemma selected this protocol from the clinical catalog.'
                              : meta.safety_mode === 'safety_override'
                              ? 'Safety-critical protocol forced over LLM selection.'
                              : 'Protocol matched. Overriding LLM follow-up with deterministic question.'}
                          </div>
                          {meta.safety_mode && meta.safety_mode !== 'static_safety' && (
                            <div className="mt-1.5 pt-1.5 border-t border-amber-200/50">
                              <span className="text-[9px] text-gray-400 uppercase font-semibold">Decision Path</span>
                              <div className="text-[10px] font-mono mt-0.5 flex items-center gap-1 flex-wrap">
                                {decisionTraceSteps(meta.safety_mode, meta.protocol).map((s, i, arr) => (
                                  <Fragment key={i}>
                                    <span className={s.color}>{s.label}</span>
                                    {i < arr.length - 1 && <span className="text-gray-300">&rarr;</span>}
                                  </Fragment>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : meta.safety_mode === 'protocol' && meta.reason_code && meta.reason_code !== 'no_protocol_match' ? (
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-2">
                          <span className="text-blue-600 text-[10px]">
                            Protocol engine evaluated — no follow-up override
                          </span>
                          <div className="text-blue-500 text-[10px] font-mono mt-1">
                            reason: {meta.reason_code}
                          </div>
                          <div className="mt-1.5 pt-1.5 border-t border-blue-200/50">
                            <span className="text-[9px] text-gray-400 uppercase font-semibold">Decision Path</span>
                            <div className="text-[10px] font-mono mt-0.5 flex items-center gap-1 flex-wrap">
                              {decisionTraceSteps(meta.safety_mode, meta.protocol).map((s, i, arr) => (
                                <Fragment key={i}>
                                  <span className={s.color}>{s.label}</span>
                                  {i < arr.length - 1 && <span className="text-gray-300">&rarr;</span>}
                                </Fragment>
                              ))}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="bg-gray-50 border border-gray-200 rounded-lg p-2">
                          <span className="text-gray-500 text-[10px]">
                            No protocol match
                            {meta.safety_mode && meta.safety_mode !== 'protocol'
                              ? ` — ${safetyModeLabel(meta.safety_mode, meta.protocol)}`
                              : ' — no follow-up override needed'}
                          </span>
                          {meta.safety_mode && meta.safety_mode !== 'static_safety' && (
                            <div className="mt-1.5 pt-1.5 border-t border-gray-200/50">
                              <span className="text-[9px] text-gray-400 uppercase font-semibold">Decision Path</span>
                              <div className="text-[10px] font-mono mt-0.5 flex items-center gap-1 flex-wrap">
                                {decisionTraceSteps(meta.safety_mode, meta.protocol).map((s, i, arr) => (
                                  <Fragment key={i}>
                                    <span className={s.color}>{s.label}</span>
                                    {i < arr.length - 1 && <span className="text-gray-300">&rarr;</span>}
                                  </Fragment>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      {meta.clinician_note && (
                        <div className="bg-orange-50 border border-orange-200 rounded-lg p-2 mt-1.5">
                          <span className="text-[10px] text-orange-600 font-semibold">Clinician Alert:</span>
                          <p className="text-[10px] text-orange-700 mt-0.5">{meta.clinician_note}</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Followup Re-extraction */}
                  {step.id === 'followup' && followupMessage && (
                    <div className="space-y-1.5">
                      <div className="bg-blue-50 rounded-lg p-2 text-gray-700">
                        <span className="text-[10px] text-gray-400 block mb-1">{patientName} (followup):</span>
                        {highlightKeywords(followupMessage.text, meta.symptoms)}
                      </div>
                      <div className="text-[10px] text-indigo-500 font-medium">
                        Re-extracted and merged into current log
                      </div>
                      <pre className="p-2 bg-[#0d1117] rounded border border-slate-700 text-emerald-400 font-mono text-[10px] overflow-x-auto whitespace-pre-wrap">
                        <code>{extractionJson}</code>
                      </pre>
                    </div>
                  )}

                  {/* MedGemma Response — streaming text + tool calls */}
                  {step.id === 'response' && agentMessage && (
                    <div className="space-y-1.5">
                      {/* Safety mode badge */}
                      {meta.safety_mode && (
                        <div className="flex items-center gap-1.5">
                          <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                            meta.safety_mode === 'protocol' && meta.protocol
                              ? 'bg-amber-100 text-amber-700'
                              : meta.safety_mode === 'static_safety'
                                ? 'bg-red-100 text-red-700'
                                : 'bg-blue-100 text-blue-700'
                          }`}>
                            {safetyModeLabel(meta.safety_mode, meta.protocol)}
                          </span>
                        </div>
                      )}
                      {/* Agent tool-calling callout */}
                      {hasToolCalls && (
                        <div className="bg-purple-50 border border-purple-300 rounded-lg p-2">
                          <div className="flex items-center gap-1.5 mb-1">
                            <span className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                            <span className="text-purple-700 font-semibold text-[10px] uppercase">
                              Agent Tool Call{meta.tool_calls!.length > 1 ? 's' : ''}
                            </span>
                            <span className="text-[9px] bg-purple-200 text-purple-700 px-1 py-0.5 rounded font-mono">
                              AUTONOMOUS
                            </span>
                          </div>
                          {meta.tool_calls!.map((tc, i) => (
                            <div key={i} className="p-1.5 bg-[#0d1117] rounded border border-slate-700 mt-1">
                              <code className={`font-mono text-[10px] ${
                                tc.startsWith('invoke_protocol:') ? 'text-cyan-400' : 'text-emerald-400'
                              }`}>
                                $&gt; agent.invoke(&quot;{tc}&quot;)
                              </code>
                            </div>
                          ))}
                          <div className="text-purple-500 text-[10px] mt-1">
                            {protocolToolCall && !hasWatchdogCall &&
                              `Agent assessed clinical situation and autonomously invoked ${protocolName?.replace(/_/g, ' ')} protocol.`}
                            {!protocolToolCall && hasWatchdogCall &&
                              'Agent detected clinical pattern requiring longitudinal review — elected to trigger tool autonomously.'}
                            {protocolToolCall && hasWatchdogCall &&
                              `Agent invoked ${protocolName?.replace(/_/g, ' ')} protocol and triggered longitudinal review.`}
                            {!protocolToolCall && !hasWatchdogCall &&
                              'Agent autonomously invoked tool based on clinical assessment.'}
                          </div>
                        </div>
                      )}
                      {/* Agent response text */}
                      <div className="bg-green-50 rounded-lg p-2 text-gray-700">
                        {isActive ? agentStream : agentMessage.text}
                        {isActive && isAgentTyping && (
                          <span className="inline-block w-1.5 h-3 bg-gray-400 ml-0.5 align-middle animate-pulse" />
                        )}
                      </div>
                      {(isCompleted || (isActive && !isAgentTyping)) && agentMessage.question && !agentMessage.text.includes(agentMessage.question) && (
                        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-2">
                          <span className="text-[10px] text-yellow-600 font-semibold">Follow-up:</span>
                          <p className="text-yellow-800 mt-0.5">{agentMessage.question}</p>
                          {meta.question_rationale && (
                            <p className="text-[10px] text-yellow-600 italic mt-1">
                              {meta.question_rationale}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Watchdog Analysis — triggered indicator */}
                  {step.id === 'watchdog' && meta.tool_calls?.includes('run_watchdog_now') && (
                    <div className="space-y-1.5">
                      <div className="flex items-center gap-1.5 mb-0.5">
                        <span className="text-[10px] bg-purple-100 text-purple-600 px-1.5 py-0.5 rounded font-semibold">
                          AGENT-TRIGGERED
                        </span>
                        <span className="text-[10px] text-purple-400">
                          via MedGemma tool-calling
                        </span>
                      </div>
                      <div className="p-2 bg-[#0d1117] rounded border border-slate-700">
                        <code className="text-amber-400 font-mono text-[10px]">
                          [WATCHDOG]: Longitudinal analysis triggered — reviewing 30-day history
                        </code>
                      </div>
                      <div className="text-[10px] text-slate-500 italic mt-1">
                        Trigger is per-log; findings are run-level (full simulation).
                        Visible in Clinical tab after playback completes.
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
