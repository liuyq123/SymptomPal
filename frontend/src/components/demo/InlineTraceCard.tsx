import PipelineStepper from './PipelineStepper'
import type { DemoLogEntry } from '../../types/demoPlayer'

interface InlineTraceCardProps {
  log: DemoLogEntry
  logIndex: number
  isExpanded: boolean
  isCurrentLog: boolean
  activeStep: number
  patientName: string
  onToggle: (logIndex: number) => void
}

export default function InlineTraceCard({
  log, logIndex, isExpanded, isCurrentLog, activeStep, patientName, onToggle
}: InlineTraceCardProps) {
  const meta = log.metadata
  const symptomCount = meta.symptoms.length
  const redFlagCount = meta.red_flags.length

  if (!isExpanded) {
    return (
      <button
        onClick={() => onToggle(logIndex)}
        className="w-full text-left flex items-center gap-2 px-3 py-2 my-2 rounded-lg
                   bg-slate-50 border border-slate-200 hover:bg-slate-100 transition-colors group"
      >
        <svg className="w-4 h-4 text-slate-400 group-hover:text-slate-600 flex-shrink-0"
             fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-xs text-slate-500 font-medium">
          Day {log.day} Agent Trace
        </span>
        {symptomCount > 0 && (
          <span className="text-[10px] bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded">
            {symptomCount} symptom{symptomCount !== 1 ? 's' : ''}
          </span>
        )}
        {redFlagCount > 0 && (
          <span className="text-[10px] bg-red-50 text-red-600 px-1.5 py-0.5 rounded font-semibold">
            {redFlagCount} red flag{redFlagCount !== 1 ? 's' : ''}
          </span>
        )}
        {meta.protocol && (
          <span className="text-[10px] bg-amber-50 text-amber-600 px-1.5 py-0.5 rounded">
            {meta.protocol}
          </span>
        )}
        <svg className="w-3.5 h-3.5 text-green-500 ml-auto flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
        </svg>
      </button>
    )
  }

  return (
    <div className="my-3 rounded-xl border border-blue-200 bg-white shadow-sm overflow-hidden">
      <button
        onClick={() => onToggle(logIndex)}
        className="w-full flex items-center gap-2 px-4 py-2.5 bg-blue-50 hover:bg-blue-100 transition-colors text-left"
      >
        <svg className="w-4 h-4 text-blue-500 rotate-90 flex-shrink-0"
             fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-xs font-semibold text-blue-700">
          Day {log.day} Agent Trace
        </span>
        {isCurrentLog && (
          <span className="text-[10px] bg-blue-500 text-white px-1.5 py-0.5 rounded font-medium">
            LIVE
          </span>
        )}
      </button>
      <div className="px-4 py-3">
        <PipelineStepper
          log={log}
          activeStep={activeStep}
          patientName={patientName}
        />
      </div>
    </div>
  )
}
