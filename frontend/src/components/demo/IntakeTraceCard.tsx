import type { DemoLogEntry } from '../../types/demoPlayer'

interface IntakeTraceCardProps {
  log: DemoLogEntry
  logIndex: number
  isExpanded: boolean
  onToggle: (logIndex: number) => void
}

const REGEX_FIELDS = new Set(['name', 'age_sex'])

export default function IntakeTraceCard({
  log, logIndex, isExpanded, onToggle
}: IntakeTraceCardProps) {
  const intake = log.intake
  if (!intake) return null

  const isRegex = REGEX_FIELDS.has(intake.question_id)
  const itemCount = intake.parsed_items.length

  // For regex-parsed demographics with empty parsed_items, show the raw answer as the result
  const displayItems = itemCount > 0
    ? intake.parsed_items
    : isRegex && intake.raw_answer
      ? [intake.raw_answer]
      : []
  const displayCount = displayItems.length

  // JSON preview — show actual value for regex fields
  const jsonPreview = isRegex && itemCount === 0 && intake.raw_answer
    ? { [intake.question_id]: intake.raw_answer }
    : { [intake.question_id]: intake.parsed_items }

  if (!isExpanded) {
    return (
      <button
        onClick={() => onToggle(logIndex)}
        className="w-full text-left flex items-center gap-2 px-3 py-2 my-2 rounded-lg
                   bg-indigo-50 border border-indigo-200 hover:bg-indigo-100 transition-colors group"
      >
        <svg className="w-4 h-4 text-indigo-400 group-hover:text-indigo-600 flex-shrink-0"
             fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-xs text-indigo-500 font-medium">
          Profile Parse: {intake.profile_field}
        </span>
        <span className="text-[10px] bg-indigo-100 text-indigo-600 px-1.5 py-0.5 rounded">
          {displayCount > 0
            ? isRegex
              ? 'regex parsed'
              : `${displayCount} item${displayCount !== 1 ? 's' : ''} extracted`
            : 'declined'}
        </span>
        <svg className="w-3.5 h-3.5 text-green-500 ml-auto flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
        </svg>
      </button>
    )
  }

  return (
    <div className="my-3 rounded-xl border border-indigo-200 bg-white shadow-sm overflow-hidden">
      <button
        onClick={() => onToggle(logIndex)}
        className="w-full flex items-center gap-2 px-4 py-2.5 bg-indigo-50 hover:bg-indigo-100 transition-colors text-left"
      >
        <svg className="w-4 h-4 text-indigo-500 rotate-90 flex-shrink-0"
             fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-xs font-semibold text-indigo-700">
          {isRegex ? 'Regex Profile Parse' : 'MedGemma Profile Parse'}
        </span>
        <span className="text-[10px] bg-indigo-500 text-white px-1.5 py-0.5 rounded font-medium">
          {intake.profile_field}
        </span>
      </button>
      <div className="px-4 py-3 space-y-3">
        {/* Terminal command */}
        <div className="text-[10px] text-gray-400 font-mono">
          {isRegex
            ? <>$&gt; regex.parse_{intake.question_id}(answer)...</>
            : <>$&gt; medgemma.parse_intake(answer, topic=&quot;{intake.question_id}&quot;)...</>
          }
        </div>

        {/* Raw answer */}
        <div>
          <span className="text-[10px] text-gray-500 font-medium block mb-1">Patient said:</span>
          <div className="bg-indigo-50 rounded-lg p-2 text-xs text-gray-700 italic">
            &ldquo;{intake.raw_answer}&rdquo;
          </div>
        </div>

        {/* Arrow */}
        <div className="flex justify-center">
          <svg className="w-4 h-4 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>

        {/* Parsed items */}
        <div>
          <span className="text-[10px] text-gray-500 font-medium block mb-1">Parsed structured data:</span>
          {displayCount > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {displayItems.map((item, i) => (
                <span key={i} className="text-[11px] bg-indigo-100 text-indigo-700 px-2 py-1 rounded-md font-medium">
                  {item}
                </span>
              ))}
            </div>
          ) : (
            <span className="text-[10px] text-gray-400 italic">No items (patient declined)</span>
          )}
        </div>

        {/* Streaming JSON result */}
        <pre className="p-2 bg-[#0d1117] rounded border border-slate-700 text-emerald-400 font-mono text-[10px] overflow-x-auto whitespace-pre-wrap">
          <code>{JSON.stringify(jsonPreview, null, 2)}</code>
        </pre>
      </div>
    </div>
  )
}
