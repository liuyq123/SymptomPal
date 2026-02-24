import { useDiffHighlight } from '../../hooks/useDiffHighlight'
import type { DiffPart, TagItem } from '../../hooks/useDiffHighlight'
import type { LogEntry } from '../../api/types'

interface DemoAnalysisPanelProps {
  currentLog: LogEntry | null
  previousLog: LogEntry | null
}

const DIFF_TEXT_CLASS: Record<DiffPart['type'], string> = {
  added: 'diff-add',
  removed: 'diff-remove',
  unchanged: '',
}

function DiffText({ parts }: { parts: DiffPart[] }) {
  return (
    <>
      {parts.map((part, i) => (
        <span key={i} className={DIFF_TEXT_CLASS[part.type]}>{part.text}</span>
      ))}
    </>
  )
}

function TagList({ items, baseClass, emptyText }: { items: TagItem[]; baseClass: string; emptyText: string }) {
  if (items.length === 0) {
    return <p className="text-gray-400 text-sm">{emptyText}</p>
  }
  return (
    <div className="flex flex-wrap">
      {items.map((item, i) => {
        const diffClass = item.type === 'added' ? ' diff-add' : item.type === 'removed' ? ' diff-remove' : ''
        return (
          <span key={i} className={`inline-block px-3 py-1 rounded-full text-sm mr-2 mb-2 ${baseClass}${diffClass}`}>
            {item.label}
          </span>
        )
      })}
    </div>
  )
}

export default function DemoAnalysisPanel({ currentLog, previousLog }: DemoAnalysisPanelProps) {
  const diff = useDiffHighlight(previousLog, currentLog)

  if (!currentLog) {
    return (
      <div className="text-center py-16 text-gray-500">
        <svg
          className="w-20 h-20 mx-auto mb-6 text-gray-300"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        <p className="text-lg font-medium mb-2">Ready for Analysis</p>
        <p className="text-sm">Record a symptom to see AI extraction in action</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Timestamp */}
      <div className="text-xs text-gray-500">
        Recorded: {new Date(currentLog.recorded_at).toLocaleString()}
      </div>

      {/* Transcript */}
      <div className="bg-gray-50 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
          <h4 className="text-sm font-semibold text-gray-700">Transcript</h4>
        </div>
        <div className="text-gray-800 leading-relaxed">
          <DiffText parts={diff.transcript} />
        </div>
      </div>

      {/* Symptoms */}
      <div className="bg-white rounded-xl border-2 border-purple-200 p-5">
        <div className="flex items-center gap-2 mb-3">
          <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
            />
          </svg>
          <h4 className="text-sm font-semibold text-gray-700">Extracted Symptoms</h4>
          {previousLog && (
            <span className="ml-auto text-xs text-gray-500">
              Green = Added, Red = Removed
            </span>
          )}
        </div>
        <TagList
          items={diff.symptoms}
          baseClass="bg-purple-100 text-purple-800"
          emptyText="No symptoms extracted yet"
        />
      </div>

      {/* Actions Taken */}
      {(currentLog.extracted.actions_taken.length > 0 || previousLog?.extracted.actions_taken.length) && (
        <div className="bg-white rounded-xl border-2 border-blue-200 p-5">
          <div className="flex items-center gap-2 mb-3">
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <h4 className="text-sm font-semibold text-gray-700">Actions Taken</h4>
          </div>
          <TagList
            items={diff.actions}
            baseClass="bg-blue-100 text-blue-800"
            emptyText="No actions taken yet"
          />
        </div>
      )}

      {/* Red Flags */}
      {(currentLog.extracted.red_flags.length > 0 || previousLog?.extracted.red_flags.length) && (
        <div className="bg-white rounded-xl border-2 border-red-200 p-5">
          <div className="flex items-center gap-2 mb-3">
            <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
            <h4 className="text-sm font-semibold text-gray-700">Attention Needed</h4>
          </div>
          <TagList
            items={diff.redFlags}
            baseClass="bg-red-100 text-red-800 font-medium"
            emptyText="No red flags detected"
          />
        </div>
      )}

      {/* Photo if present */}
      {currentLog.photo_b64 && (
        <div className="bg-gray-50 rounded-xl p-5">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Attached Photo</h4>
          <img
            src={
              currentLog.photo_b64.startsWith('data:')
                ? currentLog.photo_b64
                : `data:image/jpeg;base64,${currentLog.photo_b64}`
            }
            alt="Symptom photo"
            className="w-full h-48 object-cover rounded-lg"
          />
        </div>
      )}

      {/* Follow-up question if present */}
      {currentLog.followup_question && !currentLog.followup_answer && (
        <div className="bg-yellow-50 border-2 border-yellow-200 rounded-xl p-5">
          <div className="flex items-center gap-2 mb-2">
            <svg className="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <h4 className="text-sm font-semibold text-gray-700">AI Follow-up Question</h4>
          </div>
          <p className="text-gray-700 italic">"{currentLog.followup_question}"</p>
          <p className="text-xs text-gray-600 mt-2">Answer in the left panel to continue</p>
        </div>
      )}

      {/* Help text for first submission */}
      {!previousLog && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-sm text-gray-700">
          <p className="font-medium mb-1">First submission complete!</p>
          <p className="text-xs text-gray-600">
            Submit another symptom to see the diff highlighting in action (green = additions, red = removals)
          </p>
        </div>
      )}
    </div>
  )
}
