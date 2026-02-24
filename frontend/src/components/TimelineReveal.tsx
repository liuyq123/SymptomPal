import { useMemo } from 'react'
import type { TimelineReveal as TimelineRevealType } from '../api/types'

interface TimelineRevealProps {
  timeline: TimelineRevealType
}

export default function TimelineReveal({ timeline }: TimelineRevealProps) {
  // Sort reverse chronological (most recent first)
  const sortedPoints = useMemo(
    () =>
      [...timeline.story_points].sort(
        (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
      ),
    [timeline.story_points],
  )

  if (!sortedPoints.length) {
    return (
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Timeline</h3>
        <p className="text-sm text-gray-500">No timeline entries yet. Add logs to build your symptom timeline.</p>
      </div>
    )
  }

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    })
  }

  return (
    <div className="bg-white rounded-xl shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Timeline</h3>

      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200" />

        {/* Timeline points */}
        <div className="space-y-6">
          {sortedPoints.map((point, index) => (
            <div key={index} className="relative pl-10">
              {/* Circle marker */}
              <div className={`absolute left-2 w-5 h-5 rounded-full border-2 ${
                index === 0
                  ? 'bg-blue-500 border-blue-500'
                  : index === sortedPoints.length - 1
                  ? 'bg-green-500 border-green-500'
                  : 'bg-white border-gray-300'
              }`} />

              {/* Content */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-gray-800">{point.label}</span>
                  <span className="text-xs text-gray-500">{formatDate(point.timestamp)}</span>
                </div>
                <p className="text-gray-600 text-sm">{point.details}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
