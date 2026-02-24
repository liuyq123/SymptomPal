import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getLogsByDateRange } from '../api/client'
import type { LogEntry } from '../api/types'
import LogPreview from './LogPreview'

interface CalendarViewProps {
  userId: string
  onDeleteLog: (logId: string, isPermanent?: boolean) => void
}

const DAY_LABELS = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa']

function formatDateKey(year: number, month: number, day: number): string {
  return `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
}

function formatDayHeader(dateKey: string): string {
  const date = new Date(dateKey + 'T12:00:00')
  return date.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  })
}

export default function CalendarView({ userId, onDeleteLog }: CalendarViewProps) {
  const [currentMonth, setCurrentMonth] = useState(() => {
    const now = new Date()
    return new Date(now.getFullYear(), now.getMonth(), 1)
  })
  const [selectedDate, setSelectedDate] = useState<string | null>(null)

  const year = currentMonth.getFullYear()
  const month = currentMonth.getMonth()

  // Compute month boundaries in UTC to avoid timezone offset issues
  const startDate = new Date(Date.UTC(year, month, 1)).toISOString()
  const endDate = new Date(Date.UTC(year, month + 1, 1)).toISOString()

  const monthLogsQuery = useQuery({
    queryKey: ['logs', userId, 'month', year, month],
    queryFn: () => getLogsByDateRange(userId, startDate, endDate),
  })

  // Group logs by local date
  const logsByDay = useMemo(() => {
    const map = new Map<string, LogEntry[]>()
    if (!monthLogsQuery.data) return map
    for (const log of monthLogsQuery.data) {
      const dateKey = new Date(log.recorded_at).toLocaleDateString('en-CA')
      const existing = map.get(dateKey) || []
      existing.push(log)
      map.set(dateKey, existing)
    }
    return map
  }, [monthLogsQuery.data])

  const daysInMonth = new Date(year, month + 1, 0).getDate()
  const firstDayOfWeek = new Date(year, month, 1).getDay()
  const monthName = currentMonth.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })

  const now = new Date()
  const todayKey = formatDateKey(now.getFullYear(), now.getMonth(), now.getDate())
  const isCurrentMonth = year === now.getFullYear() && month === now.getMonth()

  const goToPreviousMonth = () => {
    setCurrentMonth(new Date(year, month - 1, 1))
    setSelectedDate(null)
  }

  const goToNextMonth = () => {
    if (!isCurrentMonth) {
      setCurrentMonth(new Date(year, month + 1, 1))
      setSelectedDate(null)
    }
  }

  const handleDayClick = (day: number) => {
    const dateKey = formatDateKey(year, month, day)
    setSelectedDate(prev => prev === dateKey ? null : dateKey)
  }

  const selectedDayLogs = selectedDate ? (logsByDay.get(selectedDate) || []) : []

  return (
    <div className="bg-white rounded-xl shadow-md p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <button
          onClick={goToPreviousMonth}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          aria-label="Previous month"
        >
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <h3 className="text-lg font-semibold text-gray-800">{monthName}</h3>
        <button
          onClick={goToNextMonth}
          disabled={isCurrentMonth}
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          aria-label="Next month"
        >
          <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>

      {/* Day-of-week headers */}
      <div className="grid grid-cols-7 gap-1 mb-1">
        {DAY_LABELS.map(d => (
          <div key={d} className="text-center text-xs font-medium text-gray-400 py-1">{d}</div>
        ))}
      </div>

      {/* Calendar grid */}
      <div className="grid grid-cols-7 gap-1">
        {/* Empty cells before first day */}
        {Array.from({ length: firstDayOfWeek }, (_, i) => (
          <div key={`empty-${i}`} className="aspect-square" />
        ))}

        {/* Day cells */}
        {Array.from({ length: daysInMonth }, (_, i) => {
          const day = i + 1
          const dateKey = formatDateKey(year, month, day)
          const count = logsByDay.get(dateKey)?.length || 0
          const isSelected = dateKey === selectedDate
          const isToday = dateKey === todayKey

          return (
            <button
              key={day}
              onClick={() => handleDayClick(day)}
              className={`
                aspect-square flex flex-col items-center justify-center rounded-lg text-sm
                transition-colors relative
                ${isSelected
                  ? 'bg-blue-500 text-white'
                  : isToday
                    ? 'bg-blue-50 text-blue-700 font-semibold'
                    : 'hover:bg-gray-100 text-gray-800'
                }
              `}
            >
              {day}
              {count > 0 && (
                <div className="flex items-center gap-0.5 mt-0.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${isSelected ? 'bg-white' : 'bg-blue-500'}`} />
                  {count > 1 && (
                    <span className={`text-[10px] leading-none ${isSelected ? 'text-blue-100' : 'text-blue-500'}`}>
                      {count}
                    </span>
                  )}
                </div>
              )}
            </button>
          )
        })}
      </div>

      {/* Loading indicator */}
      {monthLogsQuery.isLoading && (
        <div className="text-center py-4 text-gray-400 text-sm">Loading logs...</div>
      )}

      {/* Selected day detail */}
      {selectedDate && !monthLogsQuery.isLoading && (
        <div className="mt-4 pt-4 border-t border-gray-100 space-y-3">
          <h4 className="text-sm font-medium text-gray-600">
            {formatDayHeader(selectedDate)}
            {selectedDayLogs.length === 0 && ' \u2014 No logs recorded'}
          </h4>
          {selectedDayLogs.map(log => (
            <LogPreview
              key={log.id}
              log={log}
              onDelete={onDeleteLog}
              title={new Date(log.recorded_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
            />
          ))}
        </div>
      )}
    </div>
  )
}
