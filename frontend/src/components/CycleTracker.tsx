import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getCycleDays, getCycles, getCycleCorrelations,
  logCycleDay, markPeriodStart,
} from '../api/client'
import type { FlowLevel, CycleSymptomCorrelation } from '../api/types'

interface CycleTrackerProps {
  userId: string
}

const FLOW_LEVELS: { value: FlowLevel; label: string; color: string }[] = [
  { value: 'spotting', label: 'Spotting', color: 'bg-red-100 text-red-600' },
  { value: 'light', label: 'Light', color: 'bg-red-200 text-red-700' },
  { value: 'medium', label: 'Medium', color: 'bg-red-300 text-red-800' },
  { value: 'heavy', label: 'Heavy', color: 'bg-red-400 text-white' },
]

const CONFIDENCE_STYLES: Record<string, string> = {
  strong: 'bg-red-100 border-red-300 text-red-800',
  moderate: 'bg-orange-100 border-orange-300 text-orange-800',
  weak: 'bg-yellow-100 border-yellow-300 text-yellow-800',
}

function CorrelationCard({ correlation }: { correlation: CycleSymptomCorrelation }) {
  const style = CONFIDENCE_STYLES[correlation.confidence] || CONFIDENCE_STYLES.weak
  return (
    <div className={`p-4 rounded-lg border ${style}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="font-semibold capitalize">{correlation.symptom}</span>
        <span className="text-xs font-medium uppercase px-2 py-0.5 rounded-full bg-white/50">
          {correlation.confidence}
        </span>
      </div>
      <p className="text-sm">{correlation.description}</p>
    </div>
  )
}

export default function CycleTracker({ userId }: CycleTrackerProps) {
  const queryClient = useQueryClient()
  const today = new Date().toISOString().split('T')[0]
  const [selectedDate, setSelectedDate] = useState(today)
  const [selectedFlow, setSelectedFlow] = useState<FlowLevel>('medium')
  const [notes, setNotes] = useState('')

  // Queries
  const daysQuery = useQuery({
    queryKey: ['cycle-days', userId],
    queryFn: () => getCycleDays(userId),
  })

  const cyclesQuery = useQuery({
    queryKey: ['cycles', userId],
    queryFn: () => getCycles(userId),
  })

  const correlationsQuery = useQuery({
    queryKey: ['cycle-correlations', userId],
    queryFn: () => getCycleCorrelations(userId),
  })

  // Mutations
  const logDayMutation = useMutation({
    mutationFn: () => logCycleDay({
      user_id: userId,
      date: selectedDate,
      flow_level: selectedFlow,
      notes: notes || undefined,
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cycle-days', userId] })
      queryClient.invalidateQueries({ queryKey: ['cycles', userId] })
      queryClient.invalidateQueries({ queryKey: ['cycle-correlations', userId] })
      setNotes('')
    },
  })

  const periodStartMutation = useMutation({
    mutationFn: () => markPeriodStart({ user_id: userId, date: today }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['cycle-days', userId] })
      queryClient.invalidateQueries({ queryKey: ['cycles', userId] })
      queryClient.invalidateQueries({ queryKey: ['cycle-correlations', userId] })
    },
  })

  const cycles = cyclesQuery.data || []
  const correlations = correlationsQuery.data?.correlations || []
  const currentCycle = cycles.length > 0 ? cycles[cycles.length - 1] : null
  const completedCycles = cycles.filter(c => c.length_days != null)
  const avgLength = completedCycles.length > 0
    ? Math.round(completedCycles.reduce((s, c) => s + (c.length_days || 0), 0) / completedCycles.length)
    : null
  const avgPeriod = completedCycles.length > 0
    ? Math.round(completedCycles.reduce((s, c) => s + c.period_length_days, 0) / completedCycles.length * 10) / 10
    : null

  // Current cycle day
  let currentDay: number | null = null
  let currentPhase = ''
  if (currentCycle) {
    const start = new Date(currentCycle.start_date + 'T00:00:00')
    const now = new Date(today + 'T00:00:00')
    currentDay = Math.floor((now.getTime() - start.getTime()) / (1000 * 60 * 60 * 24)) + 1
    const cycleLen = currentCycle.length_days || avgLength || 28
    if (currentDay <= 5) currentPhase = 'Menstrual'
    else if (currentDay < cycleLen - 15) currentPhase = 'Follicular'
    else if (currentDay <= cycleLen - 13) currentPhase = 'Ovulatory'
    else currentPhase = 'Luteal'
  }

  // Recent period days for display
  const recentDays = (daysQuery.data || []).slice(0, 60)

  return (
    <div className="space-y-4">
      {/* Quick Period Start */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Period Tracking</h3>
        <button
          onClick={() => periodStartMutation.mutate()}
          disabled={periodStartMutation.isPending}
          className="w-full py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 disabled:opacity-50 mb-4"
        >
          {periodStartMutation.isPending ? 'Saving...' : 'Period Started Today'}
        </button>

        {/* Detailed Day Logger */}
        <div className="border-t pt-4">
          <p className="text-sm text-gray-600 mb-2">Or log a specific day:</p>
          <div className="flex gap-2 mb-3">
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
            />
          </div>
          <div className="flex gap-2 mb-3">
            {FLOW_LEVELS.map((level) => (
              <button
                key={level.value}
                onClick={() => setSelectedFlow(level.value)}
                className={`flex-1 py-2 rounded-md text-sm font-medium transition-all ${
                  selectedFlow === level.value
                    ? level.color + ' ring-2 ring-offset-1 ring-red-400'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {level.label}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Notes (optional)"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
            />
            <button
              onClick={() => logDayMutation.mutate()}
              disabled={logDayMutation.isPending}
              className="px-4 py-2 bg-red-500 text-white rounded-md text-sm font-medium hover:bg-red-600 disabled:opacity-50"
            >
              {logDayMutation.isPending ? '...' : 'Log'}
            </button>
          </div>
        </div>
      </div>

      {/* Cycle Stats */}
      {(currentCycle || completedCycles.length > 0) && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Cycle Stats</h3>
          <div className="grid grid-cols-2 gap-4">
            {currentDay && (
              <div className="text-center p-3 bg-pink-50 rounded-lg">
                <div className="text-2xl font-bold text-pink-600">Day {currentDay}</div>
                <div className="text-xs text-pink-500">{currentPhase} phase</div>
              </div>
            )}
            {avgLength && (
              <div className="text-center p-3 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{avgLength}d</div>
                <div className="text-xs text-purple-500">Avg cycle length</div>
              </div>
            )}
            {avgPeriod && (
              <div className="text-center p-3 bg-red-50 rounded-lg">
                <div className="text-2xl font-bold text-red-600">{avgPeriod}d</div>
                <div className="text-xs text-red-500">Avg period length</div>
              </div>
            )}
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{cycles.length}</div>
              <div className="text-xs text-blue-500">Cycles tracked</div>
            </div>
          </div>
        </div>
      )}

      {/* Symptom-Cycle Correlations — the crown jewel */}
      {correlations.length > 0 && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-1">Symptom-Cycle Patterns</h3>
          <p className="text-sm text-gray-500 mb-4">
            Patterns detected across {correlationsQuery.data?.analysis_window_cycles || 0} cycles
          </p>
          <div className="space-y-3">
            {correlations.map((c, i) => (
              <CorrelationCard key={i} correlation={c} />
            ))}
          </div>
        </div>
      )}

      {correlations.length === 0 && cycles.length >= 2 && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Symptom-Cycle Patterns</h3>
          <p className="text-sm text-gray-500">
            No correlations detected yet. Keep logging symptoms — patterns will appear as more cycle data accumulates.
          </p>
        </div>
      )}

      {cycles.length < 2 && cycles.length > 0 && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <p className="text-sm text-gray-500">
            Log at least 2 complete cycles to start detecting symptom-cycle patterns.
          </p>
        </div>
      )}

      {/* Recent Period Days */}
      {recentDays.length > 0 && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Recent Entries</h3>
          <div className="space-y-2">
            {recentDays.map((day) => (
              <div key={day.id} className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
                <span className="text-sm text-gray-700">{day.date}</span>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                    day.flow_level === 'heavy' ? 'bg-red-400 text-white' :
                    day.flow_level === 'medium' ? 'bg-red-300 text-red-800' :
                    day.flow_level === 'light' ? 'bg-red-200 text-red-700' :
                    day.flow_level === 'spotting' ? 'bg-red-100 text-red-600' :
                    'bg-gray-100 text-gray-600'
                  }`}>
                    {day.flow_level}
                  </span>
                  {day.notes && (
                    <span className="text-xs text-gray-400" title={day.notes}>
                      {day.notes.length > 20 ? day.notes.slice(0, 20) + '...' : day.notes}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
