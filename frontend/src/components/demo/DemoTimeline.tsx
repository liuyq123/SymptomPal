import { useQuery, useQueryClient } from '@tanstack/react-query'
import { getLogs, generateTimeline, generateDoctorPacket } from '../../api/client'
import type { Scenario } from '../../types/demo'
import { useState } from 'react'
import TimelineReveal from '../TimelineReveal'
import SymptomInput from '../SymptomInput'
import DoctorPacket from '../DoctorPacket'

const USER_ID = 'demo_user'

interface DemoTimelineProps {
  scenario: Scenario
  onBack: () => void
  onExit: () => void
}

export default function DemoTimeline({ scenario, onBack, onExit }: DemoTimelineProps) {
  const [activeView, setActiveView] = useState<'timeline' | 'doctor'>('timeline')
  const queryClient = useQueryClient()

  // Fetch real logs
  const logsQuery = useQuery({
    queryKey: ['logs', USER_ID],
    queryFn: () => getLogs(USER_ID),
  })

  // Callback to refresh timeline after symptom submission
  const handleSubmitComplete = () => {
    queryClient.invalidateQueries({ queryKey: ['logs', USER_ID] })
    queryClient.invalidateQueries({ queryKey: ['demo-timeline', USER_ID] })
    queryClient.invalidateQueries({ queryKey: ['demo-doctor', USER_ID] })
  }

  // Generate timeline if we have logs
  const timelineQuery = useQuery({
    queryKey: ['demo-timeline', USER_ID],
    queryFn: () => generateTimeline({ user_id: USER_ID, days: 7 }),
    enabled: !!logsQuery.data && logsQuery.data.length > 0,
  })

  // Generate doctor packet if we have logs
  const doctorQuery = useQuery({
    queryKey: ['demo-doctor', USER_ID],
    queryFn: () => generateDoctorPacket({ user_id: USER_ID, days: 7 }),
    enabled: !!logsQuery.data && logsQuery.data.length > 0,
  })

  const displayLogs = logsQuery.data && logsQuery.data.length > 7
    ? logsQuery.data.slice(0, 7)  // Most recent 7
    : logsQuery.data || []

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm px-6 py-4 flex justify-between items-center">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-gray-600 hover:text-gray-800 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Tour
        </button>

        <div className="flex items-center gap-3">
          <div className="text-2xl">{scenario.icon}</div>
          <div>
            <h3 className="font-semibold text-gray-800">{scenario.title} Timeline</h3>
            <p className="text-xs text-gray-500">
              {displayLogs.length} {displayLogs.length === 1 ? 'entry' : 'entries'} recorded
            </p>
          </div>
        </div>

        <button
          onClick={onExit}
          className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
        >
          Exit Demo
        </button>
      </div>

      {/* Symptom Input Section */}
      <div className="max-w-5xl mx-auto px-6 py-6">
        <div className="bg-white rounded-3xl shadow-xl p-6">
          <div className="mb-4">
            <h2 className="text-xl font-bold text-gray-800 mb-2">Try It Out</h2>
            <p className="text-sm text-gray-600">
              Record a symptom to see it appear in your timeline below. The scenario transcript is pre-filled, or you can write your own.
            </p>
          </div>
          <SymptomInput
            userId={USER_ID}
            initialTranscript={scenario.sampleTranscript}
            initialPhoto={scenario.samplePhoto}
            onSubmitComplete={handleSubmitComplete}
          />
        </div>
      </div>

      {/* View Toggle */}
      {displayLogs.length > 0 && (
        <div className="max-w-5xl mx-auto px-6 py-4">
          <div className="flex space-x-2 bg-white rounded-lg p-1 shadow-sm w-fit">
            <button
              onClick={() => setActiveView('timeline')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeView === 'timeline'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Timeline View
            </button>
            <button
              onClick={() => setActiveView('doctor')}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeView === 'doctor'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Doctor Report
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-5xl mx-auto px-6 py-6">
        {/* Empty State */}
        {displayLogs.length === 0 && (
          <div className="bg-white rounded-3xl shadow-xl p-12 text-center">
            <div className="text-6xl mb-4">{scenario.icon}</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-4">
              No Symptom Logs Yet
            </h2>
            <p className="text-gray-600 mb-8 max-w-md mx-auto">
              Start recording your symptoms using the main app to see your timeline
              and track your {scenario.title.toLowerCase()} progression over time.
            </p>
            <button
              onClick={onExit}
              className="px-8 py-3 bg-blue-500 text-white rounded-full hover:bg-blue-600 transition-colors font-semibold"
            >
              Go to Main App
            </button>
          </div>
        )}

        {/* Timeline View */}
        {displayLogs.length > 0 && activeView === 'timeline' && (
          <div className="space-y-6">
            {/* Stats Summary */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <StatCard
                title="Total Entries"
                value={displayLogs.length}
                icon="📝"
              />
              <StatCard
                title="Days Tracked"
                value={Math.ceil(
                  (new Date(displayLogs[0].recorded_at).getTime() -
                   new Date(displayLogs[displayLogs.length - 1].recorded_at).getTime()) /
                  (1000 * 60 * 60 * 24)
                ) + 1}
                icon="📅"
              />
              <StatCard
                title="Symptoms Logged"
                value={displayLogs.reduce((sum, log) => sum + log.extracted.symptoms.length, 0)}
                icon="🏥"
              />
            </div>

            {/* Timeline Component */}
            {timelineQuery.data && (
              <div className="bg-white rounded-3xl shadow-xl p-8">
                <h2 className="text-2xl font-bold text-gray-800 mb-6">Symptom Timeline</h2>
                <TimelineReveal timeline={timelineQuery.data} />
              </div>
            )}

            {timelineQuery.isLoading && (
              <div className="bg-white rounded-3xl shadow-xl p-8 text-center text-gray-500">
                Generating timeline...
              </div>
            )}

            {/* Detailed Log List */}
            <div className="bg-white rounded-3xl shadow-xl p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Detailed Entries</h2>
              <div className="space-y-4">
                {displayLogs.map((log, index) => (
                  <LogEntryCard key={log.id} log={log} index={index} totalLogs={displayLogs.length} />
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Doctor Report View */}
        {displayLogs.length > 0 && activeView === 'doctor' && (
          <div className="space-y-6">
            {doctorQuery.data && (
              <DoctorPacket packet={doctorQuery.data} />
            )}

            {doctorQuery.isLoading && (
              <div className="bg-white rounded-3xl shadow-xl p-8 text-center text-gray-500">
                Generating doctor report...
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

interface StatCardProps {
  title: string
  value: number
  icon: string
}

function StatCard({ title, value, icon }: StatCardProps) {
  return (
    <div className="bg-white rounded-2xl shadow-md p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-800">{value}</p>
        </div>
        <div className="text-4xl">{icon}</div>
      </div>
    </div>
  )
}

interface LogEntryCardProps {
  log: any
  index: number
  totalLogs: number
}

function LogEntryCard({ log, index, totalLogs }: LogEntryCardProps) {
  const date = new Date(log.recorded_at)
  const symptoms = log.extracted.symptoms
  const actions = log.extracted.actions_taken
  const redFlags = log.extracted.red_flags

  return (
    <div className="border-l-4 border-blue-500 pl-6 py-4 hover:bg-gray-50 transition-colors rounded-r-lg">
      <div className="flex items-start justify-between mb-2">
        <div>
          <p className="text-sm text-gray-500">
            {date.toLocaleDateString()} {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
          Entry #{totalLogs - index}
        </span>
      </div>

      {/* Transcript */}
      <p className="text-gray-700 mb-3 italic">"{log.transcript}"</p>

      {/* Symptoms */}
      {symptoms.length > 0 && (
        <div className="mb-2">
          <p className="text-xs text-gray-500 mb-1">Symptoms:</p>
          <div className="flex flex-wrap gap-2">
            {symptoms.map((s: any, i: number) => (
              <span
                key={i}
                className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm"
              >
                {s.symptom}
                {s.severity_1_10 && ` (${s.severity_1_10}/10)`}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      {actions.length > 0 && (
        <div className="mb-2">
          <p className="text-xs text-gray-500 mb-1">Actions taken:</p>
          <div className="flex flex-wrap gap-2">
            {actions.map((a: any, i: number) => (
              <span
                key={i}
                className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
              >
                {a.name}
                {a.dose_text && ` (${a.dose_text})`}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Red Flags */}
      {redFlags.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 mb-1">Attention needed:</p>
          <div className="flex flex-wrap gap-2">
            {redFlags.map((flag: string, i: number) => (
              <span
                key={i}
                className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium"
              >
                {flag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
