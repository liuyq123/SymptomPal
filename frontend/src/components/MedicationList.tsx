import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { deactivateMedication, updateMedication } from '../api/client'
import type { MedicationEntry, MedicationLogEntry } from '../api/types'

interface MedicationListProps {
  userId: string
  medications: MedicationEntry[]
  history: MedicationLogEntry[]
  isLoading?: boolean
}

export default function MedicationList({ userId, medications, history, isLoading }: MedicationListProps) {
  const queryClient = useQueryClient()
  const [reminderTimesDraft, setReminderTimesDraft] = useState<Record<string, string>>({})
  const [reminderError, setReminderError] = useState<string | null>(null)

  const deactivateMutation = useMutation({
    mutationFn: (medId: string) => deactivateMedication(medId, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['medications', userId] })
      queryClient.invalidateQueries({ queryKey: ['medication-history', userId] })
    },
  })

  const reminderMutation = useMutation({
    mutationFn: ({
      medId,
      reminderEnabled,
      reminderTimes,
    }: {
      medId: string
      reminderEnabled: boolean
      reminderTimes: string[]
    }) => updateMedication(
      medId,
      {
        reminder_enabled: reminderEnabled,
        reminder_times: reminderTimes,
      },
      userId
    ),
    onSuccess: () => {
      setReminderError(null)
      queryClient.invalidateQueries({ queryKey: ['medications', userId] })
      queryClient.invalidateQueries({ queryKey: ['medication-reminders', userId] })
    },
    onError: (error: Error) => setReminderError(error.message),
  })

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString()
  }

  const formatTime = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const formatDateTime = (dateStr: string) => {
    const date = new Date(dateStr)
    const today = new Date()
    const isToday = date.toDateString() === today.toDateString()

    if (isToday) {
      return `Today ${formatTime(dateStr)}`
    }
    return `${formatDate(dateStr)} ${formatTime(dateStr)}`
  }

  const parseReminderTimes = (input: string): string[] => {
    return input
      .split(',')
      .map(part => part.trim())
      .filter(Boolean)
  }

  const validReminderTime = (value: string) => /^([01]\d|2[0-3]):[0-5]\d$/.test(value)

  if (isLoading) {
    return (
      <div className="text-center py-8 text-gray-500">Loading medications...</div>
    )
  }

  return (
    <div className="space-y-4">
      {/* My Medications */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">My Medications</h3>

        {medications.length === 0 ? (
          <p className="text-gray-500 text-center py-4">
            No medications saved yet. Add one above!
          </p>
        ) : (
          <div className="space-y-3">
            {medications.map(med => (
              <div
                key={med.id}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div className="flex-1">
                  <div className="font-medium text-gray-800">{med.name}</div>
                  <div className="text-sm text-gray-500 space-x-2">
                    {med.dose && <span>{med.dose}</span>}
                    {med.frequency && <span>• {med.frequency}</span>}
                    {med.reason && <span>• for {med.reason}</span>}
                  </div>
                  {med.notes && (
                    <div className="text-xs text-gray-400 mt-1">{med.notes}</div>
                  )}
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <label className="inline-flex items-center gap-2 text-xs text-gray-600">
                      <input
                        type="checkbox"
                        checked={Boolean(med.reminder_enabled)}
                        onChange={(e) => {
                          reminderMutation.mutate({
                            medId: med.id,
                            reminderEnabled: e.target.checked,
                            reminderTimes: med.reminder_times || [],
                          })
                        }}
                        disabled={reminderMutation.isPending}
                      />
                      Reminders
                    </label>
                    <input
                      type="text"
                      value={reminderTimesDraft[med.id] ?? (med.reminder_times || []).join(', ')}
                      onChange={(e) =>
                        setReminderTimesDraft((prev) => ({
                          ...prev,
                          [med.id]: e.target.value,
                        }))
                      }
                      placeholder="HH:MM, HH:MM"
                      className="px-2 py-1 border border-gray-300 rounded text-xs w-40"
                    />
                    <button
                      className="px-2 py-1 text-xs bg-blue-500 text-white rounded disabled:opacity-50"
                      disabled={reminderMutation.isPending}
                      onClick={() => {
                        const parsedTimes = parseReminderTimes(
                          reminderTimesDraft[med.id] ?? (med.reminder_times || []).join(', ')
                        )
                        if (parsedTimes.some((value) => !validReminderTime(value))) {
                          setReminderError('Reminder times must be HH:MM (24-hour), comma separated.')
                          return
                        }
                        reminderMutation.mutate({
                          medId: med.id,
                          reminderEnabled: Boolean(med.reminder_enabled),
                          reminderTimes: parsedTimes,
                        })
                      }}
                    >
                      Save times
                    </button>
                  </div>
                </div>
                <button
                  onClick={() => {
                    if (confirm(`Remove ${med.name} from your medications?`)) {
                      deactivateMutation.mutate(med.id)
                    }
                  }}
                  className="ml-2 text-red-400 hover:text-red-600 p-1"
                  title="Remove medication"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}

        {reminderError && (
          <p className="mt-3 text-sm text-red-500">
            {reminderError}
          </p>
        )}
      </div>

      {/* Recent History */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Recent History</h3>

        {history.length === 0 ? (
          <p className="text-gray-500 text-center py-4">
            No medication logs yet. Log a dose above!
          </p>
        ) : (
          <div className="space-y-2">
            {history.map(log => (
              <div
                key={log.id}
                className="flex items-center justify-between p-3 bg-green-50 rounded-lg"
              >
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-800">{log.medication_name}</span>
                    {log.dose_taken && (
                      <span className="text-sm text-green-700 bg-green-100 px-2 py-0.5 rounded">
                        {log.dose_taken}
                      </span>
                    )}
                  </div>
                  {log.notes && (
                    <div className="text-xs text-gray-500 mt-1">{log.notes}</div>
                  )}
                </div>
                <div className="text-sm text-gray-500 ml-2">
                  {formatDateTime(log.taken_at)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
