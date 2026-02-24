import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  dismissMedicationReminder,
  getPendingMedicationReminders,
  snoozeMedicationReminder,
  takeMedicationReminder,
} from '../api/client'
import type { PendingMedicationReminder } from '../api/types'

interface MedicationRemindersProps {
  userId: string
}

function reminderKey(reminder: PendingMedicationReminder): string {
  return `${reminder.medication_id}:${reminder.due_at}`
}

function formatTime(value: string): string {
  const date = new Date(value)
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

export default function MedicationReminders({ userId }: MedicationRemindersProps) {
  const queryClient = useQueryClient()
  const [snoozeMinutesByReminder, setSnoozeMinutesByReminder] = useState<Record<string, number>>({})
  const [actionError, setActionError] = useState<string | null>(null)

  const remindersQuery = useQuery({
    queryKey: ['medication-reminders', userId],
    queryFn: () => getPendingMedicationReminders(userId),
    refetchInterval: 60_000,
  })

  const invalidateData = () => {
    queryClient.invalidateQueries({ queryKey: ['medication-reminders', userId] })
    queryClient.invalidateQueries({ queryKey: ['medication-history', userId] })
    queryClient.invalidateQueries({ queryKey: ['medications', userId] })
  }

  const takeMutation = useMutation({
    mutationFn: takeMedicationReminder,
    onSuccess: () => {
      setActionError(null)
      invalidateData()
    },
    onError: (error: Error) => setActionError(error.message),
  })

  const dismissMutation = useMutation({
    mutationFn: dismissMedicationReminder,
    onSuccess: () => {
      setActionError(null)
      invalidateData()
    },
    onError: (error: Error) => setActionError(error.message),
  })

  const snoozeMutation = useMutation({
    mutationFn: snoozeMedicationReminder,
    onSuccess: () => {
      setActionError(null)
      invalidateData()
    },
    onError: (error: Error) => setActionError(error.message),
  })

  const reminders = [...(remindersQuery.data || [])].sort((a, b) => (
    new Date(a.due_at).getTime() - new Date(b.due_at).getTime()
  ))

  return (
    <div className="bg-white rounded-xl shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-2">Medication Reminders</h3>

      {remindersQuery.isLoading && (
        <p className="text-sm text-gray-500">Checking reminders...</p>
      )}

      {!remindersQuery.isLoading && reminders.length === 0 && (
        <p className="text-sm text-gray-500">
          No reminders due. Enable reminders on a medication to see them here.
        </p>
      )}

      {reminders.length > 0 && (
        <div className="space-y-3">
          {reminders.map((reminder) => {
            const key = reminderKey(reminder)
            const snoozeMinutes = snoozeMinutesByReminder[key] ?? 15
            const busy = takeMutation.isPending || dismissMutation.isPending || snoozeMutation.isPending

            return (
              <div key={key} className="border border-gray-200 rounded-lg p-3">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <p className="font-medium text-gray-800">
                      {reminder.medication_name}
                      {reminder.dose ? ` (${reminder.dose})` : ''}
                    </p>
                    <p className="text-sm text-gray-500">
                      Scheduled {reminder.scheduled_time} • Due {formatTime(reminder.due_at)}
                    </p>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded-full ${
                      reminder.is_overdue
                        ? 'bg-red-100 text-red-700'
                        : 'bg-yellow-100 text-yellow-700'
                    }`}
                  >
                    {reminder.is_overdue ? 'Overdue' : 'Due now'}
                  </span>
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <button
                    className="px-3 py-1.5 bg-green-500 text-white rounded-md text-sm hover:bg-green-600 disabled:opacity-50"
                    disabled={busy}
                    onClick={() =>
                      takeMutation.mutate({
                        user_id: userId,
                        medication_id: reminder.medication_id,
                        due_at: reminder.due_at,
                      })
                    }
                  >
                    Taken
                  </button>

                  <button
                    className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded-md text-sm hover:bg-gray-300 disabled:opacity-50"
                    disabled={busy}
                    onClick={() =>
                      dismissMutation.mutate({
                        user_id: userId,
                        medication_id: reminder.medication_id,
                        due_at: reminder.due_at,
                      })
                    }
                  >
                    Dismiss
                  </button>

                  <select
                    className="px-2 py-1.5 border border-gray-300 rounded-md text-sm"
                    value={snoozeMinutes}
                    onChange={(e) =>
                      setSnoozeMinutesByReminder((prev) => ({
                        ...prev,
                        [key]: Number(e.target.value),
                      }))
                    }
                  >
                    <option value={10}>Snooze 10m</option>
                    <option value={15}>Snooze 15m</option>
                    <option value={30}>Snooze 30m</option>
                    <option value={60}>Snooze 60m</option>
                  </select>

                  <button
                    className="px-3 py-1.5 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600 disabled:opacity-50"
                    disabled={busy}
                    onClick={() =>
                      snoozeMutation.mutate({
                        user_id: userId,
                        medication_id: reminder.medication_id,
                        due_at: reminder.due_at,
                        snooze_minutes: snoozeMinutes,
                      })
                    }
                  >
                    Snooze
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {actionError && (
        <p className="mt-3 text-sm text-red-500">
          {actionError}
        </p>
      )}
    </div>
  )
}
