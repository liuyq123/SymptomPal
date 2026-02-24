import { useState, useCallback } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { createMedication, logMedication, extractMedicationFromVoice, audioToBase64 } from '../api/client'
import type { MedicationEntry, MedicationVoiceResponse } from '../api/types'
import { useAudioRecorder } from '../hooks/useAudioRecorder'

interface MedicationInputProps {
  userId: string
  medications: MedicationEntry[]
  onMedicationAdded?: () => void
  onDoseLogged?: () => void
}

type Mode = 'quick-log' | 'add-med'

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

export default function MedicationInput({
  userId,
  medications,
  onMedicationAdded,
  onDoseLogged,
}: MedicationInputProps) {
  const [mode, setMode] = useState<Mode>('quick-log')

  // Quick log state
  const [selectedMedId, setSelectedMedId] = useState<string>('')
  const [customMedName, setCustomMedName] = useState('')
  const [doseTaken, setDoseTaken] = useState('')
  const [logNotes, setLogNotes] = useState('')

  // Add medication state
  const [newMedName, setNewMedName] = useState('')
  const [newMedDose, setNewMedDose] = useState('')
  const [newMedFrequency, setNewMedFrequency] = useState('')
  const [newMedReason, setNewMedReason] = useState('')
  const [newMedNotes, setNewMedNotes] = useState('')

  // Voice extraction state
  const [voiceResult, setVoiceResult] = useState<MedicationVoiceResponse | null>(null)

  const queryClient = useQueryClient()

  const logMutation = useMutation({
    mutationFn: () => {
      const medName = selectedMedId
        ? medications.find(m => m.id === selectedMedId)?.name || customMedName
        : customMedName

      return logMedication({
        user_id: userId,
        medication_id: selectedMedId || undefined,
        medication_name: medName || undefined,
        dose_taken: doseTaken || undefined,
        notes: logNotes || undefined,
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['medication-history', userId] })
      setSelectedMedId('')
      setCustomMedName('')
      setDoseTaken('')
      setLogNotes('')
      setVoiceResult(null)
      onDoseLogged?.()
    },
  })

  const addMutation = useMutation({
    mutationFn: () =>
      createMedication({
        user_id: userId,
        name: newMedName,
        dose: newMedDose || undefined,
        frequency: newMedFrequency || undefined,
        reason: newMedReason || undefined,
        notes: newMedNotes || undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['medications', userId] })
      setNewMedName('')
      setNewMedDose('')
      setNewMedFrequency('')
      setNewMedReason('')
      setNewMedNotes('')
      setVoiceResult(null)
      setMode('quick-log')
      onMedicationAdded?.()
    },
  })

  const voiceMutation = useMutation({
    mutationFn: async (audioBlob: Blob) => {
      const audio_b64 = await audioToBase64(audioBlob)
      return extractMedicationFromVoice(userId, audio_b64)
    },
    onSuccess: (data) => {
      setVoiceResult(data)
      const med = data.medications[0]
      if (!med) return

      if (mode === 'add-med') {
        setNewMedName(med.name)
        if (med.dose_text) setNewMedDose(med.dose_text)
        if (data.frequency) setNewMedFrequency(data.frequency)
        if (data.reason) setNewMedReason(data.reason)
      } else {
        // Quick log: try to match against existing medications
        const match = medications.find(
          m => m.name.toLowerCase() === med.name.toLowerCase()
        )
        if (match) {
          setSelectedMedId(match.id)
          setCustomMedName('')
        } else {
          setSelectedMedId('')
          setCustomMedName(med.name)
        }
        if (med.dose_text) setDoseTaken(med.dose_text)
      }
    },
  })

  const handleRecordingComplete = useCallback((blob: Blob) => {
    voiceMutation.mutate(blob)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const { isRecording, recordingTime, startRecording, stopRecording, error: micError } =
    useAudioRecorder(handleRecordingComplete)

  const canLogDose = selectedMedId || customMedName.trim()
  const canAddMed = newMedName.trim()

  return (
    <div className="bg-white rounded-xl shadow-md p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-800">Log Medication</h2>

        {/* Voice input button */}
        <button
          onClick={isRecording ? stopRecording : startRecording}
          disabled={voiceMutation.isPending}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
            isRecording
              ? 'bg-red-100 text-red-600 animate-pulse'
              : voiceMutation.isPending
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
          title={isRecording ? 'Stop recording' : 'Say your medication'}
        >
          {voiceMutation.isPending ? (
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
              <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
            </svg>
          )}
          {isRecording ? formatTime(recordingTime) : voiceMutation.isPending ? 'Processing...' : 'Voice'}
        </button>
      </div>

      {/* Voice feedback */}
      {(micError || voiceMutation.isError) && (
        <p className="text-red-500 text-xs">
          {micError || voiceMutation.error?.message || 'Voice input failed'}
        </p>
      )}
      {voiceResult && voiceResult.medications.length === 0 && (
        <p className="text-amber-600 text-xs">
          No medication detected in: &ldquo;{voiceResult.transcript}&rdquo;. Please type it manually.
        </p>
      )}
      {voiceResult && voiceResult.medications.length > 1 && (
        <p className="text-amber-600 text-xs">
          Multiple medications detected. Showing first: &ldquo;{voiceResult.medications[0]?.name}&rdquo;.
          Also mentioned: {voiceResult.medications.slice(1).map(m => m.name).join(', ')}.
        </p>
      )}
      {voiceResult && voiceResult.medications.length === 1 && (
        <p className="text-green-600 text-xs">
          Detected: &ldquo;{voiceResult.transcript}&rdquo;
        </p>
      )}

      {/* Mode toggle */}
      <div className="flex space-x-2">
        <button
          onClick={() => setMode('quick-log')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            mode === 'quick-log'
              ? 'bg-green-100 text-green-700 border-2 border-green-500'
              : 'bg-gray-100 text-gray-600 border-2 border-transparent'
          }`}
        >
          Quick Log
        </button>
        <button
          onClick={() => setMode('add-med')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            mode === 'add-med'
              ? 'bg-green-100 text-green-700 border-2 border-green-500'
              : 'bg-gray-100 text-gray-600 border-2 border-transparent'
          }`}
        >
          Add Medication
        </button>
      </div>

      {mode === 'quick-log' && (
        <div className="space-y-3">
          {/* Medication selector */}
          {medications.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Select Medication
              </label>
              <select
                value={selectedMedId}
                onChange={e => {
                  setSelectedMedId(e.target.value)
                  if (e.target.value) setCustomMedName('')
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              >
                <option value="">-- Choose or type below --</option>
                {medications.map(med => (
                  <option key={med.id} value={med.id}>
                    {med.name} {med.dose ? `(${med.dose})` : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Or type custom */}
          {!selectedMedId && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {medications.length > 0 ? 'Or type medication name' : 'Medication Name'}
              </label>
              <input
                type="text"
                value={customMedName}
                onChange={e => setCustomMedName(e.target.value)}
                placeholder="e.g., Ibuprofen"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
          )}

          {/* Dose taken */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dose Taken (optional)
            </label>
            <input
              type="text"
              value={doseTaken}
              onChange={e => setDoseTaken(e.target.value)}
              placeholder="e.g., 400mg, 2 tablets"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>

          {/* Notes */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Notes (optional)
            </label>
            <input
              type="text"
              value={logNotes}
              onChange={e => setLogNotes(e.target.value)}
              placeholder="e.g., Took with food"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>

          {/* Submit */}
          <button
            onClick={() => logMutation.mutate()}
            disabled={!canLogDose || logMutation.isPending}
            className={`w-full py-3 rounded-lg font-medium transition-colors ${
              canLogDose && !logMutation.isPending
                ? 'bg-green-500 text-white hover:bg-green-600'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}
          >
            {logMutation.isPending ? 'Logging...' : 'Log Dose'}
          </button>

          {logMutation.isError && (
            <p className="text-red-500 text-sm text-center">
              Error: {logMutation.error.message}
            </p>
          )}
        </div>
      )}

      {mode === 'add-med' && (
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Medication Name *
            </label>
            <input
              type="text"
              value={newMedName}
              onChange={e => setNewMedName(e.target.value)}
              placeholder="e.g., Lisinopril"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dose
              </label>
              <input
                type="text"
                value={newMedDose}
                onChange={e => setNewMedDose(e.target.value)}
                placeholder="e.g., 10mg"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Frequency
              </label>
              <input
                type="text"
                value={newMedFrequency}
                onChange={e => setNewMedFrequency(e.target.value)}
                placeholder="e.g., Once daily"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Reason
            </label>
            <input
              type="text"
              value={newMedReason}
              onChange={e => setNewMedReason(e.target.value)}
              placeholder="e.g., Blood pressure"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Notes
            </label>
            <input
              type="text"
              value={newMedNotes}
              onChange={e => setNewMedNotes(e.target.value)}
              placeholder="e.g., Take in morning"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>

          <button
            onClick={() => addMutation.mutate()}
            disabled={!canAddMed || addMutation.isPending}
            className={`w-full py-3 rounded-lg font-medium transition-colors ${
              canAddMed && !addMutation.isPending
                ? 'bg-green-500 text-white hover:bg-green-600'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}
          >
            {addMutation.isPending ? 'Adding...' : 'Add Medication'}
          </button>

          {addMutation.isError && (
            <p className="text-red-500 text-sm text-center">
              Error: {addMutation.error.message}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
