import { useEffect, useState } from 'react'
import type { LogEntry } from '../api/types'

interface LogPreviewProps {
  log: LogEntry
  onDelete?: (logId: string, isPermanent?: boolean) => void
  title?: string
}

export default function LogPreview({ log, onDelete, title }: LogPreviewProps) {
  const { transcript, description, photo_b64, extracted } = log
  const dismissalKey = `contactClinicianDismissed:${log.user_id}:${log.id}`
  const [clinicianDismissed, setClinicianDismissed] = useState(false)

  useEffect(() => {
    if (!log.contact_clinician_note) {
      setClinicianDismissed(false)
      return
    }
    try {
      setClinicianDismissed(window.localStorage.getItem(dismissalKey) === 'true')
    } catch {
      setClinicianDismissed(false)
    }
  }, [dismissalKey, log.contact_clinician_note])

  const dismissClinicianNote = () => {
    try {
      window.localStorage.setItem(dismissalKey, 'true')
    } catch {
      // Ignore storage errors and keep this a client-only enhancement.
    }
    setClinicianDismissed(true)
  }

  const photoSrc = photo_b64
    ? photo_b64.startsWith('data:')
      ? photo_b64
      : `data:image/jpeg;base64,${photo_b64}`
    : null

  return (
    <div className="bg-white rounded-xl shadow-md p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-800">{title ?? 'Latest Entry'}</h3>
        {onDelete && (
          <button
            onClick={() => onDelete(log.id)}
            className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
            title="Delete log"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        )}
      </div>

      {/* Clinician contact note (persistent, dismissible) */}
      {log.contact_clinician_note && !clinicianDismissed && (
        <div className="mb-4 border border-amber-300 bg-amber-50 rounded-lg p-3 text-sm text-amber-900">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="font-semibold">Contact clinician</p>
              <p className="mt-1">{log.contact_clinician_note}</p>
            </div>
            <button
              type="button"
              onClick={dismissClinicianNote}
              className="rounded-md border border-amber-300 px-2 py-1 text-xs font-medium text-amber-900 hover:bg-amber-100"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Photo attachment + image analysis */}
      {photoSrc && (
        <div className="mb-4">
          <p className="text-sm text-gray-500 mb-1">Photo:</p>
          <img
            src={photoSrc}
            alt="Symptom photo"
            className="w-full h-48 object-cover rounded-lg"
          />
          {log.image_analysis && (
            <div className="mt-2 p-3 bg-indigo-50 rounded-lg text-sm">
              <p className="text-gray-800">{log.image_analysis.clinical_description}</p>
              <p className="mt-1 text-xs text-gray-500">
                MedSigLIP analysis {log.image_analysis.confidence > 0 && `\u00B7 ${(log.image_analysis.confidence * 100).toFixed(0)}% overall confidence`}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Description */}
      {description && (
        <div className="mb-4">
          <p className="text-sm text-gray-500 mb-1">Description:</p>
          <p className="text-gray-800 bg-blue-50 p-3 rounded-lg">
            {description}
          </p>
        </div>
      )}

      {/* Transcript */}
      <div className="mb-4">
        <p className="text-sm text-gray-500 mb-1">Transcript:</p>
        <p className="text-gray-800 bg-gray-50 p-3 rounded-lg italic">
          "{transcript}"
        </p>
      </div>

      {/* Extracted chips */}
      <div className="space-y-3">
        {/* Symptoms */}
        {extracted.symptoms.length > 0 && (
          <div>
            <p className="text-sm text-gray-500 mb-1">Symptoms:</p>
            <div className="flex flex-wrap gap-2">
              {extracted.symptoms.map((s, i) => (
                <span
                  key={i}
                  className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm"
                >
                  {s.symptom}
                  {s.severity_1_10 && ` (${s.severity_1_10}/10)`}
                  {s.location && ` • ${s.location}`}
                  {s.character && ` • ${s.character}`}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Onset */}
        {extracted.symptoms.some(s => s.onset_time_text) && (
          <div>
            <p className="text-sm text-gray-500 mb-1">Onset:</p>
            <div className="flex flex-wrap gap-2">
              {extracted.symptoms
                .filter(s => s.onset_time_text)
                .map((s, i) => (
                  <span
                    key={i}
                    className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm"
                  >
                    {s.onset_time_text}
                  </span>
                ))}
            </div>
          </div>
        )}

        {/* Actions taken */}
        {extracted.actions_taken.length > 0 && (
          <div>
            <p className="text-sm text-gray-500 mb-1">Actions taken:</p>
            <div className="flex flex-wrap gap-2">
              {extracted.actions_taken.map((a, i) => (
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

        {/* Red flags */}
        {extracted.red_flags.length > 0 && (
          <div>
            <p className="text-sm text-gray-500 mb-1">Attention needed:</p>
            <div className="flex flex-wrap gap-2">
              {extracted.red_flags.map((flag, i) => (
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
    </div>
  )
}
