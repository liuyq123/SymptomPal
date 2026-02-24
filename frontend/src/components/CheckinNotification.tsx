import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getPendingCheckins,
  respondToCheckin,
  dismissCheckin,
  triggerCheckin,
  blobToBase64,
} from '../api/client'
import type { ScheduledCheckin } from '../api/types'
import { useAudioRecorder } from '../hooks/useAudioRecorder'

interface CheckinNotificationProps {
  userId: string
}

export default function CheckinNotification({ userId }: CheckinNotificationProps) {
  const [response, setResponse] = useState('')
  const [audioResponseB64, setAudioResponseB64] = useState<string | null>(null)
  const [audioError, setAudioError] = useState<string | null>(null)
  const [activeCheckin, setActiveCheckin] = useState<ScheduledCheckin | null>(null)
  const queryClient = useQueryClient()
  const { isRecording, recordingTime, startRecording, stopRecording, error: recorderError } =
    useAudioRecorder((blob) => {
      blobToBase64(blob)
        .then((b64) => {
          setAudioResponseB64(b64)
          setAudioError(null)
        })
        .catch(() => {
          setAudioError('Failed to encode recording')
          setAudioResponseB64(null)
        })
    })

  // Poll for pending check-ins every 30 seconds
  const { data: checkins } = useQuery({
    queryKey: ['checkins', userId],
    queryFn: () => getPendingCheckins(userId),
    refetchInterval: 30000,
  })

  // When checkins arrive, show the first untriggered one
  useEffect(() => {
    if (checkins && checkins.length > 0 && !activeCheckin) {
      const firstCheckin = checkins[0]
      setActiveCheckin(firstCheckin)
      // Mark as triggered so it doesn't show again on next poll
      triggerCheckin(firstCheckin.id, userId).catch(console.error)
    }
  }, [checkins, activeCheckin])

  useEffect(() => {
    setResponse('')
    setAudioResponseB64(null)
    setAudioError(null)
  }, [activeCheckin?.id])

  const respondMutation = useMutation({
    mutationFn: () => {
      if (!activeCheckin) throw new Error('No active check-in')
      return respondToCheckin(activeCheckin.id, response, userId, audioResponseB64)
    },
    onSuccess: () => {
      setActiveCheckin(null)
      setResponse('')
      setAudioResponseB64(null)
      queryClient.invalidateQueries({ queryKey: ['checkins', userId] })
    },
  })

  const dismissMutation = useMutation({
    mutationFn: () => {
      if (!activeCheckin) throw new Error('No active check-in')
      return dismissCheckin(activeCheckin.id, userId)
    },
    onSuccess: () => {
      setActiveCheckin(null)
      setResponse('')
      setAudioResponseB64(null)
      queryClient.invalidateQueries({ queryKey: ['checkins', userId] })
    },
  })

  if (!activeCheckin) {
    return null
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (response.trim() || audioResponseB64) {
      respondMutation.mutate()
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const isIntake = activeCheckin.checkin_type === 'profile_intake'

  return (
    <div className="fixed bottom-4 right-4 max-w-sm bg-white rounded-lg shadow-lg border border-blue-200 p-4 z-50">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
          <span className="text-blue-600 text-lg">?</span>
        </div>
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-900 mb-2">
            {activeCheckin.message}
          </p>
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              placeholder={isIntake ? 'Type or record your answer' : 'How are you feeling?'}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-2"
              disabled={respondMutation.isPending}
            />
            <div className="flex items-center gap-2 mb-2">
              <button
                type="button"
                onClick={() => {
                  if (isRecording) {
                    stopRecording()
                  } else {
                    startRecording().catch(() => undefined)
                  }
                }}
                disabled={respondMutation.isPending}
                className={`px-3 py-1.5 text-xs rounded-md ${
                  isRecording
                    ? 'bg-red-100 text-red-700 hover:bg-red-200'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {isRecording ? `Stop (${formatTime(recordingTime)})` : 'Record voice'}
              </button>
              {audioResponseB64 && !isRecording && (
                <>
                  <span className="text-xs text-green-700">Voice response ready</span>
                  <button
                    type="button"
                    className="text-xs text-gray-500 hover:text-gray-700"
                    onClick={() => setAudioResponseB64(null)}
                  >
                    Clear
                  </button>
                </>
              )}
            </div>
            {(audioError || recorderError) && (
              <p className="text-xs text-red-600 mb-2">{audioError || recorderError}</p>
            )}
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={(!response.trim() && !audioResponseB64) || respondMutation.isPending}
                className="flex-1 px-3 py-1.5 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {respondMutation.isPending ? 'Sending...' : 'Reply'}
              </button>
              <button
                type="button"
                onClick={() => dismissMutation.mutate()}
                disabled={dismissMutation.isPending}
                className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md"
              >
                Dismiss
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
