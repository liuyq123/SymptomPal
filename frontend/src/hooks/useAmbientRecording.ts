import { useState, useRef, useCallback, useEffect } from 'react'
import type {
  SessionType,
  AmbientSession,
  AmbientEvent,
  AmbientSessionResult,
} from '../api/types'
import {
  startAmbientSession,
  uploadAmbientChunk,
  endAmbientSession,
  cancelAmbientSession,
  blobToBase64,
} from '../api/client'

export type RecordingState = 'idle' | 'recording' | 'processing' | 'completed' | 'error'

interface UseAmbientRecordingOptions {
  userId: string
  onError?: (error: Error) => void
}

interface UseAmbientRecordingReturn {
  state: RecordingState
  session: AmbientSession | null
  result: AmbientSessionResult | null
  events: AmbientEvent[]
  elapsedSeconds: number
  chunkCount: number
  error: string | null
  startSession: (sessionType: SessionType, label?: string) => Promise<void>
  stopSession: () => Promise<void>
  cancelSession: () => Promise<void>
  reset: () => void
}

const CHUNK_INTERVAL_MS = 30000 // 30 seconds

export function useAmbientRecording({
  userId,
  onError,
}: UseAmbientRecordingOptions): UseAmbientRecordingReturn {
  const [state, setState] = useState<RecordingState>('idle')
  const [session, setSession] = useState<AmbientSession | null>(null)
  const [result, setResult] = useState<AmbientSessionResult | null>(null)
  const [events, setEvents] = useState<AmbientEvent[]>([])
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [chunkCount, setChunkCount] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const chunkIndexRef = useRef(0)
  const uploadIntervalRef = useRef<number | null>(null)
  const timerIntervalRef = useRef<number | null>(null)
  const sessionRef = useRef<AmbientSession | null>(null)
  const chunkStartRef = useRef<number | null>(null)
  const uploadInProgressRef = useRef(false)
  const uploadPendingRef = useRef(false)
  const uploadPromiseRef = useRef<Promise<void> | null>(null)

  // Keep sessionRef in sync
  useEffect(() => {
    sessionRef.current = session
  }, [session])

  const cleanup = useCallback(() => {
    if (uploadIntervalRef.current) {
      clearInterval(uploadIntervalRef.current)
      uploadIntervalRef.current = null
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current)
      timerIntervalRef.current = null
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    mediaRecorderRef.current = null
    chunksRef.current = []
    chunkStartRef.current = null
    uploadInProgressRef.current = false
    uploadPendingRef.current = false
    uploadPromiseRef.current = null
  }, [])

  const cancelSessionSilently = useCallback(async (sessionId: string) => {
    try {
      await cancelAmbientSession(sessionId, userId)
    } catch {
      // Ignore cleanup failures.
    }
  }, [userId])

  const uploadCurrentChunk = useCallback((): Promise<void> => {
    if (uploadInProgressRef.current) {
      uploadPendingRef.current = true
      return uploadPromiseRef.current || Promise.resolve()
    }

    uploadInProgressRef.current = true
    const uploadPromise = (async () => {
      while (true) {
        uploadPendingRef.current = false

        const currentSession = sessionRef.current
        if (!currentSession || chunksRef.current.length === 0) {
          break
        }

        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        chunksRef.current = []

        if (blob.size === 0) {
          break
        }

        const now = performance.now()
        const lastStart = chunkStartRef.current ?? now
        const durationSeconds = Math.max(0, (now - lastStart) / 1000)
        chunkStartRef.current = now

        try {
          const audioB64 = await blobToBase64(blob)
          const response = await uploadAmbientChunk({
            session_id: currentSession.id,
            user_id: userId,
            chunk_index: chunkIndexRef.current,
            audio_b64: audioB64,
            duration_seconds: durationSeconds,
          })

          chunkIndexRef.current += 1
          setChunkCount(prev => prev + 1)
          setEvents(prev => [...prev, ...response.events_detected])
        } catch (err) {
          console.error('Failed to upload chunk:', err)
          // Don't fail the whole session on chunk upload error
        }

        if (!uploadPendingRef.current) {
          break
        }
      }
    })()

    uploadPromiseRef.current = uploadPromise
    return uploadPromise.finally(() => {
      if (uploadPromiseRef.current === uploadPromise) {
        uploadPromiseRef.current = null
      }
      uploadInProgressRef.current = false
    })
  }, [userId])

  const startSession = useCallback(async (sessionType: SessionType, label?: string) => {
    setError(null)
    setEvents([])
    setResult(null)
    setChunkCount(0)
    setElapsedSeconds(0)
    chunkIndexRef.current = 0

    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Start backend session
      const response = await startAmbientSession({
        user_id: userId,
        session_type: sessionType,
        label: label || null,
      })

      sessionRef.current = response.session

      // Set up MediaRecorder
      let mediaRecorder: MediaRecorder
      try {
        mediaRecorder = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=opus',
        })
      } catch (err) {
        await cancelSessionSilently(response.session.id)
        throw err
      }

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onerror = () => {
        setError('Recording error occurred')
        setState('error')
        cancelSessionSilently(response.session.id)
        cleanup()
      }

      chunkStartRef.current = performance.now()
      // Start recording with timeslice for regular data
      try {
        mediaRecorder.start(1000) // Get data every second
      } catch (err) {
        await cancelSessionSilently(response.session.id)
        throw err
      }

      mediaRecorderRef.current = mediaRecorder
      setSession(response.session)
      setState('recording')

      // Set up chunk upload interval
      uploadIntervalRef.current = window.setInterval(() => {
        uploadCurrentChunk()
      }, CHUNK_INTERVAL_MS)

      // Set up elapsed time timer
      timerIntervalRef.current = window.setInterval(() => {
        setElapsedSeconds(prev => prev + 1)
      }, 1000)

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start session'
      setError(message)
      setState('error')
      onError?.(err instanceof Error ? err : new Error(message))
      cleanup()
    }
  }, [userId, cleanup, uploadCurrentChunk, onError, cancelSessionSilently])

  const waitForRecorderStop = useCallback((recorder: MediaRecorder) => {
    if (recorder.state === 'inactive') {
      return Promise.resolve()
    }
    return new Promise<void>((resolve) => {
      const handleStop = () => {
        recorder.removeEventListener('stop', handleStop)
        resolve()
      }
      recorder.addEventListener('stop', handleStop)
    })
  }, [])

  const stopSession = useCallback(async () => {
    if (!session) return

    setState('processing')

    if (uploadIntervalRef.current) {
      clearInterval(uploadIntervalRef.current)
      uploadIntervalRef.current = null
    }

    // Stop recording and upload final chunk
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      const recorder = mediaRecorderRef.current
      const stopPromise = waitForRecorderStop(recorder)
      mediaRecorderRef.current.stop()
      await stopPromise
    }

    // Upload any remaining chunks
    await uploadCurrentChunk()

    cleanup()

    try {
      const response = await endAmbientSession({
        session_id: session.id,
        user_id: userId,
      })

      setSession(response.session)
      setResult(response.result)
      setState('completed')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to end session'
      setError(message)
      setState('error')
      onError?.(err instanceof Error ? err : new Error(message))
    }
  }, [session, userId, cleanup, uploadCurrentChunk, onError, waitForRecorderStop])

  const cancelCurrentSession = useCallback(async () => {
    if (!session) return

    cleanup()

    try {
      await cancelAmbientSession(session.id, userId)
      setSession(null)
      setState('idle')
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to cancel session'
      setError(message)
      setState('error')
    }
  }, [session, userId, cleanup])

  const reset = useCallback(() => {
    cleanup()
    setState('idle')
    setSession(null)
    setResult(null)
    setEvents([])
    setElapsedSeconds(0)
    setChunkCount(0)
    setError(null)
    chunkIndexRef.current = 0
  }, [cleanup])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup()
    }
  }, [cleanup])

  return {
    state,
    session,
    result,
    events,
    elapsedSeconds,
    chunkCount,
    error,
    startSession,
    stopSession,
    cancelSession: cancelCurrentSession,
    reset,
  }
}
