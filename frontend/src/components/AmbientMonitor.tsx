import { useState, useEffect, useRef } from 'react'
import type { SessionType, AmbientSession, AmbientSessionResult, AmbientEvent } from '../api/types'
import { useAmbientRecording } from '../hooks/useAmbientRecording'
import {
  getAmbientSessions,
  getAmbientSessionResult,
  startAmbientSession,
  uploadAmbientChunk,
  endAmbientSession,
} from '../api/client'

interface AmbientMonitorProps {
  userId: string
}

const SESSION_TYPE_INFO: Record<SessionType, { label: string; icon: string; description: string }> = {
  cough_monitor: {
    label: 'Cough Tracker',
    icon: '🫁',
    description: 'Count and analyze coughing episodes',
  },
  sleep: {
    label: 'Breath Monitor',
    icon: '🌬️',
    description: 'Track breathing patterns',
  },
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

function formatDateTime(isoString: string): string {
  const date = new Date(isoString)
  return date.toLocaleString()
}

type InputMode = 'microphone' | 'file'

export default function AmbientMonitor({ userId }: AmbientMonitorProps) {
  const [selectedType, setSelectedType] = useState<SessionType>('cough_monitor')
  const [pastSessions, setPastSessions] = useState<AmbientSession[]>([])
  const [loadingSessions, setLoadingSessions] = useState(false)
  const [inputMode, setInputMode] = useState<InputMode>('file') // Default to file for easier testing

  // File upload state
  const [fileUploadState, setFileUploadState] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'error'>('idle')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedChunks, setUploadedChunks] = useState(0)
  const [totalChunks, setTotalChunks] = useState(0)
  const [fileEvents, setFileEvents] = useState<AmbientEvent[]>([])
  const [fileResult, setFileResult] = useState<AmbientSessionResult | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Past session viewing state
  const [viewingPastSession, setViewingPastSession] = useState<AmbientSession | null>(null)
  const [pastSessionResult, setPastSessionResult] = useState<AmbientSessionResult | null>(null)
  const [loadingPastResult, setLoadingPastResult] = useState(false)
  const [pastResultError, setPastResultError] = useState<string | null>(null)

  const {
    state,
    session,
    result,
    events,
    elapsedSeconds,
    chunkCount,
    error,
    startSession,
    stopSession,
    cancelSession,
    reset,
  } = useAmbientRecording({ userId })

  // Load past sessions on mount and after completing a session
  useEffect(() => {
    const loadSessions = async () => {
      setLoadingSessions(true)
      try {
        const sessions = await getAmbientSessions(userId)
        setPastSessions(sessions)
      } catch {
        // Silently fail - sessions list is not critical
      } finally {
        setLoadingSessions(false)
      }
    }
    loadSessions()
  }, [userId, state === 'completed'])

  const handleStart = async () => {
    await startSession(selectedType)
  }

  // File upload handling
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setFileError(null)
    }
  }

  const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
    const bytes = new Uint8Array(buffer)
    let binary = ''
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i])
    }
    return btoa(binary)
  }

  const handleFileUpload = async () => {
    if (!selectedFile) return

    setFileUploadState('uploading')
    setFileError(null)
    setFileEvents([])
    setFileResult(null)
    setUploadProgress(0)
    setUploadedChunks(0)

    try {
      // Read file as ArrayBuffer
      const arrayBuffer = await selectedFile.arrayBuffer()
      const audioBytes = new Uint8Array(arrayBuffer)

      // Parse WAV header for duration
      let duration = 0
      let sampleRate = 44100
      let numChannels = 1
      let bitsPerSample = 16

      // Check for WAV header (RIFF)
      const isWav = audioBytes[0] === 0x52 && audioBytes[1] === 0x49 &&
                    audioBytes[2] === 0x46 && audioBytes[3] === 0x46

      if (isWav && audioBytes.length > 44) {
        // Parse WAV header
        const dataView = new DataView(arrayBuffer)
        numChannels = dataView.getUint16(22, true)
        sampleRate = dataView.getUint32(24, true)
        bitsPerSample = dataView.getUint16(34, true)

        // Find data chunk size
        let dataSize = 0
        for (let i = 36; i < Math.min(arrayBuffer.byteLength - 8, 100); i++) {
          if (audioBytes[i] === 0x64 && audioBytes[i + 1] === 0x61 &&
              audioBytes[i + 2] === 0x74 && audioBytes[i + 3] === 0x61) {
            dataSize = dataView.getUint32(i + 4, true)
            break
          }
        }

        if (dataSize > 0) {
          const bytesPerSample = (bitsPerSample / 8) * numChannels
          const numSamples = dataSize / bytesPerSample
          duration = numSamples / sampleRate
        }
      }

      // Fallback duration estimate based on file size
      if (duration <= 0) {
        // Assume 16-bit stereo at 44100Hz
        duration = arrayBuffer.byteLength / (44100 * 2 * 2)
      }

      // Chunk size: approximately 10 seconds worth of audio
      const chunkDuration = 10 // seconds
      const bytesPerSecond = arrayBuffer.byteLength / duration
      const chunkSize = Math.ceil(bytesPerSecond * chunkDuration)
      const numChunks = Math.ceil(arrayBuffer.byteLength / chunkSize)

      setTotalChunks(numChunks)

      // Start backend session
      const sessionResponse = await startAmbientSession({
        user_id: userId,
        session_type: selectedType,
        label: `File upload: ${selectedFile.name}`,
      })

      const sessionId = sessionResponse.session.id
      const allEvents: AmbientEvent[] = []

      // Upload chunks
      for (let i = 0; i < numChunks; i++) {
        const start = i * chunkSize
        const end = Math.min(start + chunkSize, arrayBuffer.byteLength)
        const chunkBytes = audioBytes.slice(start, end)

        // For WAV files, we need to include the header for each chunk for proper decoding
        // However, the backend should handle raw audio too
        // Let's send the complete file as base64 for the first chunk to keep header
        let chunkB64: string
        let chunkDur: number

        if (i === 0 && isWav) {
          // Send entire file for first chunk to preserve WAV header
          chunkB64 = arrayBufferToBase64(arrayBuffer)
          chunkDur = duration
          // Skip remaining chunks since we sent the whole file
          setTotalChunks(1)
        } else if (i === 0) {
          // Non-WAV file, send as is
          chunkB64 = arrayBufferToBase64(arrayBuffer)
          chunkDur = duration
          setTotalChunks(1)
        } else {
          // For chunked upload (not used with WAV since we send whole file)
          chunkB64 = arrayBufferToBase64(chunkBytes.buffer as ArrayBuffer)
          chunkDur = (end - start) / bytesPerSecond
        }

        try {
          const chunkResponse = await uploadAmbientChunk({
            session_id: sessionId,
            user_id: userId,
            chunk_index: i,
            audio_b64: chunkB64,
            duration_seconds: chunkDur,
          })

          allEvents.push(...chunkResponse.events_detected)
          setFileEvents([...allEvents])
        } catch (err) {
          console.error(`Chunk ${i} upload failed:`, err)
        }

        setUploadedChunks(i + 1)
        setUploadProgress(((i + 1) / (i === 0 ? 1 : numChunks)) * 100)

        // Break after first chunk for WAV files (sent whole file)
        if (i === 0 && isWav) break
      }

      // End session and get results
      setFileUploadState('processing')
      const endResponse = await endAmbientSession({
        session_id: sessionId,
        user_id: userId,
      })

      setFileResult(endResponse.result)
      setFileUploadState('completed')

      // Refresh past sessions
      const sessions = await getAmbientSessions(userId)
      setPastSessions(sessions)

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Upload failed'
      setFileError(message)
      setFileUploadState('error')
    }
  }

  const resetFileUpload = () => {
    setFileUploadState('idle')
    setSelectedFile(null)
    setFileEvents([])
    setFileResult(null)
    setFileError(null)
    setUploadProgress(0)
    setUploadedChunks(0)
    setTotalChunks(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleStop = async () => {
    await stopSession()
  }

  const handleCancel = async () => {
    await cancelSession()
  }

  const handleNewSession = () => {
    reset()
  }

  // Handler for viewing past session results
  const handleViewPastSession = async (session: AmbientSession) => {
    if (session.status !== 'completed') {
      return // Only completed sessions have results
    }

    setViewingPastSession(session)
    setLoadingPastResult(true)
    setPastResultError(null)
    setPastSessionResult(null)

    try {
      const result = await getAmbientSessionResult(session.id, userId)
      setPastSessionResult(result)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load session results'
      setPastResultError(message)
    } finally {
      setLoadingPastResult(false)
    }
  }

  const handleClosePastSession = () => {
    setViewingPastSession(null)
    setPastSessionResult(null)
    setPastResultError(null)
  }

  // Viewing past session result
  if (viewingPastSession) {
    if (loadingPastResult) {
      return (
        <div className="bg-white rounded-xl shadow-md p-6 text-center">
          <div className="animate-spin w-12 h-12 border-4 border-indigo-200 border-t-indigo-600 rounded-full mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-800">
            Loading Session Results...
          </h3>
        </div>
      )
    }

    if (pastResultError) {
      return (
        <div className="bg-white rounded-xl shadow-md p-6 text-center">
          <div className="text-4xl mb-4">⚠️</div>
          <h3 className="text-lg font-semibold text-red-600 mb-2">
            Failed to Load Results
          </h3>
          <p className="text-gray-600 mb-4">{pastResultError}</p>
          <button
            onClick={handleClosePastSession}
            className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
          >
            Back to Sessions
          </button>
        </div>
      )
    }

    if (pastSessionResult) {
      return (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-md p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-500">
                Viewing past session from {formatDateTime(viewingPastSession.started_at)}
              </div>
              <button
                onClick={handleClosePastSession}
                className="text-indigo-600 hover:text-indigo-700 text-sm font-medium"
              >
                ← Back to Sessions
              </button>
            </div>
          </div>

          <ResultsDisplay result={pastSessionResult} />

          <button
            onClick={handleClosePastSession}
            className="w-full py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
          >
            Back to Sessions
          </button>
        </div>
      )
    }
  }

  // File upload states
  if (fileUploadState === 'uploading') {
    return (
      <div className="bg-white rounded-xl shadow-md p-6">
        <div className="text-center mb-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-full mb-4">
            <span className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
            Uploading
          </div>

          <h3 className="text-lg font-semibold text-gray-800">
            Processing {selectedFile?.name}
          </h3>

          <div className="w-full bg-gray-200 rounded-full h-4 mt-4">
            <div
              className="bg-indigo-600 h-4 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>

          <div className="text-sm text-gray-500 mt-2">
            Chunk {uploadedChunks} of {totalChunks}
          </div>
        </div>

        {fileEvents.length > 0 && (
          <div className="mb-4">
            <p className="text-sm text-gray-500 mb-2">Events Detected:</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(
                fileEvents.reduce((acc, e) => {
                  acc[e.event_type] = (acc[e.event_type] || 0) + 1
                  return acc
                }, {} as Record<string, number>)
              ).map(([type, count]) => (
                <span
                  key={type}
                  className="px-3 py-1 bg-indigo-100 text-indigo-800 rounded-full text-sm"
                >
                  {type.replace(/_/g, ' ')}: {count}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  if (fileUploadState === 'processing') {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 text-center">
        <div className="animate-spin w-12 h-12 border-4 border-indigo-200 border-t-indigo-600 rounded-full mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-800">
          Analyzing Results...
        </h3>
        <p className="text-gray-500 mt-2">
          Processing {fileEvents.length} detected events
        </p>
      </div>
    )
  }

  if (fileUploadState === 'completed' && fileResult) {
    return (
      <div className="space-y-6">
        <ResultsDisplay result={fileResult} />

        <button
          onClick={resetFileUpload}
          className="w-full py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
        >
          Upload Another File
        </button>
      </div>
    )
  }

  if (fileUploadState === 'error') {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 text-center">
        <div className="text-4xl mb-4">Error</div>
        <h3 className="text-lg font-semibold text-red-600 mb-2">
          Upload Failed
        </h3>
        <p className="text-gray-600 mb-4">{fileError || 'Something went wrong'}</p>
        <button
          onClick={resetFileUpload}
          className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
        >
          Try Again
        </button>
      </div>
    )
  }

  // Idle state - show session type selection
  if (state === 'idle' && fileUploadState === 'idle') {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Ambient Audio Monitor
          </h3>
          <p className="text-gray-600 mb-4">
            Select a monitoring type and start a session to track health sounds.
          </p>

          {/* Input mode toggle */}
          <div className="flex rounded-lg bg-gray-100 p-1 mb-6">
            <button
              onClick={() => setInputMode('microphone')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                inputMode === 'microphone'
                  ? 'bg-white text-indigo-600 shadow'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Live Microphone
            </button>
            <button
              onClick={() => setInputMode('file')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                inputMode === 'file'
                  ? 'bg-white text-indigo-600 shadow'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Upload Audio File
            </button>
          </div>

          {/* Session type selection */}
          <div className="grid grid-cols-2 gap-3 mb-6">
            {(Object.keys(SESSION_TYPE_INFO) as SessionType[]).map(type => {
              const info = SESSION_TYPE_INFO[type]
              const isSelected = selectedType === type
              return (
                <button
                  key={type}
                  onClick={() => setSelectedType(type)}
                  className={`p-4 rounded-lg border-2 text-left transition-all ${
                    isSelected
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="text-2xl mb-1">{info.icon}</div>
                  <div className="font-medium text-gray-800">{info.label}</div>
                  <div className="text-xs text-gray-500">{info.description}</div>
                </button>
              )
            })}
          </div>

          {inputMode === 'microphone' ? (
            <>
              {/* Start button for microphone */}
              <button
                onClick={handleStart}
                className="w-full py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
              >
                Start Monitoring
              </button>

              {error && (
                <p className="mt-4 text-red-600 text-sm">{error}</p>
              )}

              <p className="mt-3 text-xs text-gray-500 text-center">
                Note: Live microphone requires HTTPS or localhost access
              </p>
            </>
          ) : (
            <>
              {/* File upload section */}
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav,audio/wav"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="audio-file-input"
                />
                <label
                  htmlFor="audio-file-input"
                  className="cursor-pointer"
                >
                  <div className="text-4xl mb-2">Upload</div>
                  <p className="text-gray-600 mb-2">
                    {selectedFile ? selectedFile.name : 'Click to select a WAV file'}
                  </p>
                  {selectedFile && (
                    <p className="text-sm text-gray-500">
                      Size: {(selectedFile.size / 1024).toFixed(1)} KB
                    </p>
                  )}
                </label>
              </div>

              <button
                onClick={handleFileUpload}
                disabled={!selectedFile}
                className={`w-full py-3 mt-4 rounded-lg font-medium transition-colors ${
                  selectedFile
                    ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                Analyze Audio File
              </button>

              {fileError && (
                <p className="mt-4 text-red-600 text-sm">{fileError}</p>
              )}
            </>
          )}
        </div>

        {/* Past sessions list */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Recent Sessions
          </h3>
          {loadingSessions ? (
            <p className="text-gray-500">Loading...</p>
          ) : pastSessions.length === 0 ? (
            <p className="text-gray-500">No sessions yet. Start one above!</p>
          ) : (
            <div className="space-y-3">
              {pastSessions.slice(0, 5).map(s => {
                const info = SESSION_TYPE_INFO[s.session_type]
                const isCompleted = s.status === 'completed'
                return (
                  <button
                    key={s.id}
                    onClick={() => isCompleted && handleViewPastSession(s)}
                    disabled={!isCompleted}
                    className={`w-full flex items-center justify-between p-3 bg-gray-50 rounded-lg text-left transition-colors ${
                      isCompleted ? 'hover:bg-gray-100 cursor-pointer' : 'cursor-default'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <span className="text-xl">{info?.icon || '🔊'}</span>
                      <div>
                        <div className="font-medium text-gray-800">
                          {info?.label || s.session_type}
                        </div>
                        <div className="text-xs text-gray-500">
                          {formatDateTime(s.started_at)}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-sm font-medium ${
                        s.status === 'completed' ? 'text-green-600' :
                        s.status === 'cancelled' ? 'text-gray-500' :
                        'text-yellow-600'
                      }`}>
                        {s.status}
                      </div>
                      <div className="text-xs text-gray-500">
                        {Math.round(s.total_duration_seconds / 60)}min
                        {isCompleted && ' • View results →'}
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </div>
    )
  }

  // Recording state
  if (state === 'recording') {
    const eventCounts: Record<string, number> = {}
    events.forEach(e => {
      eventCounts[e.event_type] = (eventCounts[e.event_type] || 0) + 1
    })

    return (
      <div className="bg-white rounded-xl shadow-md p-6">
        <div className="text-center mb-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-red-100 text-red-700 rounded-full mb-4">
            <span className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
            Recording
          </div>

          <h3 className="text-lg font-semibold text-gray-800">
            {SESSION_TYPE_INFO[session?.session_type || 'cough_monitor'].label}
          </h3>

          <div className="text-4xl font-mono text-gray-800 mt-4">
            {formatDuration(elapsedSeconds)}
          </div>

          <div className="text-sm text-gray-500 mt-2">
            {chunkCount} chunks uploaded
          </div>
        </div>

        {/* Live event counts */}
        {Object.keys(eventCounts).length > 0 && (
          <div className="mb-6">
            <p className="text-sm text-gray-500 mb-2">Events Detected:</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(eventCounts).map(([type, count]) => (
                <span
                  key={type}
                  className="px-3 py-1 bg-indigo-100 text-indigo-800 rounded-full text-sm"
                >
                  {type.replace(/_/g, ' ')}: {count}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Control buttons */}
        <div className="flex gap-3">
          <button
            onClick={handleCancel}
            className="flex-1 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleStop}
            className="flex-1 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
          >
            Stop & Get Results
          </button>
        </div>
      </div>
    )
  }

  // Processing state
  if (state === 'processing') {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 text-center">
        <div className="animate-spin w-12 h-12 border-4 border-indigo-200 border-t-indigo-600 rounded-full mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-gray-800">
          Processing Results...
        </h3>
        <p className="text-gray-500 mt-2">
          Analyzing {events.length} detected events
        </p>
      </div>
    )
  }

  // Completed state - show results
  if (state === 'completed' && result) {
    return (
      <div className="space-y-6">
        <ResultsDisplay result={result} />

        <button
          onClick={handleNewSession}
          className="w-full py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
        >
          Start New Session
        </button>
      </div>
    )
  }

  // Error state
  if (state === 'error') {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 text-center">
        <div className="text-4xl mb-4">⚠️</div>
        <h3 className="text-lg font-semibold text-red-600 mb-2">
          Error
        </h3>
        <p className="text-gray-600 mb-4">{error || 'Something went wrong'}</p>
        <button
          onClick={reset}
          className="px-6 py-2 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
        >
          Try Again
        </button>
      </div>
    )
  }

  return null
}

// Results display component
function ResultsDisplay({ result }: { result: AmbientSessionResult }) {
  const info = SESSION_TYPE_INFO[result.session_type]

  return (
    <div className="bg-white rounded-xl shadow-md p-6">
      <div className="flex items-center gap-3 mb-4">
        <span className="text-3xl">{info?.icon || '🔊'}</span>
        <div>
          <h3 className="text-lg font-semibold text-gray-800">
            {info?.label || result.session_type} Results
          </h3>
          <p className="text-sm text-gray-500">
            Duration: {result.duration_minutes.toFixed(1)} minutes
          </p>
        </div>
      </div>

      {/* Summary */}
      <div className="p-4 bg-gray-50 rounded-lg mb-4">
        <p className="text-gray-800">{result.summary}</p>
      </div>

      {/* Cough metrics */}
      {result.cough_metrics && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <MetricCard
            label="Total Coughs"
            value={result.cough_metrics.total_coughs.toString()}
          />
          <MetricCard
            label="Coughs/Hour"
            value={result.cough_metrics.coughs_per_hour.toFixed(1)}
          />
          {result.cough_metrics.peak_cough_period && (
            <MetricCard
              label="Peak Period"
              value={result.cough_metrics.peak_cough_period}
              span={2}
            />
          )}
        </div>
      )}

      {/* Cough Classification Results (observational only — no diagnosis) */}
      {result.cough_metrics && (result.cough_metrics.dominant_cough_type || result.cough_metrics.dominant_severity) && (
        <div className="mb-4">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Classification Analysis</h4>
          <div className="grid grid-cols-2 gap-3">
            {result.cough_metrics.dominant_cough_type && (
              <MetricCard
                label="Cough Type"
                value={result.cough_metrics.dominant_cough_type}
              />
            )}
            {result.cough_metrics.cough_type_confidence && (
              <MetricCard
                label="Type Confidence"
                value={`${(result.cough_metrics.cough_type_confidence * 100).toFixed(0)}%`}
              />
            )}
            {result.cough_metrics.dominant_severity && (
              <MetricCard
                label="Severity"
                value={result.cough_metrics.dominant_severity}
                highlight={
                  result.cough_metrics.dominant_severity === 'mild' ? 'green' :
                  result.cough_metrics.dominant_severity === 'moderate' ? 'yellow' : 'red'
                }
              />
            )}
            {result.cough_metrics.severity_confidence && (
              <MetricCard
                label="Severity Confidence"
                value={`${(result.cough_metrics.severity_confidence * 100).toFixed(0)}%`}
              />
            )}
          </div>
        </div>
      )}

      {/* Sleep quality metrics */}
      {result.sleep_quality && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <MetricCard
            label="Quality Rating"
            value={result.sleep_quality.quality_rating.toUpperCase()}
            highlight={
              result.sleep_quality.quality_rating === 'excellent' ? 'green' :
              result.sleep_quality.quality_rating === 'good' ? 'blue' :
              result.sleep_quality.quality_rating === 'fair' ? 'yellow' : 'red'
            }
          />
          <MetricCard
            label="Breathing Score"
            value={`${result.sleep_quality.breathing_regularity_score.toFixed(0)}%`}
          />
          <MetricCard
            label="Apnea Events"
            value={result.sleep_quality.apnea_events.toString()}
            highlight={result.sleep_quality.apnea_events > 5 ? 'red' : undefined}
          />
          <MetricCard
            label="Snoring"
            value={`${result.sleep_quality.snoring_minutes.toFixed(0)} min`}
          />
        </div>
      )}

      {/* Voice biomarkers */}
      {result.voice_biomarkers && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <MetricCard
            label="Stress Level"
            value={`${result.voice_biomarkers.stress_level.toFixed(0)}%`}
            highlight={result.voice_biomarkers.stress_level > 60 ? 'red' : undefined}
          />
          <MetricCard
            label="Fatigue Level"
            value={`${result.voice_biomarkers.fatigue_level.toFixed(0)}%`}
            highlight={result.voice_biomarkers.fatigue_level > 60 ? 'yellow' : undefined}
          />
          <MetricCard
            label="Congestion"
            value={result.voice_biomarkers.congestion_detected ? 'Detected' : 'None'}
            highlight={result.voice_biomarkers.congestion_detected ? 'yellow' : 'green'}
          />
          <MetricCard
            label="Voice Clarity"
            value={`${result.voice_biomarkers.voice_clarity_score.toFixed(0)}%`}
          />
        </div>
      )}

      {/* Event timeline (collapsed by default) */}
      {result.events_timeline.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-sm text-indigo-600 hover:text-indigo-700">
            View {result.events_timeline.length} detected events
          </summary>
          <div className="mt-2 max-h-48 overflow-y-auto space-y-1">
            {result.events_timeline.map(event => (
              <div
                key={event.id}
                className="text-sm p-2 bg-gray-50 rounded flex justify-between"
              >
                <span className="text-gray-800">
                  {event.event_type.replace(/_/g, ' ')}
                </span>
                <span className="text-gray-500">
                  {(event.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  )
}

// Metric card component
function MetricCard({
  label,
  value,
  highlight,
  span = 1,
}: {
  label: string
  value: string
  highlight?: 'green' | 'blue' | 'yellow' | 'red'
  span?: number
}) {
  const highlightClasses = {
    green: 'bg-green-100 text-green-800',
    blue: 'bg-blue-100 text-blue-800',
    yellow: 'bg-yellow-100 text-yellow-800',
    red: 'bg-red-100 text-red-800',
  }

  return (
    <div
      className={`p-3 rounded-lg ${
        highlight ? highlightClasses[highlight] : 'bg-gray-100'
      } ${span === 2 ? 'col-span-2' : ''}`}
    >
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  )
}
