import { useState, useRef, useEffect } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ingestVoice, blobToBase64, fileToDataUrl } from '../api/client'
import type { EnhancedIngestResponse } from '../api/types'

interface SymptomInputProps {
  userId: string
  onSubmitComplete: (response: EnhancedIngestResponse) => void
  initialTranscript?: string
  initialPhoto?: string
}

type InputMode = 'voice' | 'text'

export default function SymptomInput({ userId, onSubmitComplete, initialTranscript, initialPhoto }: SymptomInputProps) {
  // Input mode and content
  const [inputMode, setInputMode] = useState<InputMode>(initialTranscript ? 'text' : 'voice')
  const [textDescription, setTextDescription] = useState(initialTranscript || '')

  // Voice recording state
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)

  // Photo state
  const [photoFile, setPhotoFile] = useState<File | null>(null)
  const [photoPreview, setPhotoPreview] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const queryClient = useQueryClient()

  // Cleanup on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  useEffect(() => {
    if (initialPhoto) {
      setPhotoPreview(initialPhoto)
    }
  }, [initialPhoto])

  const mutation = useMutation({
    mutationFn: async () => {
      const request: Parameters<typeof ingestVoice>[0] = {
        user_id: userId,
        recorded_at: new Date().toISOString(),
      }

      // Add audio if recorded
      if (audioBlob) {
        request.audio_b64 = await blobToBase64(audioBlob)
      }

      // Add text description
      if (textDescription.trim()) {
        request.description_text = textDescription.trim()
      }

      // Add photo if selected
      if (photoFile) {
        request.photo_b64 = await fileToDataUrl(photoFile)
      }

      return ingestVoice(request)
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['logs', userId] })
      onSubmitComplete(data)
      // Reset form
      setTextDescription('')
      setAudioBlob(null)
      setPhotoFile(null)
      setPhotoPreview(null)
    },
  })

  // Voice recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach(track => track.stop())
        setAudioBlob(blob)
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)

      timerRef.current = window.setInterval(() => {
        setRecordingTime(t => t + 1)
      }, 1000)
    } catch (err) {
      console.error('Failed to start recording:', err)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }

  const clearRecording = () => {
    setAudioBlob(null)
    setRecordingTime(0)
  }

  // Photo functions
  const handlePhotoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setPhotoFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPhotoPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const clearPhoto = () => {
    setPhotoFile(null)
    setPhotoPreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const canSubmit = audioBlob || textDescription.trim()

  return (
    <div className="bg-white rounded-xl shadow-md p-6 space-y-4">
      <h2 className="text-lg font-semibold text-gray-800">Log Symptoms</h2>

      {/* Input mode toggle */}
      <div className="flex space-x-2">
        <button
          onClick={() => setInputMode('voice')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            inputMode === 'voice'
              ? 'bg-blue-100 text-blue-700 border-2 border-blue-500'
              : 'bg-gray-100 text-gray-600 border-2 border-transparent'
          }`}
        >
          Voice
        </button>
        <button
          onClick={() => setInputMode('text')}
          className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
            inputMode === 'text'
              ? 'bg-blue-100 text-blue-700 border-2 border-blue-500'
              : 'bg-gray-100 text-gray-600 border-2 border-transparent'
          }`}
        >
          Text
        </button>
      </div>

      {/* Voice recording */}
      {inputMode === 'voice' && (
        <div className="flex flex-col items-center space-y-3 py-4">
          {!audioBlob ? (
            <>
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={mutation.isPending}
                className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                  isRecording
                    ? 'bg-red-500 hover:bg-red-600 animate-pulse'
                    : 'bg-blue-500 hover:bg-blue-600'
                } ${mutation.isPending ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {isRecording ? (
                  <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <rect x="6" y="6" width="12" height="12" rx="2" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                    <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                  </svg>
                )}
              </button>
              {isRecording && (
                <span className="text-red-500 font-medium">{formatTime(recordingTime)}</span>
              )}
              {!isRecording && (
                <span className="text-gray-500 text-sm">Tap to record</span>
              )}
            </>
          ) : (
            <div className="flex items-center space-x-3">
              <div className="bg-green-100 text-green-700 px-4 py-2 rounded-lg">
                Recording saved ({formatTime(recordingTime)})
              </div>
              <button
                onClick={clearRecording}
                className="text-red-500 hover:text-red-700"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Text input */}
      {inputMode === 'text' && (
        <textarea
          value={textDescription}
          onChange={(e) => setTextDescription(e.target.value)}
          placeholder="Describe your symptoms... (e.g., Headache started this morning, feels like pressure behind my eyes)"
          aria-label="Describe your symptoms"
          className="w-full h-32 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
          disabled={mutation.isPending}
        />
      )}

      {/* Photo attachment */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Attach Photo (optional)
        </label>
        {!photoPreview ? (
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 transition-colors"
          >
            <svg className="w-8 h-8 mx-auto text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <span className="text-gray-500 text-sm">Tap to add photo</span>
          </div>
        ) : (
          <div className="relative">
            <img
              src={photoPreview}
              alt="Symptom photo"
              className="w-full h-48 object-cover rounded-lg"
            />
            <button
              onClick={clearPhoto}
              aria-label="Remove photo"
              className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full hover:bg-red-600"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handlePhotoSelect}
          className="hidden"
        />
      </div>

      {/* Submit button */}
      <button
        onClick={() => mutation.mutate()}
        disabled={!canSubmit || mutation.isPending}
        className={`w-full py-3 rounded-lg font-medium transition-colors ${
          canSubmit && !mutation.isPending
            ? 'bg-blue-500 text-white hover:bg-blue-600'
            : 'bg-gray-200 text-gray-400 cursor-not-allowed'
        }`}
      >
        {mutation.isPending ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            AI is thinking...
          </span>
        ) : 'Submit Symptoms'}
      </button>

      {mutation.isError && (
        <p className="text-red-500 text-sm text-center">
          Error: {mutation.error.message}
        </p>
      )}
    </div>
  )
}
