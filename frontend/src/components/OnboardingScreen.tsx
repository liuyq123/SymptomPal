import { useState, useEffect, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import {
  startOnboarding,
  getPendingCheckins,
  respondToCheckin,
} from '../api/client'
import type { ScheduledCheckin } from '../api/types'

interface OnboardingScreenProps {
  userId: string
  onComplete: () => void
}

const TOTAL_QUESTIONS = 8

export default function OnboardingScreen({ userId, onComplete }: OnboardingScreenProps) {
  const [currentCheckin, setCurrentCheckin] = useState<ScheduledCheckin | null>(null)
  const [answer, setAnswer] = useState('')
  const [questionsAnswered, setQuestionsAnswered] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchNextCheckin = useCallback(async () => {
    // Poll for the next intake checkin
    for (let attempt = 0; attempt < 10; attempt++) {
      const checkins = await getPendingCheckins(userId)
      const intake = checkins.find((c) => c.checkin_type === 'profile_intake')
      if (intake) {
        setCurrentCheckin(intake)
        setLoading(false)
        return
      }
      // Small delay between polls to let backend create next checkin
      await new Promise((r) => setTimeout(r, 500))
    }
    // No more intake checkins — onboarding is complete
    setLoading(false)
    onComplete()
  }, [userId, onComplete])

  // Initialize: call startOnboarding then fetch first checkin
  useEffect(() => {
    let cancelled = false
    async function init() {
      try {
        const result = await startOnboarding(userId)
        if (cancelled) return
        if (result.status === 'completed') {
          onComplete()
          return
        }
        await fetchNextCheckin()
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to start onboarding')
          setLoading(false)
        }
      }
    }
    init()
    return () => { cancelled = true }
  }, [userId, onComplete, fetchNextCheckin])

  const respondMutation = useMutation({
    mutationFn: () => {
      if (!currentCheckin) throw new Error('No active checkin')
      return respondToCheckin(currentCheckin.id, answer, userId)
    },
    onSuccess: async () => {
      const next = questionsAnswered + 1
      setQuestionsAnswered(next)
      setAnswer('')
      setCurrentCheckin(null)

      if (next >= TOTAL_QUESTIONS) {
        onComplete()
        return
      }

      setLoading(true)
      try {
        await fetchNextCheckin()
      } catch {
        onComplete()
      }
    },
    onError: (err: Error) => {
      setError(err.message)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (answer.trim()) {
      setError(null)
      respondMutation.mutate()
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-cyan-50 to-indigo-50 px-4">
      <div className="w-full max-w-lg">
        {/* Logo + title */}
        <div className="text-center mb-8">
          <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-cyan-600 to-indigo-700 text-lg font-semibold text-white shadow-lg mb-4">
            SP
          </div>
          <h1 className="text-2xl font-bold text-slate-900">Welcome to SymptomPal</h1>
          <p className="mt-2 text-sm text-slate-600">
            Let's set up your health profile so we can personalize your experience.
          </p>
        </div>

        {/* Progress bar */}
        <div className="mb-6">
          <div className="flex justify-between text-xs font-medium text-slate-500 mb-1.5">
            <span>Question {Math.min(questionsAnswered + 1, TOTAL_QUESTIONS)} of {TOTAL_QUESTIONS}</span>
            <span>{Math.round((questionsAnswered / TOTAL_QUESTIONS) * 100)}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-200">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-cyan-500 to-indigo-600 transition-all duration-500"
              style={{ width: `${(questionsAnswered / TOTAL_QUESTIONS) * 100}%` }}
            />
          </div>
        </div>

        {/* Question card */}
        <div className="glass-panel-strong rounded-2xl p-6">
          {loading ? (
            <div className="py-8 text-center text-slate-500">Loading...</div>
          ) : error ? (
            <div className="py-8 text-center">
              <p className="text-red-600 text-sm mb-3">{error}</p>
              <button
                onClick={() => { setError(null); onComplete() }}
                className="text-sm text-slate-600 hover:text-slate-800 underline"
              >
                Skip onboarding
              </button>
            </div>
          ) : currentCheckin ? (
            <form onSubmit={handleSubmit}>
              <p className="text-base font-medium text-slate-900 mb-4">
                {currentCheckin.message}
              </p>
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Type your answer..."
                rows={3}
                className="w-full px-4 py-3 text-sm border border-slate-300 rounded-xl focus:ring-2 focus:ring-cyan-500 focus:border-transparent resize-none"
                disabled={respondMutation.isPending}
                autoFocus
              />
              <div className="flex items-center justify-between mt-4">
                <button
                  type="button"
                  onClick={() => {
                    setAnswer('none')
                    setTimeout(() => respondMutation.mutate(), 0)
                  }}
                  disabled={respondMutation.isPending}
                  className="text-sm text-slate-500 hover:text-slate-700"
                >
                  Skip this question
                </button>
                <button
                  type="submit"
                  disabled={!answer.trim() || respondMutation.isPending}
                  className="px-6 py-2.5 text-sm font-semibold bg-gradient-to-r from-cyan-600 to-indigo-700 text-white rounded-xl shadow-lg hover:scale-[1.02] transition-transform disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
                >
                  {respondMutation.isPending ? 'Saving...' : 'Continue'}
                </button>
              </div>
            </form>
          ) : (
            <div className="py-8 text-center text-slate-500">
              Setting up your profile...
            </div>
          )}
        </div>

        <p className="mt-6 text-center text-xs text-slate-400">
          Your answers help the agent provide better, personalized responses.
        </p>
      </div>
    </div>
  )
}
