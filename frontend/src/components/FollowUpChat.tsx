import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { submitFollowup } from '../api/client'
import type { LogEntry, AgentResponse } from '../api/types'

interface FollowUpChatProps {
  log: LogEntry
  agentResponse?: AgentResponse | null
  onAnswerSubmitted: (updatedLog: LogEntry) => void
}

export default function FollowUpChat({ log, agentResponse, onAnswerSubmitted }: FollowUpChatProps) {
  const [answer, setAnswer] = useState('')
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: () => submitFollowup(log.id, answer, log.user_id),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['logs'] })
      onAnswerSubmitted(data)
      setAnswer('')
    },
  })

  // Format scheduled check-in time
  const formatCheckinTime = (isoString: string) => {
    const date = new Date(isoString)
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  }

  const exchanges = log.followup_exchanges || []
  const answeredExchanges = exchanges.filter(e => e.answer)
  const pendingExchange = exchanges.find(e => !e.answer)

  // Show acknowledgment even if no follow-up question
  const hasAcknowledgment = agentResponse?.acknowledgment
  const hasQuestion = Boolean(pendingExchange)
  const scheduledCheckin = agentResponse?.scheduled_checkin ?? null
  const hasScheduledCheckin = Boolean(scheduledCheckin)

  // Don't render if nothing to show
  if (!hasAcknowledgment && !hasQuestion && answeredExchanges.length === 0) {
    return null
  }

  return (
    <div className="bg-white rounded-xl shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        {hasQuestion ? 'Quick Follow-up' : 'Logged'}
      </h3>

      <div className="space-y-4">
        {/* Acknowledgment bubble (always shown first if present) */}
        {hasAcknowledgment && (
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div className="bg-green-50 rounded-lg p-3 max-w-xs">
              <p className="text-gray-800">{agentResponse.acknowledgment}</p>
            </div>
          </div>
        )}

        {/* Scheduled check-in notice */}
        {hasScheduledCheckin && (
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="bg-purple-50 rounded-lg p-3 max-w-xs">
              <p className="text-gray-800 text-sm">
                I'll check in at {formatCheckinTime(scheduledCheckin!.scheduled_for)} to see how you're doing.
              </p>
            </div>
          </div>
        )}

        {/* Prior answered exchanges (shown as read-only thread) */}
        {answeredExchanges.map((exchange, idx) => (
          <div key={idx} className="space-y-3">
            {/* Agent question */}
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
                <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="bg-blue-50 rounded-lg p-3 max-w-xs">
                <p className="text-gray-800">{exchange.question}</p>
              </div>
            </div>
            {/* Patient answer */}
            <div className="flex items-start space-x-3 justify-end">
              <div className="bg-gray-100 rounded-lg p-3 max-w-xs">
                <p className="text-gray-800">{exchange.answer}</p>
              </div>
            </div>
            {/* Agent response to answer */}
            {exchange.agent_response && (
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center flex-shrink-0">
                  <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div className="bg-green-50 rounded-lg p-3 max-w-xs">
                  <p className="text-gray-800">{exchange.agent_response}</p>
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Current pending question */}
        {pendingExchange && (
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="bg-blue-50 rounded-lg p-3 max-w-xs">
              <p className="text-gray-800">{pendingExchange.question}</p>
            </div>
          </div>
        )}

        {/* Answer input (only show if there's a pending question) */}
        {pendingExchange && (
          <>
            <div className="flex items-center space-x-2">
              <input
                type="text"
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Type your answer..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={mutation.isPending}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && answer.trim()) {
                    mutation.mutate()
                  }
                }}
              />
              <button
                onClick={() => mutation.mutate()}
                disabled={!answer.trim() || mutation.isPending}
                className="px-4 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {mutation.isPending ? '...' : 'Send'}
              </button>
            </div>

            {mutation.isError && (
              <p className="text-red-500 text-sm">
                Error: {mutation.error.message}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}
