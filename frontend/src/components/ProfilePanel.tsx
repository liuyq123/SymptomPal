import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { getUserProfile, updateUserProfile } from '../api/client'
import type { ProfileUpdateRequest } from '../api/types'

interface ProfilePanelProps {
  userId: string
}

function normalizeEntry(value: string): string {
  return value.trim()
}

function getIntakeQuestionTarget(): number {
  const raw = Number(import.meta.env.VITE_PROFILE_INTAKE_MAX_QUESTIONS || 8)
  if (!Number.isFinite(raw)) return 8
  return Math.max(1, Math.min(20, Math.trunc(raw)))
}

export default function ProfilePanel({ userId }: ProfilePanelProps) {
  const queryClient = useQueryClient()
  const [conditionInput, setConditionInput] = useState('')
  const [allergyInput, setAllergyInput] = useState('')
  const [regularMedInput, setRegularMedInput] = useState('')
  const [patternInput, setPatternInput] = useState('')
  const [summaryInput, setSummaryInput] = useState('')

  const profileQuery = useQuery({
    queryKey: ['profile', userId],
    queryFn: () => getUserProfile(userId),
  })

  useEffect(() => {
    if (profileQuery.data) {
      setSummaryInput(profileQuery.data.health_summary || '')
    }
  }, [profileQuery.data])

  const profileMutation = useMutation({
    mutationFn: updateUserProfile,
    onSuccess: (data) => {
      queryClient.setQueryData(['profile', userId], data)
      queryClient.invalidateQueries({ queryKey: ['profile', userId] })
    },
  })

  const updateProfile = (patch: Partial<ProfileUpdateRequest>) => {
    profileMutation.mutate({
      user_id: userId,
      ...patch,
    })
  }

  const profile = profileQuery.data
  const intakeTarget = getIntakeQuestionTarget()

  const renderTagList = (
    title: string,
    items: string[] | undefined,
    addPatchKey: string,
    removePatchKey: string,
    inputValue: string,
    setInputValue: (value: string) => void
  ) => (
    <div className="border border-gray-200 rounded-lg p-4">
      <h4 className="font-medium text-gray-800 mb-3">{title}</h4>
      <div className="flex flex-wrap gap-2 mb-3">
        {(items || []).length === 0 && (
          <span className="text-sm text-gray-500">None added</span>
        )}
        {(items || []).map((item) => (
          <span
            key={item}
            className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-gray-100 text-sm text-gray-700"
          >
            {item}
            <button
              className="text-gray-500 hover:text-red-500"
              onClick={() => updateProfile({ [removePatchKey]: [item] })}
              disabled={profileMutation.isPending}
              title={`Remove ${item}`}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md"
          placeholder={`Add ${title.toLowerCase()}...`}
        />
        <button
          className="px-3 py-2 bg-blue-500 text-white rounded-md disabled:opacity-50"
          disabled={profileMutation.isPending || !normalizeEntry(inputValue)}
          onClick={() => {
            const value = normalizeEntry(inputValue)
            if (!value) return
            updateProfile({ [addPatchKey]: [value] })
            setInputValue('')
          }}
        >
          Add
        </button>
      </div>
    </div>
  )

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Health Profile</h3>

        {profileQuery.isLoading && <p className="text-gray-500 text-sm">Loading profile...</p>}
        {profileQuery.isError && (
          <p className="text-red-500 text-sm">
            Error: {profileQuery.error.message}
          </p>
        )}

        {profile && (
          <div className="space-y-4">
            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-gray-800 mb-2">Profile Intake Progress</h4>
              {(() => {
                const answeredCount = Math.max(
                  profile.intake_questions_asked || 0,
                  profile.intake_answered_question_ids?.length || 0
                )
                const completedCount = Math.min(answeredCount, intakeTarget)
                const progressPercent = Math.round((completedCount / intakeTarget) * 100)
                if (profile.intake_completed) {
                  return (
                    <p className="text-sm text-green-700">
                      Completed ({completedCount}/{intakeTarget}). We will use this profile context to personalize follow-ups.
                    </p>
                  )
                }
                return (
                  <div className="space-y-2">
                    <p className="text-sm text-gray-700">
                      In progress: {completedCount}/{intakeTarget} questions answered.
                    </p>
                    <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-2 bg-blue-500 rounded-full"
                        style={{ width: `${progressPercent}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500">
                      Remaining questions will appear in check-ins while you keep logging symptoms.
                    </p>
                  </div>
                )
              })()}
            </div>

            <div className="border border-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-gray-800 mb-2">Health Summary</h4>
              <textarea
                value={summaryInput}
                onChange={(e) => setSummaryInput(e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
                placeholder="Add a concise health summary..."
              />
              <div className="mt-2 flex justify-end">
                <button
                  className="px-3 py-2 bg-blue-500 text-white rounded-md disabled:opacity-50"
                  disabled={profileMutation.isPending}
                  onClick={() =>
                    updateProfile({
                      health_summary: normalizeEntry(summaryInput) || null,
                    })
                  }
                >
                  Save Summary
                </button>
              </div>
            </div>

            {renderTagList(
              'Conditions',
              profile.conditions,
              'add_conditions',
              'remove_conditions',
              conditionInput,
              setConditionInput
            )}

            {renderTagList(
              'Allergies',
              profile.allergies,
              'add_allergies',
              'remove_allergies',
              allergyInput,
              setAllergyInput
            )}

            {renderTagList(
              'Regular Medications',
              profile.regular_medications,
              'add_regular_medications',
              'remove_regular_medications',
              regularMedInput,
              setRegularMedInput
            )}

            {renderTagList(
              'Patterns',
              profile.patterns,
              'add_patterns',
              'remove_patterns',
              patternInput,
              setPatternInput
            )}
          </div>
        )}

        {profileMutation.isError && (
          <p className="mt-3 text-sm text-red-500">
            Error: {profileMutation.error.message}
          </p>
        )}
      </div>
    </div>
  )
}
