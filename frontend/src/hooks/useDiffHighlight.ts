import { useEffect, useState } from 'react'
import { diffWords } from 'diff'
import type { LogEntry, SymptomEntity, ActionEntity } from '../api/types'

export interface DiffPart {
  text: string
  type: 'added' | 'removed' | 'unchanged'
}

export interface TagItem {
  label: string
  type: 'added' | 'removed' | 'unchanged'
}

export interface DiffResult {
  transcript: DiffPart[]
  symptoms: TagItem[]
  actions: TagItem[]
  redFlags: TagItem[]
}

const EMPTY: DiffResult = { transcript: [], symptoms: [], actions: [], redFlags: [] }

export function useDiffHighlight(
  previousLog: LogEntry | null,
  currentLog: LogEntry | null
): DiffResult {
  const [diff, setDiff] = useState<DiffResult>(EMPTY)

  useEffect(() => {
    if (!currentLog) {
      setDiff(EMPTY)
      return
    }

    if (!previousLog) {
      setDiff({
        transcript: [{ text: currentLog.transcript, type: 'added' }],
        symptoms: formatSymptoms(currentLog.extracted.symptoms).map(s => ({ label: s, type: 'added' })),
        actions: formatActions(currentLog.extracted.actions_taken).map(a => ({ label: a, type: 'added' })),
        redFlags: currentLog.extracted.red_flags.map(f => ({ label: f, type: 'added' })),
      })
      return
    }

    setDiff({
      transcript: computeTextDiff(previousLog.transcript, currentLog.transcript),
      symptoms: computeTagDiff(
        formatSymptoms(previousLog.extracted.symptoms),
        formatSymptoms(currentLog.extracted.symptoms),
      ),
      actions: computeTagDiff(
        formatActions(previousLog.extracted.actions_taken),
        formatActions(currentLog.extracted.actions_taken),
      ),
      redFlags: computeTagDiff(previousLog.extracted.red_flags, currentLog.extracted.red_flags),
    })
  }, [previousLog, currentLog])

  return diff
}

function formatSymptoms(symptoms: SymptomEntity[]): string[] {
  return symptoms.map(s => {
    const severity = s.severity_1_10 ? ` (${s.severity_1_10}/10)` : ''
    return s.symptom + severity
  })
}

function formatActions(actions: ActionEntity[]): string[] {
  return actions.map(a => {
    const dose = a.dose_text ? ` (${a.dose_text})` : ''
    return a.name + dose
  })
}

function computeTextDiff(oldText: string, newText: string): DiffPart[] {
  return diffWords(oldText, newText).map(part => ({
    text: part.value,
    type: part.added ? 'added' : part.removed ? 'removed' : 'unchanged',
  }))
}

function computeTagDiff(oldItems: string[], newItems: string[]): TagItem[] {
  const kept = newItems.filter(n => oldItems.includes(n))
  const added = newItems.filter(n => !oldItems.includes(n))
  const removed = oldItems.filter(n => !newItems.includes(n))

  return [
    ...kept.map(label => ({ label, type: 'unchanged' as const })),
    ...added.map(label => ({ label, type: 'added' as const })),
    ...removed.map(label => ({ label, type: 'removed' as const })),
  ]
}
