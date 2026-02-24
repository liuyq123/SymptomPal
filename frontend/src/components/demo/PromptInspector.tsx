import { useState } from 'react'
import {
  RESPONSE_SYSTEM_PROMPT,
  SAFETY_CLASSIFIER_PROMPT,
  PROTOCOL_ENGINE_SUMMARY,
} from '../../data/promptTemplates'

interface PromptInspectorProps {
  isOpen: boolean
  onClose: () => void
  initialTab?: 'system' | 'safety' | 'protocol'
}

const TABS = [
  { id: 'system' as const, label: 'System Prompt', source: 'response_generator.py:472' },
  { id: 'safety' as const, label: 'Safety Classifier', source: 'response_generator.py:89' },
  { id: 'protocol' as const, label: 'Protocol Engine', source: 'services/protocols.py' },
]

const CONTENT: Record<string, string> = {
  system: RESPONSE_SYSTEM_PROMPT,
  safety: SAFETY_CLASSIFIER_PROMPT,
  protocol: PROTOCOL_ENGINE_SUMMARY,
}

export default function PromptInspector({ isOpen, onClose, initialTab = 'system' }: PromptInspectorProps) {
  const [activeTab, setActiveTab] = useState(initialTab)

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />

      {/* Modal */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[80vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
          <div>
            <h2 className="text-lg font-bold text-gray-800">Prompt Inspector</h2>
            <p className="text-xs text-gray-500">
              Real prompts from the backend — what MedGemma 27B actually sees
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors text-gray-400 hover:text-gray-600"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tab bar */}
        <div className="px-6 pt-3 flex gap-1 flex-shrink-0">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-2 text-xs font-semibold rounded-t-lg transition-colors ${
                activeTab === tab.id
                  ? 'bg-gray-900 text-green-400'
                  : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Source file badge */}
        <div className="px-6 py-1.5 flex-shrink-0">
          <span className="text-[10px] text-gray-400 font-mono">
            Source: backend/app/{TABS.find(t => t.id === activeTab)?.source}
          </span>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto px-6 pb-6">
          <pre className="bg-gray-900 text-green-400 font-mono text-xs p-5 rounded-lg leading-relaxed whitespace-pre-wrap break-words">
            {CONTENT[activeTab]}
          </pre>
        </div>
      </div>
    </div>
  )
}
