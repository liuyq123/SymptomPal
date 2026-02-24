import { demoScenarios } from '../../data/demoScenarios'
import type { Scenario } from '../../types/demo'

interface ScenarioSelectorProps {
  onSelect: (scenario: Scenario) => void
  onExplore?: (scenario: Scenario) => void
  onBack: () => void
}

export default function ScenarioSelector({ onSelect, onExplore, onBack }: ScenarioSelectorProps) {
  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-5xl mx-auto">
        {/* Back button */}
        <button
          onClick={onBack}
          className="mb-6 inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white/75 px-3 py-2 text-slate-700 shadow-sm backdrop-blur hover:bg-white"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>

        {/* Header */}
        <div className="text-center mb-8">
          <h2 className="text-4xl font-bold text-slate-900 mb-3">Select Execution Trace</h2>
          <p className="text-slate-600 text-lg">
            Step through a pre-computed MedGemma 27B conversation trace
          </p>
        </div>

        {/* Scenario grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {demoScenarios.map((scenario) => (
            <ScenarioCard
              key={scenario.id}
              scenario={scenario}
              onReplay={() => onSelect(scenario)}
              onExplore={onExplore ? () => onExplore(scenario) : undefined}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

interface ScenarioCardProps {
  scenario: Scenario
  onReplay: () => void
  onExplore?: () => void
}

function ScenarioCard({ scenario, onReplay, onExplore }: ScenarioCardProps) {
  return (
    <div
      className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white/85 p-6 text-left shadow-md backdrop-blur"
    >
      <div className="pointer-events-none absolute -top-12 right-[-20%] h-28 w-28 rounded-full bg-indigo-200/50 blur-2xl" />

      {/* Icon */}
      <div className="mb-4 text-5xl">{scenario.icon}</div>

      {/* Title */}
      <h3 className="mb-2 text-xl font-semibold text-slate-800">{scenario.title}</h3>

      {/* Demographics */}
      {scenario.patientAge && (
        <p className="mb-2 text-sm text-slate-500">
          {scenario.patientAge}{scenario.patientGender?.toLowerCase() === 'male' ? 'M' : 'F'}
        </p>
      )}

      {/* Description */}
      <p className="mb-4 text-sm text-slate-600">{scenario.description}</p>

      {/* Features */}
      {scenario.features && scenario.features.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-4">
          {scenario.features.map((feature, i) => (
            <span key={i} className="rounded-full border border-cyan-200 bg-cyan-50 px-2 py-0.5 text-xs text-cyan-700">
              {feature}
            </span>
          ))}
        </div>
      )}

      {/* Sample transcript preview */}
      <div className="rounded-lg border border-slate-200 bg-white/80 p-3">
        <p className="mb-1 text-xs text-slate-500">Sample log:</p>
        <p className="line-clamp-2 text-sm italic text-slate-700">
          &ldquo;{scenario.sampleTranscript}&rdquo;
        </p>
      </div>

      {/* Action buttons */}
      <div className="mt-4 flex gap-2">
        <button
          onClick={onReplay}
          className="flex-1 rounded-xl border border-cyan-200 bg-cyan-50 px-3 py-2 text-sm font-semibold text-cyan-700 transition-colors hover:bg-cyan-100"
        >
          Watch Replay
        </button>
        {onExplore && (
          <button
            onClick={onExplore}
            className="flex-1 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-700 px-3 py-2 text-sm font-semibold text-white shadow transition-transform hover:scale-[1.02]"
          >
            Explore App
          </button>
        )}
      </div>
    </div>
  )
}
