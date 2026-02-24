import type { Scenario } from '../../types/demo'

interface FeatureTourProps {
  scenario: Scenario
  onStart: () => void
  onBack: () => void
}

export default function FeatureTour({ scenario, onStart, onBack }: FeatureTourProps) {
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
          <div className="text-5xl mb-3">{scenario.icon}</div>
          <h2 className="text-4xl font-bold text-slate-900 mb-3">
            {scenario.title} Demo
          </h2>
          <p className="text-slate-600 text-lg">{scenario.description}</p>
        </div>

        {/* Feature explanation */}
        <div className="rounded-3xl border border-slate-200 bg-white/85 p-8 shadow-xl backdrop-blur mb-6">
          <h3 className="text-2xl font-semibold text-slate-800 mb-6">What You'll See</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            {/* Timeline View */}
            <div className="rounded-2xl border border-cyan-200 bg-cyan-50/80 p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-cyan-600 text-white font-semibold">
                  1
                </div>
                <h4 className="text-lg font-semibold text-slate-800">Timeline View</h4>
              </div>
              <p className="text-slate-700 mb-4">
                See your symptom progression over time with a chronological timeline.
              </p>
              <ul className="space-y-2 text-sm text-slate-600">
                <li className="flex items-start gap-2">
                  <span className="text-cyan-600">•</span>
                  Day-by-day symptom tracking
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cyan-600">•</span>
                  Medication history with timing
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-cyan-600">•</span>
                  Pattern recognition over time
                </li>
              </ul>
            </div>

            {/* Doctor Report */}
            <div className="rounded-2xl border border-indigo-200 bg-indigo-50/80 p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-600 text-white font-semibold">
                  2
                </div>
                <h4 className="text-lg font-semibold text-slate-800">Doctor Report</h4>
              </div>
              <p className="text-slate-700 mb-4">
                AI-generated clinical summary you can share with your physician.
              </p>
              <ul className="space-y-2 text-sm text-slate-600">
                <li className="flex items-start gap-2">
                  <span className="text-indigo-600">•</span>
                  History of present illness (HPI)
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-600">•</span>
                  Pertinent positives and timeline
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-indigo-600">•</span>
                  Questions for your clinician
                </li>
              </ul>
            </div>
          </div>

          {/* Timeline preview */}
          <div className="mb-6 rounded-2xl border border-slate-200 bg-slate-100/80 p-6">
            <h4 className="mb-4 text-center text-lg font-semibold text-slate-800">
              Timeline Demo Preview
            </h4>
            <div className="rounded-lg border border-slate-200 bg-white p-6">
              <div className="space-y-4">
                <div className="border-l-4 border-cyan-500 pl-4 py-2">
                  <p className="mb-1 text-sm text-slate-600">Day 1 - Monday</p>
                  <p className="font-medium text-slate-800">Headache onset (3/10) -&gt; topiramate</p>
                </div>
                <div className="border-l-4 border-indigo-500 pl-4 py-2">
                  <p className="mb-1 text-sm text-slate-600">Day 1 - Afternoon</p>
                  <p className="font-medium text-slate-800">Worsened to 7/10 -&gt; sumatriptan</p>
                </div>
                <div className="border-l-4 border-emerald-500 pl-4 py-2">
                  <p className="mb-1 text-sm text-slate-600">Day 2 - Tuesday</p>
                  <p className="font-medium text-slate-800">Improving (1/10) - recovering</p>
                </div>
              </div>
            </div>
          </div>

          {/* Sample scenario */}
          <div className="rounded-2xl border border-amber-200 bg-amber-50/90 p-6">
            <h4 className="mb-3 text-lg font-semibold text-slate-800">
              Your Selected Scenario
            </h4>
            <p className="mb-2 italic text-slate-700">"{scenario.sampleTranscript}"</p>
            <p className="text-sm text-slate-600">
              You can use this sample or describe your own symptoms
            </p>
          </div>
        </div>

        {/* Start button */}
        <div className="text-center">
          <button
            onClick={onStart}
            className="rounded-full bg-gradient-to-r from-cyan-600 to-indigo-700 px-12 py-4 text-lg font-semibold text-white shadow-lg transition-transform hover:scale-[1.03]"
          >
            Start Demo &rarr;
          </button>
        </div>
      </div>
    </div>
  )
}
