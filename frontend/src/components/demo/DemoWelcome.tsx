interface DemoWelcomeProps {
  onNext: () => void
}

export default function DemoWelcome({ onNext }: DemoWelcomeProps) {
  return (
    <div className="min-h-screen px-4 py-12">
      <div className="mx-auto max-w-5xl">
        <div className="relative overflow-hidden rounded-[2rem] border border-slate-200/70 bg-white/80 p-8 shadow-2xl backdrop-blur-xl sm:p-12">
          <div className="pointer-events-none absolute -top-24 right-[-12%] h-64 w-64 rounded-full bg-indigo-300/40 blur-3xl" />
          <div className="pointer-events-none absolute -bottom-24 left-[-12%] h-64 w-64 rounded-full bg-cyan-300/40 blur-3xl" />

          <div className="relative text-center">
            <p className="mb-3 text-xs font-semibold uppercase tracking-[0.2em] text-cyan-700">Hackathon Replay Experience</p>
            <h1 className="text-5xl font-bold text-slate-900 sm:text-6xl">SymptomPal</h1>
            <p className="mx-auto mt-3 max-w-2xl text-base text-slate-600 sm:text-lg">
              Voice-first symptom tracking with AI extraction, safety guardrails, and clinician-ready summaries.
            </p>

            <div className="mx-auto mt-8 max-w-3xl rounded-2xl border border-slate-200 bg-slate-950 px-5 py-4 text-left text-sm leading-relaxed text-slate-100 shadow-xl">
              <span className="mr-1 text-yellow-300">[sandbox]</span>
              To keep judging latency near zero, this view replays pre-computed execution traces.
              Prompt logic, safety checks, and payload structures match our backend pipeline.
            </div>

            <div className="mt-8 grid gap-4 sm:grid-cols-3">
              <FeatureCard
                badge="ASR"
                title="Voice Intake"
                description="Natural language symptom capture and timeline ingestion."
              />
              <FeatureCard
                badge="LLM"
                title="MedGemma Extraction"
                description="Structured symptom, medication, and severity parsing."
              />
              <FeatureCard
                badge="SAFE"
                title="Protocol Safety"
                description="Deterministic guardrails before patient-facing responses."
              />
            </div>

            <button
              onClick={onNext}
              className="mt-10 rounded-full bg-gradient-to-r from-cyan-600 to-indigo-700 px-10 py-4 text-lg font-semibold text-white shadow-xl transition-transform hover:scale-[1.03]"
            >
              Enter Replay &rarr;
            </button>
          </div>
        </div>

        <p className="mt-5 text-center text-xs text-slate-500">
          Demonstration only. Not medical advice.
        </p>
      </div>
    </div>
  )
}

interface FeatureCardProps {
  badge: string
  title: string
  description: string
}

function FeatureCard({ badge, title, description }: FeatureCardProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white/80 p-5 text-left shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-md">
      <span className="inline-flex rounded-full border border-slate-300 bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">
        {badge}
      </span>
      <h3 className="mt-3 text-lg font-semibold text-slate-800">{title}</h3>
      <p className="mt-2 text-sm text-slate-600">{description}</p>
    </div>
  )
}
