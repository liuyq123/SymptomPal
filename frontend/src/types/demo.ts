export interface Scenario {
  id: string
  title: string
  description: string
  icon: string
  sampleTranscript: string
  samplePhoto?: string
  /** File path for pre-recorded demo data (e.g. '/demo/frank_russo.json') */
  demoDataFile?: string
  /** Patient demographics for display */
  patientAge?: number
  patientGender?: string
  /** Key features this scenario demonstrates */
  features?: string[]
}

export type DemoStage = 'welcome' | 'scenario' | 'tour' | 'live' | 'replay' | 'explore'
