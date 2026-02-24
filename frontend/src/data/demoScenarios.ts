import type { Scenario } from '../types/demo'

export const demoScenarios: Scenario[] = [
  {
    id: 'frank_russo',
    title: 'Frank Russo — COPD',
    description: 'A stoic 61-year-old whose viral URI escalates to a severe COPD exacerbation over 42 days. Tests respiratory monitoring, minimizer detection, and HeAR ambient audio.',
    icon: '🫁',
    sampleTranscript: "Morning, been coughing more than usual the past few days. Nothing major, just that deep chest cough I always get. Used my albuterol once yesterday.",
    demoDataFile: '/demo/frank_russo.json',
    patientAge: 61,
    patientGender: 'Male',
    features: ['SpO2 trend tracking', 'Minimizer detection', 'HeAR ambient audio', 'Static safety responses'],
  },
  {
    id: 'elena_martinez',
    title: 'Elena Martinez — Type 2 Diabetes',
    description: 'A 62-year-old managing T2DM and hypertension whose medication side effects signal hypoglycemia risk. Tests medication safety protocols and adverse event detection.',
    icon: '💊',
    sampleTranscript: "I've been feeling a bit off today. Some nausea after taking my morning meds. Blood sugar was 95 this morning.",
    demoDataFile: '/demo/elena_martinez.json',
    patientAge: 62,
    patientGender: 'Female',
    features: ['Medication safety', 'Hypoglycemia detection', 'Drug interaction checks'],
  },
  {
    id: 'sarah_chen',
    title: 'Sarah Chen — Endometriosis',
    description: 'A 29-year-old with progressively worsening pelvic pain across two irregular cycles. Tests cycle-symptom correlation, NSAID safety, and the Gender Pain Gap thesis.',
    icon: '🔴',
    sampleTranscript: "Period started today, flow is moderate. Having some cramping, maybe 4 out of 10. Took 400mg ibuprofen this morning.",
    demoDataFile: '/demo/sarah_chen.json',
    patientAge: 29,
    patientGender: 'Female',
    features: ['Cycle-symptom correlation', 'NSAID safety override', 'Diagnosis boundary enforcement'],
  },
]
