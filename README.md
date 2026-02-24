# SymptomPal

**Voice-first AI symptom tracking that bridges the gap between doctor visits.**

Patients with chronic diseases live with their condition every day but see their doctor a few times a year. The gap between visits relies on patient memory — a notoriously unreliable tool. SymptomPal changes this: patients speak naturally, and deploys MedGemma, HeAR, and MedSigLIP as callable tools to extract structured clinical data, detects dangerous trends days before crisis, and generates clinician-ready pre-visit packets.

**[Live Demo on Hugging Face](https://huggingface.co/spaces/yuqiliu/SymptomPal)**

## What It Does

- **Voice/Text/Photo Input** — Patients speak naturally; MedGemma extracts structured symptoms, medications, vital signs
- **Ambient Monitoring** — HeAR detects cough frequency, sleep apnea, and voice biomarkers from passive audio
- **Photo Analysis** — MedSigLIP classifies skin lesions and tracks progression across visits
- **Proactive Follow-up** — Deterministic clinical protocols evaluate MedGemma's extraction and select the single most critical missing question; MedGemma can autonomously invoke the Watchdog when it detects a pattern signal
- **Doctor Packet** — One-tap clinician-ready HPI (OLDCARTS format) with pertinent +/-, medication interactions, allergy verification
- **Longitudinal Memory** — 30-day raw logs + structured health profile + rolling baseline compression for multi-year tracking

## HAI-DEF Models Used

| Model | Role | Integration |
|-------|------|-------------|
| **MedGemma** | Symptom extraction (tool #1), contextual response + tool-calling (tool #2), safety validation (tool #3), longitudinal pattern detection (tool #4), clinical summarization | Local GPU (5-bit quantized via Ollama) |
| **MedASR** | Medical speech recognition (105M params, lower WER on medical terms) | HuggingFace |
| **HeAR** | Respiratory sound classification: lung sound F1 0.805 (ICBHI 2017 + SPRSound, 23,116 segments), cough type F1 0.663 (COUGHVID, 2,819 segments) | HuggingFace embeddings + acoustic features |
| **MedSigLIP** | Skin lesion classification (5 parallel classifiers) + progression tracking | HuggingFace zero-shot |

## Quick Start

```bash
git clone https://github.com/yuqiliu/SymptomPal.git
cd SymptomPal
```

### Demo Only (no GPU required)

The demo replays pre-computed execution traces — no backend or model access needed. Start the frontend and open http://localhost:5173?demo=true.

### Full Setup (with MedGemma)

#### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com/)
- GPU with ~20GB VRAM (e.g., RTX 4090)

#### 1. Pull MedGemma

```bash
ollama pull hf.co/unsloth/medgemma-27b-it-GGUF:Q5_K_M
```

#### 2. Start backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export API_KEY=dev_local_key
export USE_OLLAMA_MEDGEMMA=true
uvicorn app.main:app --reload --port 8000
```

#### 3. Start frontend (in a separate terminal)

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 to use the full application. HeAR, MedASR, and MedSigLIP model weights are downloaded from HuggingFace automatically on first use.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_KEY` | API key for backend auth | Required |
| `USE_OLLAMA_MEDGEMMA` | Enable Ollama MedGemma | `false` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `hf.co/unsloth/medgemma-27b-it-GGUF:Q5_K_M` |
| `MEDGEMMA_TEMPERATURE` | LLM temperature | `0.1` |
| `HF_TOKEN` | HuggingFace token (for gated models) | Optional |
| `USE_STUB_MEDGEMMA` | Use keyword stub instead of LLM | `false` |

## Demo Scenarios

| Scenario | Duration | Key Features Demonstrated |
|----------|----------|---------------------------|
| **Frank Russo** (COPD) | 21 logs, 42 days | SpO2 decline tracking, ambient acoustic monitoring, minimizer detection, static safety responses, drug-disease interaction |
| **Elena Martinez** (T2DM) | 20 logs, 30 days | Medication interaction detection, hypoglycemia crisis, contextual reasoning over history, profile evolution |
| **Sarah Chen** (Iron-deficiency anemia) | 26 logs, 73 days | Cycle-symptom correlation, drug-interaction detection (ibuprofen + iron + anemia), diagnosis boundary enforcement, Watchdog-triggered pattern alerting |

## Architecture

```
┌──────────────────────────────────────────────┐
│  React Frontend (Vite + TypeScript + Tailwind)│
│  Tabs: Record | Monitor | Meds | Cycle |     │
│        Profile | Timeline | Doctor           │
└──────────────────┬───────────────────────────┘
                   │ REST API
┌──────────────────▼───────────────────────────┐
│  FastAPI Backend                              │
│  ├─ /api/ingest/{voice,image}                 │
│  ├─ /api/checkins, /api/medications           │
│  ├─ /api/summarize, /api/ambient              │
│  ├─ /api/logs, /api/cycle, /api/profile       │
├──────────────────────────────────────────────┤
│  Models: MedGemma | HeAR | MedSigLIP | MedASR│
│  Safety: Protocols | Red flags | Validation  │
│          Watchdog | Clinician Alerts          │
├──────────────────────────────────────────────┤
│  SQLite (local, no central server)            │
└──────────────────────────────────────────────┘
```

### Safety System (Defense-in-Depth)

1. **Protocol-first** — Deterministic protocols evaluate before LLM; override if matched
2. **Red flag detection** — Standalone triggers (chest pain, severe headache, high fever) + co-occurrence rules
3. **LLM output validation** — Secondary MedGemma call screens for medication advice, diagnoses, definitive causation
4. **Clinician alerts** — Escalation-worthy patterns generate persistent notes with hedging language
5. **Watchdog Diagnostic Boundary** — 3-field architecture: private `internal_clinical_rationale` (logged, never stored), `safe_patient_nudge` (non-diagnostic), `clinician_facing_observation` (objective pattern for doctor packet)
6. **No silent degradation** — Models unavailable → clear error, not keyword fallback

## Running Tests

```bash
cd backend
pip install pytest pytest-asyncio
pytest
```

---

**Demonstration only.** This app does not provide medical advice, diagnosis, or treatment. No PHI storage — for production, implement proper data security measures.
