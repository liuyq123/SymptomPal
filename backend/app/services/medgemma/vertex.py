"""
Vertex AI MedGemma Client - Real MedGemma implementation using Google Vertex AI.

Requires environment variables:
- GCP_PROJECT_ID: Your Google Cloud project ID
- GCP_REGION: Region where endpoint is deployed (e.g., us-central1)
- MEDGEMMA_ENDPOINT_ID: The Vertex AI endpoint ID for MedGemma
"""

import os
import re
import json
from typing import List, Optional, Dict, Any

from .base import MedGemmaClient, _load_vertex_ai, _aiplatform, _Endpoint, logger
from datetime import datetime, timezone
from ...models import (
    ExtractionResult,
    MenstrualStatus,
    SymptomEntity,
    ActionEntity,
    VitalSignEntry,
    DoctorPacket,
    TimelineReveal,
    TimelinePoint,
    LogEntry,
    WatchdogResult,
)


class VertexAIMedGemmaClient(MedGemmaClient):
    """
    Real MedGemma implementation using Google Vertex AI.

    Requires environment variables:
    - GCP_PROJECT_ID: Your Google Cloud project ID
    - GCP_REGION: Region where endpoint is deployed (e.g., us-central1)
    - MEDGEMMA_ENDPOINT_ID: The Vertex AI endpoint ID for MedGemma
    """

    EXTRACTION_PROMPT = """<role>
You are a medical assistant extracting comprehensive clinical information from patient symptom descriptions.
</role>

<task>
Extract ALL clinically relevant information explicitly stated in the transcript. Be thorough - capture every symptom, medication, severity rating, location, and qualifier mentioned.
</task>

<constraints>
- Extract ONLY what the patient explicitly states - no inference or assumptions
- Capture ALL symptoms mentioned, including secondary symptoms (nausea, light sensitivity, etc.)
- Capture ALL medications with their doses if mentioned
- If a patient mentions taking MULTIPLE pills/tablets/capsules of the same medication, calculate
  the TOTAL ingested dose (e.g., "two 500mg pills" = "1000mg", "three puffs" = "3 puffs")
- If a medication is mentioned but the dose is NOT stated, you MUST include "dose" in the
  missing_fields array so the agent can ask the user for the dose
- Include severity ratings whether stated precisely or approximately (e.g., "7 out of 10" = 7, "about a 4" = 4, "maybe a 6" = 6)
- Include body locations (e.g., "left side of head")
- Include symptom character/quality (e.g., "throbbing", "dull", "sharp")
- Include sensitivities (light, sound, smell) as separate symptoms
- If information is not mentioned, use null
- For actions_taken, include medications, treatments, AND food/drinks used as treatment (e.g., juice for low sugar)
- Do NOT include rest, lifestyle actions, or "skipped" medications in actions_taken
- If the patient says they SKIPPED a medication, do NOT include it — they did not take it
- VITAL SIGNS AND MEASUREMENTS are NOT symptoms — extract them in the "vital_signs" field, NOT in "symptoms":
  Blood sugar/glucose readings, blood pressure, temperature, heart rate, weight, oxygen saturation (SpO2)
- If a patient reports IMPROVEMENT or RESOLUTION of a symptom (e.g., "sleeping better", "finally sleeping again", "less pain", "breathing is back to normal"), do NOT extract it as an active symptom. Only extract symptoms that are currently present and bothering the patient.
- For symptom names, use CONCISE CANONICAL medical terms — NOT the patient's exact colloquial phrasing:
  - "knee's been bothering me" → symptom: "knee pain"
  - "I've been coughing up phlegm" → symptom: "productive cough"
  - "feeling winded" → symptom: "shortness of breath"
  - "knee's still bad" → symptom: "knee pain" (same canonical name as previous visit)
  - Always re-use the same canonical name for the same underlying symptom
</constraints>

<examples>
What goes in actions_taken vs what does NOT:
- "Took ibuprofen 400mg" → actions_taken: [{{"name": "ibuprofen", "dose_text": "400mg"}}]
- "Took my lisinopril and metformin" → actions_taken: [{{"name": "lisinopril"}}, {{"name": "metformin"}}]
- "Drank juice to raise my sugar" → actions_taken: [{{"name": "juice"}}]  (treatment action)
- "Applied ice pack to my knee" → actions_taken: [{{"name": "ice pack"}}]  (treatment action)
- "Skipped my evening sertraline" → actions_taken: []  (skipped = did NOT take)
- "Rested in a dark room" → actions_taken: []  (rest is not a treatment)
- "Talked to my doctor's office" → actions_taken: []  (consultation is not a treatment)

Dosage math — always calculate total ingested dose:
- "I took two of my 500mg naproxen pills" → actions_taken: [{{"name": "naproxen", "dose_text": "1000mg"}}]
- "Took three puffs of my inhaler" → actions_taken: [{{"name": "inhaler", "dose_text": "3 puffs"}}]
- "Took two 25mg metoprolol tablets" → actions_taken: [{{"name": "metoprolol", "dose_text": "50mg"}}]
- "I took my insulin" → actions_taken: [{{"name": "insulin", "dose_text": null}}], missing_fields: ["dose"]

Vital signs go in vital_signs, NOT symptoms:
- "Fasting sugar 118 today" → symptoms: [], vital_signs: [{{"name": "blood sugar", "value": "118", "unit": "mg/dL"}}]
- "SpO2 was 96" → symptoms: [], vital_signs: [{{"name": "spo2", "value": "96", "unit": "%"}}]
- "Blood pressure 148 over 92" → symptoms: [], vital_signs: [{{"name": "blood pressure", "value": "148/92", "unit": "mmHg"}}]
- "Temp was 101.2" → symptoms: [], vital_signs: [{{"name": "temperature", "value": "101.2", "unit": "F"}}]
- "Heart rate was 88" → symptoms: [], vital_signs: [{{"name": "heart rate", "value": "88", "unit": "bpm"}}]

Menstrual status — detect if patient is currently on their period:
- "Period started this morning, heavy flow, cramps are a 7" → menstrual_status: {{"is_period_day": true, "flow_level": "heavy"}}
- "Day 2 of my period, just spotting now" → menstrual_status: {{"is_period_day": true, "flow_level": "spotting"}}
- "My last period was 2 weeks ago, now I have pelvic pain" → menstrual_status: {{"is_period_day": false}}
- No mention of period → omit menstrual_status or set is_period_day: false
</examples>

<transcript>
{transcript}
</transcript>

<output_format>
{{
    "symptoms": [
        {{
            "symptom": "symptom name (e.g., 'headache', 'nausea', 'light sensitivity', 'vomiting')",
            "location": "body location if mentioned (e.g., 'left side of head')" or null,
            "character": "quality/character if mentioned (e.g., 'throbbing', 'dull', 'sharp')" or null,
            "severity_1_10": number 1-10 if explicitly stated or null,
            "onset_time_text": "exact timing words" or null,
            "duration_text": "duration if mentioned" or null,
            "triggers": ["triggers if mentioned"],
            "relievers": ["what helps if mentioned"],
            "associated_symptoms": ["co-occurring symptoms"]
        }}
    ],
    "actions_taken": [
        {{
            "name": "medication name ONLY (e.g., 'ibuprofen', 'metformin', 'tylenol')",
            "dose_text": "dose with units (e.g., '100mg', '2 puffs')" or null,
            "effect_text": "effect if described" or null
        }}
    ],
    "vital_signs": [
        {{
            "name": "measurement name (e.g., 'blood sugar', 'spo2', 'blood pressure', 'temperature', 'heart rate', 'weight')",
            "value": "reading as stated (e.g., '118', '96', '148/92')",
            "unit": "unit if known (e.g., 'mg/dL', '%', 'mmHg', 'F', 'bpm', 'lbs')" or null
        }}
    ],
    "missing_fields": ["severity", "onset", "duration", "dose" - only if truly missing],
    "red_flags": ["emergency symptoms like chest pain, difficulty breathing"],
    "menstrual_status": {{
        "is_period_day": true or false,
        "flow_level": "spotting" or "light" or "medium" or "heavy" or null
    }}
}}
</output_format>

Extract from the transcript above. Respond with valid JSON only — no commentary, no examples, no explanation."""

    DOCTOR_PACKET_PROMPT = """<role>
You are a medical assistant preparing a comprehensive pre-visit summary for a Primary Care Physician.
</role>

<task>
Create a clinically useful HPI (History of Present Illness) from the patient's symptom logs. The HPI should follow the OLDCARTS format and include ALL relevant clinical details.
</task>

<patient_profile>
{patient_profile}
</patient_profile>

<logs>
Dates in [YYYY-MM-DD] brackets are authoritative timestamps.
{logs_text}
</logs>

<system_flags>
Note: dates in system flags may be approximate. Always use the log timestamps above for exact dates.
{system_flags}
</system_flags>

<instructions>
1. HPI MUST start with patient demographics extracted from the patient_profile section above. Copy the EXACT age number from the profile — if profile says "Age: 61", write "61-year-old", NOT "62-year-old". Write actual gender and conditions. NEVER use bracket placeholders like [age], [gender], [conditions]. Cross-check your HPI demographics against the profile before responding.
2. Use medication names and doses from the patient_profile section. NEVER say "[age not provided]", "[dose not specified]", or "[medication not mentioned]" — all information is available in the profile above.
3. HPI should follow OLDCARTS: Onset, Location, Duration, Character, Aggravating factors, Relieving factors, Timing, Severity
4. Note symptom progression (e.g., "worsened from 3/10 to 9/10 over 3 days")
5. Include functional impact (e.g., "had to stay in dark room", "missed work")
6. Include any image findings as pertinent positives
7. If system_flags are provided, include them verbatim in a "system_longitudinal_flags" array.
8. Analyze the patient's conversation patterns to infer implicit concerns. Look for:
   - Topics mentioned repeatedly or circled back to across multiple logs
   - Emotional language (scared, worried, frustrated, confused)
   - Functional impacts the patient emphasizes (missing work, can't exercise, affecting relationships)
   - Questions the patient asks or almost asks but doesn't fully articulate
   - Symptoms the patient seems particularly anxious about (even if clinically minor)
   Frame each concern from the patient's perspective, as they might phrase it internally — NOT as clinical questions a doctor would ask.
</instructions>

<output_format>
Respond with valid JSON:
{{
    "hpi": "Detailed clinical paragraph starting with patient demographics from profile. Use OLDCARTS format. Include severity progression (X/10), all medications with doses from profile, and functional impact.",
    "pertinent_positives": ["All symptoms present including: main symptoms, severity ratings, associated symptoms, medications with doses from profile"],
    "questions_for_clinician": ["Inferred patient concerns from conversation patterns — frame from patient perspective, e.g. 'My periods keep getting worse and I'm scared this won't stop' rather than 'Describe the character of pain'"],
    "system_longitudinal_flags": ["Verbatim system flag observations, if any were provided in system_flags"]
}}
</output_format>

IMPORTANT: Be thorough. Include severity numbers (X/10), all medication doses FROM THE PROFILE, and use the patient's actual age and conditions."""

    TIMELINE_PROMPT = """You are a medical assistant creating a symptom timeline.

Analyze these symptom logs from the past {days} days:
{logs_text}

Create meaningful timeline story points. For each significant event, provide:
- A label (e.g., "Onset", "Peak", "Improvement", "Treatment Started")
- Brief details about what happened

Respond with valid JSON only:
{{
    "story_points": [
        {{"timestamp": "ISO datetime string", "label": "Event label", "details": "What happened"}}
    ]
}}"""

    PROFILE_UPDATE_PROMPT = """You are a medical assistant analyzing a patient's symptom history to update their health profile.

The health profile is a long-term memory that helps personalize future interactions.

Current profile:
{current_profile}

Recent symptom logs (last 30 days):
{logs_text}

Analyze the logs and suggest updates to the profile. Only suggest additions when there is clear evidence:
- Chronic/recurring conditions: patterns that repeat over 3+ weeks (e.g., "Recurrent Migraines", "Chronic Back Pain")
- Observed patterns: triggers or relationships you notice (e.g., "Migraines triggered by stress", "Cough worse at night")
- DO NOT add conditions for one-off symptoms

Respond with valid JSON:
{{
    "add_conditions": ["only add if clear recurring pattern"],
    "add_patterns": ["observed triggers or relationships"],
    "health_summary": "Brief 1-2 sentence summary of their health patterns (or null if no update needed)"
}}

If no updates are needed, return empty arrays and null summary."""

    BASELINE_COMPRESSION_PROMPT = """You are an elite clinical archivist. Your task is to update a patient's long-term historical baseline by incorporating a new batch of older logs.

<existing_baseline>
{current_baseline}
</existing_baseline>

<older_logs_to_merge>
{logs_text}
</older_logs_to_merge>

INSTRUCTIONS:
1. Merge the new logs into the existing baseline.
2. Preserve all longitudinal tracking statistics (e.g., "Persistent lower back pain with intermittent flares through the reporting period").
3. Do not drop previously established chronic patterns or medication interactions.
4. Output ONLY the updated clinical baseline paragraph (max 4-6 sentences). It must be highly dense and purely objective."""

    WATCHDOG_PROMPT = """<system_role>
You are a backstage Clinical Watchdog AI. Your job is to silently monitor a patient's
longitudinal symptom timeline and detect hidden, undiagnosed, or deteriorating medical
conditions (e.g., undiagnosed thyroid disorder, progressive cardiac decompensation,
medication-induced hepatotoxicity, evolving autoimmune condition, occult malignancy).
</system_role>

<rules>
1. Evaluate the timeline for patterns of severity, frequency, or temporal correlation
   that a patient might ignore but a clinician would find highly actionable.
2. CRITICAL SAFETY: You are a background observer. You must NEVER reveal a specific
   diagnosis to the patient in safe_patient_nudge.
3. If you detect a concerning pattern, formulate a safe nudge that highlights the
   OBJECTIVE PATTERN (e.g., "your fatigue and joint stiffness have been escalating
   over the past three weeks") and recommends generating a Doctor Packet to share
   with a physician.
4. If nothing concerning is found, set concerning_pattern_detected to false.
5. If concerning, also produce a clinician_facing_observation: a dense, objective clinical
   summary using precise medical terminology that describes the EXACT data pattern (severity
   scores, temporal correlations, treatment response) WITHOUT naming any suspected diagnosis.
   This will be injected into the Doctor Packet for the physician.
</rules>

<input_timeline>
{history_context}
</input_timeline>

Respond with valid JSON matching this schema:
{{
  "concerning_pattern_detected": boolean,
  "internal_clinical_rationale": "Your private diagnostic reasoning. Name suspected diseases here.",
  "safe_patient_nudge": "The non-diagnostic, data-driven message for the patient, or null if nothing concerning.",
  "clinician_facing_observation": "Dense objective medical terminology summarizing the exact pattern WITHOUT naming the disease. Or null if nothing concerning."
}}"""

    _DEDICATED_DNS_RE = re.compile(
        r"dedicated domain name\s*[\"'](?:https?://)?([^\"'/]+\.prediction\.vertexai\.goog)/?[\"']",
        flags=re.IGNORECASE,
    )

    def __init__(self):
        super().__init__()
        self._endpoint = None
        self._initialized = False
        self._project_id = os.environ.get("GCP_PROJECT_ID")
        self._region = os.environ.get("GCP_REGION", "us-central1")
        self._endpoint_id = os.environ.get("MEDGEMMA_ENDPOINT_ID")
        self._dedicated_endpoint_dns: Optional[str] = self._normalize_dns(
            os.environ.get("MEDGEMMA_DEDICATED_ENDPOINT_DNS", "")
        )

    @staticmethod
    def _normalize_dns(value: str) -> Optional[str]:
        token = (value or "").strip()
        if not token:
            return None
        token = token.removeprefix("https://").removeprefix("http://")
        token = token.strip().strip("/")
        return token or None

    @classmethod
    def _extract_dedicated_dns_from_error(cls, error: Exception) -> Optional[str]:
        msg = str(error).strip()
        if not msg:
            return None
        match = cls._DEDICATED_DNS_RE.search(msg)
        if not match:
            return None
        return cls._normalize_dns(match.group(1))

    def _predict_via_dedicated_dns(
        self,
        dedicated_dns: str,
        *,
        instance: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Call predict against a dedicated endpoint DNS.

        Some Vertex endpoints reject requests against the shared
        `aiplatform.googleapis.com` hostname and require the dedicated DNS.
        """
        if not self._ensure_initialized():
            raise RuntimeError("MedGemma endpoint not available")

        from google.cloud.aiplatform import constants
        from google.auth.transport import requests as google_auth_requests

        credentials = self._endpoint.credentials
        # Match aiplatform Endpoint.predict() scoping behavior.
        try:
            credentials._scopes = constants.base.DEFAULT_AUTHED_SCOPES
        except Exception:
            pass

        session = google_auth_requests.AuthorizedSession(credentials)
        endpoint_resource = getattr(getattr(self._endpoint, "_gca_resource", None), "name", None)
        endpoint_resource = endpoint_resource or self._endpoint.resource_name

        body: Dict[str, Any] = {"instances": [instance]}
        if parameters is not None:
            body["parameters"] = parameters

        url = f"https://{dedicated_dns}/v1/{endpoint_resource}:predict"
        response = session.post(
            url=url,
            data=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise ValueError(
                "Failed to make prediction request via dedicated endpoint DNS. "
                f"Status code: {response.status_code}, response: {response.text}."
            )
        payload = response.json()
        predictions = payload.get("predictions") or []
        return predictions

    def _ensure_initialized(self) -> bool:
        """Initialize Vertex AI endpoint connection."""
        if self._initialized:
            return self._endpoint is not None

        self._initialized = True

        if not self._project_id or not self._endpoint_id:
            logger.warning("GCP_PROJECT_ID or MEDGEMMA_ENDPOINT_ID not set")
            return False

        if not _load_vertex_ai():
            return False

        try:
            # Re-import module-level references after lazy loading
            from .base import _aiplatform as aiplatform_mod, _Endpoint as Endpoint_cls
            aiplatform_mod.init(project=self._project_id, location=self._region)
            self._endpoint = Endpoint_cls(
                endpoint_name=self._endpoint_id,
                project=self._project_id,
                location=self._region,
            )
            logger.info(f"MedGemma Vertex AI endpoint initialized: {self._endpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MedGemma endpoint: {e}")
            return False

    def _clean_output(self, text: str) -> str:
        """Clean MedGemma output by removing thinking tokens."""
        # Remove MedGemma thinking tokens (special tokens used for chain-of-thought)
        cleaned = re.sub(r'<unused94>.*?</unused95>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    def _extract_prediction(self, pred) -> str:
        """Extract text content from a single Vertex AI prediction."""
        if isinstance(pred, str):
            if "Output:" in pred:
                pred = pred.split("Output:", 1)[1].strip()
            return self._clean_output(pred)
        if isinstance(pred, dict):
            for key in ("text", "generated_text", "output", "content"):
                if key in pred:
                    return self._clean_output(pred.get(key, ""))
            candidates = pred.get("candidates")
            if isinstance(candidates, list) and candidates:
                cand0 = candidates[0]
                if isinstance(cand0, dict):
                    for key in ("text", "generated_text", "output", "content"):
                        if key in cand0:
                            return self._clean_output(cand0.get(key, ""))
        return self._clean_output(str(pred))

    def _call_endpoint(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call the Vertex AI endpoint with a prompt."""
        if not self._ensure_initialized():
            raise RuntimeError("MedGemma endpoint not available")

        # Vertex endpoints differ in serving schema depending on deployment type.
        # responseMimeType enforces structured JSON output at the decoding layer,
        # eliminating the need for regex-based JSON extraction from free text.
        temperature = float(os.environ.get("MEDGEMMA_TEMPERATURE", "0.1"))
        seed_env = os.environ.get("MEDGEMMA_SEED")
        seed = int(seed_env) if seed_env is not None else None

        # vLLM instance
        vllm_instance = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if seed is not None:
            vllm_instance["seed"] = seed

        # Gemini params
        gemini_params = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "responseMimeType": "application/json",
        }
        if seed is not None:
            gemini_params["seed"] = seed

        # TGI params
        tgi_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "response_mime_type": "application/json",
        }
        if seed is not None:
            tgi_params["seed"] = seed

        attempts = [
            # vLLM Model Garden deployments: params embedded in instance
            (vllm_instance, None),
            # Vertex AI publisher models / Gemini API (camelCase params)
            ({"prompt": prompt}, gemini_params),
            # Custom deployments using HuggingFace TGI (snake_case params)
            ({"inputs": prompt}, tgi_params),
        ]

        # Prefer the aiplatform Endpoint client, but detect "dedicated DNS required" failures
        # and retry via the dedicated hostname when necessary.
        last_schema_error: Exception | None = None
        dedicated_dns = self._dedicated_endpoint_dns
        for instance, parameters in attempts:
            try:
                if dedicated_dns:
                    predictions = self._predict_via_dedicated_dns(
                        dedicated_dns,
                        instance=instance,
                        parameters=parameters,
                    )
                else:
                    if parameters is None:
                        response = self._endpoint.predict(instances=[instance])
                    else:
                        response = self._endpoint.predict(instances=[instance], parameters=parameters)
                    predictions = response.predictions

                if not predictions:
                    raise ValueError("Empty predictions from Vertex AI endpoint")

                result = self._extract_prediction(predictions[0])
                if result:
                    return result
                # Schema is correct (we got predictions) but content is empty.
                # Don't retry with a different schema — raise RuntimeError to
                # skip the schema-retry loop.
                raise RuntimeError("Empty content in Vertex AI prediction")
            except Exception as e:
                parsed_dns = self._extract_dedicated_dns_from_error(e)
                if parsed_dns and not dedicated_dns:
                    logger.info(
                        "Vertex endpoint requires dedicated DNS; retrying via %s",
                        parsed_dns,
                    )
                    dedicated_dns = parsed_dns
                    self._dedicated_endpoint_dns = parsed_dns
                    try:
                        predictions = self._predict_via_dedicated_dns(
                            dedicated_dns,
                            instance=instance,
                            parameters=parameters,
                        )
                        if not predictions:
                            raise ValueError("Empty predictions from Vertex AI endpoint (dedicated DNS)")
                        result = self._extract_prediction(predictions[0])
                        if result:
                            return result
                        raise RuntimeError("Empty content in Vertex AI prediction (dedicated DNS)")
                    except Exception as retry_error:
                        e = retry_error

                # Retry only when the request shape is likely wrong.
                if type(e).__name__ in ("ValueError", "InvalidArgument", "FailedPrecondition"):
                    last_schema_error = e
                    continue
                raise

        if last_schema_error is not None:
            raise last_schema_error
        raise RuntimeError("MedGemma endpoint predict call failed with unknown error")

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from model response, stripping markdown fences and prefixes."""
        text = response.strip()
        if not text:
            raise ValueError("Empty LLM response")
        # Strip markdown code fences anywhere in the response
        fence_match = re.search(r'```(?:json)?\s*', text)
        if fence_match:
            text = text[fence_match.end():]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        # If no fence, find the first { or [ to start JSON
        elif not text.startswith(("{", "[")):
            brace = text.find("{")
            bracket = text.find("[")
            starts = [i for i in (brace, bracket) if i >= 0]
            if starts:
                text = text[min(starts):]
        return json.loads(text)

    async def extract(self, transcript: str) -> ExtractionResult:
        """Extract structured symptom information using MedGemma."""
        if not self._ensure_initialized():
            raise RuntimeError("MedGemma endpoint not initialized. Check GCP_PROJECT_ID and MEDGEMMA_ENDPOINT_ID.")

        prompt = self.EXTRACTION_PROMPT.format(transcript=transcript)
        response = self._call_endpoint(prompt)
        data = self._parse_json_response(response)

        symptoms = [
            SymptomEntity(
                symptom=s.get("symptom", "unknown"),
                location=s.get("location"),
                character=s.get("character"),
                severity_1_10=s.get("severity_1_10"),
                onset_time_text=s.get("onset_time_text"),
                duration_text=s.get("duration_text"),
                triggers=s.get("triggers", []),
                relievers=s.get("relievers", []),
                associated_symptoms=s.get("associated_symptoms", []),
            )
            for s in data.get("symptoms", [])
        ]

        actions = [
            ActionEntity(
                name=a.get("name", "unknown"),
                dose_text=a.get("dose_text"),
                effect_text=a.get("effect_text"),
            )
            for a in data.get("actions_taken", [])
        ]

        vital_signs = [
            VitalSignEntry(
                name=v.get("name", "unknown"),
                value=str(v.get("value", "")),
                unit=v.get("unit"),
            )
            for v in data.get("vital_signs", [])
        ]

        menstrual_raw = data.get("menstrual_status")
        menstrual_status = None
        if menstrual_raw and menstrual_raw.get("is_period_day"):
            menstrual_status = MenstrualStatus(
                is_period_day=True,
                flow_level=menstrual_raw.get("flow_level"),
            )

        return ExtractionResult(
            transcript=transcript,
            symptoms=symptoms,
            actions_taken=actions,
            vital_signs=vital_signs,
            missing_fields=data.get("missing_fields", []),
            red_flags=data.get("red_flags", []),
            menstrual_status=menstrual_status,
        )

    async def doctor_packet(self, logs: List[LogEntry], days: int, user_id: str | None = None, user_profile=None) -> DoctorPacket:
        """Generate Doctor Packet using MedGemma."""
        if not self._ensure_initialized():
            raise RuntimeError("MedGemma endpoint not initialized.")

        # Use chunked map-reduce when logs exceed single-prompt capacity
        if len(logs) > 50:
            history_context = await self.build_full_history_context(logs)
        else:
            history_context = self._format_logs_for_prompt(logs, max_logs=50)

        # Fetch profile if not provided
        if user_profile is None and user_id:
            try:
                from ...services.storage import get_or_create_user_profile
                user_profile = get_or_create_user_profile(user_id)
            except Exception:
                pass

        # Inject watchdog observations as system flags
        flags: list[str] = []
        if user_id:
            from ...services.storage import get_watchdog_observations
            flags = get_watchdog_observations(user_id)
        system_flags = "\n".join(flags) if flags else "No automated flags."

        patient_profile_text = self._format_patient_profile(user_profile)
        prompt = (self.DOCTOR_PACKET_PROMPT
                  .replace("{patient_profile}", patient_profile_text)
                  .replace("{logs_text}", history_context)
                  .replace("{system_flags}", system_flags))
        response = self._call_endpoint(prompt, max_tokens=4096)
        data = self._parse_json_response(response)

        # Build timeline deterministically — LLM output is unreliable for this
        deterministic_timeline = self._build_timeline_bullets(logs)

        return DoctorPacket(
            hpi=self._fix_hpi_dates(self._fix_hpi_demographics(data.get("hpi", "History not available"), user_profile), logs),
            pertinent_positives=self._fix_pertinent_positives_demographics(data.get("pertinent_positives", []), user_profile),
            pertinent_negatives=[],
            timeline_bullets=deterministic_timeline,
            questions_for_clinician=data.get("questions_for_clinician", []),
            system_longitudinal_flags=data.get("system_longitudinal_flags", []),
        )

    async def timeline(self, logs: List[LogEntry], days: int) -> TimelineReveal:
        """Generate timeline using MedGemma."""
        if not self._ensure_initialized():
            raise RuntimeError("MedGemma endpoint not initialized.")

        if len(logs) > 50:
            logs_text = await self.build_full_history_context(logs)
        else:
            logs_text = self._format_logs_for_prompt(logs, max_logs=50)
        prompt = self.TIMELINE_PROMPT.format(days=days, logs_text=logs_text)
        response = self._call_endpoint(prompt, max_tokens=1024)
        data = self._parse_json_response(response)

        story_points: list[TimelinePoint] = []
        for point in data.get("story_points", []):
            ts_str = point.get("timestamp") or ""
            ts = datetime.now(timezone.utc).replace(tzinfo=None)
            if isinstance(ts_str, str) and ts_str.strip():
                try:
                    ts = datetime.fromisoformat(ts_str.strip().replace("Z", "+00:00"))
                    if ts.tzinfo is not None:
                        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
                except ValueError:
                    ts = datetime.now(timezone.utc).replace(tzinfo=None)

            story_points.append(
                TimelinePoint(
                    timestamp=ts,
                    label=point.get("label", "Event"),
                    details=point.get("details", ""),
                )
            )

        # If the model emits no points, synthesize something minimal from logs.
        if not story_points and logs:
            sorted_logs = sorted(logs, key=lambda x: x.recorded_at)
            for i, log in enumerate(sorted_logs[:10]):
                symptoms = ", ".join([s.symptom for s in log.extracted.symptoms]) or "symptoms"
                label = "Onset" if i == 0 else f"Update {i}"
                story_points.append(
                    TimelinePoint(
                        timestamp=log.recorded_at,
                        label=label,
                        details=f"Reported: {symptoms}",
                    )
                )

        return TimelineReveal(story_points=story_points)

    async def generate_agent_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate agent response JSON using MedGemma."""
        if not self._ensure_initialized():
            raise RuntimeError("MedGemma endpoint not initialized.")

        try:
            result = self._call_endpoint(prompt, max_tokens=max_tokens)
            self._clear_last_fallback()
            return result
        except Exception as e:
            logger.error(f"MedGemma agent response generation failed: {e}")
            self._set_last_fallback(f"agent_response_empty:{type(e).__name__}")
            raise

    async def generate_profile_update(self, logs: List[LogEntry], current_profile: dict) -> dict:
        """Analyze recent logs and generate profile updates using MedGemma."""
        if not self._ensure_initialized():
            raise RuntimeError("MedGemma endpoint not initialized.")

        if not logs:
            return {"add_conditions": [], "add_patterns": [], "health_summary": None}

        profile_str = json.dumps(current_profile, indent=2) if current_profile else "Empty profile"
        if len(logs) > 50:
            logs_text = await self.build_full_history_context(logs)
        else:
            logs_text = self._format_logs_for_prompt(logs, max_logs=50)
        prompt = self.PROFILE_UPDATE_PROMPT.format(
            current_profile=profile_str,
            logs_text=logs_text
        )
        response = self._call_endpoint(prompt, max_tokens=512)
        data = self._parse_json_response(response)

        return {
            "add_conditions": data.get("add_conditions", []),
            "add_patterns": data.get("add_patterns", []),
            "health_summary": data.get("health_summary"),
        }

    async def respond_to_followup(
        self,
        original_transcript: str,
        followup_question: str,
        followup_answer: str,
        patient_name: Optional[str] = None,
        user_profile=None,
    ) -> str:
        """Generate a context-aware acknowledgment of the patient's followup answer."""
        name_line = f"The patient's name is {patient_name}. " if patient_name else ""
        profile_context = ""
        if user_profile:
            profile_text = self._format_patient_profile(user_profile, include_patterns=False)
            profile_context = f"\nPatient profile:\n{profile_text}\n"
        prompt = (
            "You are a warm, empathetic health tracking assistant. "
            "Your tone is grounded and real — never use cheerleader language "
            "like 'That's fantastic!', 'Keep up the great work!', or "
            "'That's wonderful news!'. Instead say things like 'That's a real "
            "change.' or 'Noted — that's different from before.'\n\n"
            f"{name_line}"
            "The patient just answered your follow-up question.\n\n"
            f"Original: {original_transcript}\n"
            f"Your question: {followup_question}\n"
            f"Their answer: {followup_answer}\n"
            f"{profile_context}\n"
            "Respond in 2-3 sentences. Acknowledge what they said with warmth. "
            "If they expressed frustration or pain, validate it — don't just summarize data.\n\n"
            "CLINICAL RULES:\n"
            "- If the patient reports a concerning vital sign (BP >= 140/90, SpO2 < 93, "
            "blood sugar > 200 or < 70, temp >= 101), acknowledge the value and note it is "
            "elevated or abnormal. Do NOT simply accept or move on.\n"
            "- If the patient DISMISSES a concerning value ('it's fine', 'whatever', "
            "'probably nothing'), DO NOT accept the dismissal. Acknowledge their feeling "
            "but clearly state the value is elevated and recommend monitoring or discussing "
            "with their doctor. Be direct but not lecturing.\n"
            "- If the patient's profile shows conditions or medications that interact with "
            "the reported values (e.g., prednisone + hypertension, diabetes + high glucose), "
            "briefly note the connection and recommend continued monitoring.\n"
            "- You are not a doctor. Do not diagnose. But you CAN and SHOULD flag objectively "
            "concerning values and recommend the patient discuss them with their doctor.\n"
            "- NEVER fabricate clinical history or reference events not in the logs.\n"
            "- Profile patterns are for YOUR reasoning only. NEVER quote or paraphrase "
            "pattern text back to the patient. No sentences about symptoms 'often occurring "
            "together' or triggers 'preceding or coinciding with' other symptoms.\n"
            "\nRemember: grounded tone, no cheerleader language. "
            "Vary your opening — do NOT begin with 'It sounds like' or 'Sounds like'. "
            "Use constructions like 'Thanks for sharing that...', 'That makes sense...', "
            "'I appreciate you...', 'I see...', or lead with the clinical content directly.\n"
            "Response:"
        )
        return await self.generate_agent_response(prompt, max_tokens=200)

    async def watchdog_analysis(self, history_context: str) -> WatchdogResult:
        """Run background diagnostic analysis on longitudinal history."""
        if not self._ensure_initialized():
            raise RuntimeError("MedGemma endpoint not initialized.")

        prompt = self.WATCHDOG_PROMPT.format(history_context=history_context)
        response = self._call_endpoint(prompt, max_tokens=1024)
        data = self._parse_json_response(response)

        return WatchdogResult(
            concerning_pattern_detected=data.get("concerning_pattern_detected", False),
            internal_clinical_rationale=data.get("internal_clinical_rationale", ""),
            safe_patient_nudge=data.get("safe_patient_nudge"),
            clinician_facing_observation=data.get("clinician_facing_observation"),
        )
