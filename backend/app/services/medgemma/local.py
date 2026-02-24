"""Local GPU implementation of MedGemma client using HuggingFace transformers."""

import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List

from .base import MedGemmaClient, logger
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

# Global model cache so the model is loaded only once
_local_model = None
_local_tokenizer = None


class LocalMedGemmaClient(MedGemmaClient):
    """
    Local MedGemma implementation using HuggingFace transformers.

    Runs the model on your local GPU. Requires:
    - ~20GB VRAM for medgemma-27b (4-bit quantized)
    - HF_TOKEN environment variable (for gated model access)

    Set USE_LOCAL_MEDGEMMA=true in .env
    """

    def __init__(self):
        super().__init__()

        # Check for custom model ID override first
        custom_model_id = os.environ.get("MEDGEMMA_MODEL_ID", "").strip()

        if custom_model_id:
            # Use custom model ID from environment
            self.MODEL_ID = custom_model_id
            # Assume pre-quantized if model ID contains quantization indicators
            self.use_quantization = any(x in custom_model_id.lower() for x in ["4bit", "8bit", "gptq", "awq", "bnb"])
            logger.info(f"Using custom model: {self.MODEL_ID} (pre-quantized: {self.use_quantization})")
        else:
            self.MODEL_ID = "google/medgemma-27b-text-it"
            self.use_quantization = True
            logger.info("Configured for 27b-text-it model with 4-bit quantization")

        # 27b-text-it is text-only; medgemma-27b-it is multimodal
        self.is_multimodal = "27b-it" in self.MODEL_ID and "text" not in self.MODEL_ID
        logger.info(f"Model type: {'multimodal' if self.is_multimodal else 'text-only'}")

        self._model = None
        self._tokenizer = None
        self._device = None
        self._debug_logs = os.environ.get("MEDGEMMA_DEBUG_LOGS", "").lower() == "true"

    EXTRACTION_PROMPT = """You are a medical assistant. Extract ONLY symptoms and medications that are currently present in the patient's statement.

CRITICAL RULES:
1. ONLY extract CURRENT symptoms - do NOT extract resolved symptoms
2. DO NOT infer or guess symptoms not mentioned
3. If patient says "X is gone" or "X is better" → do NOT include X in symptoms. More broadly: if the patient reports IMPROVEMENT or RETURN TO BASELINE (e.g., "finally sleeping again", "breathing is back to normal", "less of the green stuff"), do NOT extract it as an active symptom
4. Extract severity numbers whether stated precisely or approximately (e.g., "7 out of 10" → 7, "about a 4" → 4, "maybe a 6" → 6)
5. Empty symptoms list is OK if nothing is mentioned
6. actions_taken should contain medications, treatments, AND food/drinks used as treatment (e.g., juice for low sugar)
7. If patient says they SKIPPED a medication, do NOT include it in actions_taken — they did not take it
8. Do NOT include rest or lifestyle actions in actions_taken
9. VITAL SIGNS AND MEASUREMENTS are NOT symptoms — extract them in the "vital_signs" field, NOT in "symptoms":
   - Blood sugar/glucose, blood pressure, temperature, heart rate, weight, oxygen saturation (SpO2)
10. Use CANONICAL medical symptom names, not the patient's exact colloquial phrasing:
    - "knee's bothering me" → symptom: "knee pain"
    - "coughing up phlegm" → symptom: "productive cough"
    - "feeling winded" → symptom: "shortness of breath"
    - Re-use the same canonical name for the same symptom across entries

EXAMPLES:

Example 1:
Patient: "I have a headache and nausea"
{{
    "symptoms": [{{"symptom": "headache"}}, {{"symptom": "nausea"}}],
    "actions_taken": []
}}

Example 2:
Patient: "Took ibuprofen and cough syrup with dextromethorphan"
{{
    "symptoms": [],
    "actions_taken": [{{"name": "ibuprofen"}}, {{"name": "dextromethorphan"}}]
}}

Example 3:
Patient: "Headache is 7 out of 10. Took aspirin 500mg"
{{
    "symptoms": [{{"symptom": "headache", "severity_1_10": 7}}],
    "actions_taken": [{{"name": "aspirin", "dose_text": "500mg"}}]
}}

Example 4 (food as treatment — goes in actions_taken):
Patient: "Drank juice to raise my sugar after it crashed to 55"
{{
    "symptoms": [{{"symptom": "hypoglycemia"}}],
    "actions_taken": [{{"name": "juice"}}]
}}

Example 5 (skipped medication — do NOT extract as action):
Patient: "I skipped my evening sertraline because I was nauseous"
{{
    "symptoms": [{{"symptom": "nausea"}}],
    "actions_taken": []
}}

Example 6 (vital signs go in vital_signs, NOT symptoms):
Patient: "Blood pressure 148 over 92 today. Doctor appointment Thursday."
{{
    "symptoms": [],
    "actions_taken": [],
    "vital_signs": [{{"name": "blood pressure", "value": "148/92", "unit": "mmHg"}}]
}}

Example 7 (SpO2 and heart rate):
Patient: "SpO2 was 96 today, heart rate 78"
{{
    "symptoms": [],
    "actions_taken": [],
    "vital_signs": [{{"name": "spo2", "value": "96", "unit": "%"}}, {{"name": "heart rate", "value": "78", "unit": "bpm"}}]
}}

Example 8 (cardiac medication with dose):
Patient: "Took my lisinopril 10mg this morning for blood pressure"
{{
    "symptoms": [],
    "actions_taken": [{{"name": "lisinopril", "dose_text": "10mg"}}]
}}

Example 9 (approximate severity — "about a", "maybe", "like" all count):
Patient: "Back pain again, maybe a 5. Took tylenol."
{{
    "symptoms": [{{"symptom": "back pain", "severity_1_10": 5}}],
    "actions_taken": [{{"name": "tylenol"}}]
}}

Example 10 (menstrual status — period is active TODAY):
Patient: "Period started this morning, really heavy flow and cramps are a 7"
{{
    "symptoms": [{{"symptom": "menstrual cramps", "severity_1_10": 7}}],
    "actions_taken": [],
    "menstrual_status": {{"is_period_day": true, "flow_level": "heavy"}}
}}

Example 11 (past period reference — NOT a current period day):
Patient: "My last period was 2 weeks ago, now I have pelvic pain"
{{
    "symptoms": [{{"symptom": "pelvic pain"}}],
    "actions_taken": [],
    "menstrual_status": {{"is_period_day": false}}
}}

Now extract from this patient statement:
Patient says: "{transcript}"

Respond with JSON only:
{{
    "symptoms": [
        {{"symptom": "symptom_name", "severity_1_10": null_or_number, "onset_time_text": null, "duration_text": null, "triggers": [], "relievers": [], "associated_symptoms": []}}
    ],
    "actions_taken": [
        {{"name": "medication_name", "dose_text": null, "effect_text": null}}
    ],
    "vital_signs": [
        {{"name": "measurement_name", "value": "reading", "unit": null}}
    ],
    "missing_fields": [],
    "red_flags": [],
    "menstrual_status": {{"is_period_day": false, "flow_level": null}}
}}"""

    AGENT_RESPONSE_PROMPT = """You are a caring health assistant. Generate a response to the patient.

{context}

Respond with JSON only:
{{
    "acknowledgment": "Brief acknowledgment of their symptoms",
    "insight_nudge": "Pattern observation if any, or null",
    "should_ask_question": true/false,
    "immediate_question": "Follow-up question or null",
    "should_schedule_checkin": true/false,
    "checkin_hours": 2,
    "checkin_message": "Check-in message or null"
}}"""

    DOCTOR_PACKET_PROMPT = """You are a medical assistant preparing a pre-visit summary for a physician.

Patient profile:
{patient_profile}

Symptom logs from the past {days} days (dates in [YYYY-MM-DD] are authoritative):
{logs_text}

System flags (dates here may be approximate — always use the log timestamps above for exact dates):
{system_flags}

Generate a Doctor Packet as JSON. Follow these rules:
1. HPI MUST start with the patient's actual age and gender from the profile above. Copy the EXACT age number — if profile says "Age: 61", write "61-year-old", NOT "62-year-old". Cross-check your HPI demographics against the profile before responding. NEVER use bracket placeholders like [age] or [gender].
2. Use medication names and doses from the profile. NEVER say "dose unspecified" — doses are in the profile.
3. HPI should cover onset, progression, and current state using OLDCARTS format. Include severity numbers (X/10) and functional impact.
4. questions_for_clinician: infer what the PATIENT is worried about from recurring themes, emotional language, and functional impacts. Frame from the patient's perspective.
5. If system_flags are provided, include them in system_longitudinal_flags.

{{
    "hpi": "Clinical paragraph with demographics, OLDCARTS, severity progression, medications with doses, functional impact.",
    "pertinent_positives": ["All symptoms with severity ratings and medications with doses"],
    "questions_for_clinician": ["Patient concerns framed from their perspective"],
    "system_longitudinal_flags": ["Verbatim system flags if any"]
}}

Respond with JSON only:"""

    TIMELINE_PROMPT = """You are a medical assistant creating a symptom timeline.

Analyze these symptom logs:
{logs_text}

Create meaningful timeline story points. For each significant event, provide:
- A label (e.g., "Onset", "Peak", "Improvement", "Treatment Started")
- Brief details about what happened

Respond with JSON only:
{{
    "story_points": [
        {{"timestamp": "ISO datetime string", "label": "Event label", "details": "What happened"}}
    ]
}}"""

    PROFILE_UPDATE_PROMPT = """You are a medical assistant analyzing patient symptom patterns.

Current patient profile:
{current_profile}

Recent symptom logs to analyze:
{logs_text}

Based on recurring patterns in the logs, suggest profile updates:
- If a symptom appears 3+ times, consider suggesting it as a chronic condition
- Look for patterns (time of day, triggers, etc.)
- Generate a brief health summary if there's enough data

Respond with JSON only:
{{
    "add_conditions": ["List of conditions to add based on recurring symptoms"],
    "add_patterns": ["List of patterns detected (e.g., 'Headaches often occur in morning')"],
    "health_summary": "Brief summary of current health status, or null if insufficient data"
}}"""

    def _ensure_loaded(self):
        """Lazy load the model."""
        global _local_model, _local_tokenizer

        if _local_model is not None:
            self._model = _local_model
            self._tokenizer = _local_tokenizer
            return

        try:
            import torch
            from transformers import BitsAndBytesConfig

            logger.info(f"Loading local MedGemma model: {self.MODEL_ID}")

            # Check for HF token
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning("HF_TOKEN not set - may fail for gated models")

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                self._device = "mps"
                logger.info("Using Apple MPS")
            else:
                self._device = "cpu"
                logger.warning("No GPU found, using CPU (will be slow)")

            trust_remote_code_setting = os.environ.get("MEDGEMMA_TRUST_REMOTE_CODE", "").strip().lower()
            trust_remote_code = trust_remote_code_setting == "true"
            if not trust_remote_code_setting and "unsloth" in self.MODEL_ID.lower():
                trust_remote_code = True
                logger.warning("MEDGEMMA_TRUST_REMOTE_CODE not set; enabling for Unsloth model.")

            use_unsloth = "unsloth" in self.MODEL_ID.lower() or os.environ.get("MEDGEMMA_USE_UNSLOTH", "").lower() == "true"

            # Load tokenizer/processor based on model type
            if self.is_multimodal:
                from transformers import AutoProcessor
                logger.info("Loading AutoProcessor for multimodal model")
                self._tokenizer = AutoProcessor.from_pretrained(
                    self.MODEL_ID,
                    token=hf_token,
                    trust_remote_code=trust_remote_code,
                )
            else:
                if use_unsloth:
                    try:
                        from unsloth import FastLanguageModel
                        max_seq_len = int(os.environ.get("MEDGEMMA_MAX_SEQ_LEN", "4096"))
                        use_bf16 = self._device == "cuda" and torch.cuda.is_bf16_supported()
                        dtype = torch.bfloat16 if use_bf16 else torch.float16
                        logger.info("Loading Unsloth FastLanguageModel for text-only model")
                        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                            model_name=self.MODEL_ID,
                            max_seq_length=max_seq_len,
                            dtype=dtype,
                            load_in_4bit=self.use_quantization,
                        )
                        FastLanguageModel.for_inference(self._model)
                        logger.info("Loaded model via Unsloth FastLanguageModel")
                    except Exception as e:
                        logger.warning(f"Unsloth FastLanguageModel load failed, falling back to HF: {e}")
                        use_unsloth = False

                if not use_unsloth:
                    from transformers import AutoTokenizer
                    logger.info("Loading AutoTokenizer for text-only model")
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.MODEL_ID,
                        token=hf_token,
                        trust_remote_code=trust_remote_code,
                    )
                    if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
                        self._tokenizer.pad_token = self._tokenizer.eos_token

            # Configure quantization if using 27b model
            model_kwargs = {
                "token": hf_token,
                "device_map": "auto",
                "trust_remote_code": trust_remote_code,
            }

            use_bf16 = self._device == "cuda" and torch.cuda.is_bf16_supported()
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

            if self.use_quantization and self._device == "cuda":
                logger.info("Using 4-bit quantization (NF4) for 27b model")
                if not use_bf16:
                    logger.warning("GPU does not support bfloat16; using float16 compute for 4-bit quantization.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,  # Nested quantization for better accuracy
                    bnb_4bit_quant_type="nf4",  # NormalFloat4
                )
                model_kwargs["quantization_config"] = quantization_config
            else:
                if self._device == "cpu":
                    model_kwargs["torch_dtype"] = torch.float32
                else:
                    model_kwargs["torch_dtype"] = torch.bfloat16 if use_bf16 else torch.float16

            # Load model based on type (unless already loaded via Unsloth)
            if self._model is None:
                if self.is_multimodal:
                    from transformers import AutoModelForImageTextToText
                    logger.info("Loading AutoModelForImageTextToText for multimodal model")
                    self._model = AutoModelForImageTextToText.from_pretrained(
                        self.MODEL_ID,
                        **model_kwargs
                    )
                else:
                    from transformers import AutoModelForCausalLM
                    logger.info("Loading AutoModelForCausalLM for text-only model")
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.MODEL_ID,
                        **model_kwargs
                    )

            # Cache globally
            _local_model = self._model
            _local_tokenizer = self._tokenizer

            # Align model/generation configs with tokenizer tokens for safe generation
            try:
                pad_id = self._tokenizer.pad_token_id
                eos_id = self._tokenizer.eos_token_id
                if pad_id is not None:
                    if getattr(self._model.config, "pad_token_id", None) != pad_id:
                        logger.warning(f"Overriding model.config.pad_token_id to {pad_id}")
                        self._model.config.pad_token_id = pad_id
                if eos_id is not None:
                    if getattr(self._model.config, "eos_token_id", None) != eos_id:
                        logger.warning(f"Overriding model.config.eos_token_id to {eos_id}")
                        self._model.config.eos_token_id = eos_id
                if hasattr(self._model, "generation_config"):
                    gen_cfg = self._model.generation_config
                    if pad_id is not None and gen_cfg.pad_token_id != pad_id:
                        logger.warning(f"Overriding generation_config.pad_token_id to {pad_id}")
                        gen_cfg.pad_token_id = pad_id
                    if eos_id is not None and gen_cfg.eos_token_id != eos_id:
                        logger.warning(f"Overriding generation_config.eos_token_id to {eos_id}")
                        gen_cfg.eos_token_id = eos_id
                    # Clear forced tokens that can cause immediate EOS/pad outputs
                    if getattr(gen_cfg, "forced_eos_token_id", None) is not None:
                        logger.warning("Clearing generation_config.forced_eos_token_id")
                        gen_cfg.forced_eos_token_id = None
                    if getattr(gen_cfg, "forced_bos_token_id", None) is not None:
                        logger.warning("Clearing generation_config.forced_bos_token_id")
                        gen_cfg.forced_bos_token_id = None
            except Exception as e:
                logger.warning(f"Failed to align generation config with tokenizer: {e}")

            logger.info("Local MedGemma model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load local MedGemma: {e}")
            raise

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate text from prompt."""
        import torch
        self._ensure_loaded()

        def _run_generation(inputs, label: str):
            input_len = inputs["input_ids"].shape[-1]
            if self._debug_logs:
                logger.debug(
                    "Generation debug label=%s input_len=%s max_new_tokens=%s input_shape=%s",
                    label,
                    input_len,
                    max_new_tokens,
                    tuple(inputs["input_ids"].shape),
                )

            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                }
                if self._tokenizer.pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
                if self._tokenizer.eos_token_id is not None:
                    gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
                generation = self._model.generate(**inputs, **gen_kwargs)
                new_tokens = generation[0][input_len:]
                if self._debug_logs:
                    logger.debug(
                        "Generation output label=%s output_shape=%s new_tokens=%s",
                        label,
                        tuple(generation.shape),
                        int(new_tokens.shape[-1]) if new_tokens.ndim > 0 else 0,
                    )

            response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            if self._debug_logs:
                logger.debug("Decoded response length=%s label=%s", len(response_text), label)
            return response_text, new_tokens

        if self.is_multimodal:
            # Multimodal model: Format as chat message (text-only, no image)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]

            # Apply chat template
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self._model.device, dtype=torch.bfloat16)

            response, _ = _run_generation(inputs, "multimodal")
            return response

        # Text-only model: Use chat template when available; otherwise fall back to raw prompt.
        force_plain = os.environ.get("MEDGEMMA_DISABLE_CHAT_TEMPLATE", "").lower() == "true"
        use_chat_template = (
            not force_plain
            and hasattr(self._tokenizer, "apply_chat_template")
            and getattr(self._tokenizer, "chat_template", None)
        )

        def _build_text_inputs(chat: bool):
            if chat:
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful medical assistant that extracts structured medical information.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
                return self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self._model.device)
            return self._tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self._model.device)

        inputs = _build_text_inputs(use_chat_template)
        response, new_tokens = _run_generation(inputs, "text-chat" if use_chat_template else "text-plain")

        pad_id = self._tokenizer.pad_token_id
        eos_id = self._tokenizer.eos_token_id
        pad_or_eos = {tid for tid in (pad_id, eos_id) if tid is not None}
        special_ids = set(getattr(self._tokenizer, "all_special_ids", []) or [])
        special_only = (
            new_tokens.numel() == 0
            or (pad_or_eos and all(tok in pad_or_eos for tok in new_tokens.tolist()))
            or (special_ids and all(tok in special_ids for tok in new_tokens.tolist()))
        )

        if use_chat_template and (not response.strip() or special_only):
            logger.warning("Chat template produced empty/special-only output. Retrying with plain prompt.")
            inputs = _build_text_inputs(False)
            response, _ = _run_generation(inputs, "text-plain-retry")

        return response

    def _parse_json(self, text: str) -> dict:
        """Parse JSON from response."""
        # Try to find JSON in response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        return json.loads(text)

    def _post_process_medications(self, transcript: str, actions: List[ActionEntity]) -> List[ActionEntity]:
        """
        Post-process to detect multiple medications connected by 'and' that the model might have missed.

        Handles patterns like:
        - "ibuprofen and cough syrup"
        - "X and Y with Z" (extracts active ingredient Z)
        """
        # Common medication keywords that indicate multiple medications
        # Pattern: "medication_name and (some|a|the)? medication_name"
        pattern = r'\b(took|taking|take|used|applied)\s+(?:more\s+)?(\w+(?:\s+\w+)?)\s+and\s+(?:some|a|the)?\s*(\w+(?:\s+\w+)?(?:\s+with\s+(\w+))?)'

        matches = re.finditer(pattern, transcript.lower(), re.IGNORECASE)
        extracted_names = {a.name.lower() for a in actions}

        for match in matches:
            # Get medication names from the match
            med1 = match.group(2).strip()
            med2_with_active = match.group(3).strip()
            active_ingredient = match.group(4)  # The part after "with"

            # If there's an active ingredient (e.g., "cough syrup with dextromethorphan"),
            # prefer the active ingredient
            med2 = active_ingredient.strip() if active_ingredient else med2_with_active.split()[0]

            # Add medications that weren't already extracted
            for med_name in [med1, med2]:
                if med_name and med_name not in extracted_names and len(med_name) > 2:
                    # Filter out common words
                    if med_name not in ['and', 'the', 'some', 'more', 'with', 'took', 'take', 'taking']:
                        actions.append(ActionEntity(
                            name=med_name,
                            dose_text=None,
                            effect_text=None
                        ))
                        extracted_names.add(med_name)

        return actions

    def _format_logs_for_prompt(self, logs: List[LogEntry]) -> str:
        """Format logs into a readable text for prompts."""
        if not logs:
            return "No logs available."

        sorted_logs = sorted(logs, key=lambda x: x.recorded_at)
        lines = []
        for log in sorted_logs:
            date_str = log.recorded_at.strftime("%Y-%m-%d %H:%M")
            symptoms = ", ".join([s.symptom for s in log.extracted.symptoms]) or "no symptoms"
            actions = ", ".join([a.name for a in log.extracted.actions_taken]) or "no actions"
            line = f"- {date_str}: Symptoms: {symptoms}. Actions: {actions}."
            if log.image_analysis:
                line += f" Image: {log.image_analysis.clinical_description}"
            lines.append(line)
        return "\n".join(lines)

    async def extract(self, transcript: str) -> ExtractionResult:
        """Extract symptoms using local MedGemma."""
        logger.info(f"LocalMedGemmaClient.extract starting, MODEL_ID={getattr(self, 'MODEL_ID', 'MISSING')}")
        try:
            prompt = self.EXTRACTION_PROMPT.format(transcript=transcript)
            response = self._generate(prompt, max_new_tokens=400)
            logger.debug("Local extraction model response length=%s", len(response))
            data = self._parse_json(response)
            self._clear_last_fallback()
            logger.debug("Parsed extraction keys=%s", list(data.keys()))

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

            # Post-processing: Detect multiple medications connected by "and" that model might have missed
            actions = self._post_process_medications(transcript, actions)

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
                symptoms=symptoms if symptoms else [SymptomEntity(symptom="general discomfort")],
                actions_taken=actions,
                vital_signs=vital_signs,
                missing_fields=data.get("missing_fields", ["severity", "onset", "duration"]),
                red_flags=data.get("red_flags", []),
                menstrual_status=menstrual_status,
            )

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Local extraction failed: {e}\n{error_details}")
            raise

    async def doctor_packet(self, logs: List[LogEntry], days: int, user_id: str | None = None, user_profile=None) -> DoctorPacket:
        """Generate a Doctor Packet using the local MedGemma model."""
        logger.info(f"LocalMedGemmaClient.doctor_packet called with {len(logs)} logs for {days} days")
        try:
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
            prompt = self.DOCTOR_PACKET_PROMPT.replace("{patient_profile}", patient_profile_text)
            prompt = prompt.replace("{logs_text}", history_context)
            if "{system_flags}" in prompt:
                prompt = prompt.replace("{system_flags}", system_flags)
            if "{days}" in prompt:
                prompt = prompt.replace("{days}", str(days))
            response = self._generate(prompt, max_new_tokens=4096)
            logger.debug("Local doctor_packet response length=%s", len(response))
            data = self._parse_json(response)
            self._clear_last_fallback()
            logger.debug("Parsed doctor_packet keys=%s", list(data.keys()))

            # Build timeline deterministically — LLM output is unreliable for this
            deterministic_timeline = self._build_timeline_bullets(logs)

            return DoctorPacket(
                hpi=self._fix_hpi_dates(self._fix_hpi_demographics(data.get("hpi", "Patient reports symptoms as noted in logs."), user_profile), logs),
                pertinent_positives=self._fix_pertinent_positives_demographics(data.get("pertinent_positives", []), user_profile),
                pertinent_negatives=[],
                timeline_bullets=deterministic_timeline,
                questions_for_clinician=data.get("questions_for_clinician", []),
                system_longitudinal_flags=data.get("system_longitudinal_flags", []),
            )
        except Exception as e:
            logger.error(f"Local doctor_packet generation failed: {e}")
            raise

    async def timeline(self, logs: List[LogEntry], days: int) -> TimelineReveal:
        """Generate a timeline using the local MedGemma model."""
        logger.info(f"LocalMedGemmaClient.timeline called with {len(logs)} logs for {days} days")
        try:
            logs_text = self._format_logs_for_prompt(logs)
            prompt = self.TIMELINE_PROMPT.format(logs_text=logs_text)
            response = self._generate(prompt, max_new_tokens=600)
            logger.debug("Local timeline response length=%s", len(response))
            data = self._parse_json(response)
            self._clear_last_fallback()
            logger.debug("Parsed timeline keys=%s", list(data.keys()))

            story_points = []
            for point in data.get("story_points", []):
                try:
                    # Parse timestamp - handle various formats
                    ts_str = point.get("timestamp", "")
                    if isinstance(ts_str, str) and ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except ValueError:
                            ts = datetime.now(timezone.utc).replace(tzinfo=None)
                    else:
                        ts = datetime.now(timezone.utc).replace(tzinfo=None)

                    story_points.append(TimelinePoint(
                        timestamp=ts,
                        label=point.get("label", "Event"),
                        details=point.get("details", ""),
                    ))
                except Exception as point_error:
                    logger.warning(f"Failed to parse timeline point: {point_error}")
                    continue

            # If no valid points, create from logs directly
            if not story_points and logs:
                sorted_logs = sorted(logs, key=lambda x: x.recorded_at)
                for i, log in enumerate(sorted_logs):
                    symptoms = ", ".join([s.symptom for s in log.extracted.symptoms]) or "symptoms"
                    label = "Onset" if i == 0 else f"Update {i}"
                    story_points.append(TimelinePoint(
                        timestamp=log.recorded_at,
                        label=label,
                        details=f"Reported: {symptoms}",
                    ))

            return TimelineReveal(story_points=story_points)
        except Exception as e:
            logger.error(f"Local timeline generation failed: {e}")
            raise

    async def generate_agent_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate agent response using local model."""
        return self._generate(prompt, max_new_tokens=max_tokens)

    async def generate_profile_update(self, logs: List[LogEntry], current_profile: dict) -> dict:
        """Generate profile update suggestions using the local MedGemma model."""
        logger.info(f"LocalMedGemmaClient.generate_profile_update called with {len(logs)} logs")
        try:
            logs_text = self._format_logs_for_prompt(logs)
            profile_text = json.dumps(current_profile, indent=2) if current_profile else "{}"
            prompt = self.PROFILE_UPDATE_PROMPT.format(
                current_profile=profile_text,
                logs_text=logs_text
            )
            response = self._generate(prompt, max_new_tokens=400)
            logger.debug("Local profile_update response length=%s", len(response))
            data = self._parse_json(response)
            self._clear_last_fallback()
            logger.debug("Parsed profile_update keys=%s", list(data.keys()))

            return {
                "add_conditions": data.get("add_conditions", []),
                "add_patterns": data.get("add_patterns", []),
                "health_summary": data.get("health_summary"),
            }
        except Exception as e:
            logger.error(f"Local profile_update generation failed: {e}")
            raise

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
        response = self._generate(prompt, max_new_tokens=200)
        # Clean up: take first paragraph, strip quotes
        text = response.strip().strip('"').strip()
        if "\n\n" in text:
            text = text.split("\n\n")[0].strip()
        return text

    async def watchdog_analysis(self, history_context: str) -> WatchdogResult:
        """Run watchdog diagnostic analysis using local MedGemma model."""
        logger.info("LocalMedGemmaClient.watchdog_analysis starting")
        try:
            from .vertex import VertexAIMedGemmaClient
            prompt = VertexAIMedGemmaClient.WATCHDOG_PROMPT.format(history_context=history_context)
            response = self._generate(prompt, max_new_tokens=1024)
            data = self._parse_json(response)
            self._clear_last_fallback()
            return WatchdogResult(
                concerning_pattern_detected=data.get("concerning_pattern_detected", False),
                internal_clinical_rationale=data.get("internal_clinical_rationale", ""),
                safe_patient_nudge=data.get("safe_patient_nudge"),
                clinician_facing_observation=data.get("clinician_facing_observation"),
            )
        except Exception as e:
            logger.error(f"Local watchdog analysis failed: {e}")
            raise
