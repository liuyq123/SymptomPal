"""
MedASR Client - Medical Automatic Speech Recognition

Uses Google's MedASR model (105M params) for medical-specific speech-to-text.
MedASR is optimized for clinical dictation with 58% fewer errors than Whisper
on medical terminology.

References:
- https://huggingface.co/google/medasr
- https://developers.google.com/health-ai-developer-foundations/medasr
"""

import os
import base64
import logging
import tempfile
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy load heavy ML dependencies
_pipeline = None
_librosa = None
_torch = None


def _load_ml_dependencies():
    """Lazy load ML dependencies to avoid startup overhead."""
    global _pipeline, _librosa, _torch
    if _pipeline is None:
        try:
            from transformers import pipeline
            import librosa
            import torch
            _pipeline = pipeline
            _librosa = librosa
            _torch = torch
            logger.info("MedASR ML dependencies loaded successfully")
        except ImportError as e:
            logger.warning(f"MedASR ML dependencies not available: {e}")
            return False
    return True


class MedASRClient(ABC):
    """Abstract base class for medical ASR (Automatic Speech Recognition)."""

    @abstractmethod
    async def transcribe(self, audio_b64: str) -> str:
        """Transcribe audio to text."""
        raise NotImplementedError


class HuggingFaceMedASRClient(MedASRClient):
    """
    Real MedASR implementation using Google's Hugging Face model.

    Model: google/medasr (105M params, Conformer-based)
    Input: 16kHz mono audio
    Output: Medical-optimized transcription
    """

    MODEL_ID = "google/medasr"

    def __init__(self):
        self._pipe = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of the ASR pipeline."""
        if self._initialized:
            return self._pipe is not None

        self._initialized = True

        if not _load_ml_dependencies():
            logger.warning("ML dependencies not available, MedASR will use stub")
            return False

        try:
            device = "cuda" if _torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing MedASR pipeline on {device}...")

            self._pipe = _pipeline(
                "automatic-speech-recognition",
                model=self.MODEL_ID,
                device=device,
            )
            logger.info("MedASR pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MedASR pipeline: {e}")
            self._pipe = None
            return False

    async def transcribe(self, audio_b64: str) -> str:
        """
        Transcribe base64-encoded audio to text using MedASR.

        Args:
            audio_b64: Base64 encoded audio (webm, wav, mp3, etc.)

        Returns:
            Transcribed text
        """
        if not self._ensure_initialized():
            logger.warning("MedASR not available — returning empty transcript")
            return ""

        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_b64)

            # Write to temp file for librosa to read
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                # Load and resample audio to 16kHz (MedASR requirement)
                speech, _ = _librosa.load(temp_path, sr=16000, mono=True)

                # Run ASR pipeline with chunking for longer audio
                result = self._pipe(
                    speech,
                    chunk_length_s=20,  # Process in 20-second chunks
                    stride_length_s=2,   # 2-second overlap between chunks
                )

                transcript = result.get("text", "").strip()
                logger.info(f"MedASR transcribed {len(speech)/16000:.1f}s audio: {transcript[:100]}...")
                return transcript

            finally:
                # Clean up temp file
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"MedASR transcription failed: {e}")
            # Return empty string on error - let caller handle
            return ""


class StubMedASRClient(MedASRClient):
    """Stub implementation that returns deterministic transcripts for testing."""

    async def transcribe(self, audio_b64: str) -> str:
        """Return a sample transcript for MVP/testing purposes."""
        return "Headache started Tuesday afternoon. Took ibuprofen. Still nauseous."


import threading

# Singleton instance for the real client
_medasr_instance: Optional[MedASRClient] = None
_medasr_lock = threading.Lock()


def get_medasr_client() -> MedASRClient:
    """
    Factory function to get the MedASR client.

    Uses real MedASR by default if:
    - ML dependencies (transformers, torch, librosa) are available
    - USE_STUB_MEDASR env var is NOT set to "true"

    Falls back to stub if dependencies unavailable or stub explicitly requested.
    """
    global _medasr_instance

    use_stub = os.environ.get("USE_STUB_MEDASR", "").lower() == "true"

    if use_stub:
        return StubMedASRClient()

    with _medasr_lock:
        if _medasr_instance is None:
            _medasr_instance = HuggingFaceMedASRClient()
        return _medasr_instance
