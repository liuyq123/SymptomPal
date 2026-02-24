"""
Ollama MedGemma Client - Local inference via Ollama.

Runs MedGemma 27B (GGUF quantized) locally using Ollama.
Inherits all prompts and parsing logic from VertexAIMedGemmaClient;
only the LLM call mechanism is replaced.

NOTE: _call_endpoint is synchronous and blocks the event loop during inference
(30-60s+ per call for 27B). This is acceptable for batch/simulation workloads.
For real-time use, run uvicorn with --workers 2+ to avoid starving health checks.

Requires:
- Ollama installed and running (ollama serve)
- Model pulled: ollama pull hf.co/unsloth/medgemma-27b-it-GGUF:Q5_K_M
- Environment: USE_OLLAMA_MEDGEMMA=true
- Optional: OLLAMA_MODEL (default: hf.co/unsloth/medgemma-27b-it-GGUF:Q5_K_M)
- Optional: OLLAMA_HOST (default: http://localhost:11434)
"""

import os
import logging

from .vertex import VertexAIMedGemmaClient
from .base import MedGemmaClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "hf.co/unsloth/medgemma-27b-it-GGUF:Q5_K_M"


class OllamaMedGemmaClient(VertexAIMedGemmaClient):
    """MedGemma client using local Ollama inference."""

    def __init__(self):
        # Skip VertexAIMedGemmaClient.__init__ — we don't need Vertex AI config.
        MedGemmaClient.__init__(self)

        self._model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self._host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Initialize Ollama client connection."""
        if self._initialized:
            return True
        try:
            import ollama
            self._client = ollama.Client(host=self._host)
            # Verify model is available
            models = self._client.list()
            model_names = [m.model for m in models.models]
            if not any(self._model in name for name in model_names):
                logger.warning(
                    "Model %s not found in Ollama. Available: %s",
                    self._model, model_names,
                )
                return False
            self._initialized = True
            logger.info(
                "Ollama MedGemma initialized: model=%s host=%s",
                self._model, self._host,
            )
            return True
        except Exception as e:
            logger.error("Failed to initialize Ollama client: %s", e)
            return False

    def _call_endpoint(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call the local Ollama model using chat API."""
        if not self._ensure_initialized():
            raise RuntimeError("Ollama not available")

        options = {
            "temperature": float(os.environ.get("MEDGEMMA_TEMPERATURE", "0.1")),
            "num_predict": max_tokens,
        }
        seed = os.environ.get("MEDGEMMA_SEED")
        if seed is not None:
            options["seed"] = int(seed)

        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            options=options,
        )
        text = response.message.content.strip()
        if not text:
            raise RuntimeError("Empty response from Ollama")
        return self._clean_output(text)

    def describe_runtime(self):
        """Override to report Ollama-specific runtime info."""
        return {
            "client": self.__class__.__name__,
            "provider": "ollama",
            "model": self._model,
            "endpoint": self._host,
        }
