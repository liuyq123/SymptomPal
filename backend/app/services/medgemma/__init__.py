"""
MedGemma Client Package - Medical Language Model for Symptom Extraction

Split into submodules for maintainability:
- base.py: Abstract base class and shared utilities
- vertex.py: Vertex AI implementation
- ollama.py: Local Ollama implementation (GGUF quantized)
- stub.py: Stub/testing implementation
- local.py: Local GPU implementation (HuggingFace transformers)

All public symbols are re-exported here for backward compatibility.
"""

from .base import MedGemmaClient, _load_vertex_ai, logger  # noqa: F401

# Import implementations lazily to avoid import-time side effects,
# but still support `from ...medgemma import StubMedGemmaClient` etc.
from .stub import StubMedGemmaClient  # noqa: F401
from .vertex import VertexAIMedGemmaClient  # noqa: F401
from .local import LocalMedGemmaClient  # noqa: F401
from .ollama import OllamaMedGemmaClient  # noqa: F401

import os
import threading
from typing import Optional

# Singleton instance
_medgemma_instance: Optional[MedGemmaClient] = None
_medgemma_lock = threading.Lock()


def get_medgemma_client() -> MedGemmaClient:
    """
    Factory function to get the MedGemma client.

    Priority:
    1. USE_STUB_MEDGEMMA=true -> StubMedGemmaClient (keyword matching, no LLM)
    2. USE_OLLAMA_MEDGEMMA=true -> OllamaMedGemmaClient (local Ollama GGUF)
    3. USE_LOCAL_MEDGEMMA=true -> LocalMedGemmaClient (runs on local GPU via HF)
    4. Default -> VertexAIMedGemmaClient (Google Cloud Vertex AI)

    Environment variables:
    - USE_STUB_MEDGEMMA: Set to "true" for keyword-based stub (no LLM needed)
    - USE_OLLAMA_MEDGEMMA: Set to "true" for local Ollama inference
    - OLLAMA_MODEL: Override model name (default: medgemma-27b-it Q5_K_M)
    - OLLAMA_HOST: Override Ollama host (default: http://localhost:11434)
    - USE_LOCAL_MEDGEMMA: Set to "true" to run MedGemma locally on GPU
    - GCP_PROJECT_ID, GCP_REGION, MEDGEMMA_ENDPOINT_ID: For Vertex AI
    - HF_TOKEN: HuggingFace token (required for local mode, gated model)
    """
    global _medgemma_instance

    use_stub = os.environ.get("USE_STUB_MEDGEMMA", "").lower() == "true"
    use_ollama = os.environ.get("USE_OLLAMA_MEDGEMMA", "").lower() == "true"
    use_local = os.environ.get("USE_LOCAL_MEDGEMMA", "").lower() == "true"

    if use_stub:
        return StubMedGemmaClient()

    with _medgemma_lock:
        if use_ollama:
            if _medgemma_instance is None or not isinstance(_medgemma_instance, OllamaMedGemmaClient):
                _medgemma_instance = OllamaMedGemmaClient()
            return _medgemma_instance

        if use_local:
            if _medgemma_instance is None or not isinstance(_medgemma_instance, LocalMedGemmaClient):
                _medgemma_instance = LocalMedGemmaClient()
            return _medgemma_instance

        # Default to Vertex AI implementation
        if _medgemma_instance is None:
            _medgemma_instance = VertexAIMedGemmaClient()
        return _medgemma_instance
