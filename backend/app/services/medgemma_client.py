"""
Backward-compatible shim — all code now lives in the medgemma/ package.

Existing imports like:
    from ..services.medgemma_client import get_medgemma_client
    from ..services.medgemma_client import StubMedGemmaClient

continue to work unchanged.
"""

from .medgemma import (  # noqa: F401
    MedGemmaClient,
    StubMedGemmaClient,
    VertexAIMedGemmaClient,
    LocalMedGemmaClient,
    OllamaMedGemmaClient,
    get_medgemma_client,
)
