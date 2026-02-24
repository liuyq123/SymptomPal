"""Unit tests for dedicated endpoint DNS detection."""

from ..services.medgemma.vertex import VertexAIMedGemmaClient


def test_extract_dedicated_dns_from_error_parses_hostname():
    msg = (
        "400 Dedicated Endpoint cannot be accessed through the shared Vertex AI domain "
        "aiplatform.googleapis.com. Please access the endpoint using its dedicated domain name "
        "'8725687994146619392.us-central1-545001441318.prediction.vertexai.goog'"
    )
    dns = VertexAIMedGemmaClient._extract_dedicated_dns_from_error(Exception(msg))
    assert dns == "8725687994146619392.us-central1-545001441318.prediction.vertexai.goog"


def test_extract_dedicated_dns_from_error_normalizes_scheme():
    msg = (
        "Please access the endpoint using its dedicated domain name "
        "'https://8725687994146619392.us-central1-545001441318.prediction.vertexai.goog/'"
    )
    dns = VertexAIMedGemmaClient._extract_dedicated_dns_from_error(Exception(msg))
    assert dns == "8725687994146619392.us-central1-545001441318.prediction.vertexai.goog"


def test_extract_dedicated_dns_from_error_none_for_other_errors():
    dns = VertexAIMedGemmaClient._extract_dedicated_dns_from_error(Exception("some other error"))
    assert dns is None

