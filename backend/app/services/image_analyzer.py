"""
Medical Image Analyzer using MedSigLIP

Uses Google's MedSigLIP model for zero-shot medical image classification
to generate clinical descriptions of skin conditions, rashes, swelling, etc.

MedSigLIP is optimized for medical image understanding and can classify
images against text descriptions of medical conditions.

References:
- https://huggingface.co/google/medsiglip-448
- https://developers.google.com/health-ai-developer-foundations/medsiglip
"""

import os
import base64
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple
from io import BytesIO

from PIL import Image

from ..models import ImageAnalysisResult, SkinLesionDescription

logger = logging.getLogger(__name__)

# Lazy load heavy dependencies
_torch = None
_AutoModel = None
_AutoProcessor = None


def _load_medsiglip_dependencies():
    """Lazy load MedSigLIP dependencies."""
    global _torch, _AutoModel, _AutoProcessor
    if _torch is None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            _torch = torch
            _AutoModel = AutoModel
            _AutoProcessor = AutoProcessor
            logger.info("MedSigLIP dependencies loaded successfully")
            return True
        except ImportError as e:
            logger.warning(f"MedSigLIP dependencies not available: {e}")
            return False
    return True


class ImageAnalyzerClient(ABC):
    """Abstract base class for medical image analysis."""

    @abstractmethod
    async def analyze_image(self, image_b64: str, context: Optional[str] = None) -> ImageAnalysisResult:
        """
        Analyze a medical image and generate a clinical description.

        Args:
            image_b64: Base64-encoded image data (PNG, JPEG)
            context: Optional context about what to look for (e.g., "rash on arm")

        Returns:
            ImageAnalysisResult with clinical description and classifications
        """
        pass

    # Concrete method — operates on data structures, not the model.
    _SIZE_ORDER = ["small", "<1cm", "< 1cm", "1-3cm", "1 to 3", "medium", "large", ">3cm", "> 3cm", "wide area", "extensive"]

    def compare_progression(
        self,
        current: ImageAnalysisResult,
        previous: ImageAnalysisResult,
        current_ts: datetime,
        previous_ts: datetime,
    ) -> Optional[str]:
        """Compare current image analysis against a previous one to detect changes.

        Returns a human-readable progression note, or None if comparison is not meaningful.
        """
        if not current.lesion_detected and not previous.lesion_detected:
            return None
        if not previous.lesion_detected and current.lesion_detected:
            return "New lesion detected (not present in previous photo)."
        if previous.lesion_detected and not current.lesion_detected:
            return "Previously documented lesion no longer visible."

        # Both have lesions — compare attributes
        delta = current_ts - previous_ts
        if delta.days > 0:
            time_desc = f"{delta.days} day{'s' if delta.days != 1 else ''}"
        else:
            hours = max(1, int(delta.total_seconds() / 3600))
            time_desc = f"{hours} hour{'s' if hours != 1 else ''}"

        notes: list[str] = []
        if current.skin_lesion and previous.skin_lesion:
            cur_sl = current.skin_lesion
            prev_sl = previous.skin_lesion

            def _size_rank(s: str) -> int:
                s_lower = s.lower()
                for i, term in enumerate(self._SIZE_ORDER):
                    if term in s_lower:
                        return i
                return -1

            cur_rank = _size_rank(cur_sl.size_estimate)
            prev_rank = _size_rank(prev_sl.size_estimate)
            if cur_rank > prev_rank >= 0:
                notes.append(f"Size appears increased (was: {prev_sl.size_estimate}, now: {cur_sl.size_estimate})")
            elif prev_rank > cur_rank >= 0:
                notes.append(f"Size appears decreased (was: {prev_sl.size_estimate}, now: {cur_sl.size_estimate})")

            if cur_sl.color.lower() != prev_sl.color.lower():
                notes.append(f"Color changed from {prev_sl.color} to {cur_sl.color}")
            if cur_sl.lesion_type.lower() != prev_sl.lesion_type.lower():
                notes.append(f"Morphology changed from {prev_sl.lesion_type} to {cur_sl.lesion_type}")
            if cur_sl.texture.lower() != prev_sl.texture.lower():
                notes.append(f"Texture changed from {prev_sl.texture} to {cur_sl.texture}")

        if notes:
            return f"Compared to photo {time_desc} ago: " + "; ".join(notes) + "."
        return f"Compared to photo {time_desc} ago: No significant changes detected."


class HuggingFaceMedSigLIPClient(ImageAnalyzerClient):
    """
    Real MedSigLIP implementation using Google's model from Hugging Face.

    MedSigLIP Details:
    - Input: 448x448 pixel images
    - Output: Image and text embeddings for zero-shot classification
    - Model ID: google/medsiglip-448
    """

    MODEL_ID = "google/medsiglip-448"
    IMAGE_SIZE = 448

    # Clinical text prompts for zero-shot classification
    LESION_TYPE_PROMPTS = [
        "a photo of skin with no visible lesion or rash",
        "a photo of a circular or round skin lesion",
        "a photo of an irregular shaped skin lesion",
        "a photo of a linear or streak-shaped skin lesion",
        "a photo of multiple scattered skin lesions",
        "a photo of a raised bump or nodule on skin",
        "a photo of a flat discolored patch on skin",
        "a photo of a blister or fluid-filled lesion",
        "a photo of skin swelling or edema",
    ]

    LESION_COLOR_PROMPTS = [
        "skin with normal coloration",
        "erythematous red or pink skin discoloration",
        "hyperpigmented dark brown skin discoloration",
        "hypopigmented light or white skin discoloration",
        "purplish or violaceous skin discoloration",
        "yellowish skin discoloration",
    ]

    LESION_SIZE_PROMPTS = [
        "a small skin lesion less than 1 centimeter",
        "a medium skin lesion 1 to 3 centimeters",
        "a large skin lesion greater than 3 centimeters",
        "skin lesions covering a wide area",
    ]

    TEXTURE_PROMPTS = [
        "smooth skin surface",
        "rough or scaly skin texture",
        "crusty or flaky skin surface",
        "weeping or oozing skin lesion",
        "dry cracked skin",
    ]

    CONDITION_PROMPTS = [
        "a photo of healthy normal skin",
        "a photo of eczema or atopic dermatitis",
        "a photo of psoriasis",
        "a photo of acne",
        "a photo of contact dermatitis",
        "a photo of a fungal skin infection such as ringworm",
        "a photo of urticaria or hives",
        "a photo of impetigo",
        "a photo of cellulitis",
        "a photo of herpes simplex cold sore",
        "a photo of shingles or herpes zoster",
        "a photo of a suspicious mole or possible melanoma",
        "a photo of an insect bite reaction",
        "a photo of a sunburn",
        "a photo of a wart or verruca",
    ]

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the MedSigLIP model."""
        if self._initialized:
            return self._model is not None

        self._initialized = True

        if not _load_medsiglip_dependencies():
            logger.warning("MedSigLIP dependencies not available, falling back to stub")
            return False

        try:
            self._device = "cuda" if _torch.cuda.is_available() else "cpu"
            logger.info(f"Loading MedSigLIP model from {self.MODEL_ID} on {self._device}...")

            self._model = _AutoModel.from_pretrained(self.MODEL_ID).to(self._device)
            self._processor = _AutoProcessor.from_pretrained(self.MODEL_ID)

            logger.info("MedSigLIP model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load MedSigLIP model: {e}")
            self._model = None
            return False

    def _decode_image(self, image_b64: str) -> Optional[Image.Image]:
        """Decode base64 image and resize to 448x448."""
        try:
            # Handle data URL format
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]

            image_bytes = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            # Resize to model's expected input size
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.Resampling.BILINEAR)
            return image
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None

    def _classify_with_prompts(
        self, image: Image.Image, prompts: List[str]
    ) -> List[Tuple[str, float]]:
        """Classify image against a set of text prompts."""
        inputs = self._processor(
            text=prompts,
            images=image,
            padding="max_length",
            return_tensors="pt"
        ).to(self._device)

        with _torch.no_grad():
            outputs = self._model(**inputs)

        # Get probabilities using softmax
        logits = outputs.logits_per_image
        probs = _torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Return sorted results
        results = [(prompts[i], float(probs[i])) for i in range(len(prompts))]
        return sorted(results, key=lambda x: -x[1])

    def _generate_clinical_description(
        self,
        lesion_type: List[Tuple[str, float]],
        color: List[Tuple[str, float]],
        size: List[Tuple[str, float]],
        texture: List[Tuple[str, float]],
        condition: Optional[List[Tuple[str, float]]] = None,
    ) -> str:
        """Generate a clinical description from classification results."""
        parts = []

        # Get top predictions with confidence threshold
        top_type = lesion_type[0] if lesion_type else None
        top_color = color[0] if color else None
        top_size = size[0] if size else None
        top_texture = texture[0] if texture else None

        # Check if there's actually a lesion
        if top_type and "no visible lesion" in top_type[0].lower() and top_type[1] > 0.5:
            return "No visible skin lesion or abnormality detected."

        # Build description
        # Color
        if top_color and top_color[1] > 0.3 and "normal" not in top_color[0].lower():
            if "erythematous" in top_color[0].lower():
                parts.append("Erythematous")
            elif "hyperpigmented" in top_color[0].lower():
                parts.append("Hyperpigmented")
            elif "hypopigmented" in top_color[0].lower():
                parts.append("Hypopigmented")
            elif "purplish" in top_color[0].lower():
                parts.append("Violaceous")
            elif "yellowish" in top_color[0].lower():
                parts.append("Yellowish")

        # Shape/Type
        if top_type and top_type[1] > 0.25:
            if "circular" in top_type[0].lower():
                parts.append("circular lesion")
            elif "irregular" in top_type[0].lower():
                parts.append("irregular-shaped lesion")
            elif "linear" in top_type[0].lower():
                parts.append("linear lesion")
            elif "multiple" in top_type[0].lower():
                parts.append("multiple scattered lesions")
            elif "bump" in top_type[0].lower() or "nodule" in top_type[0].lower():
                parts.append("raised nodule/papule")
            elif "flat" in top_type[0].lower():
                parts.append("flat macule/patch")
            elif "blister" in top_type[0].lower():
                parts.append("vesicular/bullous lesion")
            elif "swelling" in top_type[0].lower():
                parts.append("localized swelling/edema")

        # Size
        if top_size and top_size[1] > 0.3:
            if "small" in top_size[0].lower():
                parts.append("approximately <1cm")
            elif "medium" in top_size[0].lower():
                parts.append("approximately 1-3cm")
            elif "large" in top_size[0].lower():
                parts.append("approximately >3cm")
            elif "wide area" in top_size[0].lower():
                parts.append("covering extensive area")

        # Texture
        if top_texture and top_texture[1] > 0.35 and "smooth" not in top_texture[0].lower():
            if "scaly" in top_texture[0].lower():
                parts.append("with scaly surface")
            elif "crusty" in top_texture[0].lower():
                parts.append("with crusting")
            elif "weeping" in top_texture[0].lower():
                parts.append("with weeping/exudate")
            elif "dry" in top_texture[0].lower():
                parts.append("with dry/fissured appearance")

        if parts:
            return " ".join(parts) + "."
        else:
            return "Skin image analyzed. No definitive abnormalities identified with high confidence."

    async def analyze_image(self, image_b64: str, context: Optional[str] = None) -> ImageAnalysisResult:
        """Analyze a medical image using MedSigLIP."""
        if not self._ensure_initialized():
            logger.warning("MedSigLIP not available — returning unavailable result")
            return ImageAnalysisResult(
                clinical_description="Image analysis model unavailable.",
                confidence=0.0,
                lesion_detected=False,
                skin_lesion=None,
            )

        try:
            # Decode image
            image = self._decode_image(image_b64)
            if image is None:
                return ImageAnalysisResult(
                    clinical_description="Failed to process image.",
                    confidence=0.0,
                    lesion_detected=False,
                )

            # Run classifications
            lesion_type = self._classify_with_prompts(image, self.LESION_TYPE_PROMPTS)
            color = self._classify_with_prompts(image, self.LESION_COLOR_PROMPTS)
            size = self._classify_with_prompts(image, self.LESION_SIZE_PROMPTS)
            texture = self._classify_with_prompts(image, self.TEXTURE_PROMPTS)
            condition = self._classify_with_prompts(image, self.CONDITION_PROMPTS)

            # Check if lesion detected
            no_lesion_score = next(
                (score for prompt, score in lesion_type if "no visible lesion" in prompt.lower()),
                0.0
            )
            lesion_detected = no_lesion_score < 0.5

            # Generate clinical description
            description = self._generate_clinical_description(lesion_type, color, size, texture, condition)

            # Calculate overall confidence
            top_scores = [
                lesion_type[0][1] if lesion_type else 0,
                color[0][1] if color else 0,
            ]
            confidence = sum(top_scores) / len(top_scores)

            # Determine predicted condition
            top_condition = condition[0] if condition else None
            has_condition = (
                top_condition is not None
                and top_condition[1] > 0.2
                and "healthy" not in top_condition[0].lower()
            )
            pred_condition = top_condition[0].replace("a photo of ", "") if has_condition else None
            pred_confidence = top_condition[1] if has_condition else None

            # Create detailed lesion description if detected
            skin_lesion = None
            if lesion_detected:
                skin_lesion = SkinLesionDescription(
                    lesion_type=lesion_type[0][0] if lesion_type else "unknown",
                    color=color[0][0] if color else "unknown",
                    size_estimate=size[0][0] if size else "unknown",
                    texture=texture[0][0] if texture else "unknown",
                    predicted_condition=pred_condition,
                    condition_confidence=pred_confidence,
                    confidence_scores={
                        "type": lesion_type[0][1] if lesion_type else 0,
                        "color": color[0][1] if color else 0,
                        "size": size[0][1] if size else 0,
                        "texture": texture[0][1] if texture else 0,
                        "condition": condition[0][1] if condition else 0,
                    }
                )

            return ImageAnalysisResult(
                clinical_description=description,
                confidence=confidence,
                lesion_detected=lesion_detected,
                skin_lesion=skin_lesion,
                raw_classifications={
                    "lesion_type": [(p, s) for p, s in lesion_type[:3]],
                    "color": [(p, s) for p, s in color[:3]],
                    "size": [(p, s) for p, s in size[:3]],
                    "texture": [(p, s) for p, s in texture[:3]],
                    "condition": [(p, s) for p, s in condition[:3]],
                }
            )

        except Exception as e:
            logger.error(f"MedSigLIP analysis failed: {e}")
            return ImageAnalysisResult(
                clinical_description="Image analysis failed.",
                confidence=0.0,
                lesion_detected=False,
            )


class StubImageAnalyzerClient(ImageAnalyzerClient):
    """Stub implementation for MVP/testing that returns deterministic results."""

    async def analyze_image(self, image_b64: str, context: Optional[str] = None) -> ImageAnalysisResult:
        """Return a sample analysis for testing."""
        # Generate a deterministic but plausible description
        description = "Erythematous circular lesion, approximately 2-3cm, with slightly raised borders."

        return ImageAnalysisResult(
            clinical_description=description,
            confidence=0.75,
            lesion_detected=True,
            skin_lesion=SkinLesionDescription(
                lesion_type="circular lesion",
                color="erythematous (red/pink)",
                size_estimate="approximately 2-3cm",
                texture="smooth surface",
                predicted_condition="eczema or atopic dermatitis",
                condition_confidence=0.72,
                confidence_scores={
                    "type": 0.82,
                    "color": 0.78,
                    "size": 0.65,
                    "texture": 0.71,
                    "condition": 0.72,
                }
            ),
        )


import threading

# Singleton instance
_client: Optional[ImageAnalyzerClient] = None
_client_lock = threading.Lock()


def get_image_analyzer_client() -> ImageAnalyzerClient:
    """
    Factory function to get the image analyzer client.

    Uses real MedSigLIP by default if:
    - MedSigLIP dependencies (transformers, torch) are available
    - USE_STUB_MEDSIGLIP env var is NOT set to "true"

    Falls back to stub if dependencies unavailable or stub explicitly requested.
    """
    global _client

    use_stub = os.environ.get("USE_STUB_MEDSIGLIP", "").lower() == "true"

    if use_stub:
        return StubImageAnalyzerClient()

    with _client_lock:
        if _client is None:
            _client = HuggingFaceMedSigLIPClient()
        return _client
