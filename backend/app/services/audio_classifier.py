"""
Audio classifier service for ambient monitoring.

Uses Google's HeAR (Health Acoustic Representations) model for health acoustic analysis.
HeAR produces 512-dimensional embeddings from 2-second audio clips that can be used
for downstream classification tasks like cough detection, breathing analysis, etc.

References:
- https://huggingface.co/google/hear
- https://developers.google.com/health-ai-developer-foundations/hear
"""

import os
import random
import uuid
import base64
import tempfile
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

# Thread pool for TensorFlow operations (TF has issues in asyncio context)
_tf_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tf_inference")

from ..models import (
    AmbientEvent,
    AmbientEventType,
    SessionType,
    CoughMetrics,
    SleepQualityMetrics,
    VoiceBiomarkers,
    AmbientSessionResult,
)

logger = logging.getLogger(__name__)

# Lazy load heavy dependencies
_tf = None
_librosa = None
_scipy_signal = None


def _load_hear_dependencies():
    """Lazy load HeAR dependencies."""
    global _tf, _librosa, _scipy_signal
    if _tf is None:
        try:
            import tensorflow as tf
            import librosa
            from scipy import signal as scipy_signal
            _tf = tf
            _librosa = librosa
            _scipy_signal = scipy_signal
            logger.info("HeAR dependencies loaded successfully")
            return True
        except ImportError as e:
            logger.warning(f"HeAR dependencies not available: {e}")
            return False
    return True


class AudioClassifierClient(ABC):
    """Abstract base class for audio classification."""

    @abstractmethod
    def analyze_chunk(
        self,
        audio_b64: str,
        session_type: SessionType,
        session_id: str,
        user_id: str,
        chunk_index: int,
        chunk_timestamp: datetime,
    ) -> List[AmbientEvent]:
        """
        Analyze an audio chunk and return detected events.

        Args:
            audio_b64: Base64-encoded audio data
            session_type: Type of monitoring session
            session_id: ID of the ambient session
            user_id: User ID
            chunk_index: Sequential index of this chunk
            chunk_timestamp: When this chunk was recorded

        Returns:
            List of detected AmbientEvent objects
        """
        pass

    @abstractmethod
    def compute_session_metrics(
        self,
        session_type: SessionType,
        events: List[AmbientEvent],
        total_duration_seconds: float,
    ) -> AmbientSessionResult:
        """
        Compute aggregated metrics for a completed session.

        Args:
            session_type: Type of monitoring session
            events: All events detected during the session
            total_duration_seconds: Total session duration

        Returns:
            AmbientSessionResult with computed metrics
        """
        pass


class HuggingFaceHeARClient(AudioClassifierClient):
    """
    Real HeAR implementation using Google's model from Hugging Face.

    HeAR Model Details:
    - Input: 16kHz mono audio, exactly 2 seconds (32,000 samples)
    - Output: 512-dimensional embeddings
    - Model ID: google/hear

    This implementation uses embedding-based classification:
    - Computes HeAR embeddings for 2-second audio windows
    - Uses energy detection + embedding analysis for event classification
    - Applies learned thresholds for cough, breathing, and voice analysis
    """

    MODEL_ID = "google/hear"
    SAMPLE_RATE = 16000
    CLIP_SAMPLES = 32000  # 2 seconds at 16kHz
    EMBEDDING_DIM = 512

    # Thresholds for event detection (tuned for ESC-50 and real-world audio)
    COUGH_ENERGY_THRESHOLD = 0.08  # Lowered for normalized audio
    COUGH_EMBEDDING_VARIANCE_THRESHOLD = 0.01  # Lowered to be more sensitive
    SNORING_LOW_FREQ_RATIO = 0.6
    BREATHING_REGULARITY_THRESHOLD = 0.7

    def __init__(self):
        self._model = None
        self._serving_fn = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the HeAR model."""
        if self._initialized:
            return self._model is not None

        self._initialized = True

        if not _load_hear_dependencies():
            logger.warning("HeAR dependencies not available, falling back to stub")
            return False

        try:
            logger.info(f"Loading HeAR model from {self.MODEL_ID}...")
            # First, download the model using huggingface_hub
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(self.MODEL_ID)

            # Load as TFSMLayer for Keras 3 compatibility
            import keras
            self._model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
            logger.info("HeAR model loaded successfully with TFSMLayer")
            return True
        except Exception as e:
            logger.error(f"Failed to load HeAR model: {e}")
            self._model = None
            return False

    def _decode_audio(self, audio_b64: str) -> Optional[np.ndarray]:
        """Decode base64 audio and resample to 16kHz mono."""
        try:
            audio_bytes = base64.b64decode(audio_b64)

            # Detect format from magic bytes
            suffix = ".webm"  # default for browser recordings
            if audio_bytes[:4] == b'RIFF':
                suffix = ".wav"
            elif audio_bytes[:4] == b'fLaC':
                suffix = ".flac"
            elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
                suffix = ".mp3"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            try:
                # Load and resample to 16kHz mono
                audio, _ = _librosa.load(temp_path, sr=self.SAMPLE_RATE, mono=True)
                logger.info(f"Decoded audio: {len(audio)} samples ({len(audio)/self.SAMPLE_RATE:.1f}s)")
                return audio
            finally:
                os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Failed to decode audio: {e}")
            return None

    def _compute_embeddings(self, audio: np.ndarray) -> np.ndarray:
        """Compute HeAR embeddings for audio, chunking into 2-second windows."""
        embeddings = []

        # Pad audio if shorter than 2 seconds
        if len(audio) < self.CLIP_SAMPLES:
            audio = np.pad(audio, (0, self.CLIP_SAMPLES - len(audio)))

        # Process in 2-second windows with 1-second hop
        hop_samples = self.SAMPLE_RATE  # 1 second hop
        for start in range(0, len(audio) - self.CLIP_SAMPLES + 1, hop_samples):
            clip = audio[start:start + self.CLIP_SAMPLES]
            clip_batch = clip.reshape(1, -1).astype(np.float32)
            # TFSMLayer returns dict with 'output_0' key
            result = self._model(clip_batch)
            embedding = result['output_0'].numpy()[0]
            embeddings.append(embedding)

        if not embeddings:
            # Single clip if audio is exactly 2 seconds
            clip_batch = audio[:self.CLIP_SAMPLES].reshape(1, -1).astype(np.float32)
            result = self._model(clip_batch)
            embeddings.append(result['output_0'].numpy()[0])

        return np.array(embeddings)

    def _compute_audio_features(self, audio: np.ndarray) -> dict:
        """Compute additional audio features for classification."""
        features = {}

        # RMS energy
        features['rms'] = float(np.sqrt(np.mean(audio ** 2)))

        # Zero crossing rate
        features['zcr'] = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

        # Spectral features
        stft = np.abs(_librosa.stft(audio))
        freqs = _librosa.fft_frequencies(sr=self.SAMPLE_RATE)

        # Spectral centroid (average frequency)
        spectral_sum = np.sum(stft, axis=1) + 1e-10
        features['spectral_centroid'] = float(np.sum(freqs * spectral_sum) / np.sum(spectral_sum))

        # Low frequency ratio (below 500Hz)
        low_freq_mask = freqs < 500
        features['low_freq_ratio'] = float(np.sum(stft[low_freq_mask]) / (np.sum(stft) + 1e-10))

        # High frequency ratio (above 2000Hz)
        high_freq_mask = freqs > 2000
        features['high_freq_ratio'] = float(np.sum(stft[high_freq_mask]) / (np.sum(stft) + 1e-10))

        # Detect transients (sudden energy changes - indicative of coughs)
        envelope = np.abs(_scipy_signal.hilbert(audio))
        envelope_diff = np.diff(envelope)
        features['transient_count'] = int(np.sum(envelope_diff > 0.1 * np.max(envelope)))

        return features

    def _detect_coughs(
        self,
        audio: np.ndarray,
        embeddings: np.ndarray,
        features: dict,
        chunk_timestamp: datetime,
        session_id: str,
        user_id: str,
        chunk_index: int,
    ) -> List[AmbientEvent]:
        """Detect cough events using HeAR embeddings and audio features."""
        events = []

        # Coughs are characterized by:
        # 1. High transient energy (sudden onset)
        # 2. Specific embedding patterns (learned from HeAR training data)
        # 3. Duration typically 0.5-2 seconds

        # Compute embedding variance (coughs have distinctive patterns)
        embedding_variance = np.var(embeddings, axis=0).mean()

        # Energy-based cough detection
        window_size = self.SAMPLE_RATE // 4  # 250ms windows
        hop_size = self.SAMPLE_RATE // 8     # 125ms hop

        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            window_energy = np.sqrt(np.mean(window ** 2))

            # Cough detection criteria:
            # - High energy spike
            # - Sufficient embedding variance
            # - Transient characteristics
            if (window_energy > self.COUGH_ENERGY_THRESHOLD and
                embedding_variance > self.COUGH_EMBEDDING_VARIANCE_THRESHOLD and
                features['transient_count'] > 0):

                # Determine intensity based on energy
                if window_energy > 0.3:
                    intensity = "severe"
                elif window_energy > 0.2:
                    intensity = "moderate"
                else:
                    intensity = "mild"

                offset_seconds = i / self.SAMPLE_RATE

                # Build metadata with classification if available
                metadata = {
                    "intensity": intensity,
                    "energy": float(round(window_energy, 3))
                }

                # Try to classify the cough using trained classifier
                try:
                    from .cough_classifier import get_cough_classifier
                    classifier = get_cough_classifier()
                    if classifier.is_available():
                        # Use mean embedding for classification
                        mean_embedding = embeddings.mean(axis=0)
                        classification = classifier.classify(mean_embedding)

                        # Add classification results to metadata
                        if classification.condition.value != "unknown":
                            metadata["condition"] = classification.condition.value
                            metadata["condition_confidence"] = float(round(classification.condition_confidence, 3))
                            metadata["condition_probabilities"] = {
                                k: float(round(v, 3))
                                for k, v in classification.condition_probabilities.items()
                            }

                        if classification.cough_type and classification.cough_type.value != "unknown":
                            metadata["cough_type"] = classification.cough_type.value
                            metadata["type_confidence"] = float(round(classification.type_confidence, 3))

                        if classification.severity and classification.severity.value != "unknown":
                            metadata["severity"] = classification.severity.value
                            metadata["severity_confidence"] = float(round(classification.severity_confidence, 3))

                        if classification.lung_sound and classification.lung_sound.sound_type.value != "unknown":
                            metadata["lung_sound_type"] = classification.lung_sound.sound_type.value
                            metadata["lung_sound_confidence"] = float(round(classification.lung_sound.confidence, 3))
                            metadata["lung_sound_probabilities"] = {
                                k: float(round(v, 3))
                                for k, v in classification.lung_sound.probabilities.items()
                            }
                except Exception as e:
                    logger.debug(f"Cough classification not available: {e}")

                events.append(AmbientEvent(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    user_id=user_id,
                    event_type=AmbientEventType.COUGH,
                    timestamp=chunk_timestamp + timedelta(seconds=offset_seconds),
                    duration_seconds=float(0.5 + (window_energy * 2)),  # Scale duration by energy
                    confidence=float(min(0.95, 0.7 + embedding_variance * 2)),
                    metadata=metadata,
                    chunk_index=chunk_index,
                ))
                # Skip ahead to avoid double-counting the same cough
                break

        return events

    def _detect_breathing_events(
        self,
        audio: np.ndarray,
        embeddings: np.ndarray,
        features: dict,
        chunk_timestamp: datetime,
        session_id: str,
        user_id: str,
        chunk_index: int,
    ) -> List[AmbientEvent]:
        """Detect breathing and sleep-related events."""
        events = []

        # Analyze embedding patterns for breathing classification
        embedding_std = np.std(embeddings, axis=0).mean()
        embedding_mean = np.mean(np.abs(embeddings))

        # Snoring detection: low frequency dominance + specific patterns
        if features['low_freq_ratio'] > self.SNORING_LOW_FREQ_RATIO and features['rms'] > 0.05:
            events.append(AmbientEvent(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=AmbientEventType.SNORING,
                timestamp=chunk_timestamp,
                duration_seconds=float(len(audio) / self.SAMPLE_RATE * 0.7),
                confidence=float(min(0.9, 0.6 + features['low_freq_ratio'] * 0.3)),
                metadata={"low_freq_ratio": float(round(features['low_freq_ratio'], 3))},
                chunk_index=chunk_index,
            ))
            return events

        # Apnea detection: very low energy for extended period
        if features['rms'] < 0.01 and len(audio) > self.SAMPLE_RATE * 5:
            events.append(AmbientEvent(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=AmbientEventType.BREATHING_APNEA,
                timestamp=chunk_timestamp,
                duration_seconds=float(len(audio) / self.SAMPLE_RATE),
                confidence=0.75,
                metadata={"energy": float(round(features['rms'], 4))},
                chunk_index=chunk_index,
            ))
            return events

        # Normal vs irregular breathing based on embedding consistency
        if embedding_std < self.BREATHING_REGULARITY_THRESHOLD:
            event_type = AmbientEventType.BREATHING_NORMAL
            confidence = float(min(0.9, 0.7 + (1 - embedding_std) * 0.2))
        else:
            event_type = AmbientEventType.BREATHING_IRREGULAR
            confidence = float(min(0.85, 0.6 + embedding_std * 0.2))

        events.append(AmbientEvent(
            id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            timestamp=chunk_timestamp,
            duration_seconds=float(len(audio) / self.SAMPLE_RATE * 0.5),
            confidence=confidence,
            metadata={"embedding_std": float(round(embedding_std, 3))},
            chunk_index=chunk_index,
        ))

        return events

    def _detect_voice_biomarkers(
        self,
        audio: np.ndarray,
        embeddings: np.ndarray,
        features: dict,
        chunk_timestamp: datetime,
        session_id: str,
        user_id: str,
        chunk_index: int,
    ) -> List[AmbientEvent]:
        """Detect voice-related health indicators."""
        events = []

        # Check if there's actual voice activity
        if features['rms'] < 0.02:
            events.append(AmbientEvent(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=AmbientEventType.SILENCE,
                timestamp=chunk_timestamp,
                duration_seconds=float(len(audio) / self.SAMPLE_RATE),
                confidence=0.9,
                metadata=None,
                chunk_index=chunk_index,
            ))
            return events

        # Embedding-based voice analysis
        embedding_norm = np.linalg.norm(embeddings, axis=1).mean()
        embedding_variance = np.var(embeddings, axis=0).mean()

        # Stress detection: higher frequency content + embedding patterns
        if features['high_freq_ratio'] > 0.3 and embedding_variance > 0.1:
            events.append(AmbientEvent(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=AmbientEventType.VOICE_STRESS,
                timestamp=chunk_timestamp,
                duration_seconds=float(len(audio) / self.SAMPLE_RATE * 0.6),
                confidence=float(min(0.85, 0.6 + features['high_freq_ratio'] * 0.5)),
                metadata={"level": "high" if features['high_freq_ratio'] > 0.4 else "moderate"},
                chunk_index=chunk_index,
            ))

        # Fatigue detection: lower energy, reduced spectral range
        if features['spectral_centroid'] < 1500 and features['rms'] < 0.1:
            events.append(AmbientEvent(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=AmbientEventType.VOICE_FATIGUE,
                timestamp=chunk_timestamp,
                duration_seconds=float(len(audio) / self.SAMPLE_RATE * 0.5),
                confidence=0.7,
                metadata={"level": "moderate"},
                chunk_index=chunk_index,
            ))

        # Congestion detection: nasal quality, specific frequency patterns
        if features['low_freq_ratio'] > 0.4 and features['spectral_centroid'] < 1200:
            events.append(AmbientEvent(
                id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=AmbientEventType.VOICE_CONGESTION,
                timestamp=chunk_timestamp,
                duration_seconds=float(len(audio) / self.SAMPLE_RATE * 0.4),
                confidence=0.65,
                metadata={"level": "moderate"},
                chunk_index=chunk_index,
            ))

        return events

    def analyze_chunk(
        self,
        audio_b64: str,
        session_type: SessionType,
        session_id: str,
        user_id: str,
        chunk_index: int,
        chunk_timestamp: datetime,
    ) -> List[AmbientEvent]:
        """Analyze audio chunk using HeAR embeddings."""
        logger.debug("analyze_chunk called, audio_b64_length=%s", len(audio_b64))

        if not self._ensure_initialized():
            logger.warning("HeAR not available — returning empty events")
            return []

        logger.debug("HeAR initialized, processing audio")

        try:
            # Decode and process audio
            logger.debug("Decoding audio chunk")
            audio = self._decode_audio(audio_b64)
            if audio is None or len(audio) < self.SAMPLE_RATE:
                logger.warning(
                    "Audio too short or failed to decode, decode_failed=%s sample_count=%s",
                    audio is None,
                    len(audio) if audio is not None else 0,
                )
                return []

            logger.debug("Audio decoded, sample_count=%s", len(audio))

            # Compute HeAR embeddings
            logger.debug("Computing HeAR embeddings")
            embeddings = self._compute_embeddings(audio)
            logger.debug("Embeddings computed, shape=%s", embeddings.shape)

            # Compute additional audio features
            features = self._compute_audio_features(audio)

            # Detect events based on session type
            if session_type == SessionType.COUGH_MONITOR:
                return self._detect_coughs(
                    audio, embeddings, features, chunk_timestamp,
                    session_id, user_id, chunk_index
                )
            elif session_type == SessionType.SLEEP:
                return self._detect_breathing_events(
                    audio, embeddings, features, chunk_timestamp,
                    session_id, user_id, chunk_index
                )
            elif session_type == SessionType.VOICE_BIOMARKER:
                return self._detect_voice_biomarkers(
                    audio, embeddings, features, chunk_timestamp,
                    session_id, user_id, chunk_index
                )
            else:  # GENERAL
                events = []
                # Try all detectors for general monitoring
                events.extend(self._detect_coughs(
                    audio, embeddings, features, chunk_timestamp,
                    session_id, user_id, chunk_index
                ))
                if not events:
                    events.extend(self._detect_voice_biomarkers(
                        audio, embeddings, features, chunk_timestamp,
                        session_id, user_id, chunk_index
                    ))
                return events[:2]  # Limit to top 2 events

        except Exception as e:
            logger.error(f"HeAR analysis failed: {e}")
            return []

    async def analyze_chunk_async(
        self,
        audio_b64: str,
        session_type: SessionType,
        session_id: str,
        user_id: str,
        chunk_index: int,
        chunk_timestamp: datetime,
    ) -> List[AmbientEvent]:
        """
        Async version that runs TensorFlow inference in a thread pool.
        This avoids TensorFlow + asyncio compatibility issues.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _tf_executor,
            self.analyze_chunk,
            audio_b64,
            session_type,
            session_id,
            user_id,
            chunk_index,
            chunk_timestamp,
        )

    def compute_session_metrics(
        self,
        session_type: SessionType,
        events: List[AmbientEvent],
        total_duration_seconds: float,
    ) -> AmbientSessionResult:
        """Compute aggregated metrics (same logic as stub)."""
        # Reuse the stub's metrics computation - it's well-implemented
        stub = StubAudioClassifierClient()
        return stub.compute_session_metrics(session_type, events, total_duration_seconds)


class StubAudioClassifierClient(AudioClassifierClient):
    """
    Stub implementation that returns deterministic test data.
    Replace with real HeAR-based classifier for production.
    """

    def analyze_chunk(
        self,
        audio_b64: str,
        session_type: SessionType,
        session_id: str,
        user_id: str,
        chunk_index: int,
        chunk_timestamp: datetime,
    ) -> List[AmbientEvent]:
        """Generate deterministic events based on session type and chunk index."""
        events: List[AmbientEvent] = []

        # Use chunk_index as seed for deterministic but varied results
        random.seed(chunk_index * 17 + hash(session_id) % 1000)

        if session_type == SessionType.COUGH_MONITOR:
            # Simulate 0-3 coughs per 30-second chunk
            num_coughs = random.randint(0, 3) if chunk_index % 3 != 0 else 0
            for i in range(num_coughs):
                offset_seconds = random.uniform(0, 28)
                events.append(
                    AmbientEvent(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_id=user_id,
                        event_type=AmbientEventType.COUGH,
                        timestamp=chunk_timestamp + timedelta(seconds=offset_seconds),
                        duration_seconds=random.uniform(0.5, 2.0),
                        confidence=random.uniform(0.75, 0.98),
                        metadata={"intensity": random.choice(["mild", "moderate", "severe"])},
                        chunk_index=chunk_index,
                    )
                )

        elif session_type == SessionType.SLEEP:
            # Simulate breathing patterns and sleep events
            breathing_types = [
                (AmbientEventType.BREATHING_NORMAL, 0.7),
                (AmbientEventType.BREATHING_IRREGULAR, 0.15),
                (AmbientEventType.SNORING, 0.1),
                (AmbientEventType.BREATHING_APNEA, 0.05),
            ]

            # Usually detect 1-2 breathing events per chunk
            for _ in range(random.randint(1, 2)):
                event_type = random.choices(
                    [t[0] for t in breathing_types],
                    weights=[t[1] for t in breathing_types],
                )[0]

                offset_seconds = random.uniform(0, 25)
                duration = random.uniform(3.0, 15.0)

                events.append(
                    AmbientEvent(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_id=user_id,
                        event_type=event_type,
                        timestamp=chunk_timestamp + timedelta(seconds=offset_seconds),
                        duration_seconds=duration,
                        confidence=random.uniform(0.70, 0.95),
                        metadata=None,
                        chunk_index=chunk_index,
                    )
                )

        elif session_type == SessionType.VOICE_BIOMARKER:
            # Simulate voice analysis events
            voice_types = [
                (AmbientEventType.VOICE_STRESS, 0.3),
                (AmbientEventType.VOICE_FATIGUE, 0.3),
                (AmbientEventType.VOICE_CONGESTION, 0.2),
                (AmbientEventType.SILENCE, 0.2),
            ]

            # Detect 0-2 voice events per chunk
            for _ in range(random.randint(0, 2)):
                event_type = random.choices(
                    [t[0] for t in voice_types],
                    weights=[t[1] for t in voice_types],
                )[0]

                offset_seconds = random.uniform(0, 25)
                events.append(
                    AmbientEvent(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_id=user_id,
                        event_type=event_type,
                        timestamp=chunk_timestamp + timedelta(seconds=offset_seconds),
                        duration_seconds=random.uniform(2.0, 10.0),
                        confidence=random.uniform(0.65, 0.92),
                        metadata={"level": random.choice(["low", "moderate", "high"])},
                        chunk_index=chunk_index,
                    )
                )

        else:  # GENERAL
            # Mix of various events
            if random.random() > 0.5:
                events.append(
                    AmbientEvent(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        user_id=user_id,
                        event_type=random.choice([
                            AmbientEventType.COUGH,
                            AmbientEventType.NOISE,
                            AmbientEventType.SILENCE,
                        ]),
                        timestamp=chunk_timestamp + timedelta(seconds=random.uniform(0, 25)),
                        duration_seconds=random.uniform(1.0, 5.0),
                        confidence=random.uniform(0.70, 0.90),
                        metadata=None,
                        chunk_index=chunk_index,
                    )
                )

        return events

    def compute_session_metrics(
        self,
        session_type: SessionType,
        events: List[AmbientEvent],
        total_duration_seconds: float,
    ) -> AmbientSessionResult:
        """Compute metrics based on detected events."""
        duration_minutes = total_duration_seconds / 60.0
        duration_hours = total_duration_seconds / 3600.0

        cough_metrics: Optional[CoughMetrics] = None
        sleep_quality: Optional[SleepQualityMetrics] = None
        voice_biomarkers: Optional[VoiceBiomarkers] = None
        summary = ""

        if session_type == SessionType.COUGH_MONITOR:
            cough_events = [e for e in events if e.event_type == AmbientEventType.COUGH]
            total_coughs = len(cough_events)
            coughs_per_hour = total_coughs / duration_hours if duration_hours > 0 else 0

            # Find peak cough period (simplified: just report the chunk with most coughs)
            chunk_counts: dict[int, int] = {}
            for e in cough_events:
                chunk_counts[e.chunk_index] = chunk_counts.get(e.chunk_index, 0) + 1

            peak_period = None
            if chunk_counts:
                peak_chunk = max(chunk_counts.keys(), key=lambda k: chunk_counts[k])
                peak_period = f"Chunk {peak_chunk} ({chunk_counts[peak_chunk]} coughs)"

            # Aggregate classification data from cough events
            dominant_condition = None
            condition_confidence = None
            condition_probabilities = None
            dominant_cough_type = None
            cough_type_confidence = None
            dominant_severity = None
            severity_confidence = None
            dominant_lung_sound = None
            lung_sound_confidence = None

            if cough_events:
                # Get classification from the first cough with classification data
                for e in cough_events:
                    if e.metadata:
                        if "condition" in e.metadata and dominant_condition is None:
                            dominant_condition = e.metadata.get("condition")
                            condition_confidence = e.metadata.get("condition_confidence")
                            condition_probabilities = e.metadata.get("condition_probabilities")
                        if "cough_type" in e.metadata and dominant_cough_type is None:
                            dominant_cough_type = e.metadata.get("cough_type")
                            cough_type_confidence = e.metadata.get("type_confidence")
                        if "severity" in e.metadata and dominant_severity is None:
                            dominant_severity = e.metadata.get("severity")
                            severity_confidence = e.metadata.get("severity_confidence")
                        if "lung_sound_type" in e.metadata and dominant_lung_sound is None:
                            dominant_lung_sound = e.metadata.get("lung_sound_type")
                            lung_sound_confidence = e.metadata.get("lung_sound_confidence")
                        # Break once we have all classification data
                        if dominant_cough_type and dominant_lung_sound:
                            break

            cough_metrics = CoughMetrics(
                total_coughs=total_coughs,
                coughs_per_hour=round(coughs_per_hour, 1),
                peak_cough_period=peak_period,
                cough_intensity_avg=0.6 if total_coughs > 0 else None,
                dominant_condition=dominant_condition,
                condition_confidence=condition_confidence,
                condition_probabilities=condition_probabilities,
                dominant_cough_type=dominant_cough_type,
                cough_type_confidence=cough_type_confidence,
                dominant_severity=dominant_severity,
                severity_confidence=severity_confidence,
                dominant_lung_sound=dominant_lung_sound,
                lung_sound_confidence=lung_sound_confidence,
            )

            if total_coughs == 0:
                summary = f"No coughs detected during {duration_minutes:.0f} minute session."
            elif coughs_per_hour < 5:
                summary = f"Detected {total_coughs} coughs ({coughs_per_hour:.1f}/hr) - minimal coughing."
            elif coughs_per_hour < 15:
                summary = f"Detected {total_coughs} coughs ({coughs_per_hour:.1f}/hr) - moderate coughing."
            else:
                summary = f"Detected {total_coughs} coughs ({coughs_per_hour:.1f}/hr) - frequent coughing, consider tracking triggers."

            # Append doctor-visit recommendation for concerning classifications
            if (dominant_condition and dominant_condition.lower() == "abnormal"
                    and (condition_confidence or 0) > 0.6):
                summary += " Based on the cough patterns, consider consulting a healthcare provider."
            elif (dominant_severity and dominant_severity.lower() == "severe"
                    and (severity_confidence or 0) > 0.6):
                summary += " The cough intensity is elevated; consider seeing a doctor if it persists."

        elif session_type == SessionType.SLEEP:
            apnea_events = [e for e in events if e.event_type == AmbientEventType.BREATHING_APNEA]
            snoring_events = [e for e in events if e.event_type == AmbientEventType.SNORING]
            irregular_events = [e for e in events if e.event_type == AmbientEventType.BREATHING_IRREGULAR]
            normal_events = [e for e in events if e.event_type == AmbientEventType.BREATHING_NORMAL]

            snoring_duration = sum(e.duration_seconds or 0 for e in snoring_events)
            snoring_minutes = snoring_duration / 60.0

            # Calculate breathing regularity score
            total_breathing = len(normal_events) + len(irregular_events)
            regularity_score = (len(normal_events) / total_breathing * 100) if total_breathing > 0 else 80.0

            # Calculate restlessness (based on irregular breathing and events)
            restlessness = min(100, len(irregular_events) * 10 + len(apnea_events) * 15)

            # Determine quality rating
            if regularity_score >= 85 and len(apnea_events) == 0 and snoring_minutes < 10:
                quality_rating = "excellent"
            elif regularity_score >= 70 and len(apnea_events) <= 2:
                quality_rating = "good"
            elif regularity_score >= 50:
                quality_rating = "fair"
            else:
                quality_rating = "poor"

            sleep_quality = SleepQualityMetrics(
                total_sleep_duration_minutes=duration_minutes,
                breathing_regularity_score=round(regularity_score, 1),
                apnea_events=len(apnea_events),
                snoring_minutes=round(snoring_minutes, 1),
                restlessness_score=round(restlessness, 1),
                quality_rating=quality_rating,
            )

            summary = f"Sleep quality: {quality_rating}. {len(apnea_events)} apnea events, {snoring_minutes:.0f}min snoring, {regularity_score:.0f}% regular breathing."

        elif session_type == SessionType.VOICE_BIOMARKER:
            stress_events = [e for e in events if e.event_type == AmbientEventType.VOICE_STRESS]
            fatigue_events = [e for e in events if e.event_type == AmbientEventType.VOICE_FATIGUE]
            congestion_events = [e for e in events if e.event_type == AmbientEventType.VOICE_CONGESTION]

            # Calculate levels based on event frequency and confidence
            stress_level = min(100, len(stress_events) * 20)
            fatigue_level = min(100, len(fatigue_events) * 25)
            congestion_detected = len(congestion_events) > 0

            # Voice clarity inversely related to issues detected
            clarity_score = max(0, 100 - stress_level * 0.3 - fatigue_level * 0.3 - (20 if congestion_detected else 0))

            voice_biomarkers = VoiceBiomarkers(
                stress_level=round(stress_level, 1),
                fatigue_level=round(fatigue_level, 1),
                congestion_detected=congestion_detected,
                voice_clarity_score=round(clarity_score, 1),
            )

            indicators = []
            if stress_level > 40:
                indicators.append("elevated stress")
            if fatigue_level > 40:
                indicators.append("fatigue detected")
            if congestion_detected:
                indicators.append("possible congestion")

            if indicators:
                summary = f"Voice analysis indicates: {', '.join(indicators)}."
            else:
                summary = "Voice analysis shows normal patterns with no significant concerns."

        else:  # GENERAL
            event_counts: dict[str, int] = {}
            for e in events:
                event_counts[e.event_type.value] = event_counts.get(e.event_type.value, 0) + 1

            if event_counts:
                top_events = sorted(event_counts.items(), key=lambda x: -x[1])[:3]
                event_summary = ", ".join(f"{count} {etype}" for etype, count in top_events)
                summary = f"General monitoring detected: {event_summary}."
            else:
                summary = f"No significant events detected during {duration_minutes:.0f} minute session."

        return AmbientSessionResult(
            session_id="",  # Will be set by caller
            session_type=session_type,
            duration_minutes=round(duration_minutes, 1),
            cough_metrics=cough_metrics,
            sleep_quality=sleep_quality,
            voice_biomarkers=voice_biomarkers,
            events_timeline=events,
            summary=summary,
        )


import threading as _threading

# Singleton instance
_client: Optional[AudioClassifierClient] = None
_client_lock = _threading.Lock()


def get_audio_classifier_client() -> AudioClassifierClient:
    """
    Factory function to get the audio classifier client.

    Uses real HeAR model by default if:
    - HeAR dependencies (tensorflow, huggingface_hub) are available
    - USE_STUB_HEAR env var is NOT set to "true"

    Falls back to stub if dependencies unavailable or stub explicitly requested.
    """
    global _client

    use_stub = os.environ.get("USE_STUB_HEAR", "").lower() == "true"

    if use_stub:
        return StubAudioClassifierClient()

    with _client_lock:
        if _client is None:
            _client = HuggingFaceHeARClient()
        return _client
