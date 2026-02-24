"""
Cough and respiratory sound classifier using trained models on HeAR embeddings.

This module provides classification capabilities for:
- Lung sound type (normal/wheeze/crackle/rhonchi/stridor)
- Cough type (dry/wet)
- Severity level

The classifiers are trained on ICBHI, SPRSound, and COUGHVID datasets using HeAR embeddings.
"""

import os
import json
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class CoughCondition(str, Enum):
    """Possible cough conditions."""
    HEALTHY = "healthy"
    ABNORMAL = "abnormal"
    SYMPTOMATIC = "symptomatic"
    UNKNOWN = "unknown"


class CoughType(str, Enum):
    """Types of cough."""
    DRY = "dry"
    WET = "wet"
    PRODUCTIVE = "productive"
    UNKNOWN = "unknown"


class CoughSeverity(str, Enum):
    """Severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class LungSoundType(str, Enum):
    """Types of lung/respiratory sounds."""
    NORMAL = "normal"
    WHEEZE = "wheeze"
    CRACKLE = "crackle"
    BOTH = "both"  # wheeze + crackle
    RHONCHI = "rhonchi"
    STRIDOR = "stridor"
    UNKNOWN = "unknown"


@dataclass
class LungSoundClassification:
    """Classification result for lung sound type."""
    sound_type: LungSoundType
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class CoughClassification:
    """Classification result for a cough sample."""
    condition: CoughCondition
    condition_confidence: float
    condition_probabilities: Dict[str, float]

    cough_type: Optional[CoughType] = None
    type_confidence: Optional[float] = None

    severity: Optional[CoughSeverity] = None
    severity_confidence: Optional[float] = None

    lung_sound: Optional[LungSoundClassification] = None

    # Raw embedding for further analysis
    embedding: Optional[np.ndarray] = None


class CoughClassifierClient:
    """
    Client for cough classification using trained models on HeAR embeddings.

    This classifier runs on top of HeAR embeddings and provides:
    - Lung sound type classification (normal/crackle/wheeze)
    - Cough type classification
    - Severity estimation
    """

    MODELS_DIR = Path(__file__).parent.parent.parent / "ml_training" / "trained_models"

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else self.MODELS_DIR
        self._classifiers: Dict[str, Dict] = {}
        self._loaded = False

    @staticmethod
    def _verify_model_hash(model_path: Path) -> bool:
        """Verify SHA-256 hash of a model file before loading.

        Returns True if the hash matches or no hash file exists (dev/first-run).
        Returns False if the hash file exists but doesn't match (possible tampering).
        """
        hash_path = model_path.with_suffix('.sha256')
        if not hash_path.exists():
            logger.warning(f"No hash file for {model_path} — skipping verification")
            return True
        expected = hash_path.read_text().strip().split()[0]
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual != expected:
            logger.error(f"Hash mismatch for {model_path}: expected {expected[:16]}…, got {actual[:16]}…")
            return False
        return True

    def _load_classifier(self, task_name: str, prefer_ensemble: bool = True) -> Optional[Dict]:
        """Load a trained classifier from disk.

        If prefer_ensemble is True and an ensemble model exists, loads that instead
        of the single best model (ensembles typically have higher F1).
        """
        ensemble_path = self.models_dir / f"{task_name}_ensemble.pkl"
        model_path = self.models_dir / f"{task_name}_model.pkl"
        meta_path = self.models_dir / f"{task_name}_meta.json"

        # Decide which model file to load
        if prefer_ensemble and ensemble_path.exists():
            load_path = ensemble_path
            label = f"{task_name} (ensemble)"
        elif model_path.exists():
            load_path = model_path
            label = task_name
        else:
            logger.warning(f"Classifier not found: {model_path}")
            return None

        if not self._verify_model_hash(load_path):
            logger.error(f"Refusing to load {load_path} — hash verification failed")
            return None

        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)

            metadata = {}
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)

            logger.info(f"Loaded classifier '{label}' (accuracy: {metadata.get('accuracy', 'N/A')})")
            return {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'label_encoder': model_data['label_encoder'],
                'metadata': metadata
            }
        except Exception as e:
            logger.error(f"Failed to load classifier '{label}': {e}")
            return None

    def _ensure_loaded(self):
        """Lazy load classifiers."""
        if self._loaded:
            return

        # Load lung sound type classifier — prefer 3-class, then 4-class, then 6-class
        for task in ['lung_sound_3class_us2p0', 'lung_sound_4class', 'lung_sound']:
            classifier = self._load_classifier(task)
            if classifier:
                self._classifiers['lung_sound'] = classifier
                break

        # Load cough type classifier (dry/wet)
        for task in ['cough_type', 'cough_type_binary']:
            classifier = self._load_classifier(task)
            if classifier:
                self._classifiers['cough_type'] = classifier
                break

        # Load severity classifier
        for task in ['severity', 'severity_simple']:
            classifier = self._load_classifier(task)
            if classifier:
                self._classifiers['severity'] = classifier
                break

        # Fallback: load disease diagnosis classifier if no lung sound model available
        if 'lung_sound' not in self._classifiers:
            condition_classifier = self._load_classifier('diagnosis_simple')
            if condition_classifier:
                self._classifiers['condition'] = condition_classifier
            else:
                classifier = self._load_classifier('status')
                if classifier:
                    self._classifiers['condition'] = classifier

        self._loaded = True

        if not self._classifiers:
            logger.warning("No trained classifiers found. Run training first.")

    def classify(self, embedding: np.ndarray) -> CoughClassification:
        """
        Classify a cough based on its HeAR embedding.

        Args:
            embedding: HeAR embedding vector (512-dimensional)

        Returns:
            CoughClassification with predictions and confidence scores
        """
        self._ensure_loaded()

        result = CoughClassification(
            condition=CoughCondition.UNKNOWN,
            condition_confidence=0.0,
            condition_probabilities={},
            embedding=embedding
        )

        # Classify condition (healthy/abnormal/symptomatic) — legacy fallback
        if 'condition' in self._classifiers:
            clf_data = self._classifiers['condition']
            model = clf_data['model']
            scaler = clf_data['scaler']
            label_encoder = clf_data['label_encoder']

            # Scale if scaler is available (some models don't use scaling per Google's approach)
            X = embedding.reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)

            proba = model.predict_proba(X)[0]
            pred_idx = np.argmax(proba)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

            # Map to enum (support both status and diagnosis_simple label formats)
            condition_map = {
                'healthy': CoughCondition.HEALTHY,
                'COVID-19': CoughCondition.ABNORMAL,
                'infection': CoughCondition.ABNORMAL,
                'symptomatic': CoughCondition.SYMPTOMATIC,
            }
            result.condition = condition_map.get(pred_label, CoughCondition.UNKNOWN)
            result.condition_confidence = float(proba[pred_idx])
            result.condition_probabilities = {
                label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            }

        # Classify cough type
        if 'cough_type' in self._classifiers:
            clf_data = self._classifiers['cough_type']
            model = clf_data['model']
            scaler = clf_data['scaler']
            label_encoder = clf_data['label_encoder']

            X = embedding.reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)

            proba = model.predict_proba(X)[0]
            pred_idx = np.argmax(proba)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

            type_map = {
                'dry': CoughType.DRY,
                'wet': CoughType.WET,
                'productive': CoughType.PRODUCTIVE,
            }
            result.cough_type = type_map.get(pred_label.lower(), CoughType.UNKNOWN)
            result.type_confidence = float(proba[pred_idx])

        # Classify severity
        if 'severity' in self._classifiers:
            clf_data = self._classifiers['severity']
            model = clf_data['model']
            scaler = clf_data['scaler']
            label_encoder = clf_data['label_encoder']

            X = embedding.reshape(1, -1)
            if scaler is not None:
                X = scaler.transform(X)

            proba = model.predict_proba(X)[0]
            pred_idx = np.argmax(proba)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

            severity_map = {
                'mild': CoughSeverity.MILD,
                'moderate': CoughSeverity.MODERATE,
                'severe': CoughSeverity.SEVERE,
            }
            result.severity = severity_map.get(pred_label.lower(), CoughSeverity.UNKNOWN)
            result.severity_confidence = float(proba[pred_idx])

        # Classify lung sound type
        lung_result = self.classify_lung_sound(embedding)
        if lung_result:
            result.lung_sound = lung_result

        return result

    def classify_lung_sound(self, embedding: np.ndarray) -> Optional[LungSoundClassification]:
        """Classify lung sound type from a HeAR embedding.

        Returns LungSoundClassification or None if no lung sound model is loaded.
        """
        self._ensure_loaded()

        if 'lung_sound' not in self._classifiers:
            return None

        clf_data = self._classifiers['lung_sound']
        model = clf_data['model']
        scaler = clf_data['scaler']
        label_encoder = clf_data['label_encoder']

        X = embedding.reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)

        proba = model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        sound_type_map = {
            'normal': LungSoundType.NORMAL,
            'wheeze': LungSoundType.WHEEZE,
            'crackle': LungSoundType.CRACKLE,
            'both': LungSoundType.BOTH,
            'rhonchi': LungSoundType.RHONCHI,
            'stridor': LungSoundType.STRIDOR,
        }

        return LungSoundClassification(
            sound_type=sound_type_map.get(pred_label.lower(), LungSoundType.UNKNOWN),
            confidence=float(proba[pred_idx]),
            probabilities={
                label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(proba)
            },
        )

    def classify_batch(self, embeddings: np.ndarray) -> List[CoughClassification]:
        """Classify multiple cough embeddings."""
        return [self.classify(emb) for emb in embeddings]

    def is_available(self) -> bool:
        """Check if any classifiers are available."""
        self._ensure_loaded()
        return len(self._classifiers) > 0

    def available_tasks(self) -> List[str]:
        """List available classification tasks."""
        self._ensure_loaded()
        return list(self._classifiers.keys())


import threading

# Singleton instance
_classifier_client: Optional[CoughClassifierClient] = None
_classifier_lock = threading.Lock()


def get_cough_classifier() -> CoughClassifierClient:
    """Get the singleton cough classifier client."""
    global _classifier_client
    with _classifier_lock:
        if _classifier_client is None:
            _classifier_client = CoughClassifierClient()
        return _classifier_client
