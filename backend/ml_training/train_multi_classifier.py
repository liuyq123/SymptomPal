#!/usr/bin/env python3
"""
Train multiple classifiers on HeAR embeddings using techniques from Google's example.

This script follows Google's data-efficient approach:
1. Uses multiple classifier types (SVM, LogReg, GradientBoosting, RandomForest, MLP)
2. Trains directly on embeddings WITHOUT StandardScaler (as per Google's example)
3. Uses expert-labeled data for better quality labels
4. Saves an ensemble model that combines predictions

Usage:
    python train_multi_classifier.py --data-dir ../ml_data/coughvid/public_dataset
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_ffmpeg_path() -> str:
    """Get ffmpeg binary path from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def convert_audio_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """Convert audio file to WAV using ffmpeg."""
    import subprocess
    ffmpeg = get_ffmpeg_path()
    try:
        result = subprocess.run(
            [ffmpeg, "-i", input_path, "-ar", str(sample_rate), "-ac", "1", output_path, "-y"],
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Failed to convert {input_path}: {e}")
        return False


class HeAREmbeddingGenerator:
    """Generate HeAR embeddings for audio files."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._initialized = False
        self._temp_dir = None

    def _ensure_initialized(self):
        if self._initialized:
            return

        import tensorflow as tf
        import keras
        from huggingface_hub import snapshot_download
        import tempfile

        logger.info("Loading HeAR model...")

        if self.model_path and os.path.exists(self.model_path):
            model_dir = self.model_path
        else:
            model_dir = snapshot_download(
                repo_id="google/hear",
                allow_patterns=["saved_model.pb", "variables/*", "fingerprint.pb", "*.json"]
            )
            if not os.path.exists(os.path.join(model_dir, "saved_model.pb")):
                model_dir = os.path.join(model_dir, "saved_model")

        self.model = keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")
        self._temp_dir = tempfile.mkdtemp(prefix="hear_training_")
        self._initialized = True
        logger.info("HeAR model loaded successfully")

    def generate_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Generate HeAR embedding for a single audio file."""
        self._ensure_initialized()

        import tensorflow as tf
        import librosa

        try:
            # Convert to wav if needed
            if audio_path.endswith('.webm') or audio_path.endswith('.ogg'):
                wav_path = os.path.join(self._temp_dir, "temp_audio.wav")
                if not convert_audio_to_wav(audio_path, wav_path):
                    return None
                audio_path = wav_path

            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=None)

            # Resample to 16kHz if needed
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # Pad or trim to 2 seconds (32000 samples)
            target_length = 32000
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            # Run through HeAR model
            audio_tensor = tf.constant(audio.reshape(1, -1), dtype=tf.float32)
            output = self.model(audio_tensor)

            # Extract embedding
            if isinstance(output, dict):
                embedding = list(output.values())[0].numpy()
            else:
                embedding = output.numpy()

            return embedding.flatten()

        except Exception as e:
            logger.warning(f"Failed to process {audio_path}: {e}")
            return None


def load_coughvid_metadata(data_dir: str) -> pd.DataFrame:
    """Load and preprocess COUGHVID metadata."""
    possible_paths = [
        os.path.join(data_dir, "metadata_compiled.csv"),
        os.path.join(os.path.dirname(data_dir), "metadata_compiled.csv"),
    ]

    metadata_path = None
    for path in possible_paths:
        if os.path.exists(path):
            metadata_path = path
            break

    if metadata_path is None:
        raise FileNotFoundError(f"Metadata file not found. Tried: {possible_paths}")

    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} records from metadata")
    logger.info(f"Available columns: {list(df.columns)}")

    return df


def prepare_training_data(
    df: pd.DataFrame,
    data_dir: str,
    embedding_generator: HeAREmbeddingGenerator,
    target_column: str = "status",
    max_samples: int = 1000,
    cache_file: Optional[str] = None,
    min_cough_score: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare training data by generating embeddings for audio files.

    Args:
        df: Metadata DataFrame
        data_dir: Directory containing audio files
        embedding_generator: HeAR embedding generator
        target_column: Column to use as label
        max_samples: Maximum number of samples to process
        cache_file: Optional path to cache embeddings
        min_cough_score: Minimum cough_detected score to include sample
    """
    # Check for cached embeddings
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['X'], cached['y'], cached['uuids']

    # Filter samples with valid labels
    df_filtered = df[df[target_column].notna()].copy()

    # Filter by cough detection score if available
    if 'cough_detected' in df_filtered.columns:
        original_count = len(df_filtered)
        df_filtered = df_filtered[df_filtered['cough_detected'] >= min_cough_score]
        logger.info(f"Filtered to samples with cough_detected >= {min_cough_score}: {original_count} -> {len(df_filtered)}")

    # Limit samples per class for balance
    if len(df_filtered) > max_samples:
        # Stratified sampling to maintain class balance
        df_filtered = df_filtered.groupby(target_column, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_samples // df_filtered[target_column].nunique()), random_state=42)
        )

    logger.info(f"Processing {len(df_filtered)} samples for '{target_column}'")
    logger.info(f"Class distribution: {df_filtered[target_column].value_counts().to_dict()}")

    embeddings = []
    labels = []
    uuids = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Generating embeddings"):
        uuid = row['uuid']

        # Find audio file
        audio_path = None
        for ext in ['.webm', '.ogg', '.wav']:
            candidate = os.path.join(data_dir, f"{uuid}{ext}")
            if os.path.exists(candidate):
                audio_path = candidate
                break

        if audio_path is None:
            continue

        # Generate embedding
        embedding = embedding_generator.generate_embedding(audio_path)
        if embedding is not None:
            embeddings.append(embedding)
            labels.append(row[target_column])
            uuids.append(uuid)

    X = np.array(embeddings)
    y = np.array(labels)

    logger.info(f"Generated {len(X)} embeddings")

    # Cache embeddings
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'X': X, 'y': y, 'uuids': uuids}, f)
        logger.info(f"Cached embeddings to {cache_file}")

    return X, y, uuids


def create_classifiers() -> Dict[str, any]:
    """
    Create multiple classifiers following Google's example.

    Note: Google's example does NOT use StandardScaler - they train directly
    on raw embeddings. We follow the same approach here.
    """
    return {
        "SVM (linear)": SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
        "SVM (rbf)": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=128, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=128, class_weight='balanced', random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42),
    }


def train_and_evaluate_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    task_name: str = "classification"
) -> Dict:
    """
    Train multiple classifiers and evaluate them.

    Following Google's example:
    - No StandardScaler (train directly on embeddings)
    - Multiple classifier types
    - Select best performing model
    """
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logger.info(f"Classes: {list(label_encoder.classes_)}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    # Create classifiers
    classifiers = create_classifiers()

    results = {}
    best_model = None
    best_accuracy = 0
    best_f1 = 0
    best_name = None

    for name, clf in classifiers.items():
        logger.info(f"\nTraining {name}...")

        try:
            # Train
            clf.fit(X_train, y_train)

            # Evaluate
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Cross-validation score
            cv_scores = cross_val_score(clf, X, y_encoded, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            results[name] = {
                'model': clf,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
            }

            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  F1 Score: {f1:.3f}")
            logger.info(f"  CV Score: {cv_mean:.3f} (+/- {cv_std:.3f})")

            # Track best model (using F1 as primary metric for imbalanced data)
            if f1 > best_f1:
                best_f1 = f1
                best_accuracy = accuracy
                best_model = clf
                best_name = name

        except Exception as e:
            logger.error(f"  Failed to train {name}: {e}")
            continue

    # Print best model results
    logger.info(f"\n{'='*60}")
    logger.info(f"Best Model: {best_name}")
    logger.info(f"  Accuracy: {best_accuracy:.3f}")
    logger.info(f"  F1 Score: {best_f1:.3f}")

    # Detailed classification report for best model
    y_pred_best = best_model.predict(X_test)
    logger.info(f"\nClassification Report ({best_name}):")
    logger.info(f"\n{classification_report(y_test, y_pred_best, target_names=label_encoder.classes_)}")

    # Create ensemble voting classifier from top 3 models
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    top_3 = sorted_results[:3]

    logger.info(f"\nTop 3 models for ensemble:")
    for name, res in top_3:
        logger.info(f"  {name}: F1={res['f1_score']:.3f}")

    # Create soft voting ensemble
    ensemble_estimators = [(name.replace(' ', '_'), res['model']) for name, res in top_3]
    ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    ensemble.fit(X_train, y_train)

    ensemble_pred = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')

    logger.info(f"\nEnsemble (Soft Voting):")
    logger.info(f"  Accuracy: {ensemble_accuracy:.3f}")
    logger.info(f"  F1 Score: {ensemble_f1:.3f}")
    logger.info(f"\n{classification_report(y_test, ensemble_pred, target_names=label_encoder.classes_)}")

    return {
        'best_model': best_model,
        'best_name': best_name,
        'ensemble': ensemble,
        'label_encoder': label_encoder,
        'all_results': results,
        'best_accuracy': best_accuracy,
        'best_f1': best_f1,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_f1': ensemble_f1,
        'classes': list(label_encoder.classes_)
    }


def save_classifier(result: Dict, output_path: str, task_name: str):
    """Save trained classifiers to disk."""
    os.makedirs(output_path, exist_ok=True)

    scaler = result.get('scaler')  # StandardScaler, or None for legacy models

    # Save best model (for backward compatibility with existing inference code)
    model_path = os.path.join(output_path, f"{task_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': result['best_model'],
            'scaler': scaler,
            'label_encoder': result['label_encoder']
        }, f)
    logger.info(f"Saved best model to {model_path}")

    # Save ensemble model
    ensemble_path = os.path.join(output_path, f"{task_name}_ensemble.pkl")
    with open(ensemble_path, 'wb') as f:
        pickle.dump({
            'model': result['ensemble'],
            'scaler': scaler,
            'label_encoder': result['label_encoder']
        }, f)
    logger.info(f"Saved ensemble model to {ensemble_path}")

    # Save metadata
    meta_path = os.path.join(output_path, f"{task_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'task_name': task_name,
            'best_model': result['best_name'],
            'accuracy': result['best_accuracy'],
            'f1_score': result['best_f1'],
            'ensemble_accuracy': result['ensemble_accuracy'],
            'ensemble_f1': result['ensemble_f1'],
            'classes': result['classes'],
            'model_scores': {
                name: {'accuracy': res['accuracy'], 'f1_score': res['f1_score'], 'cv_mean': res['cv_mean']}
                for name, res in result['all_results'].items()
            }
        }, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Train multi-classifier on HeAR embeddings")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to COUGHVID data directory")
    parser.add_argument("--output-dir", type=str, default="./trained_models", help="Output directory for models")
    parser.add_argument("--max-samples", type=int, default=800, help="Maximum samples to process")
    parser.add_argument("--target", type=str, default="status", help="Target column to classify")
    parser.add_argument("--cache-dir", type=str, default="./embedding_cache", help="Directory to cache embeddings")
    parser.add_argument("--min-cough-score", type=float, default=0.8, help="Minimum cough detection score")
    args = parser.parse_args()

    # Load metadata
    df = load_coughvid_metadata(args.data_dir)

    # Initialize embedding generator
    embedding_gen = HeAREmbeddingGenerator()

    # Prepare cache file path
    cache_file = os.path.join(args.cache_dir, f"{args.target}_multi_embeddings.pkl")

    # Generate embeddings and prepare data
    X, y, uuids = prepare_training_data(
        df, args.data_dir, embedding_gen,
        target_column=args.target,
        max_samples=args.max_samples,
        cache_file=cache_file,
        min_cough_score=args.min_cough_score
    )

    if len(X) < 50:
        logger.error("Not enough samples with valid embeddings. Need at least 50.")
        return

    # Train classifiers
    result = train_and_evaluate_classifiers(X, y, task_name=args.target)

    # Save classifiers
    save_classifier(result, args.output_dir, args.target)

    logger.info("\nTraining complete!")
    logger.info(f"Best model: {result['best_name']} with F1={result['best_f1']:.3f}")
    logger.info(f"Ensemble F1: {result['ensemble_f1']:.3f}")


if __name__ == "__main__":
    main()
