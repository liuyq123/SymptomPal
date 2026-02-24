#!/usr/bin/env python3
"""
Train a cough classifier on top of HeAR embeddings using COUGHVID dataset.

This script:
1. Loads COUGHVID metadata and audio files
2. Generates HeAR embeddings for each sample
3. Trains classifiers for various tasks:
   - COVID-19 detection (covid_status)
   - Cough type (dry/wet/productive)
   - Severity estimation
4. Saves trained classifier weights

Usage:
    python train_cough_classifier.py --data-dir ../ml_data/coughvid/public_dataset
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
        return "ffmpeg"  # Fall back to system ffmpeg


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
            # Download from HuggingFace
            model_dir = snapshot_download(
                repo_id="google/hear",
                allow_patterns=["saved_model.pb", "variables/*", "fingerprint.pb", "*.json"]
            )
            # Check if saved_model.pb is directly in the directory (not in saved_model/ subdir)
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
        from scipy import signal

        try:
            # Convert to wav if needed (for webm, ogg, etc.)
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
    # Try multiple possible locations for metadata file
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

    # Print available columns
    logger.info(f"Available columns: {list(df.columns)}")

    return df


def prepare_training_data(
    df: pd.DataFrame,
    data_dir: str,
    embedding_generator: HeAREmbeddingGenerator,
    target_column: str = "status",
    max_samples: int = 1000,
    cache_file: Optional[str] = None
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

    Returns:
        X: Feature matrix (embeddings)
        y: Labels
        uuids: List of sample UUIDs
    """
    # Check for cached embeddings
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['X'], cached['y'], cached['uuids']

    # Filter samples with valid labels
    df_filtered = df[df[target_column].notna()].copy()

    # Limit samples
    if len(df_filtered) > max_samples:
        df_filtered = df_filtered.sample(n=max_samples, random_state=42)

    logger.info(f"Processing {len(df_filtered)} samples for '{target_column}'")

    embeddings = []
    labels = []
    uuids = []

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Generating embeddings"):
        uuid = row['uuid']

        # Find audio file (could be .webm, .ogg, or .wav)
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


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    task_name: str = "cough_classification"
) -> Dict:
    """
    Train a logistic regression classifier on HeAR embeddings.

    Returns dict with model, scaler, label_encoder, and metrics.
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

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    logger.info("Training logistic regression classifier...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"\n{task_name} Results:")
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")

    return {
        'model': clf,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'accuracy': accuracy,
        'classes': list(label_encoder.classes_)
    }


def save_classifier(classifier_dict: Dict, output_path: str, task_name: str):
    """Save trained classifier to disk."""
    os.makedirs(output_path, exist_ok=True)

    # Save model
    model_path = os.path.join(output_path, f"{task_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': classifier_dict['model'],
            'scaler': classifier_dict['scaler'],
            'label_encoder': classifier_dict['label_encoder']
        }, f)

    # Save metadata
    meta_path = os.path.join(output_path, f"{task_name}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'task_name': task_name,
            'accuracy': classifier_dict['accuracy'],
            'classes': classifier_dict['classes']
        }, f, indent=2)

    logger.info(f"Saved classifier to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train cough classifier on HeAR embeddings")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to COUGHVID data directory")
    parser.add_argument("--output-dir", type=str, default="./trained_models", help="Output directory for models")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum samples to process")
    parser.add_argument("--target", type=str, default="status", help="Target column to classify")
    parser.add_argument("--cache-dir", type=str, default="./embedding_cache", help="Directory to cache embeddings")
    args = parser.parse_args()

    # Load metadata
    df = load_coughvid_metadata(args.data_dir)

    # Initialize embedding generator
    embedding_gen = HeAREmbeddingGenerator()

    # Prepare cache file path
    cache_file = os.path.join(args.cache_dir, f"{args.target}_embeddings.pkl")

    # Generate embeddings and prepare data
    X, y, uuids = prepare_training_data(
        df, args.data_dir, embedding_gen,
        target_column=args.target,
        max_samples=args.max_samples,
        cache_file=cache_file
    )

    if len(X) < 50:
        logger.error("Not enough samples with valid embeddings. Need at least 50.")
        return

    # Train classifier
    classifier = train_classifier(X, y, task_name=args.target)

    # Save classifier
    save_classifier(classifier, args.output_dir, args.target)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
