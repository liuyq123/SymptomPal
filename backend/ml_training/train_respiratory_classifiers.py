#!/usr/bin/env python3
"""
Train respiratory sound type classifiers using HeAR embeddings.

Two classifiers:
1. Lung sound type: normal/wheeze/crackle/both/rhonchi/stridor (ICBHI + SPRSound)
2. Cough type: dry/wet (COUGHVID expert labels)

Usage:
    python train_respiratory_classifiers.py --stage extract --task lung_sound
    python train_respiratory_classifiers.py --stage embed   --task lung_sound
    python train_respiratory_classifiers.py --stage train   --task lung_sound
    python train_respiratory_classifiers.py --stage all     --task lung_sound

    python train_respiratory_classifiers.py --stage all     --task cough_type
"""

import os
import sys
import json
import pickle
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ML_DATA = Path(__file__).parent.parent / "ml_data"
CACHE_DIR = Path(__file__).parent / "embedding_cache"
OUTPUT_DIR = Path(__file__).parent / "trained_models"
SEGMENTS_DIR = Path(__file__).parent / "extracted_segments"
RANDOM_STATE = 42

SAMPLE_RATE = 16000
SEGMENT_DURATION = 2.0  # seconds, required by HeAR
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
MIN_SEGMENT_DURATION = 0.5  # discard segments shorter than this
WINDOW_HOP = 1.0  # seconds, for sliding window on long segments

# ─── Label Mappings ──────────────────────────────────────────────────────────

ICBHI_LABEL_MAP = {
    (0, 0): "normal",
    (1, 0): "crackle",
    (0, 1): "wheeze",
    (1, 1): "both",
}

SPRSOUND_LABEL_MAP = {
    "Normal": "normal",
    "Wheeze": "wheeze",
    "Fine Crackle": "crackle",
    "Coarse Crackle": "crackle",
    "Wheeze+Crackle": "both",
    "Rhonchi": "rhonchi",
    "Stridor": "stridor",
}

# Classes to drop if sample count < this threshold
MIN_CLASS_SAMPLES = 30


# ─── Audio Utilities ─────────────────────────────────────────────────────────

def load_audio_segment(audio_path: str, start_sec: float, end_sec: float) -> Optional[np.ndarray]:
    """Load a segment from an audio file, resample to 16kHz mono, normalize."""
    import librosa

    try:
        duration = end_sec - start_sec
        if duration < MIN_SEGMENT_DURATION:
            return None

        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, offset=start_sec, duration=duration)

        if len(audio) == 0:
            return None

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio

    except Exception as e:
        logger.debug(f"Failed to load {audio_path} [{start_sec:.1f}-{end_sec:.1f}]: {e}")
        return None


def window_to_segments(audio: np.ndarray, label: str, patient_id: str,
                       source: str, segment_id_prefix: str) -> List[Dict]:
    """Window audio to 2-second segments for HeAR.

    Returns list of dicts with 'audio', 'label', 'patient_id', 'source', 'segment_id'.
    """
    n_samples = len(audio)
    target = SEGMENT_SAMPLES
    segments = []

    if n_samples < int(MIN_SEGMENT_DURATION * SAMPLE_RATE):
        return segments

    if n_samples <= target:
        # Pad with zeros
        padded = np.pad(audio, (0, target - n_samples))
        segments.append({
            'audio': padded,
            'label': label,
            'patient_id': patient_id,
            'source': source,
            'segment_id': f"{segment_id_prefix}_0",
        })
    elif n_samples <= int(4.0 * SAMPLE_RATE):
        # Center-crop to 2 seconds
        start = (n_samples - target) // 2
        segments.append({
            'audio': audio[start:start + target],
            'label': label,
            'patient_id': patient_id,
            'source': source,
            'segment_id': f"{segment_id_prefix}_0",
        })
    else:
        # Sliding window: 2s window, 1s hop
        hop_samples = int(WINDOW_HOP * SAMPLE_RATE)
        idx = 0
        win_num = 0
        while idx + target <= n_samples:
            segments.append({
                'audio': audio[idx:idx + target],
                'label': label,
                'patient_id': patient_id,
                'source': source,
                'segment_id': f"{segment_id_prefix}_{win_num}",
            })
            idx += hop_samples
            win_num += 1

    return segments


def _process_one_annotation(args):
    """Worker function for parallel segment extraction."""
    row, audio_dir = args
    audio_path = row['audio_path']
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(audio_dir, audio_path)

    audio = load_audio_segment(audio_path, row['start_sec'], row['end_sec'])
    if audio is None:
        return []

    return window_to_segments(
        audio,
        label=row['label'],
        patient_id=str(row['patient_id']),
        source=row['source'],
        segment_id_prefix=row.get('segment_id', f"{row['patient_id']}_{row.name}"),
    )


# ─── Stage 1: Extract & Catalog ─────────────────────────────────────────────

def catalog_icbhi(data_dir: Path) -> pd.DataFrame:
    """Parse ICBHI annotation files into a catalog DataFrame."""
    # ICBHI structure: audio_and_txt_files/ contains .wav and .txt files
    # File naming: {patient_id}_{recording_index}_{chest_location}_{recording_equipment}.wav
    # Annotation: start end crackles wheezes (tab-separated)

    audio_dir = data_dir
    # Try common subdirectory structures (Kaggle download has double-nested dirs)
    for subdir in ['audio_and_txt_files',
                   'Respiratory_Sound_Database/audio_and_txt_files',
                   'Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files',
                   'respiratory-sound-database/audio_and_txt_files']:
        candidate = data_dir / subdir
        if candidate.exists():
            audio_dir = candidate
            break

    # Filter to only cycle annotation txt files (exclude metadata files)
    skip_names = {'patient_diagnosis.txt', 'filename_differences.txt', 'filename_format.txt'}
    txt_files = [f for f in sorted(audio_dir.glob("*.txt")) if f.name not in skip_names]

    logger.info(f"Found {len(txt_files)} ICBHI annotation files in {audio_dir}")

    records = []
    for txt_path in txt_files:
        wav_path = txt_path.with_suffix('.wav')
        if not wav_path.exists():
            continue

        # Extract patient ID from filename: {patient_id}_{rec}_{loc}_{equip}
        parts = txt_path.stem.split('_')
        patient_id = f"icbhi_{parts[0]}"

        with open(txt_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                cols = line.split('\t')
                if len(cols) < 4:
                    cols = line.split()
                if len(cols) < 4:
                    continue

                start_sec = float(cols[0])
                end_sec = float(cols[1])
                crackles = int(cols[2])
                wheezes = int(cols[3])

                label = ICBHI_LABEL_MAP.get((crackles, wheezes))
                if label is None:
                    continue

                records.append({
                    'audio_path': str(wav_path),
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'label': label,
                    'patient_id': patient_id,
                    'source': 'icbhi',
                    'segment_id': f"icbhi_{txt_path.stem}_{line_num}",
                })

    df = pd.DataFrame(records)
    logger.info(f"ICBHI catalog: {len(df)} respiratory cycles from {df['patient_id'].nunique()} patients")
    if len(df) > 0:
        logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def catalog_sprsound(data_dir: Path) -> pd.DataFrame:
    """Parse SPRSound JSON annotations into a catalog DataFrame.

    Uses the Detection folder which has event-level annotations (Normal, Wheeze,
    Fine Crackle, Coarse Crackle, Rhonchi, Stridor, Wheeze+Crackle).
    JSON and WAV files are in separate sibling directories.
    """
    records = []

    # Collect all json_dir → wav_dir pairs from Detection folders (train + valid)
    dir_pairs = []
    detection_dir = data_dir / "Detection"
    if detection_dir.exists():
        for json_dir in sorted(detection_dir.glob("*_detection_json")):
            wav_dir = json_dir.parent / json_dir.name.replace("_json", "_wav")
            if wav_dir.exists():
                dir_pairs.append((json_dir, wav_dir))

    # Also check Classification folders as fallback (these have record-level annotations
    # that we can still use for the Classification data's Normal/CAS/DAS labels)
    # But prefer Detection since it has granular event labels.

    if not dir_pairs:
        # Fallback: search recursively
        for json_path in sorted(data_dir.rglob("*_detection_json")):
            if json_path.is_dir():
                wav_dir = json_path.parent / json_path.name.replace("_json", "_wav")
                if wav_dir.exists():
                    dir_pairs.append((json_path, wav_dir))

    logger.info(f"Found {len(dir_pairs)} SPRSound Detection json/wav directory pairs")

    for json_dir, wav_dir in dir_pairs:
        json_files = sorted(json_dir.glob("*.json"))
        logger.info(f"  {json_dir.name}: {len(json_files)} JSON files")

        for json_path in json_files:
            wav_path = wav_dir / (json_path.stem + ".wav")
            if not wav_path.exists():
                continue

            try:
                with open(json_path, 'r') as f:
                    ann = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Extract patient ID from filename: {patient}_{age}_{gender}_{location}_{recording}
            parts = json_path.stem.split('_')
            patient_id = f"spr_{parts[0]}" if parts else f"spr_{json_path.stem}"

            events = ann.get('event_annotation', [])
            if not events:
                continue

            for i, event in enumerate(events):
                event_type = event.get('type', event.get('Type', ''))
                label = SPRSOUND_LABEL_MAP.get(event_type)
                if label is None:
                    continue

                # SPRSound uses milliseconds for start/end
                start_ms = float(event.get('start', event.get('Start', 0)))
                end_ms = float(event.get('end', event.get('End', 0)))
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0

                if end_sec <= start_sec:
                    continue

                records.append({
                    'audio_path': str(wav_path),
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'label': label,
                    'patient_id': patient_id,
                    'source': 'sprsound',
                    'segment_id': f"spr_{json_path.stem}_{i}",
                })

    df = pd.DataFrame(records)
    logger.info(f"SPRSound catalog: {len(df)} events from {df['patient_id'].nunique() if len(df) > 0 else 0} patients")
    if len(df) > 0:
        logger.info(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def catalog_coughvid_cough_type(data_dir: Path) -> pd.DataFrame:
    """Catalog COUGHVID samples with expert cough type labels."""
    meta_path = data_dir / "metadata_compiled.csv"
    if not meta_path.exists():
        meta_path = data_dir / "public_dataset" / "metadata_compiled.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"COUGHVID metadata not found in {data_dir}")

    df = pd.read_csv(meta_path)
    logger.info(f"COUGHVID total records: {len(df)}")

    # Filter for expert cough type labels
    type_cols = [c for c in df.columns if c.startswith('cough_type')]
    logger.info(f"Cough type columns: {type_cols}")

    records = []
    audio_dir = data_dir if (data_dir / "public_dataset").exists() is False else data_dir / "public_dataset"

    # Check if audio files are in the data_dir directly or in public_dataset
    sample_uuid = df['uuid'].iloc[0] if len(df) > 0 else None
    if sample_uuid:
        for candidate_dir in [data_dir, data_dir / "public_dataset"]:
            for ext in ['.webm', '.ogg', '.wav']:
                if (candidate_dir / f"{sample_uuid}{ext}").exists():
                    audio_dir = candidate_dir
                    break

    for _, row in df.iterrows():
        # Determine cough type from expert labels (majority vote)
        type_votes = []
        for col in type_cols:
            val = row.get(col)
            if pd.notna(val) and val not in ('', 'unknown'):
                type_votes.append(str(val).lower().strip())

        if not type_votes:
            continue

        # Majority vote
        from collections import Counter
        label = Counter(type_votes).most_common(1)[0][0]

        # Normalize: 'productive' → 'wet' (they're clinically equivalent)
        if label == 'productive':
            label = 'wet'

        if label not in ('dry', 'wet'):
            continue

        # Find audio file
        uuid = row['uuid']
        audio_path = None
        for ext in ['.webm', '.ogg', '.wav']:
            candidate = audio_dir / f"{uuid}{ext}"
            if candidate.exists():
                audio_path = str(candidate)
                break

        if audio_path is None:
            continue

        # Use cough_detected score if available
        cough_score = row.get('cough_detected', 1.0)
        if pd.notna(cough_score) and float(cough_score) < 0.5:
            continue

        records.append({
            'audio_path': audio_path,
            'start_sec': 0.0,
            'end_sec': -1.0,  # -1 means whole file
            'label': label,
            'patient_id': f"cv_{uuid}",  # Each COUGHVID sample is from a different user
            'source': 'coughvid',
            'segment_id': f"cv_{uuid}",
        })

    df_out = pd.DataFrame(records)
    logger.info(f"COUGHVID cough type catalog: {len(df_out)} samples")
    if len(df_out) > 0:
        logger.info(f"  Label distribution: {df_out['label'].value_counts().to_dict()}")
    return df_out


def extract_lung_sound_catalog() -> pd.DataFrame:
    """Combine ICBHI + SPRSound catalogs."""
    catalogs = []

    icbhi_dir = ML_DATA / "icbhi"
    if icbhi_dir.exists():
        catalogs.append(catalog_icbhi(icbhi_dir))
    else:
        logger.warning(f"ICBHI data not found at {icbhi_dir}")

    sprsound_dir = ML_DATA / "sprsound"
    if sprsound_dir.exists():
        catalogs.append(catalog_sprsound(sprsound_dir))
    else:
        logger.warning(f"SPRSound data not found at {sprsound_dir}")

    if not catalogs:
        raise FileNotFoundError("No lung sound datasets found. Download ICBHI and/or SPRSound first.")

    combined = pd.concat(catalogs, ignore_index=True)

    # Drop rare classes
    class_counts = combined['label'].value_counts()
    logger.info(f"Combined label distribution:\n{class_counts}")

    rare_classes = class_counts[class_counts < MIN_CLASS_SAMPLES].index.tolist()
    if rare_classes:
        logger.warning(f"Dropping rare classes (< {MIN_CLASS_SAMPLES} samples): {rare_classes}")
        combined = combined[~combined['label'].isin(rare_classes)]

    logger.info(f"Final lung sound catalog: {len(combined)} annotations, "
                f"{combined['patient_id'].nunique()} patients, "
                f"{combined['label'].nunique()} classes")
    return combined


def extract_segments_parallel(catalog: pd.DataFrame, task_name: str,
                              n_workers: int = None) -> List[Dict]:
    """Extract and window audio segments in parallel.

    Returns list of dicts with 'audio' (np.ndarray), 'label', 'patient_id', 'source', 'segment_id'.
    """
    if n_workers is None:
        n_workers = min(24, cpu_count())

    logger.info(f"Extracting segments with {n_workers} workers...")

    # For coughvid entries with end_sec=-1, we load the whole file
    rows_to_process = []
    for _, row in catalog.iterrows():
        row_dict = row.to_dict()
        row_dict['name'] = _  # preserve index for segment naming
        rows_to_process.append((row_dict, ''))

    all_segments = []
    # Process in chunks for memory efficiency
    chunk_size = 500
    for chunk_start in range(0, len(rows_to_process), chunk_size):
        chunk = rows_to_process[chunk_start:chunk_start + chunk_size]

        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(_process_one_annotation_wrapper, chunk),
                total=len(chunk),
                desc=f"Extracting {task_name} [{chunk_start}:{chunk_start + len(chunk)}]"
            ))

        for segs in results:
            all_segments.extend(segs)

    logger.info(f"Extracted {len(all_segments)} 2-second segments from {len(catalog)} annotations")

    # Report label distribution
    if all_segments:
        labels = [s['label'] for s in all_segments]
        from collections import Counter
        dist = Counter(labels)
        logger.info(f"Segment label distribution: {dict(dist)}")

    return all_segments


def _process_one_annotation_wrapper(args):
    """Wrapper that handles both lung sound annotations and whole-file cough type entries."""
    import librosa

    row, audio_dir = args
    audio_path = row['audio_path']

    start_sec = row['start_sec']
    end_sec = row['end_sec']

    try:
        if end_sec < 0:
            # Whole file (cough type)
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            if len(audio) == 0:
                return []
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        else:
            audio = load_audio_segment(audio_path, start_sec, end_sec)
            if audio is None:
                return []

        return window_to_segments(
            audio,
            label=row['label'],
            patient_id=str(row['patient_id']),
            source=row['source'],
            segment_id_prefix=row.get('segment_id', f"{row['patient_id']}"),
        )
    except Exception as e:
        return []


# ─── Stage 2: Generate HeAR Embeddings ──────────────────────────────────────

def generate_embeddings_batch(segments: List[Dict], batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Generate HeAR embeddings for all segments using batched GPU inference.

    Returns (X, y, patient_ids, segment_ids).
    """
    import tensorflow as tf
    from train_multi_classifier import HeAREmbeddingGenerator

    # Initialize HeAR model
    generator = HeAREmbeddingGenerator()
    generator._ensure_initialized()
    model = generator.model

    n = len(segments)
    embeddings = []
    labels = []
    patient_ids = []
    segment_ids = []

    logger.info(f"Generating embeddings for {n} segments (batch_size={batch_size})...")

    for batch_start in tqdm(range(0, n, batch_size), desc="Embedding batches"):
        batch = segments[batch_start:batch_start + batch_size]

        # Stack audio into a single tensor
        audio_batch = np.stack([s['audio'] for s in batch])  # (B, 32000)
        audio_tensor = tf.constant(audio_batch, dtype=tf.float32)

        # Run batch through HeAR
        output = model(audio_tensor)
        if isinstance(output, dict):
            emb_batch = list(output.values())[0].numpy()
        else:
            emb_batch = output.numpy()

        # Collect results
        for i, seg in enumerate(batch):
            embeddings.append(emb_batch[i])
            labels.append(seg['label'])
            patient_ids.append(seg['patient_id'])
            segment_ids.append(seg['segment_id'])

    X = np.array(embeddings)
    y = np.array(labels)
    logger.info(f"Generated {X.shape[0]} embeddings of dimension {X.shape[1]}")

    return X, y, patient_ids, segment_ids


# ─── Stage 3: Train ─────────────────────────────────────────────────────────

def train_with_patient_split(X: np.ndarray, y: np.ndarray, patient_ids: List[str],
                             task_name: str, undersample_ratio: float = 0.0,
                             tune: bool = False) -> Dict:
    """Train classifiers with patient-level train/test split to prevent data leakage.

    Args:
        undersample_ratio: If > 0, cap the majority class in training to this ratio
            of the largest minority class. E.g. 1.5 means majority gets 1.5x the
            next-largest class. 0 means no undersampling.
        tune: If True, run RandomizedSearchCV for hyperparameter tuning.
    """
    from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.ensemble import (VotingClassifier, StackingClassifier,
                                  RandomForestClassifier, GradientBoostingClassifier)
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neural_network import MLPClassifier

    try:
        from xgboost import XGBClassifier
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False
        logger.warning("XGBoost not installed — skipping. Install with: pip install xgboost")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    groups = np.array(patient_ids)

    logger.info(f"Classes: {list(label_encoder.classes_)}")
    logger.info(f"Total samples: {len(X)}, Unique patients: {len(np.unique(groups))}")
    for cls in label_encoder.classes_:
        count = np.sum(y == cls)
        logger.info(f"  {cls}: {count} samples ({100*count/len(y):.1f}%)")

    # Patient-level train/test split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    groups_train = groups[train_idx]

    logger.info(f"Train: {len(X_train)} samples ({len(np.unique(groups[train_idx]))} patients)")
    logger.info(f"Test:  {len(X_test)} samples ({len(np.unique(groups[test_idx]))} patients)")

    # Undersample majority class in training set only (test set stays full)
    if undersample_ratio > 0:
        class_counts = np.bincount(y_train)
        majority_class = np.argmax(class_counts)
        second_largest = sorted(class_counts)[-2]
        cap = int(second_largest * undersample_ratio)

        if class_counts[majority_class] > cap:
            logger.info(f"Undersampling class '{label_encoder.classes_[majority_class]}': "
                        f"{class_counts[majority_class]} → {cap} samples")
            majority_idx = np.where(y_train == majority_class)[0]
            rng = np.random.RandomState(RANDOM_STATE)
            keep_idx = rng.choice(majority_idx, size=cap, replace=False)
            minority_idx = np.where(y_train != majority_class)[0]
            train_keep = np.sort(np.concatenate([keep_idx, minority_idx]))

            X_train = X_train[train_keep]
            y_train = y_train[train_keep]
            groups_train = groups_train[train_keep]

            logger.info(f"Train after undersampling: {len(X_train)} samples")
            for i, cls in enumerate(label_encoder.classes_):
                count = np.sum(y_train == i)
                logger.info(f"  {cls}: {count} ({100*count/len(y_train):.1f}%)")

    # Feature scaling — critical for LinearSVC, LogReg, MLP, SGD
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    logger.info(f"Applied StandardScaler (mean={scaler.mean_[:3].round(4)}..., "
                f"std={scaler.scale_[:3].round(4)}...)")

    num_classes = len(np.unique(y_train))

    # Fast classifiers for large datasets (LinearSVC is O(n) vs SVC's O(n²-n³))
    classifiers = {
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE),
            cv=3
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced',
                                                   random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=128, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=128, class_weight='balanced',
                                                 random_state=RANDOM_STATE, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                             random_state=RANDOM_STATE, early_stopping=True),
        "SGD (log_loss)": CalibratedClassifierCV(
            SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=1000,
                          random_state=RANDOM_STATE, n_jobs=-1),
            cv=3
        ),
    }

    if HAS_XGBOOST:
        # Compute sample weights for XGBoost (doesn't support class_weight='balanced')
        from sklearn.utils.class_weight import compute_sample_weight
        xgb_sample_weights = compute_sample_weight('balanced', y_train)
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=256, max_depth=6, learning_rate=0.1,
            tree_method='hist', device='cuda',
            random_state=RANDOM_STATE, eval_metric='mlogloss',
            use_label_encoder=False,
        )

    # Hyperparameter grids for tuning
    param_grids = {
        "LinearSVC": {"estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "Logistic Regression": {"C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "MLP": {
            "hidden_layer_sizes": [(256, 128), (512, 256), (256, 128, 64), (512, 256, 128)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-4, 5e-4, 1e-3],
        },
        "Random Forest": {
            "n_estimators": [128, 256, 512],
            "max_depth": [None, 20, 50],
            "min_samples_leaf": [1, 2, 5],
        },
        "Gradient Boosting": {
            "n_estimators": [128, 256],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        },
        "SGD (log_loss)": {"estimator__alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
    }
    if HAS_XGBOOST:
        param_grids["XGBoost"] = {
            "n_estimators": [128, 256, 512],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "min_child_weight": [1, 3, 5],
        }

    results = {}
    best_model = None
    best_f1 = 0
    best_name = None

    for name, clf in classifiers.items():
        logger.info(f"\nTraining {name}...")

        try:
            fit_params = {}
            if name == "XGBoost" and HAS_XGBOOST:
                fit_params['sample_weight'] = xgb_sample_weights

            if tune and name in param_grids:
                logger.info(f"  Tuning hyperparameters with RandomizedSearchCV...")
                gkf = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
                search = RandomizedSearchCV(
                    clf, param_grids[name],
                    n_iter=min(20, np.prod([len(v) if isinstance(v, list) else 1
                                            for v in param_grids[name].values()])),
                    cv=gkf, scoring='f1_weighted', random_state=RANDOM_STATE,
                    n_jobs=-1, error_score='raise',
                )
                search.fit(X_train, y_train, groups=groups_train, **fit_params)
                clf = search.best_estimator_
                cv_mean = search.best_score_
                logger.info(f"  Best params: {search.best_params_}")
                logger.info(f"  CV F1: {cv_mean:.3f}")
            else:
                clf.fit(X_train, y_train, **fit_params)
                cv_mean = 0.0

            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results[name] = {
                'model': clf,
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_mean,
            }

            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  F1 Score: {f1:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_name = name

        except Exception as e:
            logger.error(f"  Failed to train {name}: {e}")
            continue

    if best_model is None:
        raise RuntimeError("All classifiers failed to train")

    # Best model report
    logger.info(f"\n{'='*60}")
    logger.info(f"Best Model: {best_name} (F1={best_f1:.3f})")
    y_pred_best = best_model.predict(X_test)
    logger.info(f"\n{classification_report(y_test, y_pred_best, target_names=label_encoder.classes_)}")

    # Stacking ensemble from top 4 models (meta-learner learns optimal weighting)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    top_n = sorted_results[:min(4, len(sorted_results))]
    logger.info(f"\nTop {len(top_n)} for stacking ensemble:")
    for name, res in top_n:
        logger.info(f"  {name}: F1={res['f1_score']:.3f}")

    ensemble_estimators = [(name.replace(' ', '_').replace('(', '').replace(')', ''),
                            res['model']) for name, res in top_n]

    # Stacking: meta-learner (LogReg) trained on base model predictions
    # Patient-level split already happened at outer level; simple 5-fold is fine for meta-learner
    stacking = StackingClassifier(
        estimators=ensemble_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)

    stacking_pred = stacking.predict(X_test)
    stacking_accuracy = accuracy_score(y_test, stacking_pred)
    stacking_f1 = f1_score(y_test, stacking_pred, average='weighted')

    logger.info(f"\nStacking Ensemble:")
    logger.info(f"  Accuracy: {stacking_accuracy:.3f}")
    logger.info(f"  F1 Score: {stacking_f1:.3f}")
    logger.info(f"\n{classification_report(y_test, stacking_pred, target_names=label_encoder.classes_)}")

    # Also try soft voting for comparison
    voting = VotingClassifier(estimators=ensemble_estimators, voting='soft')
    voting.fit(X_train, y_train)
    voting_pred = voting.predict(X_test)
    voting_f1 = f1_score(y_test, voting_pred, average='weighted')
    voting_accuracy = accuracy_score(y_test, voting_pred)
    logger.info(f"\nSoft Voting Ensemble (for comparison):")
    logger.info(f"  Accuracy: {voting_accuracy:.3f}")
    logger.info(f"  F1 Score: {voting_f1:.3f}")

    # Use whichever ensemble is better
    if stacking_f1 >= voting_f1:
        ensemble = stacking
        ensemble_f1 = stacking_f1
        ensemble_accuracy = stacking_accuracy
        ensemble_type = "stacking"
    else:
        ensemble = voting
        ensemble_f1 = voting_f1
        ensemble_accuracy = voting_accuracy
        ensemble_type = "voting"
    logger.info(f"\nUsing {ensemble_type} ensemble (F1={ensemble_f1:.3f})")

    return {
        'best_model': best_model,
        'best_name': best_name,
        'ensemble': ensemble,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'all_results': results,
        'best_accuracy': accuracy_score(y_test, y_pred_best),
        'best_f1': best_f1,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_f1': ensemble_f1,
        'classes': list(label_encoder.classes_),
    }


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_extract(task: str) -> pd.DataFrame:
    """Stage 1: Build catalog of annotations."""
    if task == 'lung_sound':
        return extract_lung_sound_catalog()
    elif task == 'cough_type':
        coughvid_dir = ML_DATA / "coughvid"
        return catalog_coughvid_cough_type(coughvid_dir)
    else:
        raise ValueError(f"Unknown task: {task}")


def run_embed(task: str, catalog: Optional[pd.DataFrame] = None,
              segments: Optional[List[Dict]] = None, batch_size: int = 64) -> str:
    """Stage 2: Extract audio segments and generate HeAR embeddings.

    Returns path to cached embeddings file.
    """
    cache_path = CACHE_DIR / f"{task}_respiratory_embeddings.pkl"

    if cache_path.exists():
        logger.info(f"Embeddings already cached at {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        logger.info(f"  Shape: {cached['X'].shape}, Classes: {np.unique(cached['y']).tolist()}")
        return str(cache_path)

    if catalog is None:
        catalog = run_extract(task)

    if segments is None:
        segments = extract_segments_parallel(catalog, task)

    if not segments:
        raise RuntimeError(f"No segments extracted for task '{task}'")

    X, y, patient_ids, segment_ids = generate_embeddings_batch(segments, batch_size=batch_size)

    # Save cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'X': X,
            'y': y,
            'patient_ids': patient_ids,
            'segment_ids': segment_ids,
        }, f)
    logger.info(f"Cached embeddings to {cache_path}")

    return str(cache_path)


LUNG_SOUND_COLLAPSE_MAP = {
    'normal': 'normal',
    'crackle': 'crackle',
    'wheeze': 'wheeze',
    'both': 'both',
    'rhonchi': 'wheeze',   # low-pitched continuous → wheeze family
    'stridor': 'wheeze',   # high-pitched continuous → wheeze family
}

LUNG_SOUND_3CLASS_MAP = {
    'normal': 'normal',
    'crackle': 'crackle',
    'wheeze': 'wheeze',
    'both': 'wheeze',      # mixed → wheeze (has wheeze component)
    'rhonchi': 'wheeze',
    'stridor': 'wheeze',
}


def run_train(task: str, cache_path: Optional[str] = None,
              collapse_classes: bool = False, task_suffix: str = "",
              undersample_ratio: float = 0.0, tune: bool = False):
    """Stage 3: Train classifiers on cached embeddings."""
    from train_multi_classifier import save_classifier

    if cache_path is None:
        cache_path = str(CACHE_DIR / f"{task}_respiratory_embeddings.pkl")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}. Run --stage embed first.")

    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)

    X = cached['X']
    y = cached['y']
    patient_ids = cached['patient_ids']

    logger.info(f"Loaded {X.shape[0]} embeddings ({X.shape[1]}-dim) for task '{task}'")

    # Collapse rare classes into broader categories
    if collapse_classes and task == 'lung_sound':
        y_original = y.copy()
        # Pick collapse map based on suffix hint
        if '3class' in task_suffix:
            cmap = LUNG_SOUND_3CLASS_MAP
        else:
            cmap = LUNG_SOUND_COLLAPSE_MAP
        y = np.array([cmap.get(label, label) for label in y])
        n_before = len(np.unique(y_original))
        n_after = len(np.unique(y))
        logger.info(f"Collapsed {n_before} classes → {n_after} classes")
        for cls in sorted(np.unique(y)):
            count = np.sum(y == cls)
            logger.info(f"  {cls}: {count} samples ({100*count/len(y):.1f}%)")

    save_task = f"{task}{task_suffix}" if task_suffix else task
    result = train_with_patient_split(X, y, patient_ids, save_task,
                                       undersample_ratio=undersample_ratio,
                                       tune=tune)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_classifier(result, str(OUTPUT_DIR), save_task)

    logger.info(f"\nTraining complete for '{save_task}'!")
    logger.info(f"Best model: {result['best_name']} (F1={result['best_f1']:.3f})")
    logger.info(f"Ensemble F1: {result['ensemble_f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train respiratory sound type classifiers")
    parser.add_argument("--stage", choices=['extract', 'embed', 'train', 'all'], required=True)
    parser.add_argument("--task", choices=['lung_sound', 'cough_type'], required=True)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for HeAR inference")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers for extraction")
    parser.add_argument("--force", action='store_true', help="Force re-run even if cache exists")
    parser.add_argument("--collapse-classes", choices=['4', '3'], default=None,
                        help="Merge rare classes: '4'=merge rhonchi/stridor→wheeze, '3'=also merge both→wheeze")
    parser.add_argument("--undersample", type=float, default=0.0,
                        help="Undersample majority class to N times the second-largest class (e.g. 1.5)")
    parser.add_argument("--tune", action='store_true',
                        help="Run RandomizedSearchCV hyperparameter tuning (slower but better)")
    args = parser.parse_args()

    start = time()

    if args.force:
        cache_path = CACHE_DIR / f"{args.task}_respiratory_embeddings.pkl"
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Removed existing cache: {cache_path}")

    if args.stage == 'extract':
        catalog = run_extract(args.task)
        logger.info(f"Catalog has {len(catalog)} entries")

    elif args.stage == 'embed':
        run_embed(args.task, batch_size=args.batch_size)

    elif args.stage == 'train':
        suffix = f"_{args.collapse_classes}class" if args.collapse_classes else ""
        if args.undersample > 0:
            suffix += f"_us{args.undersample:.1f}".replace('.', 'p')
        run_train(args.task, collapse_classes=bool(args.collapse_classes),
                  task_suffix=suffix, undersample_ratio=args.undersample,
                  tune=args.tune)

    elif args.stage == 'all':
        catalog = run_extract(args.task)
        segments = extract_segments_parallel(catalog, args.task, n_workers=args.workers)
        cache_path = run_embed(args.task, catalog=catalog, segments=segments,
                               batch_size=args.batch_size)
        suffix = f"_{args.collapse_classes}class" if args.collapse_classes else ""
        if args.undersample > 0:
            suffix += f"_us{args.undersample:.1f}".replace('.', 'p')
        run_train(args.task, cache_path=cache_path,
                  collapse_classes=args.collapse_classes, task_suffix=suffix,
                  undersample_ratio=args.undersample, tune=args.tune)

    elapsed = time() - start
    logger.info(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
