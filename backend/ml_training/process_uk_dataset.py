#!/usr/bin/env python3
"""
Download and process UK COVID-19 Vocal Audio Dataset.

Dataset: https://www.kaggle.com/datasets/andrewmvd/covid19-cough-audio-classification
- 73,000+ participants with PCR test results
- 53.7GB across 25 zip files
- Multiple audio types (cough, breathing, voice)

Pipeline:
  1. Download all 25 zip files from Kaggle
  2. Extract and catalog cough audio
  3. Generate HeAR embeddings (with checkpointing)
  4. Merge with existing data (COUGHVID + Coswara + Virufy)
  5. Retrain binary classifier

Usage:
    python process_uk_dataset.py --stage all
    python process_uk_dataset.py --stage download
    python process_uk_dataset.py --stage extract
    python process_uk_dataset.py --stage embed
    python process_uk_dataset.py --stage train
"""

import os
import sys
import pickle
import json
import logging
import argparse
import zipfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ML_DATA = Path(__file__).parent.parent / "ml_data"
UK_DATA = ML_DATA / "uk_covid_audio"
CACHE_DIR = Path(__file__).parent / "embedding_cache"
RANDOM_STATE = 42

# Kaggle dataset URL parts (25 zip files)
KAGGLE_DATASET = "andrewmvd/covid19-cough-audio-classification"
ZIP_FILES = [f"audio-{i:02d}.zip" for i in range(1, 26)]


def check_kaggle_auth():
    """Check if Kaggle API is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.error("Kaggle API not configured!")
        logger.error("Please set up Kaggle API:")
        logger.error("  1. Go to https://www.kaggle.com/settings/account")
        logger.error("  2. Click 'Create New API Token'")
        logger.error("  3. Move kaggle.json to ~/.kaggle/")
        logger.error("  4. chmod 600 ~/.kaggle/kaggle.json")
        return False
    return True


def download_uk_dataset():
    """Download UK COVID-19 audio dataset from Kaggle."""
    if not check_kaggle_auth():
        return False

    UK_DATA.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading UK COVID-19 dataset from Kaggle to {UK_DATA}")
    logger.info("This will download 53.7GB (25 zip files). Press Ctrl+C to cancel.")

    # Download entire dataset using kaggle CLI
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(UK_DATA),
        "--unzip"
    ]

    try:
        subprocess.run(cmd, check=True)
        logger.info("Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_and_catalog():
    """Extract audio files and create catalog."""
    logger.info("Cataloging audio files...")

    # Read metadata_compiled.csv which maps to the actual .webm files
    metadata_path = UK_DATA / 'metadata_compiled.csv'
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return None

    metadata = pd.read_csv(metadata_path)
    logger.info(f"Metadata entries: {len(metadata)}")

    # Filter for rows with valid status (COVID-19, healthy, symptomatic)
    metadata = metadata[metadata['status'].isin(['COVID-19', 'healthy', 'symptomatic'])].copy()
    logger.info(f"Entries with valid status: {len(metadata)}")
    logger.info(f"  COVID-19: {(metadata['status'] == 'COVID-19').sum()}")
    logger.info(f"  healthy: {(metadata['status'] == 'healthy').sum()}")
    logger.info(f"  symptomatic: {(metadata['status'] == 'symptomatic').sum()}")

    # Map to our binary labels
    # COVID-19 → COVID-19
    # healthy, symptomatic → non-COVID (symptomatic are not confirmed COVID)
    label_map = {
        'COVID-19': 'COVID-19',
        'healthy': 'non-COVID',
        'symptomatic': 'non-COVID',
    }
    metadata['label'] = metadata['status'].map(label_map)

    # Catalog audio files
    catalog = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Cataloging"):
        audio_path = UK_DATA / f"{row['uuid']}.webm"
        if audio_path.exists():
            catalog.append({
                'audio_path': str(audio_path),
                'label': row['label'],
                'participant_id': row['uuid'],
                'source': 'uk_covid',
            })

    catalog_df = pd.DataFrame(catalog)
    logger.info(f"Found {len(catalog_df)} valid cough audio files")

    if len(catalog_df) == 0:
        logger.error("No valid audio files found!")
        return None

    # Distribution
    dist = catalog_df['label'].value_counts()
    logger.info(f"Label distribution:")
    for label, count in dist.items():
        logger.info(f"  {label}: {count} ({count/len(catalog_df)*100:.1f}%)")

    # Source distribution
    src_dist = catalog_df['source'].value_counts()
    logger.info(f"Source distribution:")
    for src, count in src_dist.items():
        logger.info(f"  {src}: {count}")

    # Save catalog
    CACHE_DIR.mkdir(exist_ok=True)
    catalog_path = CACHE_DIR / "uk_dataset_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    logger.info(f"Saved catalog to {catalog_path}")

    return catalog_df


def generate_embeddings():
    """Generate HeAR embeddings for UK dataset."""
    from train_multi_classifier import HeAREmbeddingGenerator

    catalog_path = CACHE_DIR / "uk_dataset_catalog.csv"
    if not catalog_path.exists():
        logger.error("Catalog not found! Run --stage extract first")
        return

    catalog = pd.read_csv(catalog_path)
    cache_file = CACHE_DIR / "uk_dataset_embeddings.pkl"

    # Load existing embeddings if any
    existing = {}
    if cache_file.exists():
        logger.info(f"Loading existing embeddings from {cache_file}")
        with open(cache_file, 'rb') as f:
            existing = pickle.load(f)

    embed_gen = HeAREmbeddingGenerator()

    embeddings = existing.get('embeddings', {})
    labels = existing.get('labels', {})

    # Process only new files
    to_process = catalog[~catalog['audio_path'].isin(embeddings)]
    logger.info(f"Generating embeddings for {len(to_process)} new files ({len(embeddings)} cached)")

    for idx, row in tqdm(to_process.iterrows(), total=len(to_process), desc="Embedding"):
        audio_path = row['audio_path']
        try:
            embedding = embed_gen.generate_embedding(audio_path)
            if embedding is not None:
                embeddings[audio_path] = embedding
                labels[audio_path] = row['label']
        except Exception as e:
            logger.debug(f"Failed: {audio_path}: {e}")

        # Save checkpoint every 500 files
        if len(embeddings) % 500 == 0:
            with open(cache_file, 'wb') as f:
                pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    # Final save
    with open(cache_file, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    logger.info(f"Generated {len(embeddings)} total embeddings")
    label_counts = pd.Series(labels).value_counts()
    logger.info(f"Label distribution:\n{label_counts}")

    return embeddings, labels


def merge_and_retrain():
    """Merge all datasets and retrain binary classifier."""
    logger.info("Merging all datasets...")

    # Load existing merged data (COUGHVID + Coswara + Virufy)
    existing_cache = CACHE_DIR / "diagnosis_simple_merged_embeddings.pkl"
    with open(existing_cache, 'rb') as f:
        existing = pickle.load(f)
    X_existing = existing['X']
    y_existing = existing['y']

    # Convert existing 3-class labels to binary
    y_existing_binary = np.array(['COVID-19' if label == 'COVID-19' else 'non-COVID'
                                   for label in y_existing])

    logger.info(f"Existing data: {len(X_existing)} samples")
    logger.info(f"  COVID-19: {(y_existing_binary == 'COVID-19').sum()}")
    logger.info(f"  non-COVID: {(y_existing_binary == 'non-COVID').sum()}")

    # Load UK dataset
    uk_cache = CACHE_DIR / "uk_dataset_embeddings.pkl"
    if not uk_cache.exists():
        logger.error("UK embeddings not found! Run --stage embed first")
        return

    with open(uk_cache, 'rb') as f:
        uk_data = pickle.load(f)

    uk_embeddings = uk_data['embeddings']
    uk_labels = uk_data['labels']

    X_uk = np.array(list(uk_embeddings.values()))
    y_uk = np.array(list(uk_labels.values()))

    logger.info(f"UK data: {len(X_uk)} samples")
    logger.info(f"  COVID-19: {(y_uk == 'COVID-19').sum()}")
    logger.info(f"  non-COVID: {(y_uk == 'non-COVID').sum()}")

    # Merge all
    X_all = np.vstack([X_existing, X_uk])
    y_all = np.concatenate([y_existing_binary, y_uk])

    logger.info(f"\nMerged total: {len(X_all)} samples")
    logger.info(f"  COVID-19: {(y_all == 'COVID-19').sum()} ({(y_all == 'COVID-19').sum()/len(y_all)*100:.1f}%)")
    logger.info(f"  non-COVID: {(y_all == 'non-COVID').sum()} ({(y_all == 'non-COVID').sum()/len(y_all)*100:.1f}%)")

    # Save merged embeddings
    merged_cache = CACHE_DIR / "diagnosis_binary_merged_embeddings.pkl"
    with open(merged_cache, 'wb') as f:
        pickle.dump({'X': X_all, 'y': y_all}, f)
    logger.info(f"Saved merged embeddings to {merged_cache}")

    # Also save as the standard merged format so train_binary_classifier can read it
    standard_cache = CACHE_DIR / "diagnosis_simple_merged_embeddings.pkl"
    with open(standard_cache, 'wb') as f:
        pickle.dump({'X': X_all, 'y': y_all}, f)
    logger.info(f"Updated standard merged cache at {standard_cache}")

    # Retrain binary classifier
    logger.info("\n" + "=" * 60)
    logger.info("Retraining binary classifier with full dataset")
    logger.info("=" * 60)

    from train_binary_classifier import train_binary_classifier
    train_binary_classifier()


def main():
    parser = argparse.ArgumentParser(description="Process UK COVID-19 dataset")
    parser.add_argument("--stage", choices=["download", "extract", "embed", "train", "all"],
                       default="all", help="Pipeline stage to run")
    args = parser.parse_args()

    if args.stage in ["download", "all"]:
        logger.info("=" * 60)
        logger.info("STAGE 1: Download UK COVID-19 Dataset")
        logger.info("=" * 60)
        success = download_uk_dataset()
        if not success:
            logger.error("Download failed, stopping")
            return

    if args.stage in ["extract", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Extract & Catalog Audio")
        logger.info("=" * 60)
        catalog = extract_and_catalog()
        if catalog is None:
            logger.error("Extraction failed, stopping")
            return

    if args.stage in ["embed", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: Generate HeAR Embeddings")
        logger.info("=" * 60)
        generate_embeddings()

    if args.stage in ["train", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4: Merge & Retrain")
        logger.info("=" * 60)
        merge_and_retrain()


if __name__ == "__main__":
    main()
