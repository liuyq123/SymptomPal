#!/usr/bin/env python3
"""
Process Coswara and Virufy datasets: extract cough audio, generate HeAR embeddings,
merge with existing COUGHVID embeddings, and retrain classifiers.

Usage:
    python process_new_datasets.py --stage extract    # Extract & catalog audio files
    python process_new_datasets.py --stage embed      # Generate HeAR embeddings
    python process_new_datasets.py --stage train       # Retrain with merged data
    python process_new_datasets.py --stage all         # Run all stages
"""

import os
import sys
import pickle
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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
RANDOM_STATE = 42


# ─── Label Mapping ──────────────────────────────────────────────────────────

# Map Coswara covid_status to our 3 classes
COSWARA_LABEL_MAP = {
    "healthy": "healthy",
    "no_resp_illness_exposed": "healthy",  # exposed but not ill
    "positive_mild": "COVID-19",
    "positive_moderate": "COVID-19",
    "positive_asymp": "COVID-19",
    "resp_illness_not_identified": "infection",
    # Excluded:
    # "recovered_full" - ambiguous (was COVID, now recovered)
    # "under_validation" - unverified
}

# Virufy: directory-based labels
VIRUFY_LABEL_MAP = {
    "pos": "COVID-19",
    "neg": "healthy",
}


# ─── Stage 1: Extract & Catalog ────────────────────────────────────────────

def catalog_coswara(data_dir: Path) -> pd.DataFrame:
    """Catalog Coswara cough audio files with labels."""
    extracted_dir = data_dir / "Extracted_data"
    metadata_file = data_dir / "metadata_all.csv"

    if not metadata_file.exists():
        raise FileNotFoundError(f"Run metadata compilation first: {metadata_file}")

    meta = pd.read_csv(metadata_file)
    logger.info(f"Coswara metadata: {len(meta)} participants")

    records = []
    found = 0
    missing = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Cataloging Coswara"):
        pid = row['id']
        status = row.get('covid_status', '')

        if status not in COSWARA_LABEL_MAP:
            continue

        label = COSWARA_LABEL_MAP[status]

        # Each participant has cough-heavy and cough-shallow
        for cough_type in ['cough-heavy', 'cough-shallow']:
            # Search across date directories
            audio_path = None
            for date_dir in sorted(extracted_dir.glob("202*")):
                candidate = date_dir / pid / f"{cough_type}.wav"
                if candidate.exists():
                    audio_path = candidate
                    break

            if audio_path and audio_path.stat().st_size > 1000:  # Skip tiny/empty files
                records.append({
                    'source': 'coswara',
                    'audio_path': str(audio_path),
                    'label': label,
                    'original_label': status,
                    'participant_id': pid,
                    'cough_type': cough_type,
                })
                found += 1
            else:
                missing += 1

    df = pd.DataFrame(records)
    logger.info(f"Coswara: found {found} audio files, {missing} missing")
    if len(df) > 0:
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def catalog_virufy(data_dir: Path) -> pd.DataFrame:
    """Catalog Virufy cough audio files with labels."""
    segmented_dir = data_dir / "clinical" / "segmented"

    records = []
    for label_dir in ['pos', 'neg']:
        audio_dir = segmented_dir / label_dir
        if not audio_dir.exists():
            continue

        label = VIRUFY_LABEL_MAP[label_dir]
        for audio_file in sorted(audio_dir.glob("*.mp3")):
            if audio_file.stat().st_size > 500:
                records.append({
                    'source': 'virufy',
                    'audio_path': str(audio_file),
                    'label': label,
                    'original_label': label_dir,
                    'participant_id': audio_file.stem,
                    'cough_type': 'cough',
                })

    df = pd.DataFrame(records)
    logger.info(f"Virufy: {len(df)} audio files")
    if len(df) > 0:
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df


def run_extraction():
    """Stage 1: Catalog all audio files from new datasets."""
    coswara_dir = ML_DATA / "coswara"
    virufy_dir = ML_DATA / "virufy"

    catalogs = []

    if coswara_dir.exists():
        try:
            coswara_df = catalog_coswara(coswara_dir)
            catalogs.append(coswara_df)
        except Exception as e:
            logger.warning(f"Coswara cataloging failed: {e}")

    if virufy_dir.exists():
        try:
            virufy_df = catalog_virufy(virufy_dir)
            catalogs.append(virufy_df)
        except Exception as e:
            logger.warning(f"Virufy cataloging failed: {e}")

    if not catalogs:
        logger.error("No datasets found!")
        return None

    combined = pd.concat(catalogs, ignore_index=True)
    catalog_path = CACHE_DIR / "new_datasets_catalog.csv"
    CACHE_DIR.mkdir(exist_ok=True)
    combined.to_csv(catalog_path, index=False)
    logger.info(f"\nCombined catalog: {len(combined)} files saved to {catalog_path}")
    logger.info(f"Overall distribution:\n{combined['label'].value_counts()}")
    logger.info(f"By source:\n{combined.groupby(['source', 'label']).size()}")
    return combined


# ─── Stage 2: Generate HeAR Embeddings ──────────────────────────────────────

def generate_embeddings(catalog: pd.DataFrame = None, batch_size: int = 32):
    """Stage 2: Generate HeAR embeddings for all new audio files."""
    if catalog is None:
        catalog_path = CACHE_DIR / "new_datasets_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError("Run --stage extract first")
        catalog = pd.read_csv(catalog_path)

    cache_file = CACHE_DIR / "new_datasets_embeddings.pkl"

    # Load existing if partially done
    existing = {}
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            existing = pickle.load(f)
        logger.info(f"Loaded {len(existing.get('embeddings', {}))} existing embeddings")

    # Import HeAR embedding generator from training script
    from train_multi_classifier import HeAREmbeddingGenerator

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

        # Save checkpoint every 200 files
        if len(embeddings) % 200 == 0:
            with open(cache_file, 'wb') as f:
                pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    # Final save
    with open(cache_file, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)

    logger.info(f"Generated {len(embeddings)} total embeddings")
    label_counts = pd.Series(labels).value_counts()
    logger.info(f"Label distribution:\n{label_counts}")

    return embeddings, labels


# ─── Stage 3: Merge & Retrain ──────────────────────────────────────────────

def merge_and_retrain():
    """Stage 3: Merge new embeddings with COUGHVID and retrain."""
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    N_JOBS = -1
    CV_FOLDS = 5

    # Load original COUGHVID embeddings
    coughvid_cache = CACHE_DIR / "diagnosis_simple_multi_embeddings.pkl"
    with open(coughvid_cache, 'rb') as f:
        coughvid = pickle.load(f)
    X_orig = coughvid['X']
    y_orig = coughvid['y']
    logger.info(f"COUGHVID: {len(X_orig)} samples")
    logger.info(f"  Distribution: {dict(zip(*np.unique(y_orig, return_counts=True)))}")

    # Load new embeddings
    new_cache = CACHE_DIR / "new_datasets_embeddings.pkl"
    if not new_cache.exists():
        logger.error("No new embeddings found! Run --stage embed first")
        return

    with open(new_cache, 'rb') as f:
        new_data = pickle.load(f)

    new_embeddings = new_data['embeddings']
    new_labels = new_data['labels']

    X_new = np.array(list(new_embeddings.values()))
    y_new = np.array(list(new_labels.values()))
    logger.info(f"New datasets: {len(X_new)} samples")
    logger.info(f"  Distribution: {dict(zip(*np.unique(y_new, return_counts=True)))}")

    # Merge
    X_all = np.vstack([X_orig, X_new])
    y_all = np.concatenate([y_orig, y_new])
    logger.info(f"\nMerged: {len(X_all)} total samples")
    logger.info(f"  Distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)
    logger.info(f"Classes: {list(le.classes_)}")

    # Also encode original-only for comparison
    le_orig = LabelEncoder()
    y_orig_encoded = le_orig.fit_transform(y_orig)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_orig, y_orig_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_orig_encoded
    )

    logger.info(f"Train: {len(X_train)} ({len(X_train_orig)} original)")
    logger.info(f"Test:  {len(X_test)} ({len(X_test_orig)} original)")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Define classifiers
    classifiers = {
        "SVM (linear)": SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
        "LogReg": LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "GradBoost": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, random_state=RANDOM_STATE),
    }

    # ── Experiment A: Baseline (original COUGHVID only) ──
    logger.info("\n" + "=" * 60)
    logger.info("Experiment A: Baseline (COUGHVID only, 753 samples)")
    logger.info("=" * 60)

    results_orig = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_orig, y_train_orig)
        y_pred = clf.predict(X_test_orig)
        f1 = f1_score(y_test_orig, y_pred, average='weighted')
        acc = accuracy_score(y_test_orig, y_pred)
        cv_scores = cross_val_score(clf, X_orig, y_orig_encoded, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=N_JOBS)
        results_orig[name] = {'f1': f1, 'acc': acc, 'cv': cv_scores.mean()}
        logger.info(f"  {name}: F1={f1:.3f}  Acc={acc:.3f}  CV={cv_scores.mean():.3f}")

    # ── Experiment B: Merged data ──
    logger.info("\n" + "=" * 60)
    logger.info(f"Experiment B: Merged data ({len(X_all)} samples)")
    logger.info("=" * 60)

    results_merged = {}
    best_f1 = 0
    best_name = None
    best_clf = None

    for name, clf in classifiers.items():
        # Fresh instance for merged training
        clf_merged = type(clf)(**clf.get_params())
        clf_merged.fit(X_train, y_train)
        y_pred = clf_merged.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf_merged, X_all, y_encoded, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=N_JOBS)
        results_merged[name] = {'f1': f1, 'acc': acc, 'cv': cv_scores.mean()}
        logger.info(f"  {name}: F1={f1:.3f}  Acc={acc:.3f}  CV={cv_scores.mean():.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_clf = clf_merged

    # ── Experiment C: Merged + ensemble ──
    logger.info("\n" + "=" * 60)
    logger.info("Experiment C: Merged + Ensemble")
    logger.info("=" * 60)

    ensemble = VotingClassifier(
        estimators=[
            ('svm', SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE)),
            ('lr', LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=RANDOM_STATE)),
            ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE)),
            ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, random_state=RANDOM_STATE)),
        ],
        voting='soft',
        n_jobs=N_JOBS,
    )
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    f1_ens = f1_score(y_test, y_pred_ens, average='weighted')
    acc_ens = accuracy_score(y_test, y_pred_ens)
    cv_ens = cross_val_score(ensemble, X_all, y_encoded, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=N_JOBS)

    logger.info(f"  Ensemble: F1={f1_ens:.3f}  Acc={acc_ens:.3f}  CV={cv_ens.mean():.3f}")

    # Detailed report
    logger.info(f"\nBest single model ({best_name}):")
    best_clf.fit(X_train, y_train)
    y_pred_best = best_clf.predict(X_test)
    logger.info(classification_report(y_test, y_pred_best, target_names=le.classes_))

    logger.info("Ensemble:")
    logger.info(classification_report(y_test, y_pred_ens, target_names=le.classes_))

    # ── Save results ──
    results_report = generate_report(results_orig, results_merged, f1_ens, acc_ens, cv_ens.mean(),
                                      le, len(X_orig), len(X_new), len(X_all))
    report_path = Path(__file__).parent / "merged_training_results.md"
    with open(report_path, 'w') as f:
        f.write(results_report)
    logger.info(f"\nReport saved to {report_path}")

    # Save best model
    use_ensemble = f1_ens > best_f1
    save_model = ensemble if use_ensemble else best_clf
    save_name = "Ensemble" if use_ensemble else best_name
    save_f1 = f1_ens if use_ensemble else best_f1

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "diagnosis_simple_merged_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': save_model,
            'scaler': None,
            'label_encoder': le,
        }, f)
    logger.info(f"Saved {save_name} model to {model_path} (F1={save_f1:.3f})")

    meta_path = OUTPUT_DIR / "diagnosis_simple_merged_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            'task_name': 'diagnosis_simple',
            'best_model': save_name,
            'f1_score': float(save_f1),
            'accuracy': float(acc_ens if use_ensemble else results_merged[best_name]['acc']),
            'classes': list(le.classes_),
            'total_samples': len(X_all),
            'sources': {
                'coughvid': len(X_orig),
                'new_datasets': len(X_new),
            },
        }, f, indent=2)

    # Save merged embeddings for future use
    merged_cache = CACHE_DIR / "diagnosis_simple_merged_embeddings.pkl"
    with open(merged_cache, 'wb') as f:
        pickle.dump({'X': X_all, 'y': y_all}, f)
    logger.info(f"Saved merged embeddings to {merged_cache}")


def generate_report(results_orig, results_merged, f1_ens, acc_ens, cv_ens,
                     le, n_orig, n_new, n_total):
    """Generate markdown comparison report."""
    lines = [
        "# Merged Dataset Training Results",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Original data**: {n_orig} samples (COUGHVID)",
        f"**New data**: {n_new} samples (Coswara + Virufy)",
        f"**Total**: {n_total} samples",
        "",
        "## Comparison: Original vs Merged",
        "",
        "| Classifier | Original F1 | Original CV | Merged F1 | Merged CV | Change |",
        "|-----------|-------------|-------------|-----------|-----------|--------|",
    ]

    for name in results_orig:
        orig = results_orig[name]
        merged = results_merged.get(name, {'f1': 0, 'cv': 0})
        change = merged['f1'] - orig['f1']
        sign = "+" if change >= 0 else ""
        lines.append(
            f"| {name} | {orig['f1']:.3f} | {orig['cv']:.3f} | "
            f"{merged['f1']:.3f} | {merged['cv']:.3f} | {sign}{change:.3f} |"
        )

    lines.extend([
        f"| **Ensemble** | — | — | {f1_ens:.3f} | {cv_ens:.3f} | — |",
        "",
    ])

    # Best improvement
    best_orig = max(results_orig.values(), key=lambda x: x['f1'])['f1']
    best_merged = max(list(results_merged.values()) + [{'f1': f1_ens}], key=lambda x: x['f1'])['f1']
    improvement = best_merged - best_orig
    pct = improvement / best_orig * 100 if best_orig > 0 else 0

    lines.extend([
        f"**Best original F1**: {best_orig:.3f}",
        f"**Best merged F1**: {best_merged:.3f}",
        f"**Improvement**: {'+' if improvement >= 0 else ''}{improvement:.3f} ({pct:+.1f}%)",
    ])

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process new datasets and retrain")
    parser.add_argument("--stage", choices=["extract", "embed", "train", "all"], default="all")
    args = parser.parse_args()

    if args.stage in ["extract", "all"]:
        logger.info("=" * 60)
        logger.info("STAGE 1: Extract & Catalog Audio Files")
        logger.info("=" * 60)
        catalog = run_extraction()

    if args.stage in ["embed", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: Generate HeAR Embeddings")
        logger.info("=" * 60)
        generate_embeddings()

    if args.stage in ["train", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: Merge & Retrain Classifiers")
        logger.info("=" * 60)
        merge_and_retrain()


if __name__ == "__main__":
    main()
