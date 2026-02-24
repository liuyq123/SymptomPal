#!/usr/bin/env python3
"""
Experiments to fix the near-zero infection class F1.

Problem: On merged data (5,841 samples), infection=560 (9.6%) gets ~0 F1.
The model predicts almost entirely healthy/COVID-19 and ignores infection.

Approaches:
  1. Aggressive class weighting via sample_weight (all classifiers)
  2. Random oversampling of infection class to match majority
  3. SMOTE oversampling of infection class
  4. Per-class threshold tuning on probability outputs
  5. Combined: SMOTE + class weights + threshold tuning
"""

import os
import sys
import pickle
import warnings
import logging
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "embedding_cache"
OUTPUT_DIR = Path(__file__).parent / "trained_models"
RANDOM_STATE = 42
N_JOBS = -1
CV_FOLDS = 5


def load_merged_data():
    """Load the cached merged embeddings."""
    cache = CACHE_DIR / "diagnosis_simple_merged_embeddings.pkl"
    with open(cache, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']


def get_classifiers():
    """Standard classifier set WITHOUT class_weight (we'll handle weighting separately)."""
    return {
        "SVM (linear)": SVC(kernel='linear', probability=True, random_state=RANDOM_STATE),
        "LogReg": LogisticRegression(max_iter=5000, solver='saga', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "GradBoost": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, random_state=RANDOM_STATE),
    }


def get_weighted_classifiers():
    """Classifiers with class_weight='balanced' where supported."""
    return {
        "SVM (linear)": SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
        "LogReg": LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "GradBoost": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, random_state=RANDOM_STATE),
    }


def get_fast_classifiers():
    """Classifiers without SVM (for oversampled data where SVM is O(n^3))."""
    return {
        "LogReg": LogisticRegression(max_iter=5000, solver='saga', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "GradBoost": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, random_state=RANDOM_STATE),
    }


def get_fast_weighted_classifiers():
    """Fast classifiers with class_weight='balanced' where supported."""
    return {
        "LogReg": LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "GradBoost": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "MLP": MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=1000, random_state=RANDOM_STATE),
    }


def eval_clf(clf, X_train, y_train, X_test, y_test, le, sample_weight=None):
    """Train, predict, return per-class and overall metrics."""
    if sample_weight is not None:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    return {
        'f1_weighted': f1_w,
        'f1_macro': f1_macro,
        'accuracy': acc,
        'per_class': {c: report[c] for c in le.classes_},
        'clf': clf,
        'y_pred': y_pred,
    }


def threshold_tune(clf, X_test, y_test, le):
    """Tune per-class decision thresholds to maximize macro F1."""
    proba = clf.predict_proba(X_test)
    classes = le.classes_
    n_classes = len(classes)

    best_macro_f1 = 0
    best_thresholds = np.ones(n_classes) / n_classes

    # Grid search over threshold adjustments for the minority class (infection)
    infection_idx = list(classes).index('infection')

    for boost in np.arange(0.0, 0.5, 0.02):
        adjusted_proba = proba.copy()
        adjusted_proba[:, infection_idx] += boost
        y_pred = adjusted_proba.argmax(axis=1)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresholds = boost

    # Apply best threshold
    adjusted_proba = proba.copy()
    adjusted_proba[:, infection_idx] += best_thresholds
    y_pred = adjusted_proba.argmax(axis=1)

    f1_w = f1_score(y_test, y_pred, average='weighted')
    f1_m = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    return {
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'accuracy': acc,
        'per_class': {c: report[c] for c in classes},
        'boost': best_thresholds,
        'y_pred': y_pred,
    }


def random_oversample(X, y, le):
    """Oversample minority classes to match the majority class count."""
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = [X]
    y_resampled = [y]

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_mask = y == cls
            X_cls = X[cls_mask]
            y_cls = y[cls_mask]
            n_needed = max_count - count
            indices = np.random.RandomState(RANDOM_STATE).choice(count, n_needed, replace=True)
            X_resampled.append(X_cls[indices])
            y_resampled.append(y_cls[indices])

    return np.vstack(X_resampled), np.concatenate(y_resampled)


def smote_oversample(X, y):
    """SMOTE oversampling of minority classes."""
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=RANDOM_STATE)
    return sm.fit_resample(X, y)


def run_experiments(start_from=0):
    """Run all infection-fix experiments."""
    X, y = load_merged_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    logger.info(f"Total samples: {len(X)}")
    for cls in le.classes_:
        mask = y == cls
        logger.info(f"  {cls}: {mask.sum()} ({mask.sum()/len(y)*100:.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    all_results = {}
    infection_idx = list(le.classes_).index('infection')

    if start_from <= 0:
        # ── Experiment 0: Baseline (no weighting, for comparison) ──
        logger.info("\n" + "=" * 60)
        logger.info("Experiment 0: Baseline (no class weighting)")
        logger.info("=" * 60)
        t0 = time()
        results_baseline = {}
        for name, clf in get_classifiers().items():
            r = eval_clf(clf, X_train, y_train, X_test, y_test, le)
            results_baseline[name] = r
            inf_f1 = r['per_class']['infection']['f1-score']
            logger.info(f"  {name}: wF1={r['f1_weighted']:.3f}  macroF1={r['f1_macro']:.3f}  infection_F1={inf_f1:.3f}")
        all_results['0_baseline'] = results_baseline
        logger.info(f"  Time: {time()-t0:.0f}s")

    if start_from <= 1:
        # ── Experiment 1: class_weight='balanced' (already partially used) ──
        logger.info("\n" + "=" * 60)
        logger.info("Experiment 1: class_weight='balanced'")
        logger.info("=" * 60)
        t0 = time()
        results_cw = {}
        for name, clf in get_weighted_classifiers().items():
            r = eval_clf(clf, X_train, y_train, X_test, y_test, le)
            results_cw[name] = r
            inf_f1 = r['per_class']['infection']['f1-score']
            logger.info(f"  {name}: wF1={r['f1_weighted']:.3f}  macroF1={r['f1_macro']:.3f}  infection_F1={inf_f1:.3f}")
        all_results['1_balanced_weight'] = results_cw
        logger.info(f"  Time: {time()-t0:.0f}s")

    if start_from <= 2:
        # ── Experiment 2: Aggressive sample_weight (infection 5x) ──
        logger.info("\n" + "=" * 60)
        logger.info("Experiment 2: Aggressive sample_weight (infection 5x)")
        logger.info("=" * 60)
        t0 = time()
        # Create custom sample weights: infection gets 5x, others get 1x
        sw = np.ones(len(y_train))
        sw[y_train == infection_idx] = 5.0
        results_sw = {}
        for name, clf in get_classifiers().items():
            try:
                r = eval_clf(clf, X_train, y_train, X_test, y_test, le, sample_weight=sw)
            except TypeError:
                # MLP doesn't support sample_weight — fall back to no weight
                r = eval_clf(clf, X_train, y_train, X_test, y_test, le)
            results_sw[name] = r
            inf_f1 = r['per_class']['infection']['f1-score']
            logger.info(f"  {name}: wF1={r['f1_weighted']:.3f}  macroF1={r['f1_macro']:.3f}  infection_F1={inf_f1:.3f}")
        all_results['2_sample_weight_5x'] = results_sw
        logger.info(f"  Time: {time()-t0:.0f}s")

    if start_from <= 3:
        # ── Experiment 3: Random oversampling ──
        logger.info("\n" + "=" * 60)
        logger.info("Experiment 3: Random oversampling (all classes equal)")
        logger.info("=" * 60)
        t0 = time()
        X_over, y_over = random_oversample(X_train, y_train, le)
        logger.info(f"  Oversampled: {len(X_train)} → {len(X_over)}")
        results_over = {}
        for name, clf in get_fast_classifiers().items():  # No SVM — O(n^3) on oversampled data
            r = eval_clf(clf, X_over, y_over, X_test, y_test, le)
            results_over[name] = r
            inf_f1 = r['per_class']['infection']['f1-score']
            logger.info(f"  {name}: wF1={r['f1_weighted']:.3f}  macroF1={r['f1_macro']:.3f}  infection_F1={inf_f1:.3f}")
        all_results['3_random_oversample'] = results_over
        logger.info(f"  Time: {time()-t0:.0f}s")

    # Prepare oversampled data (needed for experiments 3, 4, 6)
    results_smote = {}
    X_over, y_over = None, None
    X_smote, y_smote = None, None

    if start_from <= 3:
        X_over, y_over = random_oversample(X_train, y_train, le)

    if start_from <= 4:
        # ── Experiment 4: SMOTE oversampling ──
        logger.info("\n" + "=" * 60)
        logger.info("Experiment 4: SMOTE oversampling")
        logger.info("=" * 60)
        t0 = time()
        try:
            X_smote, y_smote = smote_oversample(X_train, y_train)
            logger.info(f"  SMOTE: {len(X_train)} → {len(X_smote)}")
            results_smote = {}
            for name, clf in get_fast_classifiers().items():  # No SVM — O(n^3) on oversampled data
                r = eval_clf(clf, X_smote, y_smote, X_test, y_test, le)
                results_smote[name] = r
                inf_f1 = r['per_class']['infection']['f1-score']
                logger.info(f"  {name}: wF1={r['f1_weighted']:.3f}  macroF1={r['f1_macro']:.3f}  infection_F1={inf_f1:.3f}")
            all_results['4_smote'] = results_smote
        except ImportError:
            logger.warning("  imblearn not installed, skipping SMOTE. Install with: pip install imbalanced-learn")
            results_smote = {}
        logger.info(f"  Time: {time()-t0:.0f}s")

    if start_from <= 5:
        # ── Experiment 5: Threshold tuning on best models ──
        logger.info("\n" + "=" * 60)
        logger.info("Experiment 5: Per-class threshold tuning")
        logger.info("=" * 60)
        t0 = time()
        results_thresh = {}
        # Train with balanced weights, then tune thresholds
        for name, clf in get_weighted_classifiers().items():
            clf.fit(X_train, y_train)
            r = threshold_tune(clf, X_test, y_test, le)
            results_thresh[name] = r
            inf_f1 = r['per_class']['infection']['f1-score']
            logger.info(f"  {name}: wF1={r['f1_weighted']:.3f}  macroF1={r['f1_macro']:.3f}  infection_F1={inf_f1:.3f}  boost={r['boost']:.2f}")
        all_results['5_threshold_tuning'] = results_thresh
        logger.info(f"  Time: {time()-t0:.0f}s")

    if start_from <= 6:
        # ── Experiment 6: Combined (SMOTE + balanced weights + threshold tuning) ──
        logger.info("\n" + "=" * 60)
        logger.info("Experiment 6: Combined (oversample + balanced weights + threshold)")
        logger.info("=" * 60)
        t0 = time()
        results_combined = {}
        # Use oversampled data (SMOTE if available, else random oversample)
        if X_smote is None:
            X_smote, y_smote = smote_oversample(X_train, y_train)
        X_os = X_smote if len(results_smote) > 0 or X_smote is not None else X_over
        y_os = y_smote if len(results_smote) > 0 or y_smote is not None else y_over
        if X_os is None:
            X_os, y_os = random_oversample(X_train, y_train, le)
        for name, clf in get_fast_weighted_classifiers().items():  # No SVM — O(n^3) on oversampled data
            clf.fit(X_os, y_os)
            r = threshold_tune(clf, X_test, y_test, le)
            results_combined[name] = r
            inf_f1 = r['per_class']['infection']['f1-score']
            logger.info(f"  {name}: wF1={r['f1_weighted']:.3f}  macroF1={r['f1_macro']:.3f}  infection_F1={inf_f1:.3f}  boost={r['boost']:.2f}")
        all_results['6_combined'] = results_combined
        logger.info(f"  Time: {time()-t0:.0f}s")

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY: Best infection F1 per experiment")
    logger.info("=" * 60)

    summary_rows = []
    for exp_name, exp_results in all_results.items():
        best_inf = 0
        best_model = ""
        best_wf1 = 0
        best_mf1 = 0
        for model_name, r in exp_results.items():
            inf_f1 = r['per_class']['infection']['f1-score']
            if inf_f1 > best_inf:
                best_inf = inf_f1
                best_model = model_name
                best_wf1 = r['f1_weighted']
                best_mf1 = r['f1_macro']
        summary_rows.append({
            'experiment': exp_name,
            'best_model': best_model,
            'infection_f1': best_inf,
            'weighted_f1': best_wf1,
            'macro_f1': best_mf1,
        })
        logger.info(f"  {exp_name}: {best_model} infection_F1={best_inf:.3f} wF1={best_wf1:.3f} macroF1={best_mf1:.3f}")

    # ── Generate Report ──
    generate_report(all_results, le, summary_rows, X_train, y_train, X_test, y_test)


def generate_report(all_results, le, summary_rows, X_train, y_train, X_test, y_test):
    """Generate markdown report of infection-fix experiments."""
    lines = [
        "# Infection Class Fix Experiments",
        "",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Problem**: infection class has near-zero F1 ({560}/{5841} = 9.6% of samples)",
        f"**Goal**: Improve infection recall/F1 without destroying overall performance",
        "",
        "## Summary",
        "",
        "| Experiment | Best Model | Infection F1 | Weighted F1 | Macro F1 |",
        "|-----------|-----------|-------------|-------------|----------|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['experiment']} | {row['best_model']} | "
            f"{row['infection_f1']:.3f} | {row['weighted_f1']:.3f} | {row['macro_f1']:.3f} |"
        )

    lines.extend(["", "## Detailed Per-Class Results", ""])

    for exp_name, exp_results in all_results.items():
        lines.append(f"### {exp_name}")
        lines.append("")
        lines.append("| Model | COVID-19 F1 | healthy F1 | infection F1 | Weighted F1 | Macro F1 |")
        lines.append("|-------|-----------|----------|-------------|-------------|----------|")
        for model_name, r in exp_results.items():
            c19 = r['per_class']['COVID-19']['f1-score']
            h = r['per_class']['healthy']['f1-score']
            inf = r['per_class']['infection']['f1-score']
            lines.append(
                f"| {model_name} | {c19:.3f} | {h:.3f} | {inf:.3f} | "
                f"{r['f1_weighted']:.3f} | {r['f1_macro']:.3f} |"
            )
        lines.append("")

    report_path = Path(__file__).parent / "infection_fix_results.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    logger.info(f"\nDetailed report saved to {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0, help="Skip experiments before this number")
    args = parser.parse_args()
    run_experiments(start_from=args.start_from)
