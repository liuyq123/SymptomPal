#!/usr/bin/env python3
"""
Experiment: Improving cough classifier with augmentation, class weighting, and tuning.

Uses cached HeAR embeddings so no GPU/model loading required.
Focuses on diagnosis_simple (3-class) as the weakest task.

Key optimizations:
- SVM excluded from augmented experiments (O(n^3) doesn't scale)
- n_jobs=-1 for all parallelizable operations (uses all 32 cores)
- CV only on original data (not augmented) to avoid inflated scores
- LogReg uses saga solver with higher max_iter for convergence

Usage:
    python experiment_improvements.py
"""

import os
import pickle
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from time import time

import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Suppress convergence warnings (we handle them via solver choice)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "embedding_cache"
OUTPUT_DIR = Path(__file__).parent / "trained_models"
RESULTS_FILE = Path(__file__).parent / "experiment_results.md"

RANDOM_STATE = 42
CV_FOLDS = 5
N_JOBS = -1  # Use all 32 cores


# ─── Data Loading ───────────────────────────────────────────────────────────

def load_cached_embeddings(task: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached embeddings for a task."""
    cache_file = CACHE_DIR / f"{task}_multi_embeddings.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(f"No cached embeddings at {cache_file}")
    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)
    return cached['X'], cached['y']


# ─── Augmentation ───────────────────────────────────────────────────────────

def augment_gaussian_noise(X: np.ndarray, y: np.ndarray,
                           noise_std: float = 0.05,
                           n_copies: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise copies to embedding space."""
    rng = np.random.RandomState(RANDOM_STATE)
    augmented_X = [X]
    augmented_y = [y]
    for i in range(n_copies):
        noise = rng.normal(0, noise_std, X.shape)
        augmented_X.append(X + noise)
        augmented_y.append(y)
    return np.vstack(augmented_X), np.concatenate(augmented_y)


def augment_mixup(X: np.ndarray, y: np.ndarray,
                  alpha: float = 0.2,
                  n_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Mixup augmentation: interpolate between same-class samples."""
    rng = np.random.RandomState(RANDOM_STATE)
    if n_samples is None:
        n_samples = len(X)

    new_X = []
    new_y = []
    classes = np.unique(y)

    for cls in classes:
        cls_mask = y == cls
        cls_X = X[cls_mask]
        if len(cls_X) < 2:
            continue

        n_cls_samples = n_samples // len(classes)
        for _ in range(n_cls_samples):
            i, j = rng.choice(len(cls_X), 2, replace=False)
            lam = rng.beta(alpha, alpha)
            new_X.append(lam * cls_X[i] + (1 - lam) * cls_X[j])
            new_y.append(cls)

    return (
        np.vstack([X, np.array(new_X)]),
        np.concatenate([y, np.array(new_y)])
    )


def augment_smote_like(X: np.ndarray, y: np.ndarray,
                       target_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Simple SMOTE-like oversampling for minority classes."""
    rng = np.random.RandomState(RANDOM_STATE)
    classes, counts = np.unique(y, return_counts=True)
    max_count = int(counts.max() * target_ratio)

    new_X = [X]
    new_y = [y]

    for cls, count in zip(classes, counts):
        if count >= max_count:
            continue
        cls_mask = y == cls
        cls_X = X[cls_mask]
        n_needed = max_count - count

        synth_X = []
        for _ in range(n_needed):
            i, j = rng.choice(len(cls_X), 2, replace=True)
            lam = rng.random()
            synth_X.append(lam * cls_X[i] + (1 - lam) * cls_X[j])
        new_X.append(np.array(synth_X))
        new_y.append(np.full(n_needed, cls))

    return np.vstack(new_X), np.concatenate(new_y)


def augment_feature_dropout(X: np.ndarray, y: np.ndarray,
                            drop_rate: float = 0.1,
                            n_copies: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly zero out embedding dimensions."""
    rng = np.random.RandomState(RANDOM_STATE)
    augmented_X = [X]
    augmented_y = [y]
    for _ in range(n_copies):
        mask = rng.random(X.shape) > drop_rate
        augmented_X.append(X * mask)
        augmented_y.append(y)
    return np.vstack(augmented_X), np.concatenate(augmented_y)


# ─── Classifier Factories ──────────────────────────────────────────────────

def make_fast_classifiers():
    """Classifiers that scale well with augmented data (no SVM)."""
    return {
        "LogReg": LogisticRegression(
            max_iter=5000, solver='saga', class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        "GradBoost": GradientBoostingClassifier(
            n_estimators=128, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=128, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=1000, random_state=RANDOM_STATE
        ),
    }


def make_all_classifiers():
    """All classifiers including SVM (only for small, non-augmented data)."""
    clfs = {
        "SVM (linear)": SVC(kernel='linear', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
        "SVM (rbf)": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE),
    }
    clfs.update(make_fast_classifiers())
    return clfs


# ─── Experiments ────────────────────────────────────────────────────────────

def run_experiment(
    name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    classifiers: Dict,
    label_encoder: LabelEncoder,
    cv_X: np.ndarray = None,
    cv_y: np.ndarray = None,
) -> Dict:
    """Run a single experiment. CV is always on original (non-augmented) data."""
    t0 = time()
    results = {}
    best_f1 = 0
    best_name = None

    for clf_name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # CV on original data
            cv_mean, cv_std = 0.0, 0.0
            if cv_X is not None and cv_y is not None:
                cv_scores = cross_val_score(
                    clf, cv_X, cv_y, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=N_JOBS
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

            results[clf_name] = {
                'accuracy': acc,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
            }

            logger.info(f"  {clf_name}: F1={f1:.3f}  Acc={acc:.3f}  CV={cv_mean:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_name = clf_name

        except Exception as e:
            logger.warning(f"  {clf_name} failed: {e}")

    # Classification report for best
    best_clf = classifiers[best_name]
    best_clf.fit(X_train, y_train)
    y_pred_best = best_clf.predict(X_test)
    report = classification_report(
        y_test, y_pred_best,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    elapsed = time() - t0
    logger.info(f"  -> Best: {best_name} (F1={best_f1:.3f}) [{elapsed:.1f}s]")

    return {
        'name': name,
        'results': results,
        'best_model': best_name,
        'best_f1': best_f1,
        'best_accuracy': results[best_name]['accuracy'],
        'report': report,
        'elapsed': elapsed,
    }


# ─── Individual Experiments ─────────────────────────────────────────────────

def exp_baseline(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 0: Current baseline (all classifiers including SVM)."""
    return run_experiment(
        "Baseline (original)", X_train, y_train, X_test, y_test,
        make_all_classifiers(), le, X, y
    )


def exp_gaussian(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 1: Gaussian noise augmentation (std=0.05, 3 copies)."""
    X_aug, y_aug = augment_gaussian_noise(X_train, y_train, noise_std=0.05, n_copies=3)
    logger.info(f"  Augmented: {len(X_train)} -> {len(X_aug)} samples")
    return run_experiment(
        "Gaussian Noise (std=0.05, 3x)", X_aug, y_aug, X_test, y_test,
        make_fast_classifiers(), le, X, y
    )


def exp_gaussian_low(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 2: Gaussian noise with lower std."""
    X_aug, y_aug = augment_gaussian_noise(X_train, y_train, noise_std=0.02, n_copies=3)
    logger.info(f"  Augmented: {len(X_train)} -> {len(X_aug)} samples")
    return run_experiment(
        "Gaussian Noise (std=0.02, 3x)", X_aug, y_aug, X_test, y_test,
        make_fast_classifiers(), le, X, y
    )


def exp_mixup(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 3: Mixup augmentation."""
    X_aug, y_aug = augment_mixup(X_train, y_train, alpha=0.2)
    logger.info(f"  Augmented: {len(X_train)} -> {len(X_aug)} samples")
    return run_experiment(
        "Mixup (alpha=0.2)", X_aug, y_aug, X_test, y_test,
        make_fast_classifiers(), le, X, y
    )


def exp_smote(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 4: SMOTE-like oversampling."""
    X_aug, y_aug = augment_smote_like(X_train, y_train, target_ratio=1.0)
    logger.info(f"  Augmented: {len(X_train)} -> {len(X_aug)} samples")
    return run_experiment(
        "SMOTE-like Oversampling", X_aug, y_aug, X_test, y_test,
        make_fast_classifiers(), le, X, y
    )


def exp_dropout(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 5: Feature dropout augmentation."""
    X_aug, y_aug = augment_feature_dropout(X_train, y_train, drop_rate=0.1, n_copies=3)
    logger.info(f"  Augmented: {len(X_train)} -> {len(X_aug)} samples")
    return run_experiment(
        "Feature Dropout (10%, 3x)", X_aug, y_aug, X_test, y_test,
        make_fast_classifiers(), le, X, y
    )


def exp_combined(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 6: SMOTE + Gaussian noise."""
    X_aug, y_aug = augment_smote_like(X_train, y_train, target_ratio=1.0)
    X_aug, y_aug = augment_gaussian_noise(X_aug, y_aug, noise_std=0.03, n_copies=1)
    logger.info(f"  Augmented: {len(X_train)} -> {len(X_aug)} samples")
    return run_experiment(
        "SMOTE + Gaussian (std=0.03)", X_aug, y_aug, X_test, y_test,
        make_fast_classifiers(), le, X, y
    )


def exp_hyperparam_tuning(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 7: GridSearchCV hyperparameter tuning with SMOTE augmentation."""
    X_aug, y_aug = augment_smote_like(X_train, y_train, target_ratio=1.0)
    logger.info(f"  Augmented: {len(X_train)} -> {len(X_aug)} samples")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time()

    # Tune LogReg
    logger.info("  Tuning LogReg...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=RANDOM_STATE),
        {'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'penalty': ['l1', 'l2']},
        cv=cv, scoring='f1_weighted', n_jobs=N_JOBS
    )
    lr_grid.fit(X_aug, y_aug)
    logger.info(f"    LogReg best: {lr_grid.best_params_} -> {lr_grid.best_score_:.3f}")

    # Tune GradientBoosting
    logger.info("  Tuning GradBoost...")
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1]},
        cv=cv, scoring='f1_weighted', n_jobs=N_JOBS
    )
    gb_grid.fit(X_aug, y_aug)
    logger.info(f"    GradBoost best: {gb_grid.best_params_} -> {gb_grid.best_score_:.3f}")

    # Tune RandomForest
    logger.info("  Tuning RandomForest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_leaf': [1, 2, 5]},
        cv=cv, scoring='f1_weighted', n_jobs=N_JOBS
    )
    rf_grid.fit(X_aug, y_aug)
    logger.info(f"    RandomForest best: {rf_grid.best_params_} -> {rf_grid.best_score_:.3f}")

    # Tune MLP
    logger.info("  Tuning MLP...")
    mlp_grid = GridSearchCV(
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
        {
            'hidden_layer_sizes': [(256, 128), (512, 256), (256, 128, 64), (128,), (512, 256, 128)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.0005, 0.0001],
        },
        cv=cv, scoring='f1_weighted', n_jobs=N_JOBS
    )
    mlp_grid.fit(X_aug, y_aug)
    logger.info(f"    MLP best: {mlp_grid.best_params_} -> {mlp_grid.best_score_:.3f}")

    classifiers = {
        "LogReg (tuned)": lr_grid.best_estimator_,
        "GradBoost (tuned)": gb_grid.best_estimator_,
        "RandomForest (tuned)": rf_grid.best_estimator_,
        "MLP (tuned)": mlp_grid.best_estimator_,
    }

    result = run_experiment(
        "SMOTE + Hyperparameter Tuning", X_aug, y_aug, X_test, y_test,
        classifiers, le, X, y
    )
    result['tuning_params'] = {
        'LogReg': lr_grid.best_params_,
        'GradBoost': gb_grid.best_params_,
        'RandomForest': rf_grid.best_params_,
        'MLP': mlp_grid.best_params_,
    }
    return result


def exp_tuned_ensemble(X_train, y_train, X_test, y_test, le, X, y):
    """Experiment 8: Tuned ensemble with SMOTE."""
    X_aug, y_aug = augment_smote_like(X_train, y_train, target_ratio=1.0)
    logger.info(f"  Augmented: {len(X_train)} -> {len(X_aug)} samples")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Quick-tune each classifier type
    logger.info("  Quick-tuning for ensemble...")
    lr = GridSearchCV(
        LogisticRegression(max_iter=5000, solver='saga', class_weight='balanced', random_state=RANDOM_STATE),
        {'C': [0.1, 1.0, 10.0]}, cv=cv, scoring='f1_weighted', n_jobs=N_JOBS
    )
    lr.fit(X_aug, y_aug)

    gb = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {'n_estimators': [200, 300], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
        cv=cv, scoring='f1_weighted', n_jobs=N_JOBS
    )
    gb.fit(X_aug, y_aug)

    rf = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=N_JOBS),
        {'n_estimators': [200, 300], 'max_depth': [None, 10]},
        cv=cv, scoring='f1_weighted', n_jobs=N_JOBS
    )
    rf.fit(X_aug, y_aug)

    mlp = GridSearchCV(
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
        {'hidden_layer_sizes': [(256, 128), (512, 256)], 'alpha': [0.001, 0.01]},
        cv=cv, scoring='f1_weighted', n_jobs=N_JOBS
    )
    mlp.fit(X_aug, y_aug)

    logger.info(f"    LR={lr.best_score_:.3f} GB={gb.best_score_:.3f} RF={rf.best_score_:.3f} MLP={mlp.best_score_:.3f}")

    # Build ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr.best_estimator_),
            ('gb', gb.best_estimator_),
            ('rf', rf.best_estimator_),
            ('mlp', mlp.best_estimator_),
        ],
        voting='soft',
        n_jobs=N_JOBS,
    )
    ensemble.fit(X_aug, y_aug)
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # CV on original data
    cv_scores = cross_val_score(ensemble, X, y, cv=CV_FOLDS, scoring='f1_weighted', n_jobs=N_JOBS)

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    logger.info(f"  Ensemble: F1={f1:.3f}  Acc={acc:.3f}  CV={cv_scores.mean():.3f}")

    return {
        'name': 'Tuned Ensemble (SMOTE + GridSearch)',
        'results': {
            'Ensemble': {
                'accuracy': acc,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
            }
        },
        'best_model': 'Ensemble',
        'best_f1': f1,
        'best_accuracy': acc,
        'report': report,
        'ensemble_model': ensemble,
        'label_encoder': le,
    }


# ─── Reporting ──────────────────────────────────────────────────────────────

def format_results(experiments: List[Dict], task: str) -> str:
    """Format all experiment results as markdown."""
    lines = [
        f"# Cough Classifier Improvement Experiments",
        f"",
        f"**Task**: `{task}` (3-class: COVID-19 / healthy / infection)",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Embedding**: HeAR 512-dim (cached, 753 samples)",
        f"**Hardware**: 32 cores, RTX 4090",
        f"",
        f"## Summary",
        f"",
        f"| # | Experiment | Best Model | Test F1 | Test Acc | CV F1 | Time |",
        f"|---|-----------|-----------|---------|----------|-------|------|",
    ]

    for i, exp in enumerate(experiments):
        best = exp['results'].get(exp['best_model'], {})
        cv_str = f"{best.get('cv_mean', 0):.3f}" if best.get('cv_mean', 0) > 0 else "N/A"
        time_str = f"{exp.get('elapsed', 0):.0f}s"
        lines.append(
            f"| {i} | {exp['name']} | {exp['best_model']} | "
            f"{exp['best_f1']:.3f} | {exp['best_accuracy']:.3f} | {cv_str} | {time_str} |"
        )

    # Improvement over baseline
    if len(experiments) >= 2:
        baseline_f1 = experiments[0]['best_f1']
        best_exp = max(experiments, key=lambda e: e['best_f1'])
        improvement = best_exp['best_f1'] - baseline_f1
        pct = improvement / baseline_f1 * 100 if baseline_f1 > 0 else 0
        lines.extend([
            f"",
            f"**Best improvement**: {best_exp['name']} "
            f"(+{improvement:.3f} F1 over baseline, "
            f"{pct:.1f}% relative improvement)",
        ])

    # Detailed results per experiment
    for i, exp in enumerate(experiments):
        lines.extend([
            f"",
            f"---",
            f"## Experiment {i}: {exp['name']}",
            f"",
        ])

        # Per-classifier results
        lines.append(f"| Classifier | Test Acc | Test F1 | CV F1 |")
        lines.append(f"|-----------|----------|---------|-------|")
        for clf_name, res in sorted(exp['results'].items(), key=lambda x: x[1]['f1_score'], reverse=True):
            cv_str = f"{res.get('cv_mean', 0):.3f} +/- {res.get('cv_std', 0):.3f}" if res.get('cv_mean', 0) > 0 else "N/A"
            marker = " **(best)**" if clf_name == exp['best_model'] else ""
            lines.append(f"| {clf_name}{marker} | {res['accuracy']:.3f} | {res['f1_score']:.3f} | {cv_str} |")

        # Tuning params if available
        if 'tuning_params' in exp:
            lines.extend([f"", f"**Best hyperparameters:**", f""])
            for clf_name, params in exp['tuning_params'].items():
                lines.append(f"- **{clf_name}**: {params}")

        # Per-class report for best model
        if 'report' in exp:
            lines.extend([f"", f"**Per-class results ({exp['best_model']}):**", f""])
            lines.append(f"| Class | Precision | Recall | F1 | Support |")
            lines.append(f"|-------|-----------|--------|-----|---------|")
            for cls in ['COVID-19', 'healthy', 'infection']:
                if cls in exp['report']:
                    r = exp['report'][cls]
                    lines.append(
                        f"| {cls} | {r['precision']:.3f} | {r['recall']:.3f} | "
                        f"{r['f1-score']:.3f} | {int(r['support'])} |"
                    )

    lines.extend([
        f"",
        f"---",
        f"## Conclusions",
        f"",
        f"*(Auto-generated — review results above for details)*",
    ])

    return "\n".join(lines)


def save_best_model(experiment: Dict, task: str):
    """Save the best model from experiments."""
    if 'ensemble_model' not in experiment:
        logger.info("No ensemble model to save")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    model_path = OUTPUT_DIR / f"{task}_improved_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': experiment['ensemble_model'],
            'scaler': None,
            'label_encoder': experiment['label_encoder'],
        }, f)
    logger.info(f"Saved improved model to {model_path}")

    meta_path = OUTPUT_DIR / f"{task}_improved_meta.json"
    meta = {
        'task_name': task,
        'experiment': experiment['name'],
        'best_model': experiment['best_model'],
        'accuracy': experiment['best_accuracy'],
        'f1_score': experiment['best_f1'],
        'classes': list(experiment['label_encoder'].classes_),
        'improvements': 'SMOTE oversampling + hyperparameter tuning + soft voting ensemble',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    task = "diagnosis_simple"
    total_t0 = time()

    logger.info(f"Loading cached embeddings for '{task}'...")
    X, y = load_cached_embeddings(task)
    logger.info(f"Loaded {len(X)} samples, {len(np.unique(y))} classes")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info(f"Classes: {list(le.classes_)}")

    # Fixed train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Using {N_JOBS} jobs (all cores)")

    experiments = []
    exp_funcs = [
        ("Experiment 0: Baseline", exp_baseline),
        ("Experiment 1: Gaussian Noise (std=0.05)", exp_gaussian),
        ("Experiment 2: Gaussian Noise (std=0.02)", exp_gaussian_low),
        ("Experiment 3: Mixup", exp_mixup),
        ("Experiment 4: SMOTE-like", exp_smote),
        ("Experiment 5: Feature Dropout", exp_dropout),
        ("Experiment 6: SMOTE + Gaussian", exp_combined),
        ("Experiment 7: Hyperparameter Tuning", exp_hyperparam_tuning),
        ("Experiment 8: Tuned Ensemble", exp_tuned_ensemble),
    ]

    for name, func in exp_funcs:
        logger.info(f"\n{'=' * 60}")
        logger.info(name)
        logger.info("=" * 60)
        result = func(X_train, y_train, X_test, y_test, le, X, y_encoded)
        experiments.append(result)

    # Report
    report = format_results(experiments, task)
    with open(RESULTS_FILE, 'w') as f:
        f.write(report)
    logger.info(f"\nResults written to {RESULTS_FILE}")

    # Save best model
    best_overall = max(experiments, key=lambda e: e['best_f1'])
    logger.info(f"\nBest experiment: {best_overall['name']} (F1={best_overall['best_f1']:.3f})")

    # Save the ensemble model if it exists, otherwise save from exp 8
    if 'ensemble_model' in best_overall:
        save_best_model(best_overall, task)
    else:
        ensemble_exp = experiments[-1]  # Last experiment is always ensemble
        if 'ensemble_model' in ensemble_exp:
            save_best_model(ensemble_exp, task)

    total_elapsed = time() - total_t0
    logger.info(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    print("\n" + report)


if __name__ == "__main__":
    main()
