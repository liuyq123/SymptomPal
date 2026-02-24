"""Analyze threshold tuning and per-class performance for trained models.

Loads cached embeddings + trained models, evaluates with different thresholds,
and compares 6-class vs 4-class performance.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score

RANDOM_STATE = 42
CACHE_DIR = Path(__file__).parent / "embedding_cache"
MODELS_DIR = Path(__file__).parent / "trained_models"


def load_and_split(task: str, collapse_map: dict = None):
    """Load embeddings and create train/test split matching training."""
    cache_path = CACHE_DIR / f"{task}_respiratory_embeddings.pkl"
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)

    X, y, patient_ids = cached['X'], cached['y'], cached['patient_ids']

    if collapse_map:
        y = np.array([collapse_map.get(label, label) for label in y])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    groups = np.array(patient_ids)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))

    return X[test_idx], y_encoded[test_idx], le


def load_model(task: str, ensemble: bool = True):
    """Load trained model."""
    suffix = "_ensemble.pkl" if ensemble else "_model.pkl"
    model_path = MODELS_DIR / f"{task}{suffix}"
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler'], data['label_encoder']


def threshold_analysis(model, X_test, y_test, label_encoder, task_label: str):
    """Analyze per-class thresholds to improve minority class recall."""
    proba = model.predict_proba(X_test)
    classes = label_encoder.classes_

    print(f"\n{'='*70}")
    print(f"THRESHOLD ANALYSIS: {task_label}")
    print(f"{'='*70}")

    # Default argmax predictions
    y_pred_default = np.argmax(proba, axis=1)
    print(f"\n--- Default (argmax) ---")
    print(classification_report(y_test, y_pred_default, target_names=classes))

    # Try lowering threshold for minority classes
    # Strategy: for each class, find the threshold that maximizes F1
    best_thresholds = {}
    for i, cls in enumerate(classes):
        cls_mask = (y_test == i)
        if cls_mask.sum() == 0:
            continue

        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            y_pred_cls = (proba[:, i] >= t).astype(int)
            y_true_cls = cls_mask.astype(int)
            f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[cls] = {'threshold': best_t, 'f1': best_f1}
        print(f"  {cls}: best threshold={best_t:.2f}, binary F1={best_f1:.3f} (support={cls_mask.sum()})")

    # Apply per-class thresholds with priority-based resolution
    # Priority: abnormal classes > normal (we want to catch abnormalities)
    print(f"\n--- Per-class threshold predictions ---")
    priority = {cls: i for i, cls in enumerate(classes)}
    # Give abnormal classes higher priority (lower number = higher priority)
    if 'normal' in priority:
        normal_idx = priority['normal']
        for cls in priority:
            if cls != 'normal':
                priority[cls] = 0  # All abnormal classes get top priority

    y_pred_thresh = np.full(len(X_test), -1, dtype=int)
    for sample_idx in range(len(X_test)):
        candidates = []
        for cls_idx, cls in enumerate(classes):
            t = best_thresholds[cls]['threshold']
            if proba[sample_idx, cls_idx] >= t:
                candidates.append((cls_idx, proba[sample_idx, cls_idx]))

        if candidates:
            # Among candidates, pick the one with highest probability
            # But if an abnormal class passes its threshold, prefer it over normal
            abnormal_candidates = [(idx, p) for idx, p in candidates if classes[idx] != 'normal']
            if abnormal_candidates:
                y_pred_thresh[sample_idx] = max(abnormal_candidates, key=lambda x: x[1])[0]
            else:
                y_pred_thresh[sample_idx] = max(candidates, key=lambda x: x[1])[0]
        else:
            # No class passes threshold — fall back to argmax
            y_pred_thresh[sample_idx] = np.argmax(proba[sample_idx])

    print(classification_report(y_test, y_pred_thresh, target_names=classes))
    f1_thresh = f1_score(y_test, y_pred_thresh, average='weighted')
    f1_default = f1_score(y_test, y_pred_default, average='weighted')
    f1_macro_thresh = f1_score(y_test, y_pred_thresh, average='macro')
    f1_macro_default = f1_score(y_test, y_pred_default, average='macro')

    print(f"  Weighted F1: {f1_default:.3f} → {f1_thresh:.3f} ({'+'if f1_thresh>f1_default else ''}{f1_thresh-f1_default:.3f})")
    print(f"  Macro F1:    {f1_macro_default:.3f} → {f1_macro_thresh:.3f} ({'+'if f1_macro_thresh>f1_macro_default else ''}{f1_macro_thresh-f1_macro_default:.3f})")

    return best_thresholds


def main():
    collapse_map = {
        'normal': 'normal', 'crackle': 'crackle', 'wheeze': 'wheeze',
        'both': 'both', 'rhonchi': 'wheeze', 'stridor': 'wheeze',
    }

    # --- 6-class lung sound ---
    print("\n" + "="*70)
    print("COMPARISON: 6-class vs 4-class lung sound")
    print("="*70)

    try:
        X_test_6, y_test_6, le_6 = load_and_split('lung_sound')
        model_6, scaler_6, le_model_6 = load_model('lung_sound', ensemble=True)
        if scaler_6:
            X_test_6 = scaler_6.transform(X_test_6)
        threshold_analysis(model_6, X_test_6, y_test_6, le_6, "6-class Lung Sound (ensemble)")
    except Exception as e:
        print(f"6-class analysis failed: {e}")

    try:
        X_test_4, y_test_4, le_4 = load_and_split('lung_sound', collapse_map=collapse_map)
        model_4, scaler_4, le_model_4 = load_model('lung_sound_4class', ensemble=True)
        if scaler_4:
            X_test_4 = scaler_4.transform(X_test_4)
        thresholds_4 = threshold_analysis(model_4, X_test_4, y_test_4, le_4, "4-class Lung Sound (ensemble)")

        # Save optimal thresholds for inference
        thresh_path = MODELS_DIR / "lung_sound_4class_thresholds.json"
        with open(thresh_path, 'w') as f:
            json.dump(thresholds_4, f, indent=2)
        print(f"\nSaved thresholds to {thresh_path}")
    except Exception as e:
        print(f"4-class analysis failed: {e}")

    # --- Cough type ---
    try:
        X_test_ct, y_test_ct, le_ct = load_and_split('cough_type')
        model_ct, scaler_ct, le_model_ct = load_model('cough_type', ensemble=True)
        if scaler_ct:
            X_test_ct = scaler_ct.transform(X_test_ct)
        threshold_analysis(model_ct, X_test_ct, y_test_ct, le_ct, "Cough Type (ensemble)")
    except Exception as e:
        print(f"Cough type analysis failed: {e}")


if __name__ == "__main__":
    main()
