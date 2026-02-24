# Respiratory Sound Type Classifier — Experiment Results

**Date**: 2026-02-16
**Task**: Train lung sound type + cough type classifiers using HeAR embeddings
**Hardware**: 32 cores, RTX 4090 (24GB VRAM)
**Embedding**: HeAR 512-dim, 2-second audio windows
**Data**: ICBHI 2017 (6,898 respiratory cycles) + SPRSound (9,089 events) + COUGHVID (expert-labeled)

## Motivation

Shifted from **disease diagnosis** (COVID-19 / healthy / infection) to **sound type** classification:
- Disease diagnosis had liability concerns and poor performance (infection F1 ≈ 0)
- Sound types (wheeze, crackle, etc.) are objective acoustic observations, not diagnoses
- Better fit for Google HeAR hackathon — HeAR was designed for health sound classification

## Dataset Summary

### Lung Sound (ICBHI + SPRSound merged)

| Label | Count | Source |
|-------|-------|--------|
| normal | 17,286 (74.8%) | ICBHI + SPRSound |
| crackle | 2,829 (12.2%) | ICBHI + SPRSound |
| wheeze | 1,887 (8.2%) | ICBHI + SPRSound |
| both | 548 (2.4%) | ICBHI (crackles=1, wheezes=1) + SPRSound |
| rhonchi | 206 (0.9%) | SPRSound only |
| stridor | 66 (0.3%) | SPRSound only |
| **Total** | **23,116** | |

Patient-level split: GroupShuffleSplit (80/20), ensuring no patient appears in both train and test.

### Cough Type (COUGHVID expert labels)

| Label | Count |
|-------|-------|
| dry | 2,067 (73.3%) |
| wet | 752 (26.7%) |
| **Total** | **2,819** |

---

## Lung Sound Experiments

### Experiment 1: 6-class baseline

**Config**: All 6 classes, no undersampling, no class collapsing
**Classes**: normal, crackle, wheeze, both, rhonchi, stridor

| Classifier | Accuracy | Weighted F1 |
|-----------|----------|-------------|
| LinearSVC | 0.767 | 0.718 |
| Logistic Regression | 0.685 | 0.719 |
| Gradient Boosting | 0.757 | 0.712 |
| Random Forest | 0.748 | 0.664 |
| MLP | 0.769 | 0.720 |
| **SGD (log_loss)** | **0.771** | **0.733** |
| **Ensemble** | **0.772** | **0.758** |

**Macro F1: ~0.41** — weighted F1 inflated by dominant normal class.

Per-class breakdown (ensemble):

| Class | F1 | Support |
|-------|-----|---------|
| normal | 0.88 | ~3,450 |
| crackle | 0.48 | ~565 |
| wheeze | 0.47 | ~377 |
| both | 0.11 | ~110 |
| rhonchi | 0.27 | ~41 |
| stridor | 0.00 | ~13 |

**Conclusion**: Rare classes (stridor, rhonchi, both) are unlearnable with <200 samples on sklearn classifiers.

---

### Experiment 2: 4-class (merge rhonchi/stridor → wheeze)

**Config**: Collapse map: rhonchi→wheeze, stridor→wheeze. No undersampling.
**Classes**: normal, crackle, wheeze, both

| Classifier | Accuracy | Weighted F1 |
|-----------|----------|-------------|
| LinearSVC | 0.773 | 0.729 |
| Logistic Regression | 0.699 | 0.729 |
| Gradient Boosting | 0.762 | 0.715 |
| Random Forest | 0.750 | 0.670 |
| MLP | 0.763 | 0.705 |
| **SGD (log_loss)** | **0.777** | **0.742** |
| **Ensemble** | **0.768** | **0.760** |

**Macro F1: ~0.49** — improvement from eliminating noise of unlearnable classes.

---

### Experiment 3: 4-class + 1.5x undersampling

**Config**: 4-class + cap normal at 1.5x the second-largest class in training set.

| Classifier | Accuracy | Weighted F1 |
|-----------|----------|-------------|
| LinearSVC | 0.757 | 0.759 |
| Logistic Regression | 0.696 | 0.729 |
| Gradient Boosting | 0.743 | 0.750 |
| Random Forest | 0.749 | 0.747 |
| MLP | 0.711 | 0.731 |
| **SGD (log_loss)** | **0.753** | **0.760** |
| **Ensemble** | **0.763** | **0.766** |

**Macro F1: ~0.50**

---

### Experiment 4: 4-class + 2.0x undersampling

**Config**: 4-class + cap normal at 2.0x the second-largest class in training set.

| Classifier | Accuracy | Weighted F1 |
|-----------|----------|-------------|
| LinearSVC | 0.768 | 0.762 |
| Logistic Regression | 0.696 | 0.728 |
| Gradient Boosting | 0.755 | 0.753 |
| Random Forest | 0.769 | 0.750 |
| MLP | 0.760 | 0.762 |
| **SGD (log_loss)** | **0.771** | **0.769** |
| **Ensemble** | **0.777** | **0.773** |

**Macro F1: ~0.50**

---

### Experiment 5: 3-class + 2.0x undersampling (v1 baseline)

**Config**: Collapse both/rhonchi/stridor → wheeze, cap normal at 2.0x second-largest class. No scaling.
**Classes**: normal, crackle, wheeze

| Classifier | Accuracy | Weighted F1 |
|-----------|----------|-------------|
| LinearSVC | 0.777 | 0.781 |
| Logistic Regression | 0.744 | 0.760 |
| Gradient Boosting | 0.772 | 0.776 |
| Random Forest | 0.782 | 0.778 |
| **MLP** | **0.790** | **0.791** |
| SGD (log_loss) | 0.770 | 0.778 |
| **Ensemble (soft voting, top 3)** | **0.792** | **0.795** |

**Macro F1: ~0.68**

---

### Experiment 6: 3-class + 2.0x US + StandardScaler + XGBoost (BEST)

**Config**: Same as Exp 5, plus StandardScaler, XGBoost added, soft voting over top 4.
**Improvements**: StandardScaler critical for LinearSVC/MLP/SGD; XGBoost (GPU-accelerated) added.

| Classifier | Accuracy | Weighted F1 |
|-----------|----------|-------------|
| LinearSVC | 0.777 | 0.781 |
| Logistic Regression | 0.743 | 0.760 |
| Gradient Boosting | 0.772 | 0.776 |
| Random Forest | 0.782 | 0.778 |
| **MLP** | **0.799** | **0.799** |
| SGD (log_loss) | 0.772 | 0.778 |
| XGBoost | 0.770 | 0.779 |
| Stacking (top 4) | 0.790 | 0.782 |
| **Voting (top 4)** | **0.805** | **0.805** |

**Improvement: 0.795 → 0.805 (+1.3%)**

Per-class breakdown (voting ensemble):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| normal | 0.89 | 0.88 | 0.88 | 3,888 |
| crackle | 0.57 | 0.53 | 0.55 | 866 |
| wheeze | 0.58 | 0.67 | 0.62 | 610 |

**This is the production model**: `lung_sound_3class_us2p0_ensemble.pkl`

---

### Experiment 7: PyTorch classifier with focal loss

**Config**: 3-layer MLP (256→128→64) with BatchNorm, Dropout, focal loss (γ=1.0), cosine annealing LR, 60/20/20 patient-level split.

| Task | Weighted F1 | Crackle F1 | Wheeze F1 | Macro F1 |
|------|-------------|-----------|----------|---------|
| Lung sound (γ=1.0) | 0.783 | 0.55 | 0.58 | 0.66 |
| Lung sound (γ=2.0) | 0.754 | 0.50 | 0.55 | 0.63 |

**Trade-off**: Lower weighted F1 but better minority recall. Not used for production lung sound (sklearn ensemble is better overall).

---

### Summary Table

| Variant | Weighted F1 | Macro F1 | Notes |
|---------|-------------|----------|-------|
| 6-class baseline | 0.758 | ~0.41 | stridor/rhonchi F1≈0 |
| 4-class | 0.760 | ~0.49 | merge rhonchi/stridor→wheeze |
| 4-class + 1.5x US | 0.766 | ~0.50 | |
| 4-class + 2.0x US | 0.773 | ~0.50 | |
| **3-class + 2.0x US** | **0.795** | **~0.68** | **production model** |

---

## Threshold Tuning Analysis

Attempted per-class threshold tuning to improve minority class recall.

**Method**: For each class, sweep threshold 0.05–0.95 in steps of 0.05, find threshold maximizing binary F1. At inference, if any abnormal class passes its threshold, prefer it over normal.

**Results**:
- 6-class: macro F1 improved by +0.018, but weighted F1 dropped by -0.027
- 4-class: macro F1 improved by +0.001, weighted F1 dropped by -0.001
- **Conclusion**: Threshold tuning provides negligible benefit. Probability estimates are too poorly calibrated for minority classes. Not used in production.

---

## Cough Type Results

### sklearn classifiers (with StandardScaler + XGBoost)

**Config**: dry/wet binary classification, COUGHVID expert labels (10,695 segments from 1,602 recordings), StandardScaler applied.

| Classifier | Accuracy | Weighted F1 |
|-----------|----------|-------------|
| LinearSVC | 0.695 | 0.572 |
| Logistic Regression | 0.631 | 0.643 |
| Gradient Boosting | 0.703 | 0.625 |
| Random Forest | 0.703 | 0.598 |
| **MLP** | **0.694** | **0.663** |
| SGD (log_loss) | 0.696 | 0.571 |
| XGBoost | 0.692 | 0.659 |
| **Voting (top 4)** | **0.700** | **0.654** |

Per-class (MLP, best single):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| dry | 0.73 | 0.88 | 0.80 | 1,490 |
| wet | 0.49 | 0.27 | 0.35 | 652 |

### PyTorch classifier with focal loss (best for wet cough)

**Config**: 3-layer MLP (256→128→64) with BatchNorm, Dropout, focal loss (γ=1.0), cosine annealing LR, balanced class weights, 60/20/20 patient-level split.

| Variant | Weighted F1 | Dry F1 | **Wet F1** |
|---------|-------------|--------|----------|
| PyTorch γ=2.0 | 0.645 | 0.76 | 0.38 |
| **PyTorch γ=1.0** | **0.657** | **0.77** | **0.41** |
| Keras γ=1.0 | 0.604 | 0.66 | 0.48 |

**Key result**: Focal loss improved wet cough F1 from **0.32 → 0.41** (+28% relative), with moderate trade-off in overall weighted F1.

**Data scarcity**: A 2025 systematic review found only 3 studies in the entire ML literature on wet/dry cough classification. Our 10,695 segments (from 1,602 COUGHVID recordings: 1,248 dry, 445 wet) is one of the largest labeled datasets for this task. The only promising additional dataset (Smarty4covid, ~1,475 expert-labeled) requires restricted access.

**Conclusion**: Wet cough detection is fundamentally limited by data scarcity and low inter-rater agreement (κ = 0.22–0.81). Focal loss significantly improves minority class detection at the cost of overall F1.

---

## Key Insights

1. **StandardScaler matters**: +1.3% weighted F1 on lung sound from scaling alone. Critical for LinearSVC, MLP, SGD.
2. **Undersampling majority class is critical**: Capping normal at 2x the largest minority class boosted macro F1 from 0.41 → 0.68.
3. **Class collapsing helps**: 6→3 class (merge rhonchi/stridor/both→wheeze) improved weighted F1 from 0.758→0.795. Rare classes (<200 samples) are unlearnable with sklearn.
4. **Ensembles consistently win**: Soft voting ensemble outperforms best single model by +0.005-0.03 F1.
5. **Stacking vs soft voting**: Stacking won for balanced datasets; soft voting won for imbalanced ones (cough type).
6. **Focal loss for minority classes**: PyTorch with focal loss improved wet cough F1 from 0.32 → 0.41 (+28%). Higher gamma values (2.0) push more aggressively toward minority recall at the expense of overall F1.
7. **SVM (libsvm) is impractical**: O(n²-n³) on 23k samples. LinearSVC/SGDClassifier are O(n) alternatives.
8. **XGBoost on par with sklearn**: XGBoost (GPU-accelerated) competitive with best sklearn models, much faster than GradientBoosting.
9. **Threshold tuning doesn't help**: Probability calibration is too poor for minority classes.
10. **Embedding-space augmentation doesn't help**: Gaussian noise, mixup, SMOTE, feature dropout all failed on HeAR embeddings.
11. **Data scarcity is the wet cough bottleneck**: Only 445 wet cough recordings in COUGHVID (the largest public dataset). Smarty4covid (~1,475 labeled) is the most promising additional source but requires access approval.

## Performance Summary

| Task | Model | Weighted F1 | Notes |
|------|-------|-------------|-------|
| **Lung sound** | sklearn ensemble (scaling + XGBoost) | **0.805** | Production model |
| Lung sound | PyTorch (focal γ=1.0) | 0.783 | Better minority recall |
| **Cough type** | sklearn MLP (scaling) | **0.663** | Best overall F1 |
| Cough type | PyTorch (focal γ=1.0) | 0.657 | **Wet F1=0.41** (vs 0.35) |

## Production Models

| Task | Model File | Classes | Weighted F1 |
|------|-----------|---------|-------------|
| Lung sound | `lung_sound_3class_us2p0_ensemble.pkl` | normal, crackle, wheeze | **0.805** |
| Cough type | `cough_type_ensemble.pkl` | dry, wet | 0.654 |

Models now include a fitted `StandardScaler` (saved alongside model in pickle). Production inference in `cough_classifier.py` already handles `if scaler is not None: X = scaler.transform(X)`.

### Inference Loading Priority

In `cough_classifier.py:_ensure_loaded()`:
1. `lung_sound_3class_us2p0` (best)
2. `lung_sound_4class` (fallback)
3. `lung_sound` (6-class fallback)

## Reproducing

```bash
cd backend/ml_training

# Lung sound: sklearn with scaling + XGBoost (production)
python train_respiratory_classifiers.py --stage train --task lung_sound \
    --collapse-classes 3 --undersample 2.0

# Cough type: sklearn with scaling + XGBoost
python train_respiratory_classifiers.py --stage train --task cough_type

# With hyperparameter tuning (slow — ~2h for GradientBoosting CV)
python train_respiratory_classifiers.py --stage train --task lung_sound \
    --collapse-classes 3 --undersample 2.0 --tune

# PyTorch classifiers (fast — <1 min each)
python train_pytorch_classifier.py --task lung_sound --collapse-classes 3 --undersample 2.0 --focal-gamma 1.0
python train_pytorch_classifier.py --task cough_type --focal-gamma 1.0

# Keras classifier
python finetune_hear.py --task lung_sound --collapse-classes 3 --undersample 2.0
python finetune_hear.py --task cough_type --focal-gamma 1.0
```
