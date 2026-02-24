# Respiratory Sound Classifier Training Pipeline

Training pipeline for respiratory sound classifiers using Google's HeAR (Health Acoustic Representations) embeddings.

## Overview

The classifiers use transfer learning on HeAR 512-dim embeddings to classify:
- **Lung sound type**: normal / crackle / wheeze (3-class)
- **Cough type**: dry / wet (binary)

```
Audio File (.wav/.webm/.ogg)
        │
        ▼
   [Preprocessing]
   - Resample to 16kHz
   - Pad/trim to 2 seconds
        │
        ▼
   [HeAR Model]
   - 512-dim embedding
        │
        ▼
   [Trained Classifier]
   - Ensemble (soft voting)
        │
        ▼
   Classification Results
   - Lung sound: normal / crackle / wheeze + confidence
   - Cough type: dry / wet + confidence
```

## Training Data

| Dataset | Samples | Labels | Source |
|---------|---------|--------|--------|
| **ICBHI 2017** | 6,898 respiratory cycles | normal, crackle, wheeze, both | `ml_data/icbhi/` |
| **SPRSound** | 9,089 events | Normal, Wheeze, Crackle, Rhonchi, Stridor, etc. | `ml_data/sprsound/` |
| **COUGHVID** | 2,819 expert-labeled | dry, wet | `ml_data/coughvid/` |

Total lung sound segments after merging: **23,116** (from 6,898 ICBHI respiratory cycles + 9,089 SPRSound events, split into 2-second windows).

### Label Harmonization

ICBHI (4-class) and SPRSound (7-class) labels are merged, then collapsed to 3 classes:

| Production Label | ICBHI Source | SPRSound Source |
|---|---|---|
| `normal` | crackles=0, wheezes=0 | Normal |
| `crackle` | crackles=1, wheezes=0 | Fine Crackle, Coarse Crackle |
| `wheeze` | crackles=0, wheezes=1 | Wheeze, Rhonchi, Stridor, Wheeze+Crackle |

Rare classes (rhonchi: 206, stridor: 66, both: 548) are merged into `wheeze` — they have <200 samples and are unlearnable with sklearn.

## Training Pipeline

### Quick Start

```bash
cd backend/ml_training
conda activate mysymptom-scribe

# Lung sound: extract segments → generate HeAR embeddings → train classifiers
python train_respiratory_classifiers.py --stage all --task lung_sound \
    --collapse-classes 3 --undersample 2.0 --batch-size 64

# Cough type
python train_respiratory_classifiers.py --stage all --task cough_type --batch-size 64
```

### Stages

| Stage | What it does | Output |
|-------|-------------|--------|
| `extract` | Parse ICBHI/SPRSound annotations, cut 2s segments | Segment catalog DataFrame |
| `embed` | Generate HeAR embeddings (GPU batch inference) | `embedding_cache/{task}_respiratory_embeddings.pkl` |
| `train` | Train 7 classifiers + ensemble | `trained_models/{task}_*.pkl` |
| `all` | Run all three stages | All outputs |

### CLI Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--stage` | Pipeline stage to run | Required |
| `--task` | `lung_sound` or `cough_type` | Required |
| `--collapse-classes` | `3` (merge both/rhonchi/stridor→wheeze) or `4` (merge rhonchi/stridor→wheeze) | None |
| `--undersample` | Cap majority class to Nx second-largest (e.g. `2.0`) | 0 (disabled) |
| `--batch-size` | HeAR inference batch size | 64 |
| `--workers` | Parallel workers for segment extraction | auto |
| `--force` | Re-run even if cache exists | false |

### Classifiers Trained

7 classifiers + soft voting ensemble:

1. **LinearSVC** (CalibratedClassifierCV) — fast linear SVM, O(n)
2. **Logistic Regression** — baseline
3. **Gradient Boosting** — ensemble of trees (slow on large datasets)
4. **Random Forest** — bagging
5. **MLP** — neural network (256, 128)
6. **SGD (log_loss)** (CalibratedClassifierCV) — stochastic gradient descent
7. **XGBoost** — GPU-accelerated gradient boosting
8. **Ensemble** — soft voting over top 4 models

All models use `class_weight='balanced'` where supported.

## Results

### Lung Sound (Production: 3-class + 2x undersampling + StandardScaler + XGBoost)

| Metric | Value |
|--------|-------|
| **Weighted F1** | **0.805** |
| **Macro F1** | **~0.68** |
| Best single model | MLP (F1=0.799) |

Per-class (voting ensemble):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| normal | 0.89 | 0.88 | 0.88 | 3,888 |
| crackle | 0.57 | 0.53 | 0.55 | 866 |
| wheeze | 0.58 | 0.67 | 0.62 | 610 |

### Cough Type

| Metric | Value |
|--------|-------|
| **Weighted F1** | **0.663** |
| Best single model | MLP (F1=0.663) |

Per-class (MLP):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| dry | 0.73 | 0.88 | 0.80 | 1,490 |
| wet | 0.49 | 0.27 | 0.35 | 652 |

See [respiratory_classifier_results.md](respiratory_classifier_results.md) for full experiment comparison (6-class vs 4-class vs 3-class, undersampling ablation, threshold tuning analysis).

## Output Files

```
trained_models/
├── lung_sound_3class_us2p0_ensemble.pkl   # Production lung sound model
├── lung_sound_3class_us2p0_model.pkl      # Best single model
├── lung_sound_3class_us2p0_meta.json      # Metrics
├── cough_type_ensemble.pkl                # Production cough type model
├── cough_type_model.pkl
├── cough_type_meta.json
├── lung_sound_ensemble.pkl                # 6-class (archived)
├── lung_sound_4class_*.pkl                # 4-class variants (archived)
└── diagnosis_simple_*.pkl                 # Old disease diagnosis (deprecated)

embedding_cache/
├── lung_sound_respiratory_embeddings.pkl   # 23,116 x 512
└── cough_type_respiratory_embeddings.pkl   # 2,819 x 512
```

## Integration

```python
from app.services.cough_classifier import get_cough_classifier

classifier = get_cough_classifier()

# Lung sound classification
lung_result = classifier.classify_lung_sound(embedding)  # 512-dim HeAR embedding
print(lung_result.sound_type)    # LungSoundType.WHEEZE
print(lung_result.confidence)    # 0.73
print(lung_result.probabilities) # {'normal': 0.15, 'crackle': 0.12, 'wheeze': 0.73}

# Full classification (lung sound + cough type + severity)
result = classifier.classify(embedding)
print(result.lung_sound.sound_type)  # LungSoundType.CRACKLE
print(result.cough_type)             # CoughType.DRY
```

### Model Loading Priority

In `cough_classifier.py:_ensure_loaded()`:
1. `lung_sound_3class_us2p0` (best, preferred)
2. `lung_sound_4class` (fallback)
3. `lung_sound` (6-class fallback)

## Key Design Decisions

### Patient-Level Splitting

Uses `GroupShuffleSplit` by patient_id to prevent data leakage — multiple recordings per patient must all be in the same split.

### Undersampling > Augmentation

Embedding-space augmentation (Gaussian noise, mixup, SMOTE, feature dropout) does NOT improve sklearn classifiers on HeAR embeddings. Undersampling the majority class is far more effective.

### LinearSVC over SVM

SVM (libsvm) is O(n²-n³) — takes hours on 23k samples. LinearSVC is O(n) and produces comparable results. Wrapped in `CalibratedClassifierCV` to provide `predict_proba`.

## Requirements

```
tensorflow>=2.15
keras>=3.0
librosa>=0.10
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
huggingface_hub>=0.20
imageio-ffmpeg>=0.4
tqdm>=4.65
```

## References

- [HeAR Model (Google)](https://huggingface.co/google/hear)
- [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/)
- [SPRSound Dataset](https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound)
- [COUGHVID Dataset](https://coughvid.epfl.ch/)
- [Google HeAR Notebooks](https://github.com/google-health/hear/tree/master/notebooks)
