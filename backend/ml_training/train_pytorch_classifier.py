#!/usr/bin/env python3
"""
PyTorch classifier on frozen HeAR embeddings.

Advantages over sklearn MLP:
- BatchNorm, Dropout with proper scheduling
- Focal loss for class imbalance (critical for wet cough)
- Cosine annealing LR scheduler
- GPU acceleration (RTX 4090)
- Early stopping with patience

Usage:
    python train_pytorch_classifier.py --task lung_sound --collapse-classes 3 --undersample 2.0
    python train_pytorch_classifier.py --task cough_type
    python train_pytorch_classifier.py --task cough_type --focal-gamma 2.0  # stronger minority focus
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "embedding_cache"
OUTPUT_DIR = Path(__file__).parent / "trained_models"
RANDOM_STATE = 42

# Reuse collapse maps from main pipeline
LUNG_SOUND_3CLASS_MAP = {
    'normal': 'normal', 'crackle': 'crackle', 'wheeze': 'wheeze',
    'both': 'wheeze', 'rhonchi': 'wheeze', 'stridor': 'wheeze',
}
LUNG_SOUND_COLLAPSE_MAP = {
    'normal': 'normal', 'crackle': 'crackle', 'wheeze': 'wheeze',
    'both': 'both', 'rhonchi': 'wheeze', 'stridor': 'wheeze',
}


class FocalLoss:
    """Focal loss for class imbalance — down-weights easy examples, focuses on hard ones.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0 → standard cross-entropy
    gamma=2 → strong focus on hard examples (recommended for severe imbalance)
    """

    def __init__(self, gamma=2.0, class_weights=None):
        import torch
        self.gamma = gamma
        self.class_weights = class_weights  # tensor of per-class weights

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F

        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class HeARClassifier:
    """PyTorch classifier on frozen HeAR embeddings."""

    def __init__(self, input_dim=512, num_classes=3, hidden_dims=(256, 128, 64),
                 dropout_rates=(0.3, 0.2, 0.1), focal_gamma=2.0,
                 lr=1e-3, weight_decay=1e-4, epochs=200, patience=20,
                 batch_size=256, device='auto'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.focal_gamma = focal_gamma
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device_name = device
        self.model = None
        self.scaler = None

    def _build_model(self):
        import torch
        import torch.nn as nn

        layers = []
        in_dim = self.input_dim
        for h_dim, drop_rate in zip(self.hidden_dims, self.dropout_rates):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(drop_rate),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, self.num_classes))

        return nn.Sequential(*layers)

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler

        # Device
        if self.device_name == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.device_name)
        logger.info(f"Using device: {device}")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Convert to tensors
        X_t = torch.FloatTensor(X_train_scaled).to(device)
        y_t = torch.LongTensor(y_train).to(device)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_v = torch.FloatTensor(X_val_scaled).to(device)
            y_v = torch.LongTensor(y_val).to(device)

        # Build model
        self.model = self._build_model().to(device)
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Loss with class weights and focal loss
        if class_weights is not None:
            cw = torch.FloatTensor(class_weights).to(device)
        else:
            cw = None
        criterion = FocalLoss(gamma=self.focal_gamma, class_weights=cw)

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                       weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Data loader
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        best_val_f1 = 0
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_v)
                    val_pred = val_logits.argmax(dim=1).cpu().numpy()
                    from sklearn.metrics import f1_score
                    val_f1 = f1_score(y_val, val_pred, average='weighted')

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 20 == 0:
                    logger.info(f"  Epoch {epoch+1}/{self.epochs}: loss={total_loss/len(loader):.4f}, "
                                f"val_f1={val_f1:.4f}, best={best_val_f1:.4f}, "
                                f"lr={scheduler.get_last_lr()[0]:.6f}")

                if patience_counter >= self.patience:
                    logger.info(f"  Early stopping at epoch {epoch+1} (patience={self.patience})")
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device)
            logger.info(f"  Restored best model (val_f1={best_val_f1:.4f})")

        self.device = device
        return self

    def predict(self, X):
        import torch
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            logits = self.model(X_t)
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_scaled).to(self.device)
            logits = self.model(X_t)
            return F.softmax(logits, dim=1).cpu().numpy()


def train_pytorch(task: str, cache_path: str, collapse_classes: Optional[str] = None,
                  undersample_ratio: float = 0.0, focal_gamma: float = 2.0,
                  epochs: int = 200, lr: float = 1e-3):
    """Train PyTorch classifier on cached HeAR embeddings."""
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight

    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)

    X = cached['X']
    y = cached['y']
    patient_ids = cached['patient_ids']

    logger.info(f"Loaded {X.shape[0]} embeddings ({X.shape[1]}-dim)")

    # Collapse classes
    if collapse_classes and task == 'lung_sound':
        if collapse_classes == '3':
            cmap = LUNG_SOUND_3CLASS_MAP
        else:
            cmap = LUNG_SOUND_COLLAPSE_MAP
        y = np.array([cmap.get(label, label) for label in y])
        logger.info(f"Collapsed to {len(np.unique(y))} classes")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    groups = np.array(patient_ids)
    num_classes = len(label_encoder.classes_)

    logger.info(f"Classes: {list(label_encoder.classes_)}")
    for cls in label_encoder.classes_:
        count = np.sum(y == cls)
        logger.info(f"  {cls}: {count} ({100*count/len(y):.1f}%)")

    # Patient-level split: 60% train, 20% val, 20% test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss.split(X, y_encoded, groups))

    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y_encoded[trainval_idx], y_encoded[test_idx]
    groups_trainval = groups[trainval_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups_trainval))

    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    # Undersample majority in training only
    if undersample_ratio > 0:
        class_counts = np.bincount(y_train)
        majority_class = np.argmax(class_counts)
        second_largest = sorted(class_counts)[-2]
        cap = int(second_largest * undersample_ratio)

        if class_counts[majority_class] > cap:
            logger.info(f"Undersampling '{label_encoder.classes_[majority_class]}': "
                        f"{class_counts[majority_class]} → {cap}")
            majority_idx = np.where(y_train == majority_class)[0]
            rng = np.random.RandomState(RANDOM_STATE)
            keep_idx = rng.choice(majority_idx, size=cap, replace=False)
            minority_idx = np.where(y_train != majority_class)[0]
            train_keep = np.sort(np.concatenate([keep_idx, minority_idx]))
            X_train, y_train = X_train[train_keep], y_train[train_keep]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Class weights for focal loss
    cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    logger.info(f"Class weights: {dict(zip(label_encoder.classes_, cw.round(3)))}")

    # Train
    clf = HeARClassifier(
        input_dim=X.shape[1], num_classes=num_classes,
        focal_gamma=focal_gamma, lr=lr, epochs=epochs,
        batch_size=min(256, len(X_train) // 4),
    )
    clf.fit(X_train, y_train, X_val, y_val, class_weights=cw)

    # Evaluate on test
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    logger.info(f"\nPyTorch Classifier Results:")
    logger.info(f"  Accuracy: {test_acc:.3f}")
    logger.info(f"  Weighted F1: {test_f1:.3f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")

    # Save
    task_suffix = ""
    if collapse_classes:
        task_suffix += f"_{collapse_classes}class"
    if undersample_ratio > 0:
        task_suffix += f"_us{undersample_ratio:.1f}".replace('.', 'p')
    save_name = f"{task}{task_suffix}_pytorch"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model state dict + scaler + label_encoder
    import torch
    model_path = OUTPUT_DIR / f"{save_name}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model_state_dict': {k: v.cpu() for k, v in clf.model.state_dict().items()},
            'model_config': {
                'input_dim': X.shape[1], 'num_classes': num_classes,
                'hidden_dims': clf.hidden_dims, 'dropout_rates': clf.dropout_rates,
            },
            'scaler': clf.scaler,
            'label_encoder': label_encoder,
        }, f)
    logger.info(f"Saved PyTorch model to {model_path}")

    # Save metadata
    meta_path = OUTPUT_DIR / f"{save_name}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            'task_name': save_name,
            'classifier': 'PyTorch',
            'accuracy': test_acc,
            'f1_score': test_f1,
            'focal_gamma': focal_gamma,
            'lr': lr,
            'epochs': epochs,
            'classes': list(label_encoder.classes_),
        }, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    return {'accuracy': test_acc, 'f1_score': test_f1}


def main():
    parser = argparse.ArgumentParser(description="PyTorch classifier on HeAR embeddings")
    parser.add_argument("--task", choices=['lung_sound', 'cough_type'], required=True)
    parser.add_argument("--collapse-classes", choices=['3', '4'], default=None)
    parser.add_argument("--undersample", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0=CE, 2=strong minority focus)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cache_path = str(CACHE_DIR / f"{args.task}_respiratory_embeddings.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}. Run embedding extraction first.")

    start = time()
    train_pytorch(
        task=args.task, cache_path=cache_path,
        collapse_classes=args.collapse_classes,
        undersample_ratio=args.undersample,
        focal_gamma=args.focal_gamma,
        epochs=args.epochs, lr=args.lr,
    )
    elapsed = time() - start
    logger.info(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
