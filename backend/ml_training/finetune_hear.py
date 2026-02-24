#!/usr/bin/env python3
"""
Fine-tune HeAR model with a deeper classification head using Keras/TensorFlow.

HeAR is loaded as a TFSMLayer (frozen by default). We add a deeper classification
head with BatchNorm and Dropout, trained on raw 2-second audio clips.

Two modes:
1. --freeze-hear (default): Train only the classification head on HeAR embeddings.
   Functionally similar to PyTorch classifier but uses Keras with audio-domain
   augmentation during training.
2. --unfreeze-hear: Attempt to fine-tune HeAR layers. TFSMLayer may not support
   gradient flow — this mode will warn if gradients don't propagate.

Usage:
    python finetune_hear.py --task lung_sound --collapse-classes 3 --undersample 2.0
    python finetune_hear.py --task cough_type --focal-gamma 2.0
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from time import time
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "embedding_cache"
OUTPUT_DIR = Path(__file__).parent / "trained_models"
RANDOM_STATE = 42

LUNG_SOUND_3CLASS_MAP = {
    'normal': 'normal', 'crackle': 'crackle', 'wheeze': 'wheeze',
    'both': 'wheeze', 'rhonchi': 'wheeze', 'stridor': 'wheeze',
}


def build_hear_model(hear_model_dir: str, num_classes: int, freeze_hear: bool = True):
    """Build a Keras model with HeAR as the embedding layer + classification head."""
    import keras
    import tensorflow as tf

    # Load HeAR as a TFSMLayer
    hear_layer = keras.layers.TFSMLayer(hear_model_dir, call_endpoint='serving_default')

    # Input: raw 16kHz audio, 2 seconds = 32000 samples
    inputs = keras.Input(shape=(32000,), dtype='float32', name='audio_input')

    # HeAR embedding
    hear_output = hear_layer(inputs)
    # HeAR returns a dict with key 'output_0' → (batch, 512)
    if isinstance(hear_output, dict):
        embeddings = hear_output['output_0']
    else:
        embeddings = hear_output

    # Classification head
    x = keras.layers.Dense(256, name='head_dense1')(embeddings)
    x = keras.layers.BatchNormalization(name='head_bn1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128, name='head_dense2')(x)
    x = keras.layers.BatchNormalization(name='head_bn2')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(64, name='head_dense3')(x)
    x = keras.layers.BatchNormalization(name='head_bn3')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.1)(x)

    outputs = keras.layers.Dense(num_classes, activation='softmax', name='classifier')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='hear_classifier')

    # Log layer trainability
    trainable = sum(1 for layer in model.layers if layer.trainable)
    total = len(model.layers)
    logger.info(f"Model: {trainable}/{total} trainable layers")
    logger.info(f"Total params: {model.count_params():,}")

    return model


def focal_loss_fn(gamma=2.0, class_weights=None):
    """Create a focal loss function for Keras."""
    import tensorflow as tf

    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # One-hot encode if needed
        if len(y_true.shape) == 1 or y_true.shape[-1] == 1:
            y_true = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32),
                                depth=tf.shape(y_pred)[-1])
        ce = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = (1 - pt) ** gamma

        if class_weights is not None:
            cw = tf.constant(class_weights, dtype=tf.float32)
            weight = tf.reduce_sum(y_true * cw, axis=-1)
            return tf.reduce_mean(weight * focal_weight * tf.reduce_sum(ce, axis=-1))
        return tf.reduce_mean(focal_weight * tf.reduce_sum(ce, axis=-1))

    return focal_loss


def train_on_embeddings(task: str, cache_path: str, collapse_classes: Optional[str] = None,
                        undersample_ratio: float = 0.0, focal_gamma: float = 2.0,
                        epochs: int = 100, lr: float = 1e-3, batch_size: int = 256):
    """Train Keras classification head on pre-computed HeAR embeddings.

    This is faster than end-to-end training since we skip HeAR inference.
    """
    import keras
    import tensorflow as tf
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.utils.class_weight import compute_class_weight

    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)

    X = cached['X']
    y = cached['y']
    patient_ids = cached['patient_ids']

    if collapse_classes and task == 'lung_sound':
        if collapse_classes == '3':
            y = np.array([LUNG_SOUND_3CLASS_MAP.get(l, l) for l in y])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    groups = np.array(patient_ids)
    num_classes = len(label_encoder.classes_)

    logger.info(f"Classes: {list(label_encoder.classes_)}")
    for cls in label_encoder.classes_:
        logger.info(f"  {cls}: {np.sum(y == cls)} ({100*np.sum(y == cls)/len(y):.1f}%)")

    # Patient-level split: 60/20/20
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss.split(X, y_encoded, groups))
    X_trainval, X_test = X[trainval_idx], X[test_idx]
    y_trainval, y_test = y_encoded[trainval_idx], y_encoded[test_idx]
    groups_tv = groups[trainval_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups_tv))
    X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    # Undersample
    if undersample_ratio > 0:
        class_counts = np.bincount(y_train)
        maj = np.argmax(class_counts)
        cap = int(sorted(class_counts)[-2] * undersample_ratio)
        if class_counts[maj] > cap:
            maj_idx = np.where(y_train == maj)[0]
            rng = np.random.RandomState(RANDOM_STATE)
            keep = rng.choice(maj_idx, size=cap, replace=False)
            other = np.where(y_train != maj)[0]
            sel = np.sort(np.concatenate([keep, other]))
            X_train, y_train = X_train[sel], y_train[sel]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Class weights
    cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
    logger.info(f"Class weights: {dict(zip(label_encoder.classes_, cw.round(3)))}")

    # Build Keras model on embeddings (skip HeAR layer)
    inputs = keras.Input(shape=(512,), dtype='float32')
    x = keras.layers.Dense(256)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    loss_fn = focal_loss_fn(gamma=focal_gamma, class_weights=list(cw))
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=loss_fn,
        metrics=['accuracy'],
    )
    logger.info(f"Model params: {model.count_params():,}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6,
        ),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=0,
    )

    best_epoch = np.argmin(history.history['val_loss'])
    logger.info(f"Best epoch: {best_epoch+1}, val_loss: {history.history['val_loss'][best_epoch]:.4f}")

    # Evaluate
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    logger.info(f"\nKeras Classifier Results:")
    logger.info(f"  Accuracy: {test_acc:.3f}")
    logger.info(f"  Weighted F1: {test_f1:.3f}")
    logger.info(f"  Macro F1: {macro_f1:.3f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")

    # Save
    suffix = ""
    if collapse_classes:
        suffix += f"_{collapse_classes}class"
    if undersample_ratio > 0:
        suffix += f"_us{undersample_ratio:.1f}".replace('.', 'p')
    save_name = f"{task}{suffix}_keras"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / f"{save_name}_model.keras"
    model.save(model_path)
    logger.info(f"Saved Keras model to {model_path}")

    # Save scaler + label_encoder separately for production inference
    aux_path = OUTPUT_DIR / f"{save_name}_aux.pkl"
    with open(aux_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'label_encoder': label_encoder}, f)

    meta = {
        'task_name': save_name,
        'classifier': 'Keras',
        'accuracy': float(test_acc),
        'f1_score': float(test_f1),
        'macro_f1': float(macro_f1),
        'focal_gamma': focal_gamma,
        'lr': lr,
        'best_epoch': int(best_epoch + 1),
        'epochs_trained': len(history.history['val_loss']),
        'classes': list(label_encoder.classes_),
    }
    meta_path = OUTPUT_DIR / f"{save_name}_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HeAR classification head")
    parser.add_argument("--task", choices=['lung_sound', 'cough_type'], required=True)
    parser.add_argument("--collapse-classes", choices=['3', '4'], default=None)
    parser.add_argument("--undersample", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cache_path = str(CACHE_DIR / f"{args.task}_respiratory_embeddings.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    start = time()
    train_on_embeddings(
        task=args.task, cache_path=cache_path,
        collapse_classes=args.collapse_classes,
        undersample_ratio=args.undersample,
        focal_gamma=args.focal_gamma,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size,
    )
    elapsed = time() - start
    logger.info(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
