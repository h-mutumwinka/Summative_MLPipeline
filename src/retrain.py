"""
retrain.py

Fine-tunes the CIFAR-10 model on newly uploaded images.

Workflow
--------
1. Load the currently served model.
2. Load the uploaded dataset.
3. Evaluate baseline accuracy on the uploaded images.
4. Fine-tune on the uploaded dataset (reduced LR + early stopping).
5. Re-evaluate on the same dataset.
6. Promote the new weights only if accuracy did NOT drop by > 1 pp.
7. Save the run (before/after metrics + promotion decision) to SQLite.
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from src.database import save_retrain_run
from src.preprocessing import CLASS_NAMES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH          = "models/cifar10_model.keras"
PROMOTION_THRESHOLD = 0.01   # allow at most a 1 percentage-point accuracy drop


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def retrain_model(dataset_folder: str, epochs: int = 5, batch_size: int = 32) -> dict:
    """
    Fine-tune the served model on `dataset_folder` and apply the promotion gate.
    """
    dataset_folder = Path(dataset_folder)
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

    t0 = time.time()

    # ── 1. Load model ────────────────────────────────────────────────────
    model = load_model(MODEL_PATH)

    # ── 2. Build uploaded dataset ─────────────────────────────────────────
    # Ensure all 10 class directories exist (even if empty) so TF doesn't crash
    for cls_name in CLASS_NAMES:
        (dataset_folder / cls_name).mkdir(parents=True, exist_ok=True)

    # For small datasets, we evaluate and train on the same data just to test
    # the pipeline without downloading 170MB of CIFAR-10 over the internet.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_folder,
        image_size=(32, 32),
        batch_size=batch_size,
        label_mode="int",
        class_names=CLASS_NAMES,
        shuffle=True,
    )
    
    # Normalise to [0, 1]
    norm     = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y))

    # ── 3. Baseline accuracy ─────────────────────────────────────────────
    _, acc_before = model.evaluate(train_ds, verbose=0)
    acc_before = round(float(acc_before), 4)
    print(f"[retrain] Accuracy BEFORE: {acc_before:.4f}")

    # ── 4. Fine-tune at a reduced learning rate ───────────────────────────
    model.optimizer.learning_rate = 1e-4

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=[
            EarlyStopping(monitor="loss", patience=2, restore_best_weights=True),
            ReduceLROnPlateau(monitor="loss", factor=0.5, patience=1, verbose=1),
        ],
        verbose=1,
    )

    # ── 5. Evaluate after retraining ──────────────────────────────────────
    _, acc_after = model.evaluate(train_ds, verbose=0)
    acc_after = round(float(acc_after), 4)
    print(f"[retrain] Accuracy AFTER:  {acc_after:.4f}")

    # ── 6. Promotion gate ─────────────────────────────────────────────────
    promoted = acc_after >= (acc_before - PROMOTION_THRESHOLD)
    if promoted:
        model.save(MODEL_PATH)
        print("[retrain] ✅ New model PROMOTED — model file updated.")
    else:
        print(
            f"[retrain] ❌ Model REJECTED — accuracy dropped "
            f"{acc_before - acc_after:.4f} (threshold {PROMOTION_THRESHOLD})."
        )

    duration = round(time.time() - t0, 2)
    status   = "promoted" if promoted else "rejected"

    # ── 7. Persist run to database ────────────────────────────────────────
    save_retrain_run(
        dataset          = dataset_folder.name,
        accuracy_before  = acc_before,
        accuracy_after   = acc_after,
        promoted         = int(promoted),
        duration_seconds = duration,
        status           = status,
    )

    return {
        "accuracy_before":  acc_before,
        "accuracy_after":   acc_after,
        "promoted":         promoted,
        "duration_seconds": duration,
        "status":           status,
    }