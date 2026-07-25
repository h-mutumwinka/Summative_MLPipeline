"""Retraining.

The trigger flow, end to end:
  1. A user uploads a .zip of class folders through the UI.
  2. Every image is decoded, resized to 32x32, and written to the SQLite database.
  3. The user presses "Retrain model" (or the automatic threshold is crossed).
  4. This module loads the *currently served custom model* and continues training it
     - it is used as a pre-trained model, not rebuilt from scratch.
  5. New uploads are mixed with a replay sample of the original training data, so a
     handful of new images cannot wipe out what the model already knows.
  6. The retrained model is evaluated on the untouched test set. It is only promoted
     to production if it does not lose more than PROMOTION_TOLERANCE accuracy.
  7. Metrics before and after are written to the database and shown in the UI.
"""
from __future__ import annotations

import shutil
import threading
from datetime import datetime, timezone

import numpy as np

from src import config, database as db, model as model_lib
from src.preprocessing import load_cifar_split, load_label_names, scale

_retrain_lock = threading.Lock()


def is_running() -> bool:
    return _retrain_lock.locked()


def _replay_sample(n: int, seed: int = config.SEED):
    """A random slice of the original training set, mixed in to prevent forgetting."""
    x, y = load_cifar_split("train")
    if n >= len(x):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx], y[idx]


def run_retraining(trigger: str = "manual", epochs: int | None = None) -> dict:
    """Blocking retrain. Call from a background thread; the API does exactly that."""
    if not _retrain_lock.acquire(blocking=False):
        return {"status": "rejected", "message": "A retraining run is already in progress."}

    run_id = db.start_run(trigger)
    epochs = epochs or config.RETRAIN_EPOCHS

    try:
        from tensorflow import keras

        new_images, new_labels, upload_ids = db.fetch_unconsumed()
        if len(new_images) < config.RETRAIN_MIN_NEW_SAMPLES:
            msg = (
                f"Need at least {config.RETRAIN_MIN_NEW_SAMPLES} new images to retrain; "
                f"{len(new_images)} are pending."
            )
            db.finish_run(run_id, status="skipped", message=msg, n_new_samples=len(new_images))
            return {"status": "skipped", "run_id": run_id, "message": msg}

        if not config.BASE_MODEL_PATH.exists():
            msg = "No base model to retrain from. Train the initial model first."
            db.finish_run(run_id, status="failed", message=msg)
            return {"status": "failed", "run_id": run_id, "message": msg}

        class_names = load_label_names()
        x_test, y_test = load_cifar_split("test")
        x_test = scale(x_test)

        # --- Step 1: the model currently in production becomes the pre-trained base ---
        model = model_lib.load_model(config.BASE_MODEL_PATH)
        before = model_lib.evaluate_model(model, x_test, y_test)

        # --- Step 2: assemble the retraining set = new uploads + replay of old data ---
        x_replay, y_replay = _replay_sample(config.REPLAY_SAMPLES)
        x_all = np.concatenate([scale(new_images), scale(x_replay)])
        y_all = np.concatenate([new_labels, y_replay])

        rng = np.random.default_rng(config.SEED)
        perm = rng.permutation(len(x_all))
        x_all, y_all = x_all[perm], y_all[perm]

        n_val = max(1, int(0.1 * len(x_all)))
        x_val, y_val = x_all[:n_val], y_all[:n_val]
        x_fit, y_fit = x_all[n_val:], y_all[n_val:]

        # --- Step 3: continue training at a low learning rate -----------------------
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.RETRAIN_LR),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
            ],
        )
        model_lib.train_model(
            model,
            x_fit,
            y_fit,
            x_val,
            y_val,
            epochs=epochs,
            callbacks=model_lib.default_callbacks(patience=3),
        )

        # --- Step 4: evaluate and decide whether to promote -------------------------
        after = model_lib.evaluate_model(model, x_test, y_test)
        version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
        version_path = config.MODEL_VERSIONS_DIR / f"cifar_cnn_{version}.keras"
        model_lib.save_model(model, version_path)

        promoted = after["accuracy"] >= before["accuracy"] - config.PROMOTION_TOLERANCE
        if promoted:
            shutil.copy(version_path, config.BASE_MODEL_PATH)
            db.mark_consumed(upload_ids, run_id)
            from src import prediction

            prediction.reload_model()
            message = (
                f"Promoted. Test accuracy {before['accuracy']:.4f} -> {after['accuracy']:.4f}."
            )
        else:
            message = (
                f"Not promoted: test accuracy dropped {before['accuracy']:.4f} -> "
                f"{after['accuracy']:.4f}, beyond the {config.PROMOTION_TOLERANCE:.2%} tolerance. "
                f"The previous model is still serving. Version saved at {version_path.name}."
            )

        db.finish_run(
            run_id,
            status="completed",
            n_new_samples=len(new_images),
            n_replay_samples=len(x_replay),
            epochs=epochs,
            metrics_before={k: v for k, v in before.items() if k not in ("report", "confusion_matrix")},
            metrics_after={k: v for k, v in after.items() if k not in ("report", "confusion_matrix")},
            promoted=int(promoted),
            model_version=version,
            message=message,
        )
        return {
            "status": "completed",
            "run_id": run_id,
            "promoted": promoted,
            "metrics_before": before,
            "metrics_after": after,
            "message": message,
        }

    except Exception as exc:  # noqa: BLE001 - the failure must reach the UI, not the void
        db.finish_run(run_id, status="failed", message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _retrain_lock.release()


def run_retraining_async(trigger: str = "manual", epochs: int | None = None) -> int | None:
    """Kick the job off in a daemon thread and return immediately with the run id."""
    if is_running():
        return None
    run_id_holder: dict[str, int] = {}

    def _target():
        run_retraining(trigger=trigger, epochs=epochs)

    thread = threading.Thread(target=_target, daemon=True, name=f"retrain-{trigger}")
    thread.start()
    return run_id_holder.get("id", -1)
