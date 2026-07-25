"""Single-datapoint prediction.

The served model is held as a module-level singleton so it is loaded once per
process, not once per request. `reload_model()` is called after a retrain so the
API picks up the new weights without a restart.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import numpy as np

from src import config, model as model_lib
from src.preprocessing import load_label_names, preprocess_for_prediction

_model = None
_model_loaded_at: datetime | None = None
_class_names: list[str] | None = None
_lock = threading.Lock()

PROCESS_STARTED_AT = datetime.now(timezone.utc)


def get_model():
    global _model, _model_loaded_at, _class_names
    with _lock:
        if _model is None:
            if not config.BASE_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"No model at {config.BASE_MODEL_PATH}. Train one first: "
                    "run the notebook, or `python -m scripts.train`."
                )
            _model = model_lib.load_model(config.BASE_MODEL_PATH)
            _model_loaded_at = datetime.now(timezone.utc)
            _class_names = load_label_names()
        return _model


def reload_model():
    """Drop the cached model so the next request loads the freshly retrained one."""
    global _model
    with _lock:
        _model = None
    get_model()


def get_class_names() -> list[str]:
    get_model()
    return _class_names or []


def model_info() -> dict:
    loaded = config.BASE_MODEL_PATH.exists()
    return {
        "model_path": str(config.BASE_MODEL_PATH),
        "model_available": loaded,
        "model_last_modified": (
            datetime.fromtimestamp(config.BASE_MODEL_PATH.stat().st_mtime, tz=timezone.utc).isoformat()
            if loaded
            else None
        ),
        "model_loaded_at": _model_loaded_at.isoformat() if _model_loaded_at else None,
        "label_mode": config.LABEL_MODE,
        "num_classes": config.NUM_CLASSES,
        "process_started_at": PROCESS_STARTED_AT.isoformat(),
        "uptime_seconds": (datetime.now(timezone.utc) - PROCESS_STARTED_AT).total_seconds(),
    }


def predict(raw: bytes, top_k: int = 5) -> dict:
    """Classify one uploaded image. Returns the winning class plus the top-k ranking."""
    started = time.perf_counter()
    mdl = get_model()
    names = get_class_names()

    batch = preprocess_for_prediction(raw)
    probs = mdl.predict(batch, verbose=0)[0]

    order = np.argsort(probs)[::-1][:top_k]
    return {
        "predicted_class": names[int(order[0])],
        "predicted_index": int(order[0]),
        "confidence": float(probs[order[0]]),
        "top_k": [
            {"class": names[int(i)], "probability": float(probs[int(i)])} for i in order
        ],
        "inference_ms": round((time.perf_counter() - started) * 1000, 2),
    }
