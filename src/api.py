"""
api.py

FastAPI application — the central traffic controller for the
CIFAR-10 MLOps pipeline.

Endpoints
---------
GET  /health           uptime, model version, request count, avg latency
GET  /classes          list of all 10 class labels
POST /predict          upload one image → prediction + top-5
POST /upload-image     upload one or more images with a class label for retraining
POST /retrain          trigger retraining on user-uploaded images
GET  /retrain/status   whether retraining is running + run history
GET  /stats            total predictions, uploads, and per-class image counts
"""

import shutil
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.prediction import predict_image
from src.retrain import retrain_model
from src.database import (
    initialize_database,
    save_prediction,
    save_uploaded_dataset,
    get_all_predictions,
    get_all_uploads,
)
from src.preprocessing import CLASS_NAMES

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CIFAR-10 MLOps API",
    description=(
        "End-to-end image classification pipeline.\n\n"
        "Upload an image → get a prediction. "
        "Upload labelled images → retrain the model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

_start_time      = time.time()
_request_count   = 0
_total_latency   = 0.0
_state_lock      = threading.Lock()

_retrain_running = False
_retrain_history: list[dict] = []

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    initialize_database()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Monitoring"])
def health():
    """API uptime, total requests served, average latency, model version."""
    with _state_lock:
        count   = _request_count
        latency = _total_latency

    uptime      = time.time() - _start_time
    avg_latency = (latency / count) if count > 0 else 0.0

    return {
        "status":          "ok",
        "uptime_seconds":  round(uptime, 2),
        "model_version":   "v1.0",
        "requests_served": count,
        "avg_latency_ms":  round(avg_latency * 1000, 2),
    }


@app.get("/classes", tags=["Model"])
def get_classes():
    """Returns the list of all 10 CIFAR-10 class labels."""
    return {"classes": CLASS_NAMES, "total": len(CLASS_NAMES)}


@app.post("/predict", tags=["Model"])
async def predict(file: UploadFile = File(..., description="PNG/JPG image to classify")):
    """
    Classify one image.
    Returns the predicted class, confidence, and the full top-5 ranking.
    Every prediction is logged to the SQLite database.
    """
    global _request_count, _total_latency

    allowed = {".png", ".jpg", ".jpeg"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use PNG or JPG.",
        )

    temp_path = UPLOAD_FOLDER / f"tmp_{uuid.uuid4().hex}{suffix}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    t0 = time.time()
    try:
        result = predict_image(str(temp_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    elapsed = time.time() - t0

    with _state_lock:
        _request_count += 1
        _total_latency  += elapsed

    save_prediction(file.filename, result["class"], result["confidence"])

    return {
        "filename":        file.filename,
        "predicted_class": result["class"],
        "confidence":      result["confidence"],
        "top5":            result["top5"],
        "latency_ms":      round(elapsed * 1000, 2),
    }


@app.post("/upload-image", tags=["Data"])
async def upload_image(
    label: str         = Form(..., description="CIFAR-10 class name e.g. 'cat'"),
    file:  UploadFile  = File(..., description="PNG/JPG image"),
):
    """
    Upload a single labelled image for retraining.

    - **label**: must be one of the 10 CIFAR-10 class names
    - **file**: PNG or JPG image file

    Images are saved to `uploads/{label}/` so the retraining job can
    read them directly as a class-labelled dataset.
    """
    label = label.strip().lower()
    if label not in CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown class '{label}'. Valid classes: {CLASS_NAMES}",
        )

    allowed = {".png", ".jpg", ".jpeg"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'.")

    # Save to uploads/{label}/
    class_folder = UPLOAD_FOLDER / label
    class_folder.mkdir(exist_ok=True)

    unique_name = f"{uuid.uuid4().hex}{suffix}"
    dest = class_folder / unique_name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    save_uploaded_dataset(f"{label}/{unique_name}")

    # Count total images for this class
    class_count = sum(1 for p in class_folder.iterdir() if p.is_file())

    return {
        "message":       f"Image saved as class '{label}'.",
        "label":         label,
        "saved_as":      unique_name,
        "class_total":   class_count,
    }


# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------

def _run_retrain(dataset_path: str) -> None:
    """Background thread — fine-tunes the model and applies the promotion gate."""
    global _retrain_running, _retrain_history

    t0 = time.time()
    try:
        result  = retrain_model(dataset_path)
        elapsed = round(time.time() - t0, 2)
        _retrain_history.append({
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
            "status":           result["status"],
            "dataset":          Path(dataset_path).name,
            "accuracy_before":  result["accuracy_before"],
            "accuracy_after":   result["accuracy_after"],
            "promoted":         result["promoted"],
            "duration_seconds": elapsed,
        })
    except Exception as exc:
        _retrain_history.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "status":    "failed",
            "error":     str(exc),
        })
    finally:
        _retrain_running = False


@app.post("/retrain", tags=["Training"])
def retrain(background_tasks: BackgroundTasks):
    """
    Trigger model retraining on all user-uploaded labelled images.

    Images must have been uploaded via POST /upload-image first.
    Retraining runs in the background — poll GET /retrain/status to track progress.
    """
    global _retrain_running

    if _retrain_running:
        raise HTTPException(
            status_code=409,
            detail="Retraining is already running. Check /retrain/status.",
        )

    # Count total labelled images available
    total_images = sum(
        1
        for cls in CLASS_NAMES
        for p in (UPLOAD_FOLDER / cls).glob("*")
        if p.is_file() and (UPLOAD_FOLDER / cls).exists()
    )

    if total_images == 0:
        raise HTTPException(
            status_code=404,
            detail="No labelled images found. Upload images via POST /upload-image first.",
        )

    _retrain_running = True
    background_tasks.add_task(_run_retrain, str(UPLOAD_FOLDER))

    return {
        "message":      "Retraining started in the background.",
        "total_images": total_images,
        "dataset_path": str(UPLOAD_FOLDER),
    }


@app.get("/retrain/status", tags=["Training"])
def retrain_status():
    """Whether retraining is running, and the history of the last 10 runs."""
    return {
        "is_running": _retrain_running,
        "history":    _retrain_history[-10:],
    }


@app.get("/stats", tags=["Monitoring"])
def stats():
    """Prediction counts, upload counts, and per-class image totals."""
    predictions = get_all_predictions()
    uploads     = get_all_uploads()

    # Count images per class in the uploads folder
    class_counts = {}
    for cls in CLASS_NAMES:
        folder = UPLOAD_FOLDER / cls
        if folder.exists():
            class_counts[cls] = sum(1 for p in folder.iterdir() if p.is_file())
        else:
            class_counts[cls] = 0

    return {
        "total_predictions":  len(predictions),
        "total_uploads":      sum(class_counts.values()),
        "class_image_counts": class_counts,
        "recent_predictions": [
            {
                "id":         row[0],
                "image":      row[1],
                "prediction": row[2],
                "confidence": row[3],
                "date":       row[4],
            }
            for row in predictions[:5]
        ],
    }