"""FastAPI service.

Endpoints
  GET  /health          liveness + uptime + which model version is loaded
  GET  /classes         the label vocabulary the model was trained on
  POST /predict         one image in, one prediction (plus top-5) out
  POST /upload          bulk training images: a .zip of class folders, or loose files + a label
  POST /retrain         trigger retraining on everything uploaded since the last run
  GET  /retrain/status  progress + history of retrain runs
  GET  /stats           upload counts per class, for the dashboard
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src import config, database as db, prediction, retrain
from src.preprocessing import decode_image, load_label_names, read_zip_of_class_folders

REQUEST_COUNT = {"predict": 0, "errors": 0}
LATENCIES: list[float] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    try:
        prediction.get_model()  # warm the model so the first request is not slow
    except FileNotFoundError:
        pass  # the API still starts; /health reports model_available=false
    yield


app = FastAPI(
    title="CIFAR-100 MLOps API",
    description="Prediction, bulk upload and retraining for a CIFAR-100 image classifier.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class RetrainRequest(BaseModel):
    trigger: str = "manual"
    epochs: int | None = None


@app.get("/health")
def health():
    info = prediction.model_info()
    info.update(
        {
            "status": "ok",
            "retraining_in_progress": retrain.is_running(),
            "predictions_served": REQUEST_COUNT["predict"],
            "errors": REQUEST_COUNT["errors"],
            "avg_latency_ms": round(sum(LATENCIES) / len(LATENCIES), 2) if LATENCIES else None,
        }
    )
    return info


@app.get("/classes")
def classes():
    return {"label_mode": config.LABEL_MODE, "classes": load_label_names()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    raw = await file.read()
    if not raw:
        REQUEST_COUNT["errors"] += 1
        raise HTTPException(400, "Empty file. Attach an image.")
    try:
        result = prediction.predict(raw)
    except FileNotFoundError as exc:
        REQUEST_COUNT["errors"] += 1
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        REQUEST_COUNT["errors"] += 1
        raise HTTPException(400, f"Could not read that image: {exc}") from exc

    REQUEST_COUNT["predict"] += 1
    elapsed = (time.perf_counter() - started) * 1000
    LATENCIES.append(elapsed)
    del LATENCIES[:-1000]  # keep the last 1000 only
    result["request_ms"] = round(elapsed, 2)
    result["filename"] = file.filename
    return result


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), class_name: str | None = Form(None)):
    """Two accepted shapes:

    - one .zip laid out as `class_name/image.png` -> labels are read from the folder names
    - one or more loose image files + a `class_name` form field -> all get that label
    """
    class_names = load_label_names()
    records, skipped = [], []

    for f in files:
        raw = await f.read()
        if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
            skipped.append({"file": f.filename, "reason": f"larger than {config.MAX_UPLOAD_MB} MB"})
            continue

        if f.filename and f.filename.lower().endswith(".zip"):
            recs, skips = read_zip_of_class_folders(raw, class_names)
            records.extend(recs)
            skipped.extend({"file": p, "reason": r} for p, r in skips)
        else:
            if not class_name:
                skipped.append(
                    {"file": f.filename, "reason": "no class_name given for a loose image"}
                )
                continue
            lookup = {n.lower(): i for i, n in enumerate(class_names)}
            key = class_name.strip().lower()
            if key not in lookup:
                skipped.append({"file": f.filename, "reason": f"unknown class '{class_name}'"})
                continue
            try:
                img = decode_image(raw)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"file": f.filename, "reason": f"unreadable image ({exc})"})
                continue
            records.append((lookup[key], class_names[lookup[key]], f.filename, img))

    saved = db.save_uploads(records)
    stats = db.upload_stats()

    return {
        "saved": saved,
        "skipped": skipped,
        "pending_for_retraining": stats["pending_uploads"],
        "ready_to_retrain": stats["pending_uploads"] >= config.RETRAIN_MIN_NEW_SAMPLES,
        "minimum_required": config.RETRAIN_MIN_NEW_SAMPLES,
    }


@app.post("/retrain")
def trigger_retrain(req: RetrainRequest, background: BackgroundTasks):
    if retrain.is_running():
        raise HTTPException(409, "A retraining run is already in progress.")

    pending = db.upload_stats()["pending_uploads"]
    if pending < config.RETRAIN_MIN_NEW_SAMPLES:
        raise HTTPException(
            400,
            f"Only {pending} new images are pending. Upload at least "
            f"{config.RETRAIN_MIN_NEW_SAMPLES} before retraining.",
        )

    background.add_task(retrain.run_retraining, req.trigger, req.epochs)
    return {
        "status": "started",
        "trigger": req.trigger,
        "new_samples": pending,
        "message": "Retraining started. Poll /retrain/status for progress.",
    }


@app.get("/retrain/status")
def retrain_status(limit: int = 10):
    return {
        "in_progress": retrain.is_running(),
        "active_run": db.active_run(),
        "history": db.list_runs(limit),
    }


@app.get("/stats")
def stats():
    return db.upload_stats()
