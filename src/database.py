"""SQLite persistence.

Two tables:
  uploads      - every image a user uploads for retraining, stored as a PNG blob
                 alongside its class label. `consumed_by_run` records which retrain
                 run has already learned from it.
  retrain_runs - one row per retraining job: what triggered it, how many new samples
                 it saw, the metrics before and after, and whether it was promoted.

Images are stored in the database itself (32x32 PNGs are ~1-2 KB each), so the
whole persistent state of the system is one file. Point DB_PATH at a mounted
persistent disk and nothing is lost on restart.
"""
from __future__ import annotations

import io
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL,
    class_index     INTEGER NOT NULL,
    class_name      TEXT    NOT NULL,
    image_png       BLOB    NOT NULL,
    label_mode      TEXT    NOT NULL,
    uploaded_at     TEXT    NOT NULL,
    consumed_by_run INTEGER
);
CREATE INDEX IF NOT EXISTS idx_uploads_unconsumed ON uploads(consumed_by_run);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    status          TEXT    NOT NULL,
    trigger         TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    n_new_samples   INTEGER DEFAULT 0,
    n_replay_samples INTEGER DEFAULT 0,
    epochs          INTEGER,
    metrics_before  TEXT,
    metrics_after   TEXT,
    promoted        INTEGER DEFAULT 0,
    model_version   TEXT,
    message         TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
def _to_png(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def _from_png(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)


def save_uploads(records) -> int:
    """records: iterable of (class_index, class_name, filename, image_uint8). Returns count."""
    rows = [
        (fname, int(idx), name, _to_png(img), config.LABEL_MODE, _now())
        for idx, name, fname, img in records
    ]
    if not rows:
        return 0
    with connect() as conn:
        conn.executemany(
            "INSERT INTO uploads (filename, class_index, class_name, image_png, label_mode, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def fetch_unconsumed():
    """Return (images_uint8, labels, ids) for uploads not yet used by a retrain run."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, class_index, image_png FROM uploads "
            "WHERE consumed_by_run IS NULL AND label_mode = ?",
            (config.LABEL_MODE,),
        ).fetchall()
    if not rows:
        return np.empty((0, config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8), np.empty(0, dtype=np.int64), []
    images = np.stack([_from_png(r["image_png"]) for r in rows])
    labels = np.asarray([r["class_index"] for r in rows], dtype=np.int64)
    ids = [r["id"] for r in rows]
    return images, labels, ids


def mark_consumed(ids, run_id: int) -> None:
    if not ids:
        return
    with connect() as conn:
        conn.executemany(
            "UPDATE uploads SET consumed_by_run = ? WHERE id = ?", [(run_id, i) for i in ids]
        )


def upload_stats() -> dict:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM uploads").fetchone()["c"]
        pending = conn.execute(
            "SELECT COUNT(*) c FROM uploads WHERE consumed_by_run IS NULL"
        ).fetchone()["c"]
        per_class = conn.execute(
            "SELECT class_name, COUNT(*) c FROM uploads GROUP BY class_name ORDER BY c DESC"
        ).fetchall()
        recent = conn.execute(
            "SELECT filename, class_name, uploaded_at, consumed_by_run FROM uploads "
            "ORDER BY id DESC LIMIT 10"
        ).fetchall()
    return {
        "total_uploads": total,
        "pending_uploads": pending,
        "per_class": {r["class_name"]: r["c"] for r in per_class},
        "recent": [dict(r) for r in recent],
    }


# ---------------------------------------------------------------------------
# Retrain runs
# ---------------------------------------------------------------------------
def start_run(trigger: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO retrain_runs (status, trigger, started_at) VALUES ('running', ?, ?)",
            (trigger, _now()),
        )
        return int(cur.lastrowid)


def finish_run(run_id: int, **fields) -> None:
    for key in ("metrics_before", "metrics_after"):
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key])
    fields.setdefault("finished_at", _now())
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with connect() as conn:
        conn.execute(
            f"UPDATE retrain_runs SET {assignments} WHERE id = ?",
            (*fields.values(), run_id),
        )


def get_run(run_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM retrain_runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def active_run() -> dict | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM retrain_runs WHERE status = 'running' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None
