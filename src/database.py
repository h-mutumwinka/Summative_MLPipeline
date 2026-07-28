"""
database.py

Creates and manages the SQLite database used by the CIFAR-10 MLOps project.

Tables
------
predictions   – every inference the model makes
uploads       – every zip dataset a user uploads
retrain_runs  – before/after metrics for every retraining job
"""

import sqlite3
from pathlib import Path

DATABASE_FOLDER = Path("database")
DATABASE_FOLDER.mkdir(exist_ok=True)
DATABASE_PATH = DATABASE_FOLDER / "mlops.db"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection():
    """Return a new SQLite connection."""
    return sqlite3.connect(DATABASE_PATH)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def initialize_database():
    """Create all required tables (idempotent)."""
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name       TEXT,
            predicted_class  TEXT,
            confidence       REAL,
            prediction_date  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT NOT NULL,
            upload_date   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status        TEXT,
            model_version TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS retrain_runs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            dataset          TEXT,
            accuracy_before  REAL,
            accuracy_after   REAL,
            promoted         INTEGER,
            duration_seconds REAL,
            status           TEXT
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def save_prediction(image_name: str, predicted_class: str, confidence: float):
    """Insert one prediction record."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO predictions (image_name, predicted_class, confidence) VALUES (?, ?, ?)",
        (image_name, predicted_class, confidence),
    )
    conn.commit()
    conn.close()


def get_all_predictions():
    """Return all prediction rows ordered by most recent first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY prediction_date DESC"
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

def save_uploaded_dataset(filename: str, status: str = "Uploaded", model_version: str = "v1"):
    """Insert one upload record."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO uploads (filename, status, model_version) VALUES (?, ?, ?)",
        (filename, status, model_version),
    )
    conn.commit()
    conn.close()


def get_all_uploads():
    """Return all upload rows ordered by most recent first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM uploads ORDER BY upload_date DESC"
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Retrain runs
# ---------------------------------------------------------------------------

def save_retrain_run(
    dataset: str,
    accuracy_before: float,
    accuracy_after: float,
    promoted: int,
    duration_seconds: float,
    status: str,
):
    """Insert one retraining run record with before/after metrics."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO retrain_runs
           (dataset, accuracy_before, accuracy_after, promoted, duration_seconds, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (dataset, accuracy_before, accuracy_after, promoted, duration_seconds, status),
    )
    conn.commit()
    conn.close()


def get_all_retrain_runs():
    """Return all retrain run rows ordered by most recent first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM retrain_runs ORDER BY run_date DESC"
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    initialize_database()
    print("Database initialised successfully.")