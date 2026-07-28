"""
database.py

Creates and manages the SQLite database used by the
CIFAR-10 MLOps project.
"""

import sqlite3
from pathlib import Path

# Database folder and file
DATABASE_FOLDER = Path("database")
DATABASE_FOLDER.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_FOLDER / "mlops.db"


def get_connection():
    """
    Create and return a database connection.
    """
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    """
    Create all required database tables.
    """

    connection = get_connection()
    cursor = connection.cursor()

    # Table for uploaded datasets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT,
            model_version TEXT
        )
    """)

    # Table for prediction history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            predicted_class TEXT,
            confidence REAL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_uploaded_dataset(filename, status="Uploaded", model_version="v1"):
    """
    Save uploaded dataset information.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO uploads
        (filename, status, model_version)
        VALUES (?, ?, ?)
    """, (filename, status, model_version))

    connection.commit()
    connection.close()


def save_prediction(image_name, predicted_class, confidence):
    """
    Save prediction history.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (image_name, predicted_class, confidence)
        VALUES (?, ?, ?)
    """, (image_name, predicted_class, confidence))

    connection.commit()
    connection.close()


def get_all_predictions():
    """
    Return all prediction records.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY prediction_date DESC
    """)

    results = cursor.fetchall()

    connection.close()

    return results


def get_all_uploads():
    """
    Return all uploaded datasets.
    """

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT *
        FROM uploads
        ORDER BY upload_date DESC
    """)

    results = cursor.fetchall()

    connection.close()

    return results


if __name__ == "__main__":
    initialize_database()
    print("Database created successfully.")