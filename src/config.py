"""
config.py

Project configuration.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cifar10_model.keras"

DATABASE_PATH = BASE_DIR / "database" / "mlops.db"

UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(exist_ok=True)

IMAGE_SIZE = (32, 32)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]