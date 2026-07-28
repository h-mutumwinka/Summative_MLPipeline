"""
api.py

API functions for prediction and retraining.
"""

from pathlib import Path
import shutil
import zipfile

from src.prediction import predict_image
from src.retrain import retrain_model

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


def predict(uploaded_image_path):
    """
    Predict the class of an uploaded image.
    """

    result = predict_image(uploaded_image_path)

    return result


def retrain(uploaded_zip_path):
    """
    Retrain the model using an uploaded ZIP dataset.
    """

    uploaded_zip_path = Path(uploaded_zip_path)

    if not uploaded_zip_path.exists():
        raise FileNotFoundError("Uploaded ZIP file not found.")

    # Folder where dataset will be extracted
    extract_folder = UPLOAD_FOLDER / uploaded_zip_path.stem

    # Remove old extracted folder if it exists
    if extract_folder.exists():
        shutil.rmtree(extract_folder)

    extract_folder.mkdir(parents=True, exist_ok=True)

    # Extract ZIP
    with zipfile.ZipFile(uploaded_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    # Retrain model
    history = retrain_model(extract_folder)

    return {
        "message": "Retraining completed successfully.",
        "dataset": uploaded_zip_path.name
    }