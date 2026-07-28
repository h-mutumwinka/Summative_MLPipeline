"""
prediction.py

Loads the trained CNN model once at startup and exposes
predict_image() for the API to call.
"""

import numpy as np
from tensorflow.keras.models import load_model

from src.preprocessing import preprocess_uploaded_image, CLASS_NAMES

# Load model once — shared across all requests
model = load_model("models/cifar10_model.keras")


def predict_image(image_path: str) -> dict:
    """
    Predict the class of one image and return the top-5 ranking.

    Parameters
    ----------
    image_path : str
        Path to the image file on disk.

    Returns
    -------
    dict with keys:
        class       – predicted class name (str)
        confidence  – confidence of top prediction (float 0-1)
        top5        – list of {class, confidence} dicts, best first
    """
    img = preprocess_uploaded_image(image_path)

    # Raw softmax probabilities — shape (1, 10)
    probs = model.predict(img, verbose=0)[0]

    # Top-5 indices (descending confidence)
    top5_indices = np.argsort(probs)[::-1][:5]

    top5 = [
        {
            "rank":       int(rank + 1),
            "class":      CLASS_NAMES[idx],
            "confidence": round(float(probs[idx]), 4),
        }
        for rank, idx in enumerate(top5_indices)
    ]

    return {
        "class":      top5[0]["class"],
        "confidence": top5[0]["confidence"],
        "top5":       top5,
    }