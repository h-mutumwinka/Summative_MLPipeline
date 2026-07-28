"""
prediction.py

Loads the trained CNN model and predicts
the class of an uploaded image.
"""

import numpy as np
from tensorflow.keras.models import load_model

from src.preprocessing import (
    preprocess_uploaded_image,
    CLASS_NAMES,
)

# Load the trained model only once
model = load_model("models/cifar10_model.keras")


def predict_image(image_path):
    """
    Predict the class of one uploaded image.
    """

    # Preprocess image
    img = preprocess_uploaded_image(image_path)

    # Make prediction
    predictions = model.predict(img, verbose=0)

    # Predicted class index
    class_index = np.argmax(predictions)

    # Confidence score
    confidence = float(np.max(predictions))

    return {
        "class": CLASS_NAMES[class_index],
        "confidence": confidence
    }