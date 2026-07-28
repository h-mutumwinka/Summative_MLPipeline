"""
retrain.py

Retrains the existing CIFAR-10 model using newly uploaded images.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from src.database import save_uploaded_dataset

# Model path
MODEL_PATH = "models/cifar10_model.keras"

# Folder where uploaded images will be stored
UPLOAD_FOLDER = "uploads"


def retrain_model(dataset_folder, epochs=5, batch_size=32):
    """
    Retrain the saved model using an uploaded dataset.

    Parameters
    ----------
    dataset_folder : str
        Path to the uploaded training dataset.
    """

    dataset_folder = Path(dataset_folder)

    if not dataset_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_folder}"
        )

    # Save upload information in database
    save_uploaded_dataset(dataset_folder.name)

    # Load uploaded dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_folder,
        image_size=(32, 32),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True
    )

    # Normalize images
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    train_dataset = train_dataset.map(
        lambda x, y: (normalization_layer(x), y)
    )

    # Load existing trained model
    model = load_model(MODEL_PATH)

    # Early stopping
    early_stop = EarlyStopping(
        monitor="loss",
        patience=2,
        restore_best_weights=True
    )

    # Continue training
    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )

    # Save updated model
    model.save(MODEL_PATH)

    return history