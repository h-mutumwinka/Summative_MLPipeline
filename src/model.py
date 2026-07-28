"""
model.py

Builds and compiles the CNN model for CIFAR-10.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


def build_model():
    """
    Build and compile the CNN model.
    """

    model = Sequential([

        Input(shape=(32, 32, 3)),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.30),

        Flatten(),

        Dense(256, activation="relu"),
        Dropout(0.50),

        Dense(10, activation="softmax")

    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model