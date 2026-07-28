"""
preprocessing.py

This module handles loading and preprocessing data for
training, prediction, and retraining.
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image

# CIFAR-10 class names
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
    "truck",
]


def load_dataset():
    """
    Load the CIFAR-10 dataset.
    """
    return cifar10.load_data()


def normalize_images(x_train, x_test):
    """
    Normalize images from pixel range [0,255] to [0,1].
    """
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, x_test


def preprocess_uploaded_image(image_path):
    """
    Preprocess an uploaded image before prediction.
    """

    img = image.load_img(image_path, target_size=(32, 32))

    img = image.img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img
















































































































































































































































































































