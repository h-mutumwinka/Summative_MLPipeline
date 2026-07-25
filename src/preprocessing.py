"""Data acquisition + preprocessing for CIFAR-100.

Handles two sources of data:
  1. The original CIFAR-100 python pickles (data/train/train, data/test/test, data/meta)
  2. Images a user uploads through the UI (single images or a .zip of class folders)

Both end up in the same shape: float32 arrays of (N, 32, 32, 3) scaled to [0, 1],
with integer labels, ready for the model.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from typing import Iterable

import numpy as np
from PIL import Image

from src import config


# ---------------------------------------------------------------------------
# 1. Loading the original CIFAR-100 pickles
# ---------------------------------------------------------------------------
def _unpickle(path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_label_names(mode: str | None = None) -> list[str]:
    """Human-readable class names, index-aligned with the integer labels."""
    mode = mode or config.LABEL_MODE
    meta = _unpickle(config.META_FILE)
    key = "coarse_label_names" if mode == "coarse" else "fine_label_names"
    return list(meta[key])


def _reshape_flat(flat: np.ndarray) -> np.ndarray:
    """CIFAR stores each image as 3072 bytes: 1024 R, then 1024 G, then 1024 B."""
    return flat.reshape(-1, 3, config.IMG_SIZE, config.IMG_SIZE).transpose(0, 2, 3, 1)


def load_cifar_split(split: str, mode: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (images_uint8, labels_int) for split in {"train", "test"}."""
    mode = mode or config.LABEL_MODE
    path = config.TRAIN_FILE if split == "train" else config.TEST_FILE
    batch = _unpickle(path)
    images = _reshape_flat(np.asarray(batch["data"], dtype=np.uint8))
    key = "coarse_labels" if mode == "coarse" else "fine_labels"
    labels = np.asarray(batch[key], dtype=np.int64)
    return images, labels


def load_dataset(mode: str | None = None):
    """Full pipeline: load, scale to [0,1], carve a validation split off the training set.

    Returns (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names.
    """
    mode = mode or config.LABEL_MODE
    x_train_full, y_train_full = load_cifar_split("train", mode)
    x_test, y_test = load_cifar_split("test", mode)
    class_names = load_label_names(mode)

    # Shuffle before splitting: the pickle is not guaranteed to be class-shuffled.
    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(x_train_full))
    x_train_full, y_train_full = x_train_full[idx], y_train_full[idx]

    n_val = int(len(x_train_full) * config.VAL_SPLIT)
    x_val, y_val = x_train_full[:n_val], y_train_full[:n_val]
    x_train, y_train = x_train_full[n_val:], y_train_full[n_val:]

    return (
        (scale(x_train), y_train),
        (scale(x_val), y_val),
        (scale(x_test), y_test),
        class_names,
    )


def scale(images: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]. The single normalisation rule used everywhere."""
    return images.astype("float32") / 255.0


# ---------------------------------------------------------------------------
# 2. Preprocessing images that a user uploads
# ---------------------------------------------------------------------------
def decode_image(raw: bytes) -> np.ndarray:
    """Bytes of any common image format -> (32, 32, 3) uint8.

    Converts to RGB (drops alpha / grayscale) and resizes to the network's input size.
    """
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (config.IMG_SIZE, config.IMG_SIZE):
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def preprocess_for_prediction(raw: bytes) -> np.ndarray:
    """Bytes -> (1, 32, 32, 3) float32 batch of one, ready for model.predict()."""
    return scale(decode_image(raw))[np.newaxis, ...]


def read_zip_of_class_folders(raw: bytes, class_names: Iterable[str]):
    """Parse an uploaded .zip laid out as class_name/image.png.

    Returns (records, skipped) where records is a list of
    (class_index, class_name, filename, image_uint8) and skipped is a list of
    (path, reason) for anything that could not be used.
    """
    lookup = {name.lower(): i for i, name in enumerate(class_names)}
    records, skipped = [], []

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = [p for p in info.filename.split("/") if p and not p.startswith("__MACOSX")]
            if len(parts) < 2:
                skipped.append((info.filename, "not inside a class folder"))
                continue
            label_name = parts[-2].strip().lower()
            if label_name not in lookup:
                skipped.append((info.filename, f"unknown class '{parts[-2]}'"))
                continue
            try:
                img = decode_image(zf.read(info))
            except Exception as exc:  # noqa: BLE001 - report, do not crash the upload
                skipped.append((info.filename, f"unreadable image ({exc})"))
                continue
            records.append((lookup[label_name], class_names[lookup[label_name]], parts[-1], img))

    return records, skipped


# ---------------------------------------------------------------------------
# 3. Augmentation (used during training only)
# ---------------------------------------------------------------------------
def build_augmentation():
    """Random flip / translate / rotate / zoom, applied on-GPU as part of the model.

    Augmentation is a regularisation technique: it enlarges the effective training set
    and is the single biggest accuracy win on CIFAR at this resolution.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=config.SEED),
            layers.RandomTranslation(0.1, 0.1, seed=config.SEED),
            layers.RandomRotation(0.06, seed=config.SEED),
            layers.RandomZoom(0.1, seed=config.SEED),
        ],
        name="augmentation",
    )
