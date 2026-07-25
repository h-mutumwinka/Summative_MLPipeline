"""Central configuration. Everything else imports from here."""
import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# On Render, mount a Persistent Disk at /data and set DATA_DIR=/data so that
# uploaded images, the SQLite database and retrained models survive restarts.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = Path(os.getenv("RAW_DIR", PROJECT_ROOT / "data"))
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))

TRAIN_FILE = RAW_DIR / "train" / "train"
TEST_FILE = RAW_DIR / "test" / "test"
META_FILE = RAW_DIR / "meta"

DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "mlops.db"))
BASE_MODEL_PATH = MODELS_DIR / "cifar_cnn.keras"      # the model served + used as the pre-trained base
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"           # every retrain writes a timestamped copy here

for _d in (DATA_DIR, MODELS_DIR, MODEL_VERSIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Data --------------------------------------------------------------------
# "coarse" -> 20 superclasses (recommended: higher accuracy, reliable live demo)
# "fine"   -> 100 classes (harder; expect ~45-55% top-1 on 32x32)
LABEL_MODE = os.getenv("LABEL_MODE", "coarse")
NUM_CLASSES = 20 if LABEL_MODE == "coarse" else 100

IMG_SIZE = 32
CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)

VAL_SPLIT = 0.1
SEED = 42

# --- Training ----------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 60
LEARNING_RATE = 1e-3
L2_REG = 1e-4
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1

# --- Retraining --------------------------------------------------------------
# Retraining loads BASE_MODEL_PATH (our own custom model, used as a pre-trained
# model) and continues training on old data + newly uploaded data at a lower LR.
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", 8))
RETRAIN_LR = float(os.getenv("RETRAIN_LR", 2e-4))
RETRAIN_MIN_NEW_SAMPLES = int(os.getenv("RETRAIN_MIN_NEW_SAMPLES", 10))
# How many original training images to mix in, so the model does not catastrophically
# forget the other classes when it sees a small batch of new uploads.
REPLAY_SAMPLES = int(os.getenv("REPLAY_SAMPLES", 10000))
# Only promote the retrained model to production if it does not lose more than this
# much test accuracy against the currently served model.
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", 0.01))

# --- API ---------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_UPLOAD_MB = 25
