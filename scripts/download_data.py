"""Fetch CIFAR-100 into data/.

    python -m scripts.download_data

The raw pickles are 186 MB and `train` alone is 155 MB, which is over GitHub's 100 MB
per-file hard limit. So they are not committed. This script pulls them from the original
source at Toronto and lays them out where src/config.py expects them:

    data/train/train
    data/test/test
    data/meta

It runs during the Docker build, so containers get the data without it ever touching git.
"""
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def main():
    targets = {
        "train": config.TRAIN_FILE,
        "test": config.TEST_FILE,
        "meta": config.META_FILE,
    }
    if all(p.exists() for p in targets.values()):
        print("CIFAR-100 already present, nothing to do.")
        return

    for p in targets.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "cifar-100-python.tar.gz"
        print(f"Downloading {URL} (169 MB)...")
        urllib.request.urlretrieve(URL, archive)

        print("Extracting...")
        with tarfile.open(archive) as tf:
            tf.extractall(tmp)

        extracted = Path(tmp) / "cifar-100-python"
        for name, dest in targets.items():
            shutil.copy(extracted / name, dest)
            print(f"  {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    main()
