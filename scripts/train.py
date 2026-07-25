"""Train the initial model from the command line.

    python -m scripts.train
    LABEL_MODE=fine python -m scripts.train

Writes models/cifar_cnn.keras, which is what the API serves and what retraining
later loads as its pre-trained starting point.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, model as model_lib
from src.preprocessing import load_dataset


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = load_dataset()
    print(f"train={x_train.shape} val={x_val.shape} test={x_test.shape} classes={len(class_names)}")

    model = model_lib.build_model()
    model.summary()

    model_lib.train_model(model, x_train, y_train, x_val, y_val)

    metrics = model_lib.evaluate_model(model, x_test, y_test, class_names)
    print("\nTest metrics:", model_lib.summarise(metrics))

    path = model_lib.save_model(model)
    print(f"Saved model to {path}")

    out = config.MODELS_DIR / "metrics.json"
    out.write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2)
    )
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
