"""Streamlit front end.

Four sections, matching what the assignment asks a user to be able to do:
  Status      - is the model up, which version is serving, how many requests has it seen
  Insights    - visualisations and interpretations of the dataset
  Predict     - upload one image, get one prediction
  Retrain     - upload bulk data, press a button, watch the model retrain
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.preprocessing import load_cifar_split, load_label_names  # noqa: E402

API = config.API_URL.rstrip("/")

st.set_page_config(page_title="CIFAR-100 MLOps", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached: the pickles are read once per container, not per click)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading CIFAR-100...")
def get_data():
    x_train, y_train = load_cifar_split("train")
    x_test, y_test = load_cifar_split("test")
    return x_train, y_train, x_test, y_test, load_label_names()


def api_get(path: str, **kwargs):
    try:
        r = requests.get(f"{API}{path}", timeout=15, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("CIFAR-100 MLOps")
page = st.sidebar.radio("Go to", ["Status", "Insights", "Predict", "Retrain"], label_visibility="collapsed")

health = api_get("/health")
if "_error" in health:
    st.sidebar.error(f"API unreachable at {API}")
else:
    st.sidebar.success("API online")
    st.sidebar.caption(f"Model: {'loaded' if health.get('model_available') else 'missing'}")
    st.sidebar.caption(f"Classes: {health.get('num_classes')} ({health.get('label_mode')})")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
if page == "Status":
    st.title("Model status")

    if "_error" in health:
        st.error(f"Cannot reach the API at {API}. Start it, then refresh this page.")
    else:
        uptime = health.get("uptime_seconds", 0)
        h, rem = divmod(int(uptime), 3600)
        m, s = divmod(rem, 60)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Uptime", f"{h}h {m}m {s}s")
        c2.metric("Predictions served", health.get("predictions_served", 0))
        c3.metric("Average latency", f"{health.get('avg_latency_ms') or 0:.0f} ms")
        c4.metric("Errors", health.get("errors", 0))

        st.subheader("Serving")
        st.json(
            {
                "model_available": health.get("model_available"),
                "model_last_modified": health.get("model_last_modified"),
                "label_mode": health.get("label_mode"),
                "num_classes": health.get("num_classes"),
                "retraining_in_progress": health.get("retraining_in_progress"),
            }
        )

        st.subheader("Retraining history")
        status = api_get("/retrain/status")
        history = status.get("history", [])
        if history:
            df = pd.DataFrame(history)[
                ["id", "status", "trigger", "started_at", "n_new_samples", "promoted", "message"]
            ]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No retraining runs yet. Upload data on the Retrain page to start one.")


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------
elif page == "Insights":
    st.title("What the data looks like")
    x_train, y_train, x_test, y_test, class_names = get_data()

    st.caption(
        f"{len(x_train):,} training images and {len(x_test):,} test images, "
        f"32x32 RGB, {len(class_names)} classes ({config.LABEL_MODE} labels)."
    )

    # --- Feature 1: class balance -----------------------------------------
    st.header("1. Class balance")
    counts = pd.Series(y_train).value_counts().sort_index()
    counts.index = [class_names[i] for i in counts.index]
    st.bar_chart(counts, height=320)
    st.markdown(
        f"Every class holds **{int(counts.min()):,}** training images and the largest holds "
        f"**{int(counts.max()):,}**. "
        "> _Your interpretation: what does a perfectly balanced dataset let you assume about "
        "accuracy as a metric, and what would change if it were skewed?_"
    )

    # --- Feature 2: colour signature --------------------------------------
    st.header("2. Colour signature per class")
    rgb = np.stack(
        [x_train[y_train == i].reshape(-1, 3).mean(axis=0) for i in range(len(class_names))]
    )
    colour_df = pd.DataFrame(rgb, columns=["red", "green", "blue"], index=class_names)
    st.bar_chart(colour_df, height=380)

    greenest = colour_df["green"].sub(colour_df[["red", "blue"]].mean(axis=1)).nlargest(3)
    bluest = colour_df["blue"].sub(colour_df[["red", "green"]].mean(axis=1)).nlargest(3)
    st.markdown(
        f"Most green-dominant classes: **{', '.join(greenest.index)}**. "
        f"Most blue-dominant: **{', '.join(bluest.index)}**.\n\n"
        "> _Your interpretation: which classes could a model separate on average colour alone, "
        "and which pairs would that fool?_"
    )

    # --- Feature 3: contrast / texture ------------------------------------
    st.header("3. Contrast, as a proxy for texture")
    contrast = pd.Series(
        [x_train[y_train == i].std(axis=(1, 2, 3)).mean() for i in range(len(class_names))],
        index=class_names,
    ).sort_values()
    st.bar_chart(contrast, height=380)
    st.markdown(
        f"Flattest classes: **{', '.join(contrast.head(3).index)}** "
        f"(mean pixel std {contrast.head(3).mean():.1f}). "
        f"Busiest: **{', '.join(contrast.tail(3).index)}** "
        f"(mean pixel std {contrast.tail(3).mean():.1f}).\n\n"
        "> _Your interpretation: what does within-image pixel variance stand in for, and why "
        "would a convolutional network find the busy classes easier or harder?_"
    )

    # --- Sample grid -------------------------------------------------------
    st.header("Sample images")
    pick = st.selectbox("Class", class_names)
    idx = np.where(y_train == class_names.index(pick))[0][:12]
    cols = st.columns(12)
    for col, i in zip(cols, idx):
        col.image(x_train[i], width=60)
    st.caption(
        "These are 32x32 pixels shown enlarged. The resolution is the hard ceiling on accuracy: "
        "if you cannot tell the class from the thumbnail, the network is unlikely to either."
    )


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
elif page == "Predict":
    st.title("Classify one image")
    st.caption("Any size, any common format. It is resized to 32x32 before it reaches the model.")

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if uploaded is None:
        st.info("Pick an image to classify. Grab one from the Insights page if you need a test file.")
    else:
        left, right = st.columns([1, 2])
        left.image(uploaded, caption=uploaded.name, width=220)

        if right.button("Predict", type="primary"):
            with st.spinner("Predicting..."):
                try:
                    r = requests.post(
                        f"{API}/predict",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=60,
                    )
                    r.raise_for_status()
                    result = r.json()
                except Exception as exc:  # noqa: BLE001
                    right.error(f"Prediction failed: {exc}")
                else:
                    right.success(
                        f"**{result['predicted_class']}** "
                        f"({result['confidence']:.1%} confidence, {result['request_ms']:.0f} ms)"
                    )
                    top = pd.DataFrame(result["top_k"]).set_index("class")
                    right.bar_chart(top, height=260)


# ---------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------
elif page == "Retrain":
    st.title("Upload data and retrain")

    stats = api_get("/stats")
    pending = stats.get("pending_uploads", 0)

    c1, c2 = st.columns(2)
    c1.metric("Images uploaded in total", stats.get("total_uploads", 0))
    c2.metric("Waiting to be learned from", pending)

    st.subheader("1. Upload training images")
    st.markdown(
        "Upload a **.zip** whose top-level folders are class names, for example:\n\n"
        "```\nnew_data.zip\n  flowers/rose_01.png\n  flowers/rose_02.png\n  insects/bee_01.png\n```\n"
        "Or upload loose images and pick a single class for all of them below."
    )

    class_names = api_get("/classes").get("classes", [])
    mode = st.radio("Upload type", ["Zip of class folders", "Loose images with one label"], horizontal=True)
    label = None
    if mode == "Loose images with one label":
        label = st.selectbox("Class for these images", class_names)

    files = st.file_uploader(
        "Choose files",
        type=["zip"] if mode == "Zip of class folders" else ["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if files and st.button("Upload to database", type="primary"):
        payload = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
        data = {"class_name": label} if label else {}
        with st.spinner("Saving to the database..."):
            try:
                r = requests.post(f"{API}/upload", files=payload, data=data, timeout=300)
                r.raise_for_status()
                out = r.json()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Upload failed: {exc}")
            else:
                st.success(f"Saved {out['saved']} images. {out['pending_for_retraining']} now pending.")
                if out["skipped"]:
                    with st.expander(f"{len(out['skipped'])} files were skipped"):
                        st.dataframe(pd.DataFrame(out["skipped"]), hide_index=True)
                st.rerun()

    st.subheader("2. Trigger retraining")
    status = api_get("/retrain/status")
    running = status.get("in_progress", False)

    if running:
        st.warning("A retraining run is in progress.")
        if st.button("Refresh"):
            st.rerun()
    else:
        epochs = st.slider("Epochs", 1, 20, config.RETRAIN_EPOCHS)
        disabled = pending < config.RETRAIN_MIN_NEW_SAMPLES
        if disabled:
            st.info(
                f"Upload at least {config.RETRAIN_MIN_NEW_SAMPLES} images before retraining. "
                f"{pending} are pending."
            )
        if st.button("Retrain model", type="primary", disabled=disabled):
            try:
                r = requests.post(
                    f"{API}/retrain", json={"trigger": "ui_button", "epochs": epochs}, timeout=30
                )
                r.raise_for_status()
                st.success(r.json()["message"])
                time.sleep(1)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not start retraining: {exc}")

    st.subheader("3. Results")
    history = status.get("history", [])
    if not history:
        st.info("No runs yet.")
    else:
        for run in history[:5]:
            icon = {"completed": "✅", "running": "⏳", "failed": "❌", "skipped": "⏭️"}.get(run["status"], "•")
            with st.expander(f"{icon} Run {run['id']} — {run['status']} — {run['started_at'][:19]}"):
                st.write(run.get("message") or "")
                before, after = run.get("metrics_before"), run.get("metrics_after")
                if before and after:
                    import json

                    b, a = json.loads(before), json.loads(after)
                    rows = []
                    for k in ["accuracy", "top5_accuracy", "loss", "f1_macro", "precision_macro", "recall_macro"]:
                        if k in b and k in a:
                            rows.append({"metric": k, "before": round(b[k], 4), "after": round(a[k], 4),
                                         "change": round(a[k] - b[k], 4)})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
