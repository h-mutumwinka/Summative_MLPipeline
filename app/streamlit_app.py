"""
streamlit_app.py

CIFAR-10 MLOps Dashboard — talks to the FastAPI backend via HTTP.
"""

import os
import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image

# Use dark background for all matplotlib charts
plt.rcParams.update({
    'figure.facecolor':  '#161b27',
    'axes.facecolor':    '#161b27',
    'axes.edgecolor':    '#30363d',
    'axes.labelcolor':   '#8b949e',
    'xtick.color':       '#8b949e',
    'ytick.color':       '#8b949e',
    'text.color':        '#c9d1d9',
    'grid.color':        '#21262d',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'axes.titlecolor':   '#e6edf3',
    'axes.titleweight':  'bold',
    'axes.titlesize':    13,
    'figure.titlesize':  14,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CLASS_COLORS = [
    "#4C9BE8", "#E8834C", "#4CE8A0", "#E84C4C", "#A04CE8",
    "#E8D44C", "#4CE8E8", "#E84CA0", "#4C6BE8", "#E8A04C",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CIFAR-10 MLOps Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background — deep dark ── */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] hr { border-color: #21262d !important; }

/* ── Headings ── */
h1 { color: #e6edf3 !important; font-weight: 700; letter-spacing: -0.5px; }
h2, h3 { color: #c9d1d9 !important; font-weight: 600; }
p, li, label { color: #8b949e !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #161b27;
    border: 1px solid #21262d;
    border-top: 3px solid #4f8ef7;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(79,142,247,0.15);
}
div[data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #7c6af7);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.3px;
    transition: all 0.2s;
    box-shadow: 0 4px 14px rgba(79,142,247,0.3);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6ba3f9, #9484fa);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(79,142,247,0.45);
    color: white !important;
}

/* ── Inputs & uploaders ── */
.stFileUploader, [data-testid="stFileUploadDropzone"] {
    background: #161b27 !important;
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
    color: #8b949e !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #4f8ef7 !important;
}

/* ── DataFrames / tables ── */
.stDataFrame, [data-testid="stDataFrame"] {
    background: #161b27 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}

/* ── Info / success / warning / error alerts ── */
.stAlert {
    background: #161b27 !important;
    border-radius: 10px !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4f8ef7 !important; }

/* ── Radio buttons ── */
.stRadio > div { gap: 6px; }
.stRadio label {
    background: #1c2333;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 8px 14px;
    transition: all 0.15s;
}
.stRadio label:hover { border-color: #4f8ef7; }

/* ── Divider ── */
hr { border-color: #21262d !important; }

/* ── Custom card ── */
.dark-card {
    background: #161b27;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}

/* ── Prediction result ── */
.prediction-result {
    background: linear-gradient(135deg, #1c3557, #1a2a4a);
    border: 1px solid #4f8ef7;
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(79,142,247,0.2);
}

/* ── Status badges ── */
.status-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-green  { background: rgba(35,134,54,0.2);  color: #3fb950; border: 1px solid #238636; }
.badge-red    { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f85149; }
.badge-yellow { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid #9e6a03; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4f8ef7; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🚀 CIFAR-10 MLOps")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Predict", "📊 Insights", "🔄 Retrain", "📋 History"],
)

st.sidebar.markdown("---")

# Live health badge in sidebar
try:
    h = requests.get(f"{API_URL}/health", timeout=2).json()
    st.sidebar.markdown(
        '<span class="status-badge badge-green">● API Online</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"Uptime: {round(h['uptime_seconds'])}s  |  Requests: {h['requests_served']}")
except Exception:
    st.sidebar.markdown(
        '<span class="status-badge badge-red">● API Offline</span>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption("African Leadership University")
st.sidebar.caption("Machine Learning Pipeline")
st.sidebar.caption("Heroine Mutumwinka")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(path: str):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def api_post_file(path: str, file_bytes, filename: str, content_type: str = "image/jpeg"):
    try:
        r = requests.post(
            f"{API_URL}{path}",
            files={"file": (filename, file_bytes, content_type)},
            timeout=30,
        )
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


# ===========================================================================
# HOME PAGE
# ===========================================================================

if page == "🏠 Home":
    st.title("🚀 CIFAR-10 MLOps Dashboard")
    st.markdown("An end-to-end image classification pipeline — predict, upload data, and retrain the model in the browser.")
    st.markdown("---")

    health = api_get("/health")
    stats  = api_get("/stats")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🤖 API Status", "Online ✅" if health else "Offline ❌")
    with c2:
        st.metric("⏱ Uptime", f"{round(health['uptime_seconds'])}s" if health else "—")
    with c3:
        st.metric("🔮 Predictions Made", stats["total_predictions"] if stats else "—")
    with c4:
        st.metric("📦 Datasets Uploaded", stats["total_uploads"] if stats else "—")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("📋 Project Overview")
        st.markdown("""
<div class="dark-card">
<span style='color:#c9d1d9'><b>Dataset:</b> CIFAR-10 — 60,000 32×32 colour images across 10 classes</span><br><br>
<span style='color:#c9d1d9'><b>Model:</b> CNN with Conv → BN → Pool → Dropout → Dense layers</span><br><br>
<span style='color:#8b949e'><b>Pipeline:</b></span>
<ol style='color:#8b949e'>
<li>Train CNN offline on CIFAR-10</li>
<li>Serve predictions via FastAPI</li>
<li>Visualise insights in Streamlit</li>
<li>Accept user uploads for retraining</li>
<li>Retrain with promotion gate</li>
</ol>
</div>
""", unsafe_allow_html=True)

    with col_r:
        st.subheader("🔧 Tech Stack")
        st.markdown("""
<div class="dark-card">
<table style='color:#c9d1d9; border-collapse:collapse; width:100%'>
<tr style='border-bottom:1px solid #21262d'><td style='padding:8px 0'>🧠 <b>ML</b></td><td style='color:#8b949e'>TensorFlow / Keras</td></tr>
<tr style='border-bottom:1px solid #21262d'><td style='padding:8px 0'>⚡ <b>API</b></td><td style='color:#8b949e'>FastAPI + Uvicorn</td></tr>
<tr style='border-bottom:1px solid #21262d'><td style='padding:8px 0'>🎨 <b>UI</b></td><td style='color:#8b949e'>Streamlit</td></tr>
<tr style='border-bottom:1px solid #21262d'><td style='padding:8px 0'>💾 <b>DB</b></td><td style='color:#8b949e'>SQLite</td></tr>
<tr style='border-bottom:1px solid #21262d'><td style='padding:8px 0'>🐳 <b>Deploy</b></td><td style='color:#8b949e'>Docker + nginx + Render</td></tr>
<tr><td style='padding:8px 0'>🦗 <b>Load test</b></td><td style='color:#8b949e'>Locust</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    if health:
        st.markdown("---")
        st.subheader("📡 Live API Health")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Model Version", health.get("model_version", "—"))
        hc2.metric("Avg Latency", f"{health.get('avg_latency_ms', 0):.1f} ms")
        hc3.metric("Requests Served", health.get("requests_served", 0))

    if stats and stats.get("recent_predictions"):
        st.markdown("---")
        st.subheader("🕐 Recent Predictions")
        df = pd.DataFrame(stats["recent_predictions"])
        st.dataframe(df[["image", "prediction", "confidence", "date"]], use_container_width=True)


# ===========================================================================
# PREDICTION PAGE
# ===========================================================================

elif page == "🔍 Predict":
    st.title("🔍 Image Prediction")
    st.markdown("Upload any image and the model will classify it into one of the 10 CIFAR-10 classes.")
    st.markdown("---")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.caption(f"Size: {img.size}  |  Mode: {img.mode}")

            if st.button("🚀 Predict", use_container_width=True):
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)

                with st.spinner("Running inference..."):
                    result = api_post_file("/predict", img_bytes, uploaded.name, "image/png")

                if "error" in result:
                    st.error(f"API error: {result['error']}")
                else:
                    with col_result:
                        conf_pct = result['confidence'] * 100
                        latency  = result['latency_ms']
                        cls_name = result['predicted_class'].upper()
                        st.markdown(f"""
<div class="prediction-result">
  <div style='font-size:13px;color:#4f8ef7;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px'>Prediction</div>
  <h2 style='color:#e6edf3;margin:0;font-size:32px;font-weight:700'>{cls_name}</h2>
  <div style='margin:12px 0;background:rgba(79,142,247,0.15);border-radius:20px;height:6px;overflow:hidden'>
    <div style='background:linear-gradient(90deg,#4f8ef7,#7c6af7);height:100%;width:{conf_pct:.0f}%;border-radius:20px'></div>
  </div>
  <p style='color:#8b949e;margin:4px 0 0 0;font-size:15px'>{conf_pct:.2f}% confidence &nbsp;·&nbsp; {latency:.1f} ms</p>
</div>
""", unsafe_allow_html=True)

                        st.markdown("#### Top-5 Predictions")
                        top5 = result.get("top5", [])
                        if top5:
                            classes = [t["class"] for t in top5]
                            confs   = [t["confidence"] for t in top5]
                            colors  = ["#0B3D91" if i == 0 else "#93b4e8" for i in range(len(top5))]

                            fig, ax = plt.subplots(figsize=(5, 3))
                            bars = ax.barh(classes[::-1], confs[::-1], color=colors[::-1])
                            ax.set_xlabel("Confidence")
                            ax.set_xlim(0, 1)
                            ax.bar_label(bars, labels=[f"{c*100:.1f}%" for c in confs[::-1]], padding=4)
                            ax.spines[["top", "right"]].set_visible(False)
                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        st.success("✅ Prediction saved to database!")
                        st.balloons()


# ===========================================================================
# INSIGHTS PAGE
# ===========================================================================

elif page == "📊 Insights":
    st.title("📊 Dataset Insights")
    st.markdown("Visual analysis of the CIFAR-10 dataset and live prediction activity.")
    st.markdown("---")

    # ── Visualisation 1: Dataset class distribution ──────────────────────
    st.subheader("1️⃣  Dataset Balance — Is CIFAR-10 Fair?")
    st.markdown("""
**Interpretation:** CIFAR-10 is a perfectly balanced dataset. Every class has exactly
**5,000 training images** and **1,000 test images**, giving the model an equal chance to
learn each category without bias. This means accuracy differences across classes come
purely from visual complexity, not data imbalance.
""")

    train_counts = [5000] * 10
    test_counts  = [1000] * 10

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    x = np.arange(len(CIFAR10_CLASSES))
    w = 0.4
    b1 = ax1.bar(x - w/2, train_counts, w, label="Train", color="#0B3D91", alpha=0.85)
    b2 = ax1.bar(x + w/2, test_counts,  w, label="Test",  color="#4C9BE8", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha="right")
    ax1.set_ylabel("Images per class")
    ax1.set_title("CIFAR-10 Class Distribution (Train vs Test)", fontweight="bold")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("---")

    # ── Visualisation 2: Live prediction distribution ─────────────────────
    st.subheader("2️⃣  What Has the Model Been Predicting?")
    st.markdown("""
**Interpretation:** This chart shows the real-world distribution of predictions made by
your deployed model. A uniform spread suggests the model is generalising well across
all classes. A skew toward one class might indicate that class dominates uploaded images,
or the model has a bias worth investigating.
""")

    stats = api_get("/stats")
    if stats and stats.get("total_predictions", 0) > 0:
        recent = stats.get("recent_predictions", [])
        if recent:
            df_pred = pd.DataFrame(recent)
            counts  = df_pred["prediction"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            bars = ax2.bar(counts.index, counts.values, color=CLASS_COLORS[:len(counts)], alpha=0.85)
            ax2.set_ylabel("Number of Predictions")
            ax2.set_title("Prediction Class Distribution (Live from DB)", fontweight="bold")
            ax2.bar_label(bars, padding=3)
            ax2.spines[["top", "right"]].set_visible(False)
            plt.xticks(rotation=30, ha="right")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        st.info("📭 No predictions in the database yet — make some predictions first!")
        # Show a demo chart
        demo_counts = {"airplane": 8, "automobile": 5, "bird": 3, "cat": 7, "deer": 2,
                       "dog": 6, "frog": 4, "horse": 3, "ship": 9, "truck": 4}
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(demo_counts.keys(), demo_counts.values(), color=CLASS_COLORS, alpha=0.7)
        ax2.set_title("Example — Prediction Distribution (Demo Data)", fontweight="bold", style="italic")
        ax2.set_ylabel("Predictions")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30, ha="right")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")

    # ── Visualisation 3: Confidence distribution ──────────────────────────
    st.subheader("3️⃣  Model Confidence — How Sure Is the AI?")
    st.markdown("""
**Interpretation:** Confidence scores reveal how decisive the model is. Scores clustered
near **1.0** mean the model is highly certain — typical for very clear, canonical images.
A spread toward **0.5–0.7** suggests ambiguous or out-of-distribution images. Very low
confidence (< 0.5) is a signal to **distrust the prediction** and consider uploading
that image for retraining.
""")

    # Build histogram from DB data or demo
    recent = (stats or {}).get("recent_predictions", [])
    if recent:
        confs = [row["confidence"] for row in recent]
        label = "Live prediction confidence"
    else:
        # Synthetic demo distribution
        np.random.seed(42)
        confs = list(np.clip(np.random.beta(8, 2, 200), 0, 1))
        label = "Demo confidence distribution"

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    n, bins, patches = ax3.hist(confs, bins=20, color="#0B3D91", alpha=0.8, edgecolor="white")
    # Colour low-confidence bars red
    for patch, left in zip(patches, bins[:-1]):
        if left < 0.5:
            patch.set_facecolor("#E84C4C")
        elif left < 0.75:
            patch.set_facecolor("#E8D44C")
    ax3.axvline(0.5,  color="#E84C4C", linestyle="--", linewidth=1.5, label="Low confidence (<0.5)")
    ax3.axvline(0.75, color="#E8D44C", linestyle="--", linewidth=1.5, label="Medium confidence (<0.75)")
    ax3.set_xlabel("Confidence Score")
    ax3.set_ylabel("Count")
    ax3.set_title(f"Confidence Distribution — {label}", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.spines[["top", "right"]].set_visible(False)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)


# ===========================================================================
# RETRAIN PAGE
# ===========================================================================

elif page == "🔄 Retrain":
    st.title("🔄 Model Retraining")
    st.markdown("Upload new labelled images and trigger retraining. The model is only promoted if accuracy holds.")
    st.markdown("---")

    # ── Upload section ────────────────────────────────────────────────────
    st.subheader("📦 Step 1 — Upload Training Data")
    st.markdown("Select the correct class and upload an image to add it to the training dataset.")

    col1, col2 = st.columns(2)
    with col1:
        selected_class = st.selectbox("Select Class Label", CIFAR10_CLASSES)
    with col2:
        uploaded_img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_img:
        st.image(uploaded_img, caption=f"To be labelled as: {selected_class.upper()}", use_container_width=True)
        if st.button("📤 Upload Image", use_container_width=True):
            img_bytes = uploaded_img.getvalue()
            with st.spinner("Uploading image..."):
                try:
                    r = requests.post(
                        f"{API_URL}/upload-image",
                        data={"label": selected_class},
                        files={"file": (uploaded_img.name, img_bytes, "image/jpeg")},
                        timeout=10,
                    )
                    result = r.json() if r.ok else {"error": r.text}
                except Exception as e:
                    result = {"error": str(e)}

            if "error" in result:
                st.error(f"Upload failed: {result['error']}")
            else:
                st.success(f"✅ {result['message']}")
                st.info(f"Total uploaded images for **{selected_class}**: {result.get('class_total', '?')}")

    st.markdown("---")

    # ── Retrain trigger ───────────────────────────────────────────────────
    st.subheader("🚀 Step 2 — Trigger Retraining")
    st.markdown("Retraining runs in the background. A **promotion gate** ensures the new model is only saved if accuracy does not drop by more than 1%.")

    status = api_get("/retrain/status") or {}
    is_running = status.get("is_running", False)

    if is_running:
        st.warning("⏳ Retraining is currently running... Refresh this page to check progress.")
    else:
        if st.button("🧠 Start Retraining", use_container_width=True):
            result = requests.post(f"{API_URL}/retrain", timeout=10)
            if result.ok:
                st.success(f"✅ {result.json()['message']}")
                st.info("Refresh this page to track progress via the history table below.")
            else:
                st.error(f"Error: {result.json().get('detail', 'Unknown error')}")

    st.markdown("---")

    # ── Run history ───────────────────────────────────────────────────────
    st.subheader("📜 Retraining History")

    history = status.get("history", [])
    if history:
        df_hist = pd.DataFrame(history)
        df_hist["result"] = df_hist["status"].apply(
            lambda s: "✅ Promoted" if s == "promoted" else "❌ Rejected"
        )
        if "accuracy_before" in df_hist.columns:
            df_hist["Δ accuracy"] = (
                df_hist["accuracy_after"] - df_hist["accuracy_before"]
            ).map(lambda x: f"+{x:.4f}" if x >= 0 else f"{x:.4f}")
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No retraining runs yet.")


# ===========================================================================
# HISTORY PAGE
# ===========================================================================

elif page == "📋 History":
    st.title("📋 Prediction & Upload History")
    st.markdown("---")

    # ── Prediction history ────────────────────────────────────────────────
    st.subheader("🔮 Prediction Log")

    stats = api_get("/stats")
    preds = (stats or {}).get("recent_predictions", [])

    if preds:
        df = pd.DataFrame(preds)
        df["confidence"] = df["confidence"].apply(lambda c: f"{c*100:.2f}%")
        st.dataframe(df.rename(columns={
            "id": "ID", "image": "Image", "prediction": "Class",
            "confidence": "Confidence", "date": "Timestamp",
        }), use_container_width=True)

        # Mini bar chart
        counts = pd.DataFrame(preds)["prediction"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(counts.index, counts.values, color="#0B3D91", alpha=0.85)
        ax.set_title("Prediction Distribution", fontweight="bold")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No predictions recorded yet.")

    st.markdown("---")

    # ── Upload history ────────────────────────────────────────────────────
    st.subheader("📦 Upload Log")

    retrain_status = api_get("/retrain/status") or {}
    history = retrain_status.get("history", [])

    if history:
        df_rt = pd.DataFrame(history)
        st.dataframe(df_rt, use_container_width=True)
    else:
        st.info("No retraining runs recorded yet.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6b7280; font-size:13px'>"
    "🚀 <b>CIFAR-10 MLOps Dashboard</b> · Developed by <b>Heroine Mutumwinka</b> · "
    "African Leadership University · Machine Learning Pipeline Summative"
    "</div>",
    unsafe_allow_html=True,
)