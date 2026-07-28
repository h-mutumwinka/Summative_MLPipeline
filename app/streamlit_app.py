import streamlit as st
from PIL import Image
import pandas as pd

from src.api import predict, retrain
from src.database import (
    get_all_predictions,
    get_all_uploads
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="CIFAR-10 MLOps Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background-color:white;
}

section[data-testid="stSidebar"]{
    background-color:#0B3D91;
}

section[data-testid="stSidebar"] *{
    color:white;
}

h1,h2,h3{
    color:#0B3D91;
}

div[data-testid="metric-container"]{
    background-color:white;
    border:2px solid #0B3D91;
    padding:20px;
    border-radius:12px;
    box-shadow:2px 2px 8px rgba(0,0,0,0.15);
}

.stButton>button{
    background-color:#0B3D91;
    color:white;
    border-radius:8px;
    border:none;
    font-weight:bold;
}

.stButton>button:hover{
    background-color:#1357c5;
}

hr{
    border:1px solid #dcdcdc;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🚀 CIFAR-10 MLOps")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Prediction",
        "🔄 Retraining",
        "📊 History"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Model Status: Active")

st.sidebar.info("Version: v1.0")

st.sidebar.markdown("---")

st.sidebar.write("African Leadership University")
st.sidebar.write("Machine Learning Pipeline")
st.sidebar.write("Heroine Mutumwinka")

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":

    st.title("🚀 CIFAR-10 MLOps Dashboard")

    st.write(
        """
Welcome Back 
Murakaza Neza!

This application demonstrates a complete Machine Learning Operations
(MLOps) pipeline for the CIFAR-10 image classification dataset.

The system allows you to:

- Predict image classes
- Upload new datasets
- Retrain the model
- Store records in SQLite
- Monitor prediction history
"""
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🤖 Model",
            "Active"
        )

    with col2:
        st.metric(
            "📷 Classes",
            "10"
        )

    with col3:
        st.metric(
            "💾 Database",
            "Connected"
        )

    with col4:
        st.metric(
            "🔄 Retraining",
            "Enabled"
        )

    st.markdown("---")

    st.subheader("Project Overview")

    st.info("""
This project was developed using:

• TensorFlow / Keras

• Streamlit

• SQLite

• Python

• CNN (Convolutional Neural Network)

Dataset:

CIFAR-10
""")

    st.markdown("---")

    st.subheader("Pipeline")

    st.write("""
1. Load Dataset

2. Preprocess Images

3. Train CNN

4. Save Model

5. Predict Images

6. Upload New Dataset

7. Retrain Model

8. Save Prediction History
""")

    st.success("System Ready ✅")
# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🔍 Prediction":

    st.title("🔍 Image Prediction")

    st.write("Upload an image from one of the CIFAR-10 classes.")

    uploaded_image = st.file_uploader(
        "Choose an Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded Image",
            width=300
        )

        if st.button("🚀 Predict"):

            # Save temporarily
            temp_path = "temp_prediction_image.png"

            image.save(temp_path)

            with st.spinner("Making prediction..."):

                result = predict(temp_path)

            st.success("Prediction Completed!")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Class",
                    result["class"].capitalize()
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{result['confidence']*100:.2f}%"
                )

            st.balloons()

# =====================================================
# RETRAINING PAGE
# =====================================================

elif page == "🔄 Retraining":

    st.title("🔄 Model Retraining")

    st.write(
        """
Upload a ZIP file containing a new training dataset.

The system will:

✅ Save upload information to SQLite

✅ Extract the dataset

✅ Preprocess the images

✅ Retrain the CNN

✅ Save the updated model
"""
    )

    uploaded_zip = st.file_uploader(
        "Upload Dataset (.zip)",
        type=["zip"]
    )

    if uploaded_zip is not None:

        st.success(f"Selected: {uploaded_zip.name}")

        if st.button("Start Retraining"):

            zip_path = uploaded_zip.name

            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())

            progress = st.progress(0)

            status = st.empty()

            status.write("Saving uploaded dataset...")

            progress.progress(20)

            status.write("Extracting dataset...")

            progress.progress(40)

            status.write("Preprocessing images...")

            progress.progress(60)

            status.write("Retraining model...")

            retrain(zip_path)

            progress.progress(90)

            status.write("Saving updated model...")

            progress.progress(100)

            st.success("🎉 The Model was retrained successfully!")

            st.balloons()
    # =====================================================
# HISTORY PAGE
# =====================================================

elif page == "📊 History":

    st.title("📊 Prediction & Upload History")

    st.markdown("### 🖼 Prediction History")

    predictions = get_all_predictions()

    if predictions:

        prediction_df = pd.DataFrame(
            predictions,
            columns=[
                "ID",
                "Image",
                "Prediction",
                "Confidence",
                "Date"
            ]
        )

        st.dataframe(
            prediction_df,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("Prediction Distribution")

        chart = prediction_df["Prediction"].value_counts()

        st.bar_chart(chart)

    else:

        st.info("No prediction history found.")

    st.markdown("---")

    st.subheader("📁 Uploaded Datasets")

    uploads = get_all_uploads()

    if uploads:

        upload_df = pd.DataFrame(
            uploads,
            columns=[
                "ID",
                "Filename",
                "Upload Date",
                "Status",
                "Model Version"
            ]
        )

        st.dataframe(
            upload_df,
            use_container_width=True
        )

    else:

        st.info("No uploaded datasets found.")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
<div style='text-align:center; color:gray;'>

<h4 style='color:#0B3D91;'>🚀 CIFAR-10 MLOps Dashboard</h4>

Developed by <b>Heroine Mutumwinka</b><br>

African Leadership University Year 3<br>

Machine Learning Pipeline Summative Assignment

</div>
""",
    unsafe_allow_html=True
)
            