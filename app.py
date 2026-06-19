# app.py
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ── Page config ──
st.set_page_config(page_title="Butterfly Classifier 🦋", layout="centered")

# ── Load model & encoder ──
@st.cache_resource
def load_artifacts():
    model = load_model("models/Effnet_model.keras")
    le = joblib.load("models/label_encoder.joblib")
    return model, le

model, label_encoder = load_artifacts()

# ── UI ──
st.title("🦋 Butterfly Species Classifier")
st.write("Upload a butterfly image — model will predict the species.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # EfficientNet preprocessing

    # Predict
    preds = model.predict(img_array, verbose=0)
    top3_idx = np.argsort(preds[0])[::-1][:3]

    st.markdown("---")
    st.subheader("Prediction Results")

    # Top prediction
    top_class = label_encoder.inverse_transform([top3_idx[0]])[0]
    top_conf = preds[0][top3_idx[0]]
    st.success(f"**{top_class}** — {top_conf:.2%} confidence")

    # Top 3
    st.markdown("**Top 3 Predictions:**")
    for i, idx in enumerate(top3_idx):
        name = label_encoder.inverse_transform([idx])[0]
        conf = preds[0][idx]
        st.write(f"{i+1}. {name} — {conf:.2%}")