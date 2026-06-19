import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Lepidoptera AI",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Modern Glassmorphism CSS
# -------------------------------
st.markdown("""
<style>

/* Animated background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0b1220, #111827);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass card */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
}

/* Header */
.header {
    text-align: center;
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    box-shadow: 0 10px 40px rgba(99,102,241,0.4);
    animation: glow 3s infinite alternate;
}

@keyframes glow {
    from {box-shadow: 0 0 20px #6366f1;}
    to {box-shadow: 0 0 40px #a855f7;}
}

.header h1 {
    font-size: 3rem;
    margin: 0;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #22c55e, #06b6d4);
    color: white;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 10px 25px rgba(34,197,94,0.4);
}

/* Image styling */
img {
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
}

/* Result box */
.result {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid #22c55e;
    padding: 20px;
    border-radius: 15px;
    animation: pop 0.5s ease;
}

@keyframes pop {
    from {transform: scale(0.9); opacity: 0;}
    to {transform: scale(1); opacity: 1;}
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/Effnet_model.keras")
    encoder = joblib.load("models/label_encoder.joblib")
    return model, encoder

model, encoder = load_model()

# -------------------------------
# Header
# -------------------------------
st.markdown("""
<div class="header">
    <h1>🦋 Lepidoptera AI</h1>
    <p>Next-Gen Butterfly Species Classifier</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Butterfly Image")

    uploaded_file = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔬 AI Prediction")

    if uploaded_file:
        if st.button("Predict Now 🚀"):
            with st.spinner("AI is analyzing wings pattern..."):
                time.sleep(1)

                img_resized = img.resize((224, 224))
                arr = image.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0)
                arr = preprocess_input(arr)

                pred = model.predict(arr)
                idx = np.argmax(pred)
                label = encoder.inverse_transform([idx])[0]
                conf = np.max(pred)

                st.markdown(f"""
                <div class="result">
                    <h2>🦋 {label}</h2>
                    <h4>Confidence: {conf:.2%}</h4>
                </div>
                """, unsafe_allow_html=True)

                st.success("Prediction completed!")
    else:
        st.info("Upload an image first")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<center>✨ Built with Streamlit + EfficientNet</center>", unsafe_allow_html=True)