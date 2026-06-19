# Import Streamlit for web app UI
import streamlit as st

# Import required libraries for image handling and prediction
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------
# Page Configuration (UI ONLY)
# -------------------------------
st.set_page_config(
    page_title="Butterfly Classifier",
    page_icon="🦋",
    layout="centered"
)

# -------------------------------
# Custom CSS Styling (ONLY DESIGN)
# -------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
    animation: fadeIn 1.5s ease-in-out;
}

/* Subtitle */
.sub-title {
    text-align: center;
    color: #cfd8dc;
    font-size: 18px;
    margin-bottom: 25px;
}

/* Upload box */
.upload-box {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 20px rgba(0,0,0,0.3);
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 16px;
    padding: 10px 20px;
    border-radius: 10px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 15px #00c6ff;
}

/* Fade animation */
@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(-20px);}
    100% {opacity: 1; transform: translateY(0);}
}

.result-box {
    background: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 12px;
    margin-top: 15px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model and Encoder
# -------------------------------
model = tf.keras.models.load_model("models/Effnet_model.keras")
label_encoder = joblib.load("models/label_encoder.joblib")

# -------------------------------
# UI Header
# -------------------------------
st.markdown('<div class="main-title">🦋 Butterfly Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image and discover the butterfly species</div>', unsafe_allow_html=True)

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Butterfly Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# Prediction Function (UNCHANGED LOGIC)
# -------------------------------
def predict_butterfly(img):

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    class_name = label_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction)

    return class_name, confidence

# -------------------------------
# Output Section
# -------------------------------
if uploaded_file is not None:

    st.image(uploaded_file, caption="Uploaded Butterfly", use_column_width=True)

    img = image.load_img(uploaded_file)

    if st.button("✨ Predict Species"):

        label, confidence = predict_butterfly(img)

        st.markdown(f"""
        <div class="result-box">
            <h3>🧬 Prediction Result</h3>
            <p><b>Species:</b> {label}</p>
            <p><b>Confidence:</b> {confidence:.2%}</p>
        </div>
        """, unsafe_allow_html=True)