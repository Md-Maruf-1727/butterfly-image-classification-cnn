# Import Streamlit for web app UI
import streamlit as st
# Import required libraries for image handling and prediction
import numpy as np  # numerical operations
import joblib  # load label encoder
import tensorflow as tf  # load trained model
from tensorflow.keras.preprocessing import image  # image loading utility
from tensorflow.keras.applications.efficientnet import preprocess_input  # preprocessing
import requests  # fetch image bytes from a URL
from io import BytesIO  # wrap downloaded bytes as a file-like object
from PIL import Image as PILImage  # open image bytes from URL

# -------------------------------
# Page config (must be first Streamlit call)
# -------------------------------
st.set_page_config(
    page_title="Lepidoptera | Species Identification",
    page_icon="🦋",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# Load trained model and encoder
# -------------------------------
# Load the trained EfficientNet model
model = tf.keras.models.load_model("models/Effnet_model.keras")
# Load the label encoder (class mapping)
label_encoder = joblib.load("models/label_encoder.joblib")

# -------------------------------
# Custom CSS — design system
# -------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --ink: #f4efe4;
        --ink-dim: #b8b09c;
        --forest: #14201a;
        --forest-deep: #0c1410;
        --forest-card: #1b2a22;
        --gold: #c89b4a;
        --gold-bright: #e0b65f;
        --line: rgba(200, 155, 74, 0.22);
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(200,155,74,0.10), transparent),
            linear-gradient(180deg, var(--forest-deep) 0%, var(--forest) 100%);
        color: var(--ink);
    }

    .block-container {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    /* -------- Hero -------- */
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--gold);
        text-align: center;
        margin-bottom: 0.9rem;
        opacity: 0;
        animation: fadeUp 0.7s ease-out 0.05s forwards;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 500;
        font-size: 3rem;
        line-height: 1.08;
        text-align: center;
        color: var(--ink);
        margin: 0 0 0.6rem 0;
        opacity: 0;
        animation: fadeUp 0.8s ease-out 0.15s forwards;
    }

    .hero-title em {
        font-style: italic;
        color: var(--gold-bright);
    }

    .hero-sub {
        text-align: center;
        color: var(--ink-dim);
        font-size: 1.02rem;
        max-width: 460px;
        margin: 0 auto 2.6rem auto;
        line-height: 1.6;
        opacity: 0;
        animation: fadeUp 0.8s ease-out 0.28s forwards;
    }

    .hero-rule {
        width: 56px;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
        margin: 0 auto 2.6rem auto;
        opacity: 0;
        animation: fadeUp 0.8s ease-out 0.4s forwards;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* -------- Specimen card (uploader) -------- */
    .specimen-frame {
        position: relative;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: linear-gradient(160deg, var(--forest-card), rgba(27,42,34,0.5));
        padding: 2.2rem 2rem 1.6rem 2rem;
        margin-bottom: 1.6rem;
        opacity: 0;
        animation: fadeUp 0.8s ease-out 0.5s forwards;
    }

    .specimen-frame::before,
    .specimen-frame::after {
        content: "";
        position: absolute;
        width: 14px;
        height: 14px;
        border: 1px solid var(--gold);
        opacity: 0.55;
    }
    .specimen-frame::before { top: 10px; left: 10px; border-right: none; border-bottom: none; }
    .specimen-frame::after { bottom: 10px; right: 10px; border-left: none; border-top: none; }

    .specimen-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--ink-dim);
        text-align: center;
        margin-bottom: 1.1rem;
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed var(--line);
        border-radius: 4px;
        padding: 0.4rem;
        background: rgba(12, 20, 16, 0.35);
        transition: border-color 0.25s ease, background 0.25s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--gold);
        background: rgba(200,155,74,0.05);
    }
    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
    }
    [data-testid="stFileUploader"] section {
        background: transparent;
        border: none;
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: var(--ink-dim) !important;
    }

    /* Uploaded image styling */
    [data-testid="stImage"] img {
        border-radius: 4px;
        border: 1px solid var(--line);
        animation: revealImg 0.6s ease-out;
    }
    @keyframes revealImg {
        from { opacity: 0; transform: scale(0.97); }
        to { opacity: 1; transform: scale(1); }
    }

    /* -------- Predict button -------- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--gold), #a87c34);
        color: #1a1208;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.92rem;
        letter-spacing: 0.02em;
        border: none;
        border-radius: 4px;
        padding: 0.72rem 1rem;
        margin-top: 0.6rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        box-shadow: 0 4px 18px rgba(200,155,74,0.18);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.07);
        box-shadow: 0 6px 24px rgba(200,155,74,0.3);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* -------- Result card -------- */
    .result-card {
        border: 1px solid var(--line);
        border-radius: 6px;
        background: linear-gradient(160deg, var(--forest-card), rgba(27,42,34,0.4));
        padding: 1.8rem 1.9rem;
        margin-top: 1.4rem;
        animation: resultIn 0.55s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes resultIn {
        from { opacity: 0; transform: translateY(10px) scale(0.985); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .result-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.5rem;
    }

    .result-species {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.85rem;
        color: var(--ink);
        line-height: 1.2;
        margin-bottom: 1.1rem;
        text-transform: capitalize;
    }

    .confidence-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: var(--ink-dim);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .confidence-value {
        color: var(--gold-bright);
        font-size: 0.9rem;
    }

    .confidence-track {
        width: 100%;
        height: 5px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #a87c34, var(--gold-bright));
        border-radius: 3px;
        width: 0%;
        animation: fillBar 1.1s cubic-bezier(0.16, 1, 0.3, 1) 0.15s forwards;
    }

    @keyframes fillBar {
        to { width: var(--target-width); }
    }

    /* -------- Tabs (Upload / URL) -------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: rgba(12, 20, 16, 0.4);
        padding: 0.3rem;
        border-radius: 6px;
        border: 1px solid var(--line);
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-dim);
        background: transparent;
        border-radius: 4px;
        padding: 0.5rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(200, 155, 74, 0.14) !important;
        color: var(--gold-bright) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent;
    }

    [data-testid="stTextInput"] input {
        background: rgba(12, 20, 16, 0.35);
        border: 1px dashed var(--line);
        border-radius: 4px;
        color: var(--ink);
        font-family: 'Inter', sans-serif;
        padding: 0.6rem 0.8rem;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: var(--gold);
        box-shadow: none;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        color: var(--ink-dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 3rem;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Hero section
# -------------------------------
st.markdown("""
<div class="hero-eyebrow">Field Identification Tool</div>
<h1 class="hero-title">Lepidoptera<br><em>Species Classifier</em></h1>
<p class="hero-sub">Upload a photograph of a butterfly and the model will identify its species from 75 known classifications, trained on EfficientNetB0.</p>
<div class="hero-rule"></div>
""", unsafe_allow_html=True)

# -------------------------------
# Upload card — Upload File or Paste URL
# -------------------------------
st.markdown('<div class="specimen-frame">', unsafe_allow_html=True)
st.markdown('<div class="specimen-label">Specimen Input — Upload or Paste a Link</div>', unsafe_allow_html=True)

# Two tabs: one for local file upload, one for pasting an image URL
tab_upload, tab_url = st.tabs(["Upload Image", "Paste URL"])

# This will hold a PIL image regardless of which tab is used
pil_image = None

with tab_upload:
    # File uploader widget (same as before)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        pil_image = image.load_img(uploaded_file)

with tab_url:
    # Text input for pasting an online image link
    image_url = st.text_input("Image URL", placeholder="https://example.com/butterfly.jpg", label_visibility="collapsed")
    if image_url:
        try:
            # Download the image bytes from the given URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            # Open the downloaded bytes as a PIL image and ensure RGB mode
            pil_image = PILImage.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            st.error("Couldn't load image from that URL. Please check the link and try again.")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Prediction function
# -------------------------------
def predict_butterfly(img):
    # Resize image to model input size (224x224)
    img = img.resize((224, 224))
    # Convert image to array
    img_array = image.img_to_array(img)
    # Add batch dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Apply EfficientNet preprocessing
    img_array = preprocess_input(img_array)
    # Predict using model
    prediction = model.predict(img_array)
    # Get highest probability class index
    predicted_class = np.argmax(prediction)
    # Convert index to class name
    class_name = label_encoder.inverse_transform([predicted_class])[0]
    # Get confidence score
    confidence = np.max(prediction)
    return class_name, confidence

# -------------------------------
# Run prediction when an image is available (from upload or URL)
# -------------------------------
if pil_image is not None:
    # Display the image (works for both upload and URL cases)
    st.image(pil_image, caption="Uploaded specimen", use_column_width=True)

    # Predict button
    if st.button("Identify Species"):
        with st.spinner("Analyzing wing patterns…"):
            # Get prediction
            label, confidence = predict_butterfly(pil_image)

        conf_pct = confidence * 100

        # Show result as a styled specimen result card
        st.markdown(f"""
        <div class="result-card">
            <div class="result-eyebrow">Identification Result</div>
            <div class="result-species">{label.title()}</div>
            <div class="confidence-row">
                <span>Model Confidence</span>
                <span class="confidence-value">{conf_pct:.1f}%</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="--target-width: {conf_pct}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer-note">EfficientNetB0 · 75 Species · Trained Classifier</div>', unsafe_allow_html=True)