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
        --forest-deep: #0a120e;
        --forest-card: #1b2a22;
        --gold: #c89b4a;
        --gold-bright: #e8c272;
        --line: rgba(200, 155, 74, 0.22);
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* -------- Ambient background -------- */
    .stApp {
        background:
            radial-gradient(ellipse 70% 45% at 18% 8%, rgba(200,155,74,0.13), transparent 60%),
            radial-gradient(ellipse 60% 40% at 85% 95%, rgba(200,155,74,0.08), transparent 60%),
            linear-gradient(180deg, var(--forest-deep) 0%, var(--forest) 100%);
        color: var(--ink);
        overflow: hidden;
    }

    /* Compress Streamlit's default spacing so everything fits one screen */
    .block-container {
        max-width: 600px;
        padding-top: 1.4rem !important;
        padding-bottom: 1rem !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div.element-container) {
        gap: 0 !important;
    }
    .element-container { margin-bottom: 0 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { gap: 0 !important; }

    /* Floating ambient specks (pure decoration, signature atmosphere) */
    .motes span {
        position: fixed;
        width: 3px;
        height: 3px;
        border-radius: 50%;
        background: var(--gold);
        opacity: 0;
        animation: drift 9s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    .motes span:nth-child(1) { left: 8%;  top: 70%; animation-delay: 0s;   }
    .motes span:nth-child(2) { left: 88%; top: 20%; animation-delay: 1.8s; }
    .motes span:nth-child(3) { left: 15%; top: 25%; animation-delay: 3.4s; }
    .motes span:nth-child(4) { left: 92%; top: 65%; animation-delay: 5s;   }
    .motes span:nth-child(5) { left: 50%; top: 88%; animation-delay: 2.5s; }
    @keyframes drift {
        0%   { opacity: 0; transform: translateY(0) translateX(0); }
        20%  { opacity: 0.6; }
        50%  { opacity: 0.35; transform: translateY(-26px) translateX(10px); }
        80%  { opacity: 0.5; }
        100% { opacity: 0; transform: translateY(-55px) translateX(-6px); }
    }

    /* -------- Signature: animated butterfly mark -------- */
    .butterfly-mark {
        display: flex;
        justify-content: center;
        margin-bottom: 0.3rem;
        opacity: 0;
        animation: fadeUp 0.7s ease-out 0.05s forwards;
    }
    .butterfly-mark svg { width: 46px; height: 46px; overflow: visible; }
    .wing-right {
        transform-origin: 32px 32px;
        animation: flapRight 2.6s ease-in-out infinite;
    }
    .wing-left {
        transform-origin: 32px 32px;
        transform: scaleX(-1) translateX(-64px);
        animation: flapLeft 2.6s ease-in-out infinite;
    }
    @keyframes flapRight {
        0%, 100% { transform: rotateY(0deg) scaleX(1); }
        50% { transform: rotateY(50deg) scaleX(0.78); }
    }
    @keyframes flapLeft {
        0%, 100% { transform: scaleX(-1) translateX(-64px) rotateY(0deg) scaleX(1); }
        50% { transform: scaleX(-1) translateX(-64px) rotateY(50deg) scaleX(0.78); }
    }

    /* -------- Hero -------- */
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: var(--gold);
        text-align: center;
        margin-bottom: 0.5rem;
        opacity: 0;
        animation: fadeUp 0.7s ease-out 0.12s forwards;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 500;
        font-size: 2.05rem;
        line-height: 1.05;
        text-align: center;
        color: var(--ink);
        margin: 0 0 0.4rem 0;
        opacity: 0;
        animation: fadeUp 0.8s ease-out 0.2s forwards;
    }

    .hero-title em {
        font-style: italic;
        background: linear-gradient(90deg, var(--gold-bright), var(--gold));
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-sub {
        text-align: center;
        color: var(--ink-dim);
        font-size: 0.84rem;
        max-width: 380px;
        margin: 0 auto 1rem auto;
        line-height: 1.5;
        opacity: 0;
        animation: fadeUp 0.8s ease-out 0.3s forwards;
    }

    .hero-rule {
        width: 40px;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--gold), transparent);
        margin: 0 auto 1rem auto;
        opacity: 0;
        animation: fadeUp 0.8s ease-out 0.38s forwards;
        position: relative;
    }
    .hero-rule::after {
        content: "✦";
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        color: var(--gold);
        font-size: 0.6rem;
        background: var(--forest);
        padding: 0 6px;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* -------- Specimen card (uploader) -------- */
    .specimen-frame {
        position: relative;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(160deg, var(--forest-card), rgba(27,42,34,0.5));
        padding: 1.1rem 1.1rem 0.8rem 1.1rem;
        margin-bottom: 0.7rem;
        opacity: 0;
        animation: fadeUp 0.8s ease-out 0.48s forwards;
        transition: border-color 0.4s ease, box-shadow 0.4s ease;
    }
    .specimen-frame:hover {
        border-color: rgba(200,155,74,0.4);
        box-shadow: 0 0 32px rgba(200,155,74,0.06);
    }

    .specimen-frame::before,
    .specimen-frame::after {
        content: "";
        position: absolute;
        width: 12px;
        height: 12px;
        border: 1px solid var(--gold);
        opacity: 0.55;
        transition: opacity 0.3s ease;
    }
    .specimen-frame::before { top: 8px; left: 8px; border-right: none; border-bottom: none; }
    .specimen-frame::after { bottom: 8px; right: 8px; border-left: none; border-top: none; }
    .specimen-frame:hover::before, .specimen-frame:hover::after { opacity: 1; }

    .specimen-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ink-dim);
        text-align: center;
        margin-bottom: 0.6rem;
    }

    /* -------- Tabs (Upload / URL) -------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.3rem;
        background: rgba(10, 18, 14, 0.5);
        padding: 0.25rem;
        border-radius: 6px;
        border: 1px solid var(--line);
        margin-bottom: 0.6rem;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-dim);
        background: transparent;
        border-radius: 4px;
        padding: 0.35rem 0.6rem;
        height: auto;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(200, 155, 74, 0.16) !important;
        color: var(--gold-bright) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs { margin-bottom: -0.4rem; }

    [data-testid="stTextInput"] input {
        background: rgba(10, 18, 14, 0.45);
        border: 1px dashed var(--line);
        border-radius: 4px;
        color: var(--ink);
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        padding: 0.5rem 0.7rem;
        transition: border-color 0.25s ease;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: var(--gold);
        box-shadow: 0 0 0 1px var(--gold);
    }

    [data-testid="stFileUploader"] {
        border: 1px dashed var(--line);
        border-radius: 4px;
        padding: 0.3rem;
        background: rgba(10, 18, 14, 0.4);
        transition: border-color 0.3s ease, background 0.3s ease, transform 0.2s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--gold);
        background: rgba(200,155,74,0.06);
    }
    [data-testid="stFileUploaderDropzone"] { background: transparent; min-height: 0 !important; }
    [data-testid="stFileUploader"] section { background: transparent; border: none; padding: 0.5rem !important; }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span {
        color: var(--ink-dim) !important;
        font-size: 0.74rem !important;
    }
    [data-testid="stFileUploader"] button {
        font-size: 0.74rem !important;
        padding: 0.3rem 0.7rem !important;
    }

    /* Uploaded image styling — kept small so layout stays in one screen */
    [data-testid="stImage"] img {
        border-radius: 6px;
        border: 1px solid var(--line);
        max-height: 130px;
        object-fit: cover;
        width: 100%;
        animation: revealImg 0.55s cubic-bezier(0.16, 1, 0.3, 1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stImage"] img:hover {
        transform: scale(1.015);
        box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    }
    @keyframes revealImg {
        from { opacity: 0; transform: scale(0.96); filter: blur(4px); }
        to { opacity: 1; transform: scale(1); filter: blur(0); }
    }
    [data-testid="stImageCaption"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.62rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-dim) !important;
        text-align: center;
    }

    /* -------- Predict button -------- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--gold-bright), #a87c34);
        background-size: 200% 200%;
        color: #1a1208;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.84rem;
        letter-spacing: 0.02em;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin-top: 0.3rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, background-position 0.5s ease;
        box-shadow: 0 4px 18px rgba(200,155,74,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        background-position: 100% 50%;
        box-shadow: 0 6px 22px rgba(200,155,74,0.32);
    }
    .stButton > button:active { transform: translateY(0px) scale(0.99); }

    /* -------- Result card -------- */
    .result-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(160deg, var(--forest-card), rgba(27,42,34,0.4));
        padding: 0.9rem 1.1rem;
        margin-top: 0.6rem;
        animation: resultIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent 30%, rgba(200,155,74,0.08) 50%, transparent 70%);
        background-size: 200% 100%;
        animation: shimmer 2.2s ease-out 0.3s 1;
    }
    @keyframes shimmer {
        from { background-position: 150% 0; }
        to { background-position: -50% 0; }
    }
    @keyframes resultIn {
        from { opacity: 0; transform: translateY(8px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .result-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .result-eyebrow::before {
        content: "";
        width: 5px; height: 5px;
        border-radius: 50%;
        background: var(--gold-bright);
        box-shadow: 0 0 8px var(--gold-bright);
        animation: pulse 1.6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }

    .result-species {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.35rem;
        color: var(--ink);
        line-height: 1.15;
        margin-bottom: 0.6rem;
        text-transform: capitalize;
    }

    .confidence-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.62rem;
        color: var(--ink-dim);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .confidence-value { color: var(--gold-bright); font-size: 0.78rem; }

    .confidence-track {
        width: 100%;
        height: 4px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #a87c34, var(--gold-bright));
        border-radius: 3px;
        width: 0%;
        animation: fillBar 1.1s cubic-bezier(0.16, 1, 0.3, 1) 0.2s forwards;
        position: relative;
    }
    @keyframes fillBar { to { width: var(--target-width); } }

    /* Footer */
    .footer-note {
        text-align: center;
        color: var(--ink-dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.58rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.8rem;
        opacity: 0.45;
    }

    @media (max-height: 750px) {
        .hero-title { font-size: 1.7rem; }
        .hero-sub { display: none; }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Ambient floating motes (decorative, behind everything)
# -------------------------------
st.markdown("""
<div class="motes"><span></span><span></span><span></span><span></span><span></span></div>
""", unsafe_allow_html=True)

# -------------------------------
# Signature mark + Hero section
# -------------------------------
st.markdown("""
<div class="butterfly-mark">
    <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
        <g class="wing-right">
            <path d="M32 30 C 44 6, 62 10, 58 26 C 55 38, 40 36, 32 30 Z" fill="#c89b4a" opacity="0.9"/>
            <path d="M32 34 C 42 44, 56 46, 52 56 C 48 64, 36 56, 32 34 Z" fill="#e8c272" opacity="0.85"/>
        </g>
        <g class="wing-left">
            <path d="M32 30 C 44 6, 62 10, 58 26 C 55 38, 40 36, 32 30 Z" fill="#c89b4a" opacity="0.9"/>
            <path d="M32 34 C 42 44, 56 46, 52 56 C 48 64, 36 56, 32 34 Z" fill="#e8c272" opacity="0.85"/>
        </g>
        <line x1="32" y1="22" x2="32" y2="46" stroke="#f4efe4" stroke-width="2" stroke-linecap="round"/>
        <circle cx="32" cy="22" r="2.4" fill="#f4efe4"/>
    </svg>
</div>
<div class="hero-eyebrow">Field Identification Tool</div>
<h1 class="hero-title">Lepidoptera <em>Classifier</em></h1>
<p class="hero-sub">Identify a butterfly's species instantly — trained on 75 species with EfficientNetB0.</p>
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