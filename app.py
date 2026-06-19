# Import Streamlit for web app UI
import streamlit as st

# Import required libraries for image handling and prediction
import numpy as np  # numerical operations
import joblib  # load label encoder
import tensorflow as tf  # load trained model
from tensorflow.keras.preprocessing import image  # image loading utility
from tensorflow.keras.applications.efficientnet import preprocess_input  # preprocessing

# -------------------------------
# Load trained model and encoder
# -------------------------------

# Load the trained EfficientNet model
model = tf.keras.models.load_model("models/Effnet_model.keras")

# Load the label encoder (class mapping)
label_encoder = joblib.load("models/label_encoder.joblib")

# -------------------------------
# Streamlit UI setup
# -------------------------------

# App title
st.title("🦋 Butterfly Classification App")

# App description
st.write("Upload a butterfly image and the model will predict its species.")

# File uploader widget
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

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
# Run prediction when image uploaded
# -------------------------------

if uploaded_file is not None:

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert uploaded file to PIL image
    img = image.load_img(uploaded_file)

    # Predict button
    if st.button("Predict Species"):

        # Get prediction
        label, confidence = predict_butterfly(img)

        # Show result
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2%}")