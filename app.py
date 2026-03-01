import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

MODEL_PATH = "retinopathy_model.keras"
FILE_ID = "1ff9RWXpabyTPGC5F7AMj3qakCgH_dICF"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retinal image to predict DR stage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
